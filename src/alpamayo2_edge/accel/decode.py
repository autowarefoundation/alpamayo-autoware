# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt prefill and CoT decode over a static flash-layout KV cache.

``DecodeAccelerator`` replaces ``Alpamayo2Edge._generate_with_shared_prefill`` on one
model instance and keeps its contract. The prompt is prefilled once into a
``FlashKVCache``: as one graph replay for a single-CoT image prompt, otherwise by the
original eager prefill. ``generate`` then runs ``GraphSampler`` through
``custom_generate``. Each token is one replay of [compiled decoder step ->
generate's logits processors -> sampling -> stopping bookkeeping], followed by one
host read of the "all rows finished" flag, as in ``GenerationMixin._sample``.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.cosmos3_edge.modeling_cosmos3_edge import apply_rotary_pos_emb

from alpamayo2_edge.accel.graphs import (
    AccelStats,
    GraphedCall,
    autocast_state,
    bucket,
    freeze,
    maybe_compile,
    override_forward,
)
from alpamayo2_edge.accel.kernels import fa3_attention, kvcache_attention
from alpamayo2_edge.models.alpamayo2_edge import (
    MaskDiscreteTrajectoryLogitsProcessor,
    MaskTokenIdsLogitsProcessor,
)
from alpamayo2_edge.models.expert_utils import StopAfterEOS


class _FlashLayer(CacheLayerMixin):
    """HF-layout ``[rows, kv_heads, length, head_dim]`` views of one ``FlashKVCache`` layer."""

    is_initialized = True
    is_sliding = False
    is_croppable = True

    def __init__(self, cache: "FlashKVCache", index: int) -> None:
        self.cache, self.index = cache, index

    @property
    def keys(self) -> torch.Tensor:
        return self.cache.view(self.cache.k[self.index])

    @property
    def values(self) -> torch.Tensor:
        return self.cache.view(self.cache.v[self.index])

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        del key_states, value_states

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        return self.cache.update(key_states, value_states, self.index)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cache.length + query_length, 0

    def get_seq_length(self) -> int:
        return self.cache.length

    def get_max_length(self) -> int:
        return self.cache.max_len


class FlashKVCache(Cache):
    """Static ``[rows, max_len, kv_heads, head_dim]`` KV cache per layer.

    The decode graphs write it through ``flash_attn_with_kvcache``. ``generate`` and
    the model see a ``DynamicCache``-like object: ``get_seq_length`` is a host integer
    and ``crop`` only moves the valid length. Once ``frozen`` (after generation),
    eager ``update`` calls store at the current length, so the anchor completion and
    the eager expert see the same keys and values as with a ``DynamicCache``.
    """

    def __init__(
        self,
        n_layers: int,
        rows: int,
        max_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (rows, max_len, n_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.rows, self.max_len = rows, max_len
        self.seqlens = torch.zeros(rows, dtype=torch.int32, device=device)
        self.rope_delta = torch.zeros(rows, dtype=torch.long, device=device)
        self.input_ids = torch.zeros(rows, 1, dtype=torch.long, device=device)
        self.length, self.frozen = 0, False
        super().__init__(layers=[_FlashLayer(self, index) for index in range(n_layers)])

    def view(self, buffer: torch.Tensor) -> torch.Tensor:
        return buffer[:, : self.length].transpose(1, 2)

    def start(self, length: int, rope_delta: torch.Tensor) -> None:
        """Mark ``length`` prompt tokens as cached, ready for decoding."""
        if length + 1 > self.max_len:
            raise ValueError(f"Prompt length {length} exceeds the cache length {self.max_len}")
        self.length, self.frozen = length, False
        self.seqlens.fill_(length)
        self.rope_delta.copy_(rope_delta)

    def load(self, prompt_cache: Cache, rope_deltas: torch.Tensor, repeats: int) -> None:
        """Copy an eager prefill cache, repeating each batch row ``repeats`` times."""
        length = prompt_cache.get_seq_length()
        if length + 1 > self.max_len:
            raise ValueError(f"Prompt length {length} exceeds the cache length {self.max_len}")
        for keys_out, values_out, layer in zip(self.k, self.v, prompt_cache.layers, strict=True):
            keys, values = layer.keys, layer.values
            if repeats > 1:
                keys = keys.repeat_interleave(repeats, dim=0)
                values = values.repeat_interleave(repeats, dim=0)
            keys_out[:, :length].copy_(keys.transpose(1, 2))
            values_out[:, :length].copy_(values.transpose(1, 2))
        self.start(length, rope_deltas.reshape(-1).repeat_interleave(repeats))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.length + query_length, 0

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return self.max_len

    @property
    def is_compileable(self) -> bool:
        # Keeps generate from building 4D masks or compiling its own forward.
        return False

    @property
    def batch_size(self) -> int:
        return self.rows

    def crop(self, tokens_to_remove: int) -> None:
        """``DynamicCache.crop``: a positive value is the legacy absolute length."""
        if tokens_to_remove > 0:
            self.length = min(self.length, tokens_to_remove)
        elif tokens_to_remove < 0:
            self.length = max(self.length + tokens_to_remove, 0)

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("FlashKVCache is allocated with one row per sampled CoT")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        raise NotImplementedError("FlashKVCache rows are fixed")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frozen:
            raise RuntimeError("FlashKVCache is written by the decode graphs until generation ends")
        start = self.length
        end = start + key_states.shape[-2]
        if key_states.shape[0] != self.rows or end > self.max_len:
            raise RuntimeError(
                f"FlashKVCache holds {self.rows} rows x {self.max_len} tokens; got "
                f"{key_states.shape[0]} rows ending at {end}"
            )
        self.k[layer_idx][:, start:end].copy_(key_states.transpose(1, 2))
        self.v[layer_idx][:, start:end].copy_(value_states.transpose(1, 2))
        if layer_idx == len(self.k) - 1:
            self.length = end
        return (
            self.k[layer_idx][:, :end].transpose(1, 2),
            self.v[layer_idx][:, :end].transpose(1, 2),
        )


def decoder_block(layer, hidden, cos, sin, k_cache, v_cache, seqlens, causal: bool):
    """``Cosmos3EdgeTextDecoderLayer.forward`` with the cache update and attention fused.

    Every other operation is the layer's own module, so autocast dtypes match
    eager. Eager feeds attention keys/queries after RoPE (FP32 when the residual
    stream is FP32, as in the expert) and autocast casts them to BF16;
    ``.to(cache dtype)`` is the same cast.
    """
    attn = layer.self_attn
    rows, seq, _ = hidden.shape
    x = layer.input_layernorm(hidden)
    q = attn.q_proj(x).view(rows, seq, -1, attn.head_dim)
    k = attn.k_proj(x).view(rows, seq, -1, attn.head_dim)
    v = attn.v_proj(x).view(rows, seq, -1, attn.head_dim)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    dtype = k_cache.dtype
    out = kvcache_attention(
        q.to(dtype).contiguous(),
        k_cache,
        v_cache,
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
        seqlens,
        attn.scaling,
        causal,
    )
    hidden = hidden + attn.o_proj(out.reshape(rows, seq, -1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def prefill_block(layer, hidden, cos, sin, k_cache, v_cache, kv_only: bool, fa3: bool):
    """``Cosmos3EdgeTextDecoderLayer.forward`` over the uncached prompt, writing its K/V.

    Attention is HF's causal SDPA call (or FA3); ``kv_only`` stops after the K/V
    write because the last layer's prompt output is never used.
    """
    attn = layer.self_attn
    rows, seq, _ = hidden.shape
    x = layer.input_layernorm(hidden)
    q = attn.q_proj(x).view(rows, seq, -1, attn.head_dim)
    k = attn.k_proj(x).view(rows, seq, -1, attn.head_dim)
    v = attn.v_proj(x).view(rows, seq, -1, attn.head_dim)
    if fa3:
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        k_cache[:, :seq].copy_(k)
        v_cache[:, :seq].copy_(v)
        if kv_only:
            return hidden
        out = fa3_attention(q, k, v, attn.scaling, True)
    else:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k_cache[:, :seq].copy_(k.transpose(1, 2))
        v_cache[:, :seq].copy_(v.transpose(1, 2))
        if kv_only:
            return hidden
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=attn.scaling, enable_gqa=True
        ).transpose(1, 2)
    hidden = hidden + attn.o_proj(out.reshape(rows, seq, -1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class DecodeStep(torch.nn.Module):
    """One token per row through the VLM text model and LM head.

    Positions are ``cache length + rope delta``, as ``generate`` computes them
    once the prompt is cached; ``seqlens`` advances in place.
    """

    def __init__(self, vlm: torch.nn.Module) -> None:
        super().__init__()
        self.text = vlm.model.language_model
        self.lm_head = vlm.lm_head

    def forward(self, input_ids, rope_delta, seqlens, k_caches, v_caches):
        text = self.text
        hidden = text.embed_tokens(input_ids)
        rows = hidden.shape[0]
        positions = (seqlens.to(torch.long) + rope_delta).view(1, rows, 1).expand(3, rows, 1)
        cos, sin = text.rotary_emb(hidden, positions)
        for index, layer in enumerate(text.layers):
            hidden = decoder_block(
                layer, hidden, cos, sin, k_caches[index], v_caches[index], seqlens, True
            )
        logits = self.lm_head(text.norm(hidden))
        seqlens.add_(1)
        return logits


class PrefillStep(torch.nn.Module):
    """``Cosmos3EdgeTextModel.forward`` over right-padded prompt embeddings.

    Padding follows every real token, so the causal mask hides it from them; its
    K/V land beyond the cache length and decoding overwrites them.
    """

    def __init__(self, vlm: torch.nn.Module, fa3: bool) -> None:
        super().__init__()
        self.text = vlm.model.language_model
        self.fa3 = fa3

    def forward(self, hidden, positions, k_caches, v_caches):
        text = self.text
        cos, sin = text.rotary_emb(hidden, positions)
        last = len(text.layers) - 1
        for index, layer in enumerate(text.layers):
            hidden = prefill_block(
                layer,
                hidden,
                cos,
                sin,
                k_caches[index],
                v_caches[index],
                index == last,
                self.fa3,
            )
        return hidden


class MaskColumns:
    """``MaskTokenIdsLogitsProcessor`` as one capturable ``index_fill_`` (same result)."""

    def __init__(self, token_ids: list[int], device: torch.device) -> None:
        self.index = torch.tensor(list(token_ids), dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        return scores.index_fill_(1, self.index, float("-inf"))


class _SamplerPlan(NamedTuple):
    cache: FlashKVCache
    processors: list[Callable[[], Any]]
    max_length: int
    stop_after: int | None
    eos_ids: torch.Tensor | None
    has_eos: bool
    pad: torch.Tensor | None
    do_sample: bool
    key: tuple


_PLAIN_PROCESSORS = (
    MaskDiscreteTrajectoryLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class GraphSampler:
    """``GenerationMixin._sample`` with each decoding step as one CUDA-graph replay.

    It is passed as ``generate(custom_generate=...)``, so generate's own setup
    (generation config, special tokens, logits processors, stopping criteria) is
    unchanged. Sampling stays eager PyTorch inside the graph, with the same kernels
    and FP32 math as ``_sample``. CUDA-graph RNG is Philox-offset based, so a replay
    draws what eager would at the same generator state. Unsupported processors,
    criteria or outputs run the stock ``_sample``.
    """

    def __init__(self, decode: "DecodeAccelerator") -> None:
        self.decode = decode
        self.graphs: dict[tuple, tuple] = {}

    def _plan(self, input_ids, logits_processor, stopping_criteria, config, model_kwargs):
        cache = model_kwargs.get("past_key_values")
        if (
            not isinstance(cache, FlashKVCache)
            or cache.frozen
            or config.output_scores
            or config.output_logits
            or config.output_attentions
            or config.output_hidden_states
            or not config.return_dict_in_generate
            or config.prefill_chunk_size is not None
            or (config.num_beams or 1) != 1
            or input_ids.shape[0] != cache.rows
            or input_ids.shape[1] != cache.length + 1
        ):
            return None
        processors, key = [], []
        for processor in logits_processor:
            kind = type(processor)
            if kind in _PLAIN_PROCESSORS:
                processors.append(lambda processor=processor: processor)
            elif kind is MaskTokenIdsLogitsProcessor:
                # ``scores[:, ids] = -inf`` copies its index list to the GPU on every
                # call, which cannot be captured; index_fill_ is the same write.
                processors.append(
                    lambda ids=processor.token_ids: MaskColumns(ids, input_ids.device)
                )
            else:
                return None
            key.append((kind.__name__, freeze(vars(processor))))
        max_length, stop_after, eos_ids = None, None, None
        for criterion in stopping_criteria:
            kind = type(criterion)
            if kind is MaxLengthCriteria:
                max_length = min(max_length or criterion.max_length, criterion.max_length)
            elif kind is StopAfterEOS and stop_after is None and criterion.eos_found is None:
                stop_after = int(criterion.eos_token_id)
            elif kind is EosTokenCriteria and eos_ids is None:
                eos_ids = criterion.eos_token_id.to(input_ids.device)
            else:
                return None
        has_eos = any(hasattr(criterion, "eos_token_id") for criterion in stopping_criteria)
        pad = config._pad_token_tensor
        if max_length is None or (has_eos and pad is None):
            return None
        if pad is not None:
            pad = pad.to(input_ids.device)
        key = (
            cache.rows,
            cache.max_len,
            id(cache.k[0]),
            bool(config.do_sample),
            tuple(key),
            stop_after,
            None if eos_ids is None else tuple(eos_ids.reshape(-1).tolist()),
            None if pad is None else int(pad),
            autocast_state(input_ids.device.type),
        )
        return _SamplerPlan(
            cache,
            processors,
            max_length,
            stop_after,
            eos_ids,
            has_eos,
            pad,
            bool(config.do_sample),
            key,
        )

    def _graph(self, plan: _SamplerPlan):
        entry = self.graphs.get(plan.key)
        if entry is not None:
            return entry
        cache, device = plan.cache, plan.cache.input_ids.device
        rows = cache.rows
        processors = [factory() for factory in plan.processors]
        tokens = torch.zeros(rows, cache.max_len, dtype=torch.long, device=device)
        step = torch.zeros(1, dtype=torch.long, device=device)
        unfinished = torch.ones(rows, dtype=torch.long, device=device)
        eos_found = torch.zeros(rows, dtype=torch.bool, device=device)
        finished = torch.zeros((), dtype=torch.bool, device=device)
        compiled_step = self.decode.compiled_step
        stop_after, eos_ids, has_eos, pad = plan.stop_after, plan.eos_ids, plan.has_eos, plan.pad
        do_sample = plan.do_sample

        def run() -> torch.Tensor:
            logits = compiled_step(
                cache.input_ids, cache.rope_delta, cache.seqlens, cache.k, cache.v
            )
            scores = logits[:, -1, :].to(dtype=torch.float32)
            for processor in processors:
                scores = processor(cache.input_ids, scores)
            if do_sample:
                probs = F.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(scores, dim=-1)
            if has_eos:
                next_tokens = next_tokens * unfinished + pad * (1 - unfinished)
            tokens.index_copy_(1, step, next_tokens[:, None])
            done = torch.zeros(rows, dtype=torch.bool, device=device)
            if stop_after is not None:
                # StopAfterEOS: stop once every row emitted the anchor on an earlier step.
                all_found = eos_found.all()
                eos_found.copy_(eos_found | (next_tokens == stop_after))
                done = done | all_found
            if eos_ids is not None:
                done = done | torch.isin(next_tokens, eos_ids)
            unfinished.copy_(unfinished & ~done)
            finished.copy_(unfinished.max() == 0)
            cache.input_ids.copy_(next_tokens[:, None])
            step.add_(1)
            return finished

        def reset(length: int, first_tokens: torch.Tensor) -> None:
            cache.seqlens.fill_(length)
            cache.input_ids.copy_(first_tokens)
            step.zero_()
            unfinished.fill_(1)
            eos_found.zero_()
            finished.zero_()

        length, first = cache.length, cache.input_ids.clone()
        graph = self.decode.graphed(
            "decode_sample", plan.key[:2] + plan.key[3:4], run, (), lambda: reset(length, first)
        )
        entry = (graph, tokens, reset)
        self.graphs[plan.key] = entry
        return entry

    def __call__(
        self,
        model,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs,
    ):
        plan = None
        if not synced_gpus and streamer is None:
            plan = self._plan(
                input_ids, logits_processor, stopping_criteria, generation_config, model_kwargs
            )
        if plan is None:
            self.decode.stats.fallback("sampler_unsupported")
            return GenerationMixin._sample(
                model,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        cache = plan.cache
        cache.input_ids.copy_(input_ids[:, -1:])
        graph, tokens, reset = self._graph(plan)
        reset(cache.length, input_ids[:, -1:])
        max_new_tokens = plan.max_length - input_ids.shape[1]
        generated = 0
        while True:
            finished = graph()
            generated += 1
            cache.length += 1
            if generated >= max_new_tokens or bool(finished):
                break
        return GenerateDecoderOnlyOutput(
            sequences=torch.cat([input_ids, tokens[:, :generated]], dim=1),
            past_key_values=cache,
        )


class DecodeAccelerator:
    """Graphed prefill and CoT decoding for one ``Alpamayo2Edge`` instance."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        stats: AccelStats,
        compile: bool,
        cuda_graphs: bool,
        warmup: int,
        kv_bucket: int,
        decode_mode: str,
        mode: str,
        graph_prefill: bool,
        prefill_fa3: bool,
    ) -> None:
        self.model, self.vlm, self.stats = model, model.vlm, stats
        self.cuda_graphs, self.warmup, self.kv_bucket = cuda_graphs, warmup, kv_bucket
        text_config = self.vlm.config.text_config
        self.n_layers = text_config.num_hidden_layers
        self.n_kv_heads, self.head_dim = text_config.num_key_value_heads, text_config.head_dim
        self.n_expert_tokens = model.expert.action_space.get_action_space_dims()[0]
        self.caches: dict[tuple, FlashKVCache] = {}
        self.step_graphs: dict[tuple, GraphedCall] = {}
        self.prefill_graphs: dict[tuple, GraphedCall] = {}
        self.compiled_step = maybe_compile(DecodeStep(self.vlm), compile, decode_mode)
        self.compiled_prefill = maybe_compile(PrefillStep(self.vlm, prefill_fa3), compile, mode)
        self.graph_prefill = graph_prefill
        self.sampler = GraphSampler(self)
        self.original_generate = model._generate_with_shared_prefill
        self.original_forward = self.vlm.forward
        self.installed_generate = self.generate_with_shared_prefill
        self.installed_forward = override_forward(self.original_forward, self.forward)
        model._generate_with_shared_prefill = self.installed_generate
        self.vlm.forward = self.installed_forward

    def remove(self) -> None:
        if (
            vars(self.model).get("_generate_with_shared_prefill") is not self.installed_generate
            or vars(self.vlm).get("forward") is not self.installed_forward
        ):
            raise RuntimeError("Remove wrappers installed after the decode accelerator first")
        del self.model._generate_with_shared_prefill
        del self.vlm.forward

    def graphed(self, name, key, fn, args, restore=None) -> GraphedCall:
        device = self.model.device
        return GraphedCall(
            name,
            key,
            fn,
            args,
            stats=self.stats,
            warmup=self.warmup if self.cuda_graphs else 0,
            cuda_graph=self.cuda_graphs,
            device=device,
            restore=restore,
        )

    def _cache(self, rows: int, max_len: int) -> FlashKVCache:
        key = (rows, max_len)
        cache = self.caches.get(key)
        if cache is None:
            parameter = next(self.vlm.parameters())
            cache = FlashKVCache(
                self.n_layers,
                rows,
                max_len,
                self.n_kv_heads,
                self.head_dim,
                parameter.dtype,
                parameter.device,
            )
            self.caches[key] = cache
        return cache

    @staticmethod
    def _unsupported(tokenized_data: dict[str, Any]) -> str | None:
        if tokenized_data.get("image_grid_thw") is None:
            return "decode_without_images"
        if tokenized_data.get("pixel_values_videos") is not None:
            return "decode_video"
        if not bool(tokenized_data["attention_mask"].all()):
            # Cached columns are attended unmasked: padding would be visible.
            return "decode_padding"
        return None

    def generate_with_shared_prefill(
        self,
        tokenized_data: dict[str, Any],
        generation_config,
        n_samples_total: int,
        **generate_kwargs: Any,
    ):
        """``Alpamayo2Edge._generate_with_shared_prefill`` over a ``FlashKVCache``."""
        reason = self._unsupported(tokenized_data)
        if reason is None and generation_config.max_new_tokens is None:
            reason = "decode_without_max_new_tokens"
        if reason is not None:
            self.stats.fallback(reason)
            return self.original_generate(
                tokenized_data, generation_config, n_samples_total, **generate_kwargs
            )
        generation_config.num_return_sequences = 1
        generation_config.output_logits = False
        input_ids = tokenized_data["input_ids"]
        prompt_mask = tokenized_data["attention_mask"]
        rows = input_ids.shape[0] * n_samples_total
        needed = input_ids.shape[1] + generation_config.max_new_tokens + self.n_expert_tokens
        cache = self._cache(rows, bucket(needed, self.kv_bucket))
        if not (self.graph_prefill and self._graph_prefill(tokenized_data, cache, n_samples_total)):
            self._eager_prefill(tokenized_data, cache, n_samples_total)
        rope_deltas = self.vlm.model.rope_deltas.repeat_interleave(n_samples_total, dim=0)
        self.vlm.model.rope_deltas = rope_deltas
        outputs = self.vlm.generate(
            input_ids=input_ids.repeat_interleave(n_samples_total, dim=0),
            attention_mask=prompt_mask.repeat_interleave(n_samples_total, dim=0),
            past_key_values=cache,
            generation_config=generation_config,
            custom_generate=self.sampler,
            **generate_kwargs,
        )
        cache.frozen = True
        outputs.rope_deltas = rope_deltas
        return outputs

    def _eager_prefill(self, tokenized_data, cache: FlashKVCache, n_samples_total: int) -> None:
        """The original shared prefill, copied into the flash cache."""
        input_ids = tokenized_data["input_ids"]
        vision_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        prefill_inputs = {key: tokenized_data[key] for key in vision_keys if key in tokenized_data}
        if "mm_token_type_ids" in tokenized_data:
            prefill_inputs["mm_token_type_ids"] = tokenized_data["mm_token_type_ids"][:, :-1]
        outputs = self.vlm.model(
            input_ids=input_ids[:, :-1],
            attention_mask=tokenized_data["attention_mask"][:, :-1],
            use_cache=True,
            **prefill_inputs,
        )
        rope_deltas = self.vlm.model.rope_deltas
        if rope_deltas is None:
            raise RuntimeError("The VLM did not produce multimodal rotary position offsets.")
        cache.load(outputs.past_key_values, rope_deltas, n_samples_total)

    def _graph_prefill(self, tokenized_data, cache: FlashKVCache, n_samples_total: int) -> bool:
        """Prefill one image prompt into the flash cache as one graph replay.

        Embeddings, vision features and their scatter, and ``get_rope_index`` are
        ``Cosmos3EdgeModel.forward``'s own steps; the rope index runs on host copies,
        which gives the same integer positions without per-image device syncs. The
        decoder stack runs over the prompt right-padded to a ``kv_bucket`` multiple.
        """
        data = tokenized_data
        input_ids = data["input_ids"][:, :-1]
        if (
            input_ids.shape[0] != 1
            or n_samples_total != 1
            or data.get("pixel_values") is None
            or data.get("mm_token_type_ids") is None
        ):
            self.stats.fallback("prefill_unsupported")
            return False
        vlm_model = self.vlm.model
        embeds = vlm_model.get_input_embeddings()(input_ids)
        image_outputs = vlm_model.get_image_features(
            data["pixel_values"], data["image_grid_thw"], return_dict=True
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(embeds.device, embeds.dtype)
        image_mask, _ = vlm_model.get_placeholder_mask(
            input_ids, inputs_embeds=embeds, image_features=image_embeds
        )
        embeds = embeds.masked_scatter(image_mask, image_embeds)
        positions, rope_deltas = vlm_model.get_rope_index(
            input_ids.cpu(),
            data["mm_token_type_ids"][:, :-1].cpu(),
            data["image_grid_thw"].cpu(),
            None,
            attention_mask=data["attention_mask"][:, :-1].cpu(),
        )
        vlm_model.rope_deltas = rope_deltas.to(embeds.device)
        n_prompt = input_ids.shape[1]
        padded = bucket(n_prompt, self.kv_bucket)
        key = (cache.max_len, padded, id(cache.k[0]), autocast_state(embeds.device.type))
        graph = self.prefill_graphs.get(key)
        if graph is None:
            static = (
                embeds.new_zeros(1, padded, embeds.shape[-1]),
                torch.zeros(3, 1, padded, dtype=positions.dtype, device=embeds.device),
                cache.k,
                cache.v,
            )
            graph = self.graphed("prefill", key[:2], self.compiled_prefill, static)
            self.prefill_graphs[key] = graph
        hidden_in, positions_in = graph.args[:2]
        hidden_in[:, :n_prompt].copy_(embeds)
        hidden_in[:, n_prompt:].zero_()
        positions = positions.to(embeds.device)
        positions_in[:, :, :n_prompt].copy_(positions)
        positions_in[:, :, n_prompt:].copy_(positions[:, :, -1:])
        graph()
        cache.start(n_prompt, vlm_model.rope_deltas.reshape(-1))
        return True

    def _step_graph(self, cache: FlashKVCache) -> GraphedCall:
        key = (cache.rows, cache.max_len, id(cache.k[0]), autocast_state(cache.k[0].device.type))
        graph = self.step_graphs.get(key)
        if graph is None:
            length = cache.length

            def restore() -> None:
                cache.seqlens.fill_(length)

            args = (cache.input_ids, cache.rope_delta, cache.seqlens, cache.k, cache.v)
            graph = self.graphed("decode", key[:2], self.compiled_step, args, restore)
            self.step_graphs[key] = graph
        return graph

    def forward(self, *args, **kwargs):
        """``vlm.forward`` override: replay single-token steps on a live flash cache."""
        cache = kwargs.get("past_key_values")
        if not isinstance(cache, FlashKVCache) or cache.frozen or args:
            return self.original_forward(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        if (
            input_ids is None
            or input_ids.shape[1] != 1
            or kwargs.get("inputs_embeds") is not None
            or kwargs.get("pixel_values") is not None
        ):
            raise RuntimeError("The flash decode cache only supports single-token steps")
        cache.input_ids.copy_(input_ids)
        cache.seqlens.fill_(cache.length)
        logits = self._step_graph(cache)()
        cache.length += 1
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
