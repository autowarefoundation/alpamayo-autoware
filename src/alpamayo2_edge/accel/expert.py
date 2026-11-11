# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flow-matching denoiser as one CUDA graph per sampling call.

``ExpertAccelerator`` replaces ``Alpamayo2Edge._sample_actions`` on one model
instance. Each Euler step runs the eager ``action_in_proj``, the compiled expert
transformer over the decode cache's flash buffers and its own 64 action tokens
(non-causal), the eager ``action_out_proj`` and ``x + dt * v``; the whole loop is
one replay. Eager attention instead re-concatenates the prompt cache on every step
(in FP32, because the expert's residual stream is FP32 under BF16 autocast) and
expands the KV heads for SDPA. The initial noise is drawn eagerly from the call's
generator, exactly as ``FlowMatching._euler`` draws it.

Several CoTs replay the graph only when they all have the same length; otherwise
the eager loop runs. The graph is still captured on the first call with that
shape, normally the warm-up, so that the first equal-length call does not stall.
"""

from typing import Any

import torch

from alpamayo2_edge.accel.decode import FlashKVCache, decoder_block
from alpamayo2_edge.accel.graphs import AccelStats, GraphedCall, autocast_state, maybe_compile
from alpamayo2_edge.diffusion.flow_matching import FlowMatching

_LOOP_KWARGS = frozenset({"inference_step", "generator", "temperature", "int_method"})


class ExpertStep(torch.nn.Module):
    """All expert layers over the cached prompt ``[0, seqlens)`` and the action tokens."""

    def __init__(self, expert_text: torch.nn.Module) -> None:
        super().__init__()
        self.text = expert_text

    def forward(self, hidden, cos, sin, seqlens, k_caches, v_caches):
        for index, layer in enumerate(self.text.layers):
            hidden = decoder_block(
                layer, hidden, cos, sin, k_caches[index], v_caches[index], seqlens, False
            )
        return self.text.norm(hidden)


def _mask_hides_tokens(attention_mask: torch.Tensor | None) -> bool:
    """Whether the expert mask removes any cache column (padding or another row's CoT).

    Additive float masks hide where nonzero; FlashAttention 2 integer and boolean
    masks keep where one/True.
    """
    if attention_mask is None:
        return False
    if attention_mask.is_floating_point():
        return bool((attention_mask != 0).any())
    return not bool(attention_mask.bool().all())


class ExpertAccelerator:
    """Graphed denoising for one ``Alpamayo2Edge`` instance."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        stats: AccelStats,
        compile: bool,
        cuda_graphs: bool,
        warmup: int,
        mode: str,
    ) -> None:
        self.model, self.expert, self.stats = model, model.expert, stats
        self.cuda_graphs, self.warmup = cuda_graphs, warmup
        self.text = model.expert.expert
        self.compiled_step = maybe_compile(ExpertStep(self.text), compile, mode)
        self.graphs: dict[tuple, GraphedCall] = {}
        self.embed_dtypes: dict[tuple, torch.dtype] = {}
        self.original = model._sample_actions
        self.installed = self.sample_actions
        model._sample_actions = self.installed

    def remove(self) -> None:
        if vars(self.model).get("_sample_actions") is not self.installed:
            raise RuntimeError("Remove wrappers installed after the expert accelerator first")
        del self.model._sample_actions

    def _unsupported(self, cache, attention_mask, diffusion_kwargs) -> str | None:
        diffusion = self.expert.diffusion
        kwargs = diffusion_kwargs or {}
        if not isinstance(cache, FlashKVCache):
            return "expert_without_flash_cache"
        if (
            type(diffusion) is not FlowMatching
            or set(kwargs) - _LOOP_KWARGS
            or (kwargs.get("int_method") or diffusion.int_method) != "euler"
            or diffusion.use_classifier_free_guidance
        ):
            return "expert_sampler"
        if _mask_hides_tokens(attention_mask):
            # Several CoTs of different lengths.
            return "expert_mask"
        return None

    def sample_actions(
        self,
        cache: Any,
        prefix_length: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expert_batch: int,
        device: torch.device,
        diffusion_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        """``Alpamayo2Edge._sample_actions`` with the whole Euler loop graphed."""
        reason = self._unsupported(cache, attention_mask, diffusion_kwargs)
        kwargs = diffusion_kwargs or {}
        if reason == "expert_mask":
            # Capture now with placeholder noise (no random draw); the eager loop
            # below rewrites the cache columns the capture wrote.
            self._graph(cache, prefix_length, position_ids, expert_batch, device, kwargs, None)
        if reason is not None:
            self.stats.fallback(reason)
            return self.original(
                cache,
                prefix_length,
                position_ids,
                attention_mask,
                expert_batch,
                device,
                diffusion_kwargs,
            )
        dims = tuple(self.expert.diffusion.x_dims)
        noise = torch.randn(expert_batch, *dims, device=device, generator=kwargs.get("generator"))
        noise = noise * kwargs.get("temperature", 1.0)
        graph, cos, sin = self._graph(
            cache, prefix_length, position_ids, expert_batch, device, kwargs, noise
        )
        noise_in, cos_in, sin_in, seqlens = graph.args[:4]
        noise_in.copy_(noise)
        cos_in.copy_(cos)
        sin_in.copy_(sin)
        seqlens.fill_(prefix_length)
        return graph().clone()

    def _graph(self, cache, prefix_length, position_ids, expert_batch, device, kwargs, noise):
        """The graph for this call's shapes (built on first use) and its rotary tables."""
        diffusion = self.expert.diffusion
        dims = tuple(diffusion.x_dims)
        steps = kwargs.get("inference_step") or diffusion.num_inference_steps
        if noise is None:
            noise = torch.zeros(expert_batch, *dims, device=device)
        autocast = autocast_state(noise.device.type)
        if autocast not in self.embed_dtypes:
            probe_t = torch.zeros(1, *[1] * len(dims), device=device)
            self.embed_dtypes[autocast] = self.expert.action_in_proj(noise[:1], probe_t).dtype
        probe = torch.empty(0, dtype=self.embed_dtypes[autocast], device=device)
        cos, sin = self.text.rotary_emb(probe, position_ids)
        key = (expert_batch, dims, cache.max_len, id(cache.k[0]), steps, autocast)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self._build(key, noise, cos, sin, cache, prefix_length, steps)
            self.graphs[key] = graph
        return graph, cos, sin

    def _build(self, key, noise, cos, sin, cache, prefix_length, steps) -> GraphedCall:
        rows, dims = key[0], key[1]
        n_dims = len(dims)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=noise.device)
        expert, compiled_step = self.expert, self.compiled_step

        def loop(x, cos, sin, seqlens, k_caches, v_caches):
            # FlowMatching._euler with Alpamayo2Edge._sample_actions' step function.
            for index in range(steps):
                dt = (timesteps[index + 1] - timesteps[index]).view(1, *[1] * n_dims)
                t = timesteps[index].view(1, *[1] * n_dims).expand(rows, *[1] * n_dims)
                embeddings = expert.action_in_proj(x, t).reshape(rows, dims[0], -1)
                hidden = compiled_step(embeddings, cos, sin, seqlens, k_caches, v_caches)
                velocity = expert.action_out_proj(hidden).reshape(-1, *dims)
                x = x + dt.expand(rows, *[1] * n_dims) * velocity
            return x

        static = (
            noise.clone(),
            cos.clone(),
            sin.clone(),
            torch.full((rows,), prefix_length, dtype=torch.int32, device=noise.device),
            cache.k,
            cache.v,
        )
        return GraphedCall(
            "expert",
            (rows, cache.max_len, f"steps={steps}"),
            loop,
            static,
            stats=self.stats,
            warmup=self.warmup if self.cuda_graphs else 0,
            cuda_graph=self.cuda_graphs,
            device=noise.device,
        )
