# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference for the trained Alpamayo 2 Edge checkpoint."""

import copy
from typing import Any

import hydra.utils as hyu
import numpy as np
import torch
import transformers
from transformers import AutoModel, GenerationConfig, PreTrainedModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerateDecoderOnlyOutput

from alpamayo2_edge.config import (
    Alpamayo2EdgeConfig,
    build_alpamayo2_edge_tokenizer,
    resolve_checkpoint_name_or_path,
)
from alpamayo2_edge.models.expert import ExpertModel
from alpamayo2_edge.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_edge.models.token_utils import extract_text_tokens
from alpamayo2_edge.models.utils import fuse_traj_tokens


class MaskDiscreteTrajectoryLogitsProcessor(LogitsProcessor):
    """Keep discrete trajectory codes out of the generated reasoning text."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int) -> None:
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[
            :, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size
        ] = -torch.inf
        return scores


class MaskTokenIdsLogitsProcessor(LogitsProcessor):
    """Prevent text EOS from terminating a CoT before the expert anchor."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.token_ids] = -torch.inf
        return scores


class Alpamayo2Edge(PreTrainedModel):
    """Cosmos3-Edge reasoning followed by ten-step trajectory flow matching.

    The VLM and expert retain their checkpoint module names and KV geometry.
    Every trajectory samples its own CoT from a shared prompt prefill.
    ``num_traj_sets`` and ``num_traj_samples`` group these independent candidates.
    """

    config_class = Alpamayo2EdgeConfig
    base_model_prefix = "vlm"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: Alpamayo2EdgeConfig) -> None:
        config.set_dtype(config.dtype)
        super().__init__(config)
        implementation = config._attn_implementation or "sdpa"
        if implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError(f"Unsupported attention implementation: {implementation!r}")
        config.vlm_config._attn_implementation = implementation
        vlm_cls = getattr(transformers, config.vlm_class)
        self.vlm = vlm_cls._from_config(config.vlm_config)
        checkpoint = resolve_checkpoint_name_or_path(config)
        if checkpoint is None:
            raise ValueError(
                "Alpamayo2Edge requires the checkpoint path for its trained tokenizer."
            )
        self.tokenizer = build_alpamayo2_edge_tokenizer(
            checkpoint, config.history_vocab_size, config.future_vocab_size
        )
        self.history_traj_tokenizer = hyu.instantiate(
            config.hist_traj_tokenizer_cfg, load_weights=False
        )
        self.future_traj_tokenizer = hyu.instantiate(
            config.future_traj_tokenizer_cfg, load_weights=False
        )
        if config.enable_expert:
            config.expert_config._attn_implementation = implementation
            config.expert_config.llm_config._attn_implementation = implementation
            self.expert = ExpertModel._from_config(config.expert_config)
            vlm_text = config.vlm_config.text_config
            expert_text = config.expert_config.llm_config
            for field in ("model_type", "num_hidden_layers", "num_key_value_heads", "head_dim"):
                if getattr(vlm_text, field) != getattr(expert_text, field):
                    raise ValueError(f"VLM and expert must share KV-cache geometry ({field}).")
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.vlm.get_input_embeddings()

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.vlm.get_output_embeddings()

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        super().tie_weights(*args, **kwargs)

    def _generate_with_shared_prefill(
        self,
        tokenized_data: dict[str, Any],
        generation_config: GenerationConfig,
        n_samples_total: int,
        **generate_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        """Prefill each prompt once and sample independent CoTs from its cache."""
        generation_config.num_return_sequences = 1
        generation_config.output_logits = False
        input_ids = tokenized_data["input_ids"]
        prompt_mask = tokenized_data["attention_mask"]
        vision_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        prefill_inputs = {key: tokenized_data[key] for key in vision_keys if key in tokenized_data}
        if "mm_token_type_ids" in tokenized_data:
            prefill_inputs["mm_token_type_ids"] = tokenized_data["mm_token_type_ids"][:, :-1]
        if all(tokenized_data.get(key) is None for key in ("image_grid_thw", "video_grid_thw")):
            # Direct HF5 text forwards do not initialize RoPE deltas or remove
            # left padding from positions as generate() does.
            text_positions = prompt_mask.long().cumsum(-1) - 1
            prefill_inputs["position_ids"] = text_positions.masked_fill(prompt_mask == 0, 0)[:, :-1]
            self.vlm.model.rope_deltas = input_ids.new_zeros((input_ids.shape[0], 1))
        # The final prompt token supplies the first generation logits. All image
        # tokens precede it, so vision encoding runs only at the original batch size.
        prefill_outputs = self.vlm.model(
            input_ids=input_ids[:, :-1],
            attention_mask=prompt_mask[:, :-1],
            use_cache=True,
            **prefill_inputs,
        )
        prompt_cache = prefill_outputs.past_key_values
        del prefill_outputs
        prompt_cache.batch_repeat_interleave(n_samples_total)
        rope_deltas = self.vlm.model.rope_deltas
        if rope_deltas is None:
            raise RuntimeError("The VLM did not produce multimodal rotary position offsets.")
        rope_deltas = rope_deltas.repeat_interleave(n_samples_total, dim=0)
        self.vlm.model.rope_deltas = rope_deltas
        outputs = self.vlm.generate(
            input_ids=input_ids.repeat_interleave(n_samples_total, dim=0),
            attention_mask=prompt_mask.repeat_interleave(n_samples_total, dim=0),
            past_key_values=prompt_cache,
            generation_config=generation_config,
            **generate_kwargs,
        )
        outputs.rope_deltas = rope_deltas
        return outputs

    def _sample_actions(
        self,
        cache: Any,
        prefix_length: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expert_batch: int,
        device: torch.device,
        diffusion_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Denoise expert actions conditioned on the VLM cache cropped to ``prefix_length``."""
        action_dims = self.expert.action_space.get_action_space_dims()

        # SDPA/eager require the caller's explicit 4D noncausal mask: is_causal
        # alone would leave HF's upstream causal-mask construction in effect.
        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            embeddings = self.expert.action_in_proj(x, t).reshape(expert_batch, action_dims[0], -1)
            try:
                outputs = self.expert.expert(
                    inputs_embeds=embeddings,
                    position_ids=position_ids,
                    past_key_values=cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    is_causal=False,
                )
            finally:
                # Each diffusion step conditions on the original CoT only.
                cache.crop(prefix_length)
            return self.expert.action_out_proj(outputs.last_hidden_state).reshape(-1, *action_dims)

        return self.expert.diffusion.sample(
            batch_size=expert_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **(diffusion_kwargs or {}),
        )

    @torch.inference_mode()
    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        max_generation_length: int | None = None,
        diffusion_kwargs: dict[str, Any] | None = None,
        return_extra: bool = False,
    ) -> tuple:
        """Return XYZ, rotations and placeholder log probabilities.

        Trajectory dimensions are ``[batch, sets, samples, waypoints, ...]``.
        Each candidate has an independent CoT. Decoded text uses the same first
        three dimensions; generated token IDs flatten those dimensions.
        A missing ``<|traj_future_start|>`` raises before running the expert.
        """
        if not self.config.enable_expert:
            raise ValueError("Alpamayo2Edge trajectory inference requires an expert checkpoint.")
        for name, value in (
            ("num_traj_sets", num_traj_sets),
            ("num_traj_samples", num_traj_samples),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if max_generation_length is not None and max_generation_length < 1:
            raise ValueError("max_generation_length must be positive.")
        tokenized_data = dict(data["tokenized_data"])
        traj_data = {key: data[key] for key in ("ego_history_xyz", "ego_history_rot")}
        input_ids = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )
        tokenized_data["input_ids"] = input_ids
        tokenized_data.setdefault("attention_mask", torch.ones_like(input_ids))
        if "mm_token_type_ids" not in tokenized_data and (
            "image_grid_thw" in tokenized_data or "video_grid_thw" in tokenized_data
        ):
            # Older packed fixtures omit this HF5 processor field. Without it,
            # generation silently uses text positions for vision tokens.
            token_types = torch.zeros_like(input_ids)
            token_types[input_ids == self.config.vlm_config.image_token_id] = 1
            token_types[input_ids == self.config.vlm_config.video_token_id] = 2
            tokenized_data["mm_token_type_ids"] = token_types
        batch_size, prompt_length = input_ids.shape
        n_samples_total = num_traj_sets * num_traj_samples
        anchor = self.config.traj_ids["future_start"]
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.top_k = top_k
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_beams = 1
        generation_config.max_new_tokens = max_generation_length or max(
            256, self.config.tokens_per_future_traj
        )
        generation_config.output_logits = False
        generation_config.output_scores = False
        generation_config.return_dict_in_generate = True
        generation_config.use_cache = True
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        eos_ids = generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        # The custom criterion waits one step after the last anchor to cache it.
        generation_config.eos_token_id = None
        processors = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    self.config.traj_ids["history_id0"], self.config.history_vocab_size
                ),
                MaskDiscreteTrajectoryLogitsProcessor(
                    self.config.traj_ids["future_id0"], self.config.future_vocab_size
                ),
            ]
        )
        blocked_eos = [token_id for token_id in eos_ids if token_id != anchor]
        if blocked_eos:
            processors.append(MaskTokenIdsLogitsProcessor(blocked_eos))
        vlm_outputs = self._generate_with_shared_prefill(
            tokenized_data,
            generation_config=generation_config,
            n_samples_total=n_samples_total,
            logits_processor=processors,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(anchor)]),
        )
        sequences = vlm_outputs.sequences
        # Search only the generated continuation, never an example in the prompt.
        offsets = (
            find_eos_offset(sequences[:, prompt_length:], anchor, input_ids.device) + prompt_length
        )
        cache = vlm_outputs.past_key_values
        if cache is None:
            raise RuntimeError("The VLM did not return the KV cache needed by the expert.")
        rope_deltas = vlm_outputs.rope_deltas
        expert_batch = batch_size * n_samples_total
        prefix_mask = tokenized_data["attention_mask"].repeat_interleave(n_samples_total, dim=0)
        # HF5 expresses MRoPE deltas relative to unpadded text positions. Cache
        # offsets include prompt padding, so remove it for both continuation and
        # expert rotary positions (without changing the cache-column masks).
        rope_deltas = rope_deltas - prefix_mask.eq(0).sum(dim=-1, keepdim=True)
        prefix_length = int(offsets.max().item())
        cached_length = cache.get_seq_length()
        if cached_length < prefix_length:
            # If the length limit lands on the anchor, generate() has not yet
            # forwarded it. Finish only the uncached text, reusing the vision cache.
            missing_ids = sequences[:, cached_length:prefix_length]
            text_positions = torch.arange(cached_length, prefix_length, device=input_ids.device)
            positions = text_positions[None, None, :].expand(3, expert_batch, -1)
            positions = positions + rope_deltas[None, :, :]
            completed_mask = torch.cat(
                [
                    prefix_mask,
                    prefix_mask.new_ones((expert_batch, prefix_length - prompt_length)),
                ],
                dim=1,
            )
            self.vlm.model.language_model(
                input_ids=missing_ids,
                position_ids=positions,
                attention_mask=completed_mask,
                past_key_values=cache,
                use_cache=True,
            )
        cache.crop(prefix_length)
        sequences = sequences.clone()
        sequences[:, prompt_length:] = replace_padding_after_eos(
            sequences[:, prompt_length:], anchor, self.tokenizer.pad_token_id
        )
        action_dims = self.expert.action_space.get_action_space_dims()
        position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
            offset=offsets,
            rope_deltas=rope_deltas,
            kv_cache_seq_len=prefix_length,
            n_diffusion_tokens=action_dims[0],
            b_star=expert_batch,
            device=input_ids.device,
            prefix_mask=prefix_mask,
        )
        implementation = self.expert.expert.config._attn_implementation
        if implementation == "flash_attention_2":
            # FA2 takes a padding mask; is_causal=False controls its kernel.
            attention_mask = attention_mask[:, 0, 0].eq(0).to(prefix_mask.dtype)
        sampled_action = self._sample_actions(
            cache,
            prefix_length,
            position_ids,
            attention_mask,
            expert_batch,
            input_ids.device,
            diffusion_kwargs,
        )
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(
            sampled_action,
            traj_data["ego_history_xyz"][:, -1].repeat_interleave(n_samples_total, dim=0),
            traj_data["ego_history_rot"][:, -1].repeat_interleave(n_samples_total, dim=0),
        )
        shape = (batch_size, num_traj_sets, num_traj_samples)
        pred_xyz = pred_xyz.reshape(*shape, *pred_xyz.shape[1:])
        pred_rot = pred_rot.reshape(*shape, *pred_rot.shape[1:])
        logprob = torch.zeros_like(pred_xyz[..., 0])
        if not return_extra:
            return pred_xyz, pred_rot, logprob
        decoded = extract_text_tokens(self.tokenizer, sequences)
        extra = {key: np.asarray(value).reshape(shape) for key, value in decoded.items()}
        generated_ids = vlm_outputs.sequences[:, prompt_length:].clone()
        generated_offsets = offsets - prompt_length
        columns = torch.arange(generated_ids.shape[1], device=generated_ids.device)
        generated_ids.masked_fill_(columns[None, :] >= generated_offsets[:, None], -1)
        extra["generated_ids"] = generated_ids
        return pred_xyz, pred_rot, logprob, extra


AutoModel.register(Alpamayo2EdgeConfig, Alpamayo2Edge)
