# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Navigation classifier-free guidance for Alpamayo 2 Super on a single GPU.

Alpamayo 2 Super conditions its 2B flow-matching expert only through the KV cache the
32B VLM produces, so the single way to tell the model where to go is to put the
navigation instruction into the VLM prompt. That instruction carries no dedicated
structure tokens (``chat_template/conversation.py: construct_nav_instruction`` inserts
raw text with empty ``start_str``/``end_str``), which makes it easy to drown in the
~5k-token visual prefix. The checkpoint was therefore trained for classifier-free
guidance -- ``train_ignore_guidance_rate = 0.1`` in ``config.json`` -- and ships an
``inference_guidance_weight`` of 3.0, i.e. the navigation effect is meant to be
extrapolated, not merely conditioned on::

    v = unguided_v + w * (guided_v - unguided_v)

The released API cannot do this: ``sample_trajectories_from_data`` passes a single
``step_fn`` and never an ``unguided_step_fn``, and ``diffusion.sample`` refuses CFG
without one. This module reimplements the sampling loop with both branches, following
upstream's ``examples/two_gpu_nav_cfg_demo.py``.

No monkey-patching is needed. ``ExpertModel.__init__`` only raises when the *config*
enables CFG, and the checkpoint sets ``use_classifier_free_guidance = False``; CFG is
switched on per call at ``diffusion.sample(use_classifier_free_guidance=True, ...)``.

Deviation from upstream's demo: that script requires two GPUs because it targets 80GB
cards, where the 72GB of weights leave no room to prefill 24 images twice. The extra
cost of CFG is in fact small -- Qwen3-VL's 8 KV heads over 64 layers make one cache
256 KiB/token, so a second 5k-token cache is ~1.2 GiB -- so on a 96GB card both
branches fit alongside the weights and every cross-device transfer disappears. This
module asserts single-device placement rather than moving caches around.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.models.alpamayo2_super import (
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import fuse_traj_tokens

#: Prompt layout with the navigation instruction. Upstream's default order
#: (``helper.create_messages``) is the unguided one below and has no nav slot at all.
GUIDED_COMPONENTS_ORDER = ["image", "traj_history", "nav_instruction", "prompt"]

#: Prompt layout without navigation -- identical to the released inference path.
UNGUIDED_COMPONENTS_ORDER = ["image", "traj_history", "prompt"]


def _empty_cuda_cache() -> None:
    """Release cached CUDA blocks between the two large VLM phases."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize_nav_prompts(
    payload: dict[str, Any],
    model: Any,
    processor: Any,
    nav_text: str,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Tokenize the guided (nav) and unguided (no-nav) prompts for one sample.

    Both prompts share the same images; only the text differs. The unguided prompt drops
    its own ``pixel_values`` because the guided visual payload is reused for its prefill,
    which keeps the vision tower's output identical across branches.

    Args:
        payload: Same dict the node feeds to ``helper.create_messages``.
        model: A loaded ``Alpamayo2Super``.
        processor: Cached processor from ``helper.get_processor``.
        nav_text: Navigation instruction, e.g. ``"Turn left in 30m"``.
        device: Device for the processor's image preprocessing.

    Returns:
        Dict with ``tokenized_data``, ``unguided_tokenized_data`` and the ego history.
    """
    if not nav_text:
        raise ValueError("nav_text must be a non-empty string")

    data = dict(payload)
    data["nav_text"] = [nav_text]
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()

    def tokenize(components_order: list[str]) -> dict[str, Any]:
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=model.config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=["cot", "traj_future"],
            generation_mode=True,
            include_camera_ids=model.config.include_camera_ids,
            camera_ids=data["camera_indices"],
            include_frame_nums=model.config.frame_label == "frame_num",
        )
        if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages = messages[:-1]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )
        return dict(
            processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
                device=device,
            )
        )

    tokenized_data = tokenize(GUIDED_COMPONENTS_ORDER)
    unguided_tokenized_data = tokenize(UNGUIDED_COMPONENTS_ORDER)
    unguided_tokenized_data.pop("pixel_values", None)
    unguided_tokenized_data.pop("image_grid_thw", None)

    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("nav CFG expects exactly one sample per call")

    return {
        "tokenized_data": tokenized_data,
        "unguided_tokenized_data": unguided_tokenized_data,
        "ego_history_xyz": payload["ego_history_xyz"],
        "ego_history_rot": payload["ego_history_rot"],
    }


def _build_unguided_continuation(
    guided_sequences: torch.Tensor,
    guided_prefix_length: int,
    unguided_input_ids: torch.Tensor,
    unguided_prefix_mask: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Replay the guided CoT on top of the unguided prefix.

    The chain of causation is generated once, by the guided branch. Feeding those same
    tokens through the nav-free prefix is what makes the two caches differ *only* by the
    navigation instruction -- and it keeps the expensive autoregressive decode from
    being paid twice.
    """
    generated = guided_sequences[:, guided_prefix_length:]
    attention_mask = torch.cat([unguided_prefix_mask, generated.ne(pad_token_id).long()], dim=1)
    cache_position = torch.arange(
        unguided_input_ids.shape[1],
        unguided_input_ids.shape[1] + generated.shape[1],
        device=unguided_input_ids.device,
        dtype=torch.long,
    )
    return {
        "input_ids": generated,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "full_sequences": torch.cat([unguided_input_ids, generated], dim=1),
    }


@torch.inference_mode()
def sample_with_nav_cfg(
    model: Any,
    model_inputs: dict[str, Any],
    diffusion_steps: int = 10,
    top_p: float = 0.98,
    temperature: float = 0.6,
    guidance_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], float]:
    """Sample one trajectory with navigation classifier-free guidance.

    Mirrors ``Alpamayo2Super.sample_trajectories_from_data`` but runs the VLM prefill
    twice -- once per prompt variant -- and integrates the flow field with both caches.

    Args:
        model: Loaded ``Alpamayo2Super`` with VLM and expert on the same CUDA device.
        model_inputs: Output of :func:`tokenize_nav_prompts`, already on device.
        diffusion_steps: Euler steps for the expert. Each step costs two expert forwards.
        top_p: Nucleus sampling parameter for the CoT rollout.
        temperature: Sampling temperature for the CoT rollout.
        guidance_weight: Overrides the checkpoint's ``inference_guidance_weight`` (3.0).
            1.0 reduces to plain conditioning; 0.0 discards the navigation instruction.

    Returns:
        ``(pred_xyz, pred_rot, logprob, extra, guidance_weight)``. ``logprob`` is a zeros
        placeholder, matching upstream -- it is not a confidence.
    """
    tokenized_data = dict(model_inputs["tokenized_data"])
    unguided_tokenized_data = dict(model_inputs["unguided_tokenized_data"])
    traj_data = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    num_traj_samples = 1

    tokenized_data["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        tokenized_data["input_ids"],
        traj_data,
        model.config.traj_ids,
    )
    unguided_input_ids = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        unguided_tokenized_data["input_ids"],
        traj_data,
        model.config.traj_ids,
    )
    unguided_tokenized_data["input_ids"] = unguided_input_ids

    vlm_device = tokenized_data["input_ids"].device
    expert_device = next(model.expert.parameters()).device
    if torch.device(expert_device) != torch.device(vlm_device):
        raise ValueError(
            "nav CFG in this module assumes the VLM and expert share one device "
            f"(got vlm={vlm_device}, expert={expert_device}). Use upstream's "
            "two_gpu_nav_cfg_demo.py for split placement."
        )

    # ---- guided branch: CoT rollout + KV cache ------------------------------------
    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max(256, model.config.tokens_per_future_traj)
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    logits_processor = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(
                traj_token_offset=min(
                    model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"]
                ),
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    eos_token_id = model.config.traj_ids["future_start"]
    _append_text_eos_mask(
        logits_processor, generation_config.eos_token_id, preserved_token_id=eos_token_id
    )

    vlm_outputs = model.vlm.generate(
        **tokenized_data,
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
        logits_processor=logits_processor,
    )
    if hasattr(vlm_outputs, "logits"):
        del vlm_outputs.logits
    _empty_cuda_cache()

    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    guided_prompt_cache = vlm_outputs.past_key_values
    batch_size = vlm_outputs.sequences.shape[0]
    num_diffusion_tokens = model.expert.action_space.get_action_space_dims()[0]

    guided_position_ids, guided_attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=find_eos_offset(
            sequences=vlm_outputs.sequences, eos_token_id=eos_token_id, device=vlm_device
        ),
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=guided_prompt_cache.get_seq_length(),
        n_diffusion_tokens=num_diffusion_tokens,
        b_star=batch_size,
        device=vlm_device,
        prefix_mask=tokenized_data.get("attention_mask"),
    )

    # ---- unguided branch: same images and same CoT, no navigation text -------------
    unguided_prefix_mask = unguided_tokenized_data.get("attention_mask")
    if unguided_prefix_mask is None:
        unguided_prefix_mask = unguided_input_ids.ne(model.tokenizer.pad_token_id).long()

    unguided_prefill = model.vlm(
        input_ids=unguided_input_ids,
        attention_mask=unguided_prefix_mask,
        image_grid_thw=tokenized_data.get("image_grid_thw"),
        pixel_values=tokenized_data.get("pixel_values"),
        use_cache=True,
        logits_to_keep=1,
    )
    unguided_prompt_cache = unguided_prefill.past_key_values
    del unguided_prefill
    _empty_cuda_cache()

    continuation = _build_unguided_continuation(
        guided_sequences=vlm_outputs.sequences,
        guided_prefix_length=tokenized_data["input_ids"].shape[1],
        unguided_input_ids=unguided_input_ids,
        unguided_prefix_mask=unguided_prefix_mask,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    guided_generated_tokens = continuation["input_ids"]
    unguided_vlm_outputs = model.vlm(
        input_ids=guided_generated_tokens,
        attention_mask=continuation["attention_mask"],
        past_key_values=unguided_prompt_cache,
        cache_position=continuation["cache_position"],
        use_cache=True,
        logits_to_keep=1,
    )
    unguided_prompt_cache = unguided_vlm_outputs.past_key_values
    unguided_rope_deltas = getattr(
        unguided_vlm_outputs, "rope_deltas", model.vlm.model.rope_deltas
    )
    if hasattr(unguided_vlm_outputs, "logits"):
        del unguided_vlm_outputs.logits
    _empty_cuda_cache()

    unguided_position_ids, unguided_attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=find_eos_offset(
            sequences=continuation["full_sequences"],
            eos_token_id=eos_token_id,
            device=vlm_device,
            warn=False,
        ),
        rope_deltas=unguided_rope_deltas,
        kv_cache_seq_len=unguided_prompt_cache.get_seq_length(),
        n_diffusion_tokens=num_diffusion_tokens,
        b_star=batch_size,
        device=vlm_device,
        prefix_mask=unguided_prefix_mask,
    )

    # ---- integrate the guided flow field ------------------------------------------
    forward_kwargs = {}
    if model.expert.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def expert_step(
        action: torch.Tensor,
        timestep: torch.Tensor,
        past_key_values: Any,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the flow field once against one prompt cache."""
        future_token_embeds = model.expert.action_in_proj(action, timestep)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, num_diffusion_tokens, -1)
        cache_len = past_key_values.get_seq_length()
        expert_outputs = model.expert.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        # The expert appends its action tokens to the prompt cache; drop them so the next
        # Euler step -- and the other branch -- see the prompt-only cache again.
        past_key_values.crop(cache_len)
        pred = model.expert.action_out_proj(expert_outputs.last_hidden_state)
        return pred.view(-1, *model.expert.action_space.get_action_space_dims())

    def guided_step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return expert_step(x, t, guided_prompt_cache, guided_position_ids, guided_attention_mask)

    def unguided_step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return expert_step(
            x, t, unguided_prompt_cache, unguided_position_ids, unguided_attention_mask
        )

    if guidance_weight is None:
        guidance_weight = float(model.expert.diffusion.inference_guidance_weight)

    action = model.expert.diffusion.sample(
        batch_size=batch_size,
        step_fn=guided_step_fn,
        unguided_step_fn=unguided_step_fn,
        device=expert_device,
        return_all_steps=False,
        inference_step=diffusion_steps,
        use_classifier_free_guidance=True,
        inference_guidance_weight=guidance_weight,
    )

    pred_xyz, pred_rot = model.expert.action_space.action_to_traj(
        action,
        traj_data["ego_history_xyz"][:, -1],
        traj_data["ego_history_rot"][:, -1],
    )
    pred_xyz = pred_xyz.reshape(batch_size, 1, 1, *pred_xyz.shape[1:])
    pred_rot = pred_rot.reshape(batch_size, 1, 1, *pred_rot.shape[1:])
    logprob = torch.zeros_like(pred_xyz[..., 0])

    extra = extract_text_tokens(model.tokenizer, guided_generated_tokens)
    for key in extra:
        extra[key] = np.array(extra[key]).reshape([1, 1, 1])
    return pred_xyz, pred_rot, logprob, extra, guidance_weight
