# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-native input preparation for Alpamayo 2 Edge inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torchvision.transforms.functional import InterpolationMode, resize
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from alpamayo2_edge.chat_template import (
    CLOSED_THINKING,
    build_conversation,
    build_vqa_conversation,
)
from alpamayo2_edge.config import resolve_checkpoint_name_or_path
from alpamayo2_edge.sample import canonicalize_sample, canonicalize_vqa_sample


def _check_image_layout(model_config: Any) -> None:
    if (
        getattr(model_config, "frame_label", "relative_time") != "relative_time"
        or getattr(model_config, "token_layout", "camera_ts") != "camera_ts"
        or getattr(model_config, "encode_as_video", False)
    ):
        raise ValueError(
            "Edge input preparation requires relative_time labels, camera_ts layout, and image encoding"
        )


def _canonical_inputs(data: dict[str, Any], model_config: Any) -> dict[str, Any]:
    history_steps = getattr(model_config, "history_num_steps", 20)
    expected_tokens = (history_steps - 1) * 3
    if model_config.tokens_per_history_traj != expected_tokens:
        raise ValueError(f"{history_steps} history poses require {expected_tokens} history tokens")
    _check_image_layout(model_config)
    return canonicalize_sample(data, history_num_steps=history_steps)


def create_messages(data: dict[str, Any], model_config: Any) -> list[dict[str, Any]]:
    """Build the driving prompt from raw capture-ordered camera data."""
    return build_conversation(
        _canonical_inputs(data, model_config),
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        include_camera_ids=model_config.include_camera_ids,
        system_prompt=getattr(model_config, "system_prompt", None),
    )


def get_processor(tokenizer: Any, model_config: Any) -> Any:
    """Read processor settings and the chat template from the model checkpoint."""
    checkpoint = resolve_checkpoint_name_or_path(model_config)
    if checkpoint is None:
        raise ValueError("model_config must include a checkpoint path or vlm_name_or_path")
    processor = AutoProcessor.from_pretrained(checkpoint, fix_mistral_regex=True)
    processor.tokenizer = tokenizer
    return processor


def _normalize_patches(patches: torch.Tensor, image_processor: Any) -> torch.Tensor:
    """Apply the checkpoint's fused uint8 rescale and normalization operation.

    Each flattened patch is laid out (height, width, channel, time).
    """
    channels = len(image_processor.image_mean)
    temporal = image_processor.temporal_patch_size
    patch_dim = channels * temporal * image_processor.patch_size**2
    if patches.dtype != torch.uint8 or patches.ndim != 2 or patches.shape[1] != patch_dim:
        raise ValueError(f"Processor must return uint8 image patches [N, {patch_dim}]")
    scale = 1.0 / image_processor.rescale_factor
    mean = (
        torch.tensor(image_processor.image_mean, dtype=torch.float32, device=patches.device) * scale
    )
    std = (
        torch.tensor(image_processor.image_std, dtype=torch.float32, device=patches.device) * scale
    )
    result = patches.float().reshape(-1, image_processor.patch_size**2, channels, temporal)
    result.sub_(mean.reshape(1, 1, channels, 1)).div_(std.reshape(1, 1, channels, 1))
    return result.reshape(-1, patch_dim)


def _tokenize_messages(
    messages: list[dict[str, Any]], processor: Any, *, include_closed_thinking: bool
) -> dict[str, Any]:
    """Tokenize messages and normalize resized uint8 camera images."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
        continue_final_message=False,
        enable_thinking=False,
    )
    if not text.endswith(CLOSED_THINKING):
        raise ValueError(f"The chat template must end the generation prompt with {CLOSED_THINKING}")
    if not include_closed_thinking:
        # Remove only the assistant suffix. Keep the question text unchanged.
        text = text[: -len(CLOSED_THINKING)]
    image_processor = processor.image_processor
    images = []
    for message in messages:
        for entry in message["content"]:
            if entry["type"] != "image":
                continue
            frame = entry["image"].squeeze(0)
            height, width = smart_resize(
                *frame.shape[-2:],
                factor=image_processor.patch_size * image_processor.merge_size,
                min_pixels=entry["min_pixels"],
                max_pixels=entry["max_pixels"],
            )
            images.append(resize(frame, [height, width], interpolation=InterpolationMode.BICUBIC))
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            padding=False,
            return_tensors="pt",
            return_mm_token_type_ids=True,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
    )
    tokenized_data["pixel_values"] = _normalize_patches(
        tokenized_data["pixel_values"], image_processor
    )
    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("Edge input preparation expects one sample at a time")
    return tokenized_data


def prepare_model_inputs(data: dict[str, Any], model_config: Any, tokenizer: Any) -> dict[str, Any]:
    """Pack one raw sample for ``sample_trajectories_from_data``.

    Resizing occurs on uint8 before normalization, preserving the training path.
    The returned history tensors include batch and trajectory-group dimensions:
    ``[1, 1, 20, 3]`` for positions and ``[1, 1, 20, 3, 3]`` for rotations.
    """
    canonical = _canonical_inputs(data, model_config)
    messages = build_conversation(
        canonical,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        include_camera_ids=model_config.include_camera_ids,
        system_prompt=getattr(model_config, "system_prompt", None),
    )
    tokenized_data = _tokenize_messages(
        messages, get_processor(tokenizer, model_config), include_closed_thinking=True
    )
    xyz, rot = canonical["ego_history_xyz"], canonical["ego_history_rot"]
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": xyz.unsqueeze(0) if xyz.ndim == 3 else xyz,
        "ego_history_rot": rot.unsqueeze(0) if rot.ndim == 4 else rot,
    }


def prepare_vqa_inputs(
    data: dict[str, Any], model_config: Any, tokenizer: Any, question: str
) -> dict[str, Any]:
    """Pack six cameras with four frames each and a question for text generation.

    Navigation and ego history are not required and do not enter the prompt.
    The assistant prefix has no empty thinking block.
    """
    _check_image_layout(model_config)
    messages = build_vqa_conversation(
        canonicalize_vqa_sample(data),
        question,
        include_camera_ids=model_config.include_camera_ids,
        system_prompt=getattr(model_config, "system_prompt", None),
    )
    return {
        "tokenized_data": _tokenize_messages(
            messages, get_processor(tokenizer, model_config), include_closed_thinking=False
        )
    }


def to_device(
    data: Any, device: str | torch.device | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Move nested tensors, casting only floating-point tensors to ``dtype``."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype if data.is_floating_point() else None)
    if isinstance(data, Mapping):
        return {key: to_device(value, device, dtype) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(to_device(value, device, dtype) for value in data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(value, device, dtype) for value in data]
    return data
