# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone configuration and compatibility for Edge release checkpoints."""

import copy
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

_TARGETS = {
    "alpamayo.tokenizer.trajectory.delta_tokenizer.DeltaTrajectoryTokenizer": "alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer",
    "alpamayo.action_space.discrete_action_space.DiscreteTrajectoryTokenizer": "alpamayo2_edge.action_space.discrete_action_space.DiscreteTrajectoryTokenizer",
    "alpamayo.action_space.UnicycleAccelCurvatureActionSpace": "alpamayo2_edge.action_space.UnicycleAccelCurvatureActionSpace",
    "alpamayo.action_space.UnicycleAccelCurvatureCourseActionSpace": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
    "alpamayo.diffusion.flow_matching.FlowMatching": "alpamayo2_edge.diffusion.flow_matching.FlowMatching",
    "reasoning_vla.submodules.action_in_proj.PerWaypointActionInProjV2": "alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2",
}
_ALLOWED_TARGETS = set(_TARGETS.values()) | {"torch.nn.Linear"}


def normalize_checkpoint_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Copy a legacy config, replacing only known release class/module names.

    Tensor dimensions, RoPE settings, token IDs and all trained numerical parameters
    are preserved. Unknown instantiation targets fail before Hydra imports anything.
    """
    result = copy.deepcopy(raw)
    result["model_type"] = "alpamayo2_edge"
    result["architectures"] = ["Alpamayo2Edge"]
    if isinstance(result.get("expert_config"), dict):
        result["expert_config"]["model_type"] = "alpamayo2_edge_expert"

    def visit(value):
        if isinstance(value, dict):
            if "_target_" in value:
                target = _TARGETS.get(value["_target_"], value["_target_"])
                if target not in _ALLOWED_TARGETS:
                    raise ValueError(f"Unsupported checkpoint target: {target!r}")
                value["_target_"] = target
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def resolve_checkpoint_name_or_path(config: PretrainedConfig) -> str | None:
    """Prefer the resolved checkpoint, keeping all processor assets self-contained."""
    return getattr(config, "_name_or_path", None) or getattr(config, "vlm_name_or_path", None)


def build_alpamayo2_edge_tokenizer(
    name_or_path: str,
    history_vocab_size: int = 1000,
    future_vocab_size: int = 3000,
) -> AutoTokenizer:
    """Load saved tokens without adding tokens or resizing trained embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, padding_side="left")
    for index in (0, history_vocab_size, history_vocab_size + future_vocab_size - 1):
        token = f"<i{index}>"
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Checkpoint tokenizer is missing required trajectory token {token}")
    if tokenizer.pad_token_id is None:
        raise ValueError("Checkpoint tokenizer must define pad_token_id")
    return tokenizer


class Alpamayo2EdgeConfig(PretrainedConfig):
    """Cosmos3-Edge backbone with the trained trajectory diffusion expert."""

    model_type = "alpamayo2_edge"

    def __init__(
        self,
        vlm_config: dict | PretrainedConfig | None = None,
        expert_config: dict | PretrainedConfig | None = None,
        hist_traj_tokenizer_cfg: dict | None = None,
        future_traj_tokenizer_cfg: dict | None = None,
        vlm_name_or_path: str | None = None,
        vlm_class: str = "Cosmos3EdgeForConditionalGeneration",
        history_vocab_size: int = 1000,
        future_vocab_size: int = 3000,
        tokens_per_history_traj: int = 57,
        tokens_per_future_traj: int = 128,
        history_num_steps: int | None = None,
        frame_label: str = "relative_time",
        token_layout: str = "camera_ts",
        include_camera_ids: bool = True,
        min_pixels: int = 163840,
        max_pixels: int = 196608,
        padding_side: str = "left",
        traj_ids: dict | None = None,
        enable_expert: bool = True,
        dtype: str | torch.dtype = "bfloat16",
        **kwargs: Any,
    ):
        kwargs.pop("model_type", None)
        kwargs.pop("architectures", None)
        super().__init__(dtype=dtype, architectures=["Alpamayo2Edge"], **kwargs)
        normalized = normalize_checkpoint_config(
            {
                "hist_traj_tokenizer_cfg": hist_traj_tokenizer_cfg
                or {
                    "_target_": "alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer",
                    "pad_origin_at_beginning": False,
                },
                "future_traj_tokenizer_cfg": future_traj_tokenizer_cfg or {},
                "expert_config": expert_config if isinstance(expert_config, dict) else None,
            }
        )
        self.hist_traj_tokenizer_cfg = normalized["hist_traj_tokenizer_cfg"]
        self.future_traj_tokenizer_cfg = normalized["future_traj_tokenizer_cfg"]
        axes = 4 if self.hist_traj_tokenizer_cfg.get("predict_yaw", False) else 3
        if tokens_per_history_traj <= 0 or tokens_per_history_traj % axes:
            raise ValueError(f"tokens_per_history_traj must be positive and divisible by {axes}")
        expected_steps = tokens_per_history_traj // axes + (
            not self.hist_traj_tokenizer_cfg.get("pad_origin_at_beginning", True)
        )
        if history_num_steps is not None and history_num_steps != expected_steps:
            raise ValueError(f"history_num_steps must be {expected_steps}, got {history_num_steps}")
        self.history_num_steps = int(expected_steps)
        if frame_label not in ("relative_time", "frame_num", "none"):
            raise ValueError(f"Unsupported frame_label: {frame_label!r}")
        if token_layout != "camera_ts":
            raise ValueError('Edge inference currently requires token_layout="camera_ts"')
        if getattr(self, "encode_as_video", False):
            raise ValueError("Edge inference expects image inputs, not encode_as_video")
        if vlm_class != "Cosmos3EdgeForConditionalGeneration":
            raise ValueError(f"Unsupported Edge backbone: {vlm_class!r}")
        self.vlm_config = (
            AutoConfig.for_model(**vlm_config) if isinstance(vlm_config, dict) else vlm_config
        )
        if isinstance(expert_config, dict):
            from alpamayo2_edge.models.expert import ExpertModelConfig

            self.expert_config = ExpertModelConfig(**normalized["expert_config"])
        else:
            self.expert_config = expert_config
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_class = vlm_class
        self.history_vocab_size = history_vocab_size
        self.future_vocab_size = future_vocab_size
        self.traj_vocab_size = history_vocab_size + future_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.frame_label = frame_label
        self.token_layout = token_layout
        self.include_camera_ids = include_camera_ids
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.padding_side = padding_side
        self.traj_ids = traj_ids or {}
        self.enable_expert = enable_expert
        self.set_dtype(dtype)

    def set_dtype(self, dtype: str | torch.dtype | None = None) -> None:
        dtype = dtype or self.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if self.vlm_config is not None:
            self.vlm_config.dtype = dtype
            self.vlm_config.text_config.dtype = dtype
            self.vlm_config.vision_config.dtype = dtype
        if self.expert_config is not None and self.expert_config.llm_config is not None:
            self.expert_config.dtype = dtype
            self.expert_config.llm_config.dtype = dtype


AutoConfig.register(Alpamayo2EdgeConfig.model_type, Alpamayo2EdgeConfig)
