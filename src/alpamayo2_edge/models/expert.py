# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-compatible Cosmos3-Edge action expert for trajectory inference."""

import math
from collections.abc import Mapping
from typing import Any

import hydra.utils as hyu
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers import initialization as init

from alpamayo2_edge.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from alpamayo2_edge.models.action_in_proj import FourierEncoderV2


class ExpertModelConfig(PretrainedConfig):
    """Configuration of the expert backbone, action projections and flow sampler."""

    model_type = "alpamayo2_edge_expert"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        llm_config: PretrainedConfig | Mapping[str, Any] | None = None,
        action_space_cfg: Mapping[str, Any] | None = None,
        diffusion_cfg: Mapping[str, Any] | None = None,
        action_in_proj_cfg: Mapping[str, Any] | None = None,
        action_out_proj_cfg: Mapping[str, Any] | None = None,
        expert_non_causal_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(llm_config, Mapping):
            llm_config = AutoConfig.for_model(**dict(llm_config))
        self.llm_config = llm_config
        self.action_space_cfg = action_space_cfg
        self.diffusion_cfg = diffusion_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_non_causal_attention = expert_non_causal_attention


class ExpertModel(PreTrainedModel):
    """The 512-wide expert sharing the trained VLM's KV head geometry.

    Module names are intentionally identical to the checkpoint's ``expert.*``
    state dictionary. Denoising is orchestrated by ``Alpamayo2Edge``.
    """

    config_class = ExpertModelConfig
    base_model_prefix = "expert"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: ExpertModelConfig) -> None:
        super().__init__(config)
        if not config.expert_non_causal_attention:
            raise ValueError("Alpamayo2Edge requires noncausal expert attention.")
        self.expert = AutoModel.from_config(config.llm_config)
        del self.expert.embed_tokens
        self.action_space = hyu.instantiate(config.action_space_cfg)
        self.diffusion = hyu.instantiate(
            config.diffusion_cfg, x_dims=self.action_space.get_action_space_dims()
        )
        if self.diffusion.use_classifier_free_guidance:
            raise ValueError("Classifier-free guidance is not supported by Alpamayo2Edge.")
        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=self.expert.config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=self.expert.config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )
        # The release saves learned tensors only. These seven deterministic
        # constants are reconstructed from config, including after meta loading.
        for module in self.action_in_proj.modules():
            if isinstance(module, FourierEncoderV2):
                module._non_persistent_buffers_set.add("freqs")
        self.action_space._non_persistent_buffers_set.update(
            {"accel_mean", "accel_std", "curvature_mean", "curvature_std"}
        )
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, FourierEncoderV2):
            max_freq = self.config.action_in_proj_cfg.get("max_freq", 100.0)
            frequencies = torch.logspace(
                0,
                math.log10(max_freq),
                module.out_dim // 2,
                device=module.freqs.device,
                dtype=module.freqs.dtype,
            )[None, :]
            init.copy_(module.freqs, frequencies)
        elif isinstance(module, UnicycleAccelCurvatureActionSpace):
            for name, default in (
                ("accel_mean", 0.0),
                ("accel_std", 1.0),
                ("curvature_mean", 0.0),
                ("curvature_std", 1.0),
            ):
                value = self.config.action_space_cfg.get(name, default)
                init.copy_(getattr(module, name), torch.as_tensor(value))


AutoConfig.register(ExpertModelConfig.model_type, ExpertModelConfig)
AutoModel.register(ExpertModelConfig, ExpertModel)
