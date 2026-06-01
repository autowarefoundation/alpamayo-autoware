"""ONNX export of Alpamayo's expert denoiser step.

The exported (FP16/BF16) ONNX is the input to the FP8 quantization +
TensorRT-engine build in ``export_fp8`` / ``trt_fp8_runtime``. The expert must
run with **eager** attention before export (flash-attention is not
ONNX-exportable) — the build script sets
``model.expert.config._attn_implementation = "eager"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import torch
from transformers.cache_utils import DynamicCache

from alpamayo1_5.trt.common import (
    build_dynamic_axes,
    flatten_past_key_values,
    legacy_cache_from_prompt_cache,
    past_key_value_input_names,
)


class ExpertDenoiserExportModule(torch.nn.Module):
    """Wrap Alpamayo's expert denoiser step into a single exportable module."""

    def __init__(self, model: Any):
        super().__init__()
        self.action_in_proj = model.action_in_proj
        self.expert = model.expert
        self.action_out_proj = model.action_out_proj
        self.n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
        self.action_dims = tuple(model.action_space.get_action_space_dims())
        self.num_hidden_layers = int(self.expert.config.num_hidden_layers)
        self.is_causal = not bool(model.config.expert_non_causal_attention)
        try:
            self.action_in_proj_dtype = next(self.action_in_proj.parameters()).dtype
        except StopIteration:
            self.action_in_proj_dtype = torch.float32
        try:
            self.expert_dtype = next(self.expert.parameters()).dtype
        except StopIteration:
            self.expert_dtype = torch.float32

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        legacy_cache = []
        for idx in range(0, len(past_key_values), 2):
            past_key = past_key_values[idx].to(dtype=self.expert_dtype)
            past_value = past_key_values[idx + 1].to(dtype=self.expert_dtype)
            legacy_cache.append((past_key, past_value))
        prompt_cache = DynamicCache.from_legacy_cache(tuple(legacy_cache))

        future_token_embeds = self.action_in_proj(
            x.to(dtype=self.action_in_proj_dtype),
            t.to(dtype=self.action_in_proj_dtype),
        ).to(dtype=self.expert_dtype)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, self.n_diffusion_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids.to(dtype=torch.int64),
            past_key_values=prompt_cache,
            attention_mask=attention_mask.to(dtype=self.expert_dtype),
            use_cache=False,
            is_causal=self.is_causal,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens :]
        pred = self.action_out_proj(last_hidden).view(-1, *self.action_dims)
        return pred.to(dtype=self.expert_dtype)


def load_calibration_sample(sample_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a captured denoiser sample to a target device."""
    sample = torch.load(Path(sample_path), map_location="cpu", weights_only=False)
    return {k: v.to(device=device) for k, v in sample.items() if isinstance(v, torch.Tensor)}


def export_expert_denoiser_onnx(
    model: Any,
    sample: dict[str, torch.Tensor],
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    """Export Alpamayo's expert denoiser step to ONNX (external data)."""
    export_module = ExpertDenoiserExportModule(model).eval().to(device=sample["x"].device)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    past_key_values = legacy_cache_from_prompt_cache(
        tuple(
            (sample[f"past_key_{layer_idx:02d}"], sample[f"past_value_{layer_idx:02d}"])
            for layer_idx in range(export_module.num_hidden_layers)
        )
    )
    flat_past = flatten_past_key_values(past_key_values)
    input_names = [
        "x",
        "t",
        "position_ids",
        "attention_mask",
        *past_key_value_input_names(export_module.num_hidden_layers),
    ]

    torch.onnx.export(
        export_module,
        (
            sample["x"],
            sample["t"],
            sample["position_ids"],
            sample["attention_mask"],
            *flat_past,
        ),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=["pred"],
        dynamic_axes=build_dynamic_axes(export_module.num_hidden_layers),
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        external_data=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output_path))
    return output_path
