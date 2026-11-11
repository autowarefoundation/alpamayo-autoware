# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in inference acceleration for Alpamayo 2 Edge.

``enable_acceleration`` installs ``torch.compile`` plus CUDA graphs on one loaded
model: the prompt prefill, CoT decoding with a graphed sampler and the whole
flow-matching loop. Nothing here changes the default inference path.
"""

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import torch

from alpamayo2_edge.accel.decode import DecodeAccelerator, FlashKVCache
from alpamayo2_edge.accel.expert import ExpertAccelerator
from alpamayo2_edge.accel.graphs import AccelStats
from alpamayo2_edge.accel.kernels import ATTENTION_CHOICES, PARTS, resolve_kernels

MODES = ("compile", "tuned")
# Inductor mode of the decode step; every other part uses the default mode.
_DECODE_MODES = {"compile": "default", "tuned": "max-autotune-no-cudagraphs"}
# One compiled function serves every shape of its part, with one recompile per shape.
_RECOMPILE_LIMIT = 64
_recompile_limit = {"users": 0, "saved": None}


def _raise_recompile_limit() -> None:
    config = torch._dynamo.config
    if _recompile_limit["users"] == 0:
        _recompile_limit["saved"] = config.recompile_limit
        config.recompile_limit = max(config.recompile_limit, _RECOMPILE_LIMIT)
    _recompile_limit["users"] += 1


def _restore_recompile_limit() -> None:
    """Restore dynamo's limit once no accelerated model is left (unless changed since)."""
    _recompile_limit["users"] -= 1
    config = torch._dynamo.config
    saved = _recompile_limit["saved"]
    if _recompile_limit["users"] == 0 and config.recompile_limit == max(saved, _RECOMPILE_LIMIT):
        config.recompile_limit = saved


@dataclass(frozen=True)
class AccelConfig:
    """Acceleration settings. The defaults are the validated configuration.

    ``mode="tuned"`` compiles the decode step with inductor's
    ``max-autotune-no-cudagraphs`` mode, and ``"compile"`` uses the default mode
    everywhere. ``attention`` selects the prefill kernel: ``"auto"`` uses
    FlashAttention 3 on Hopper when it is installed, and PyTorch SDPA otherwise.
    Decode and expert attention always use FlashAttention 2 ``flash_attn_with_kvcache``.
    Shapes are bucketed to ``kv_bucket`` tokens. ``warmup`` eager runs precede each
    CUDA-graph capture. ``compile=False`` and ``cuda_graphs=False`` run the same code
    eagerly, for debugging and CPU tests.
    """

    mode: str = "tuned"
    attention: str = "auto"
    parts: tuple[str, ...] = PARTS
    kv_bucket: int = 256
    warmup: int = 2
    compile: bool = True
    cuda_graphs: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.attention not in ATTENTION_CHOICES:
            raise ValueError(f"attention must be one of {ATTENTION_CHOICES}")
        unknown = set(self.parts) - set(PARTS)
        if unknown:
            raise ValueError(f"Unknown parts {sorted(unknown)}; choose from {PARTS}")
        if self.kv_bucket < 1 or self.warmup < 0:
            raise ValueError("kv_bucket must be positive and warmup nonnegative")
        if self.cuda_graphs and self.warmup < 1:
            # Compilation and cuBLAS initialization cannot run inside a graph capture.
            raise ValueError("CUDA graphs need at least one warm-up run before capture")


class AccelHandle:
    """Installed accelerators of one model: their settings, graph builds and fallbacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: AccelConfig,
        kernels: dict[str, str | None],
        skipped: dict[str, str],
        stats: AccelStats,
        accelerators: list[Any],
    ) -> None:
        self.model, self.config, self.kernels = model, config, kernels
        self.skipped, self.stats, self.accelerators = skipped, stats, accelerators
        self.parts = tuple(part for part in PARTS if part in config.parts and part not in skipped)

    def describe(self) -> dict[str, Any]:
        """JSON-ready settings, resolved kernels, graph builds and fallback counts."""
        return {
            "config": asdict(self.config),
            "parts": list(self.parts),
            "skipped": dict(self.skipped),
            "kernels": {part: self.kernels[part] for part in self.parts},
            **self.stats.as_dict(),
        }

    def warmup(self, inputs: dict[str, Any], **sample_kwargs: Any) -> None:
        """Compile and capture the graphs for ``inputs`` without changing RNG state."""
        device = self.model.device
        with (
            torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
            torch.inference_mode(),
            torch.autocast(device.type, dtype=self.model.dtype, enabled=device.type == "cuda"),
        ):
            self.model.sample_trajectories_from_data(inputs, **sample_kwargs)

    def remove(self) -> None:
        """Restore the original methods (wrappers installed later must be removed first)."""
        if self.model._alpamayo2_edge_accel is not self:
            return
        for accelerator in reversed(self.accelerators):
            accelerator.remove()
        self.accelerators = []
        self.model._alpamayo2_edge_accel = None
        _restore_recompile_limit()


def enable_acceleration(model: torch.nn.Module, config: AccelConfig | None = None) -> AccelHandle:
    """Install the selected accelerators on ``model`` (an ``Alpamayo2Edge`` instance).

    The first call for each new shape compiles and captures graphs inline (tens of
    seconds); later calls replay them. Inputs outside the supported envelope run
    the original eager code, counted in ``handle.describe()["fallbacks"]``. Without
    FlashAttention 2, or without the decode part that the prefill and expert build
    on, the requested parts all stay eager: ``handle.skipped`` lists why, and a
    ``RuntimeWarning`` reports it.
    """
    config = config or AccelConfig()
    if getattr(model, "_alpamayo2_edge_accel", None) is not None:
        raise RuntimeError("Acceleration is already enabled on this model")
    if not model.config.enable_expert:
        raise ValueError("Acceleration requires the trajectory expert")
    device = model.device
    if config.cuda_graphs and device.type != "cuda":
        raise ValueError("CUDA graphs require a model on a CUDA device")
    if device.type == "cuda" and model.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FlashAttention kernels require a BF16 or FP16 model")
    kernels = resolve_kernels(config.attention, device)
    requested = set(config.parts)
    skipped: dict[str, str] = {}
    for part in ("decode", "expert"):
        if part in requested and kernels[part] is None:
            skipped[part] = "requires flash_attn (FlashAttention 2)"
    # Prefill writes, and the expert reads, the decode part's flash KV cache.
    has_decode = "decode" in requested and "decode" not in skipped
    for part in ("prefill", "expert"):
        if part in requested and part not in skipped and not has_decode:
            skipped[part] = "requires the decode part"
    parts = [part for part in PARTS if part in requested and part not in skipped]
    if skipped:
        skipped = {part: skipped[part] for part in PARTS if part in skipped}
        reasons = "; ".join(f"{part} {reason}" for part, reason in skipped.items())
        warnings.warn(
            f"Acceleration runs {', '.join(skipped)} eagerly ({reasons}); accelerated parts: "
            f"{', '.join(parts) or 'none'}. See docs/acceleration.md#requirements.",
            RuntimeWarning,
            stacklevel=2,
        )
    stats = AccelStats()
    common = dict(
        stats=stats,
        compile=config.compile,
        cuda_graphs=config.cuda_graphs,
        warmup=config.warmup,
    )
    accelerators: list[Any] = []
    if "decode" in parts:
        accelerators.append(
            DecodeAccelerator(
                model,
                **common,
                kv_bucket=config.kv_bucket,
                decode_mode=_DECODE_MODES[config.mode],
                mode="default",
                graph_prefill="prefill" in parts,
                prefill_fa3=kernels["prefill"] == "fa3",
            )
        )
    if "expert" in parts:
        accelerators.append(ExpertAccelerator(model, **common, mode="default"))
    handle = AccelHandle(model, config, kernels, skipped, stats, accelerators)
    model._alpamayo2_edge_accel = handle
    _raise_recompile_limit()
    return handle


__all__ = [
    "AccelConfig",
    "AccelHandle",
    "FlashKVCache",
    "enable_acceleration",
    "resolve_kernels",
]
