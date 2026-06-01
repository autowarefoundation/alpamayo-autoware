"""Raw-TensorRT FP8 runtime for the Alpamayo expert denoiser.

Drop-in replacement for ``expert_runtime.TrtExpertEngine`` (same
``prepare_context`` / ``step`` surface, so it plugs into
``Alpamayo1_5.set_expert_step_runner``), but executes a **STRONGLY_TYPED**
TensorRT engine built from an FP8-Q/DQ ONNX (see ``export_fp8``).

We use the TensorRT Python API directly rather than ONNX Runtime's TRT EP
because TRT 10/11 derives precision from the network's types (STRONGLY_TYPED)
and consumes the FP8 Q/DQ nodes natively, giving real FP8 tensor-core kernels
on Blackwell. IO is bound directly to torch CUDA tensors via ``data_ptr`` —
no host copies.

Measured on RTX PRO 6000 Blackwell (single denoiser step): PyTorch 14.8ms →
TRT FP8 7.3ms (~2.0x), engine 2.29GB (half the FP16 engine), max|Δ|≈0.023.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import tensorrt as trt
import torch

from alpamayo1_5.trt.common import (
    flatten_past_key_values,
    legacy_cache_from_prompt_cache,
    past_key_value_input_names,
    resolve_artifact_dir,
)

_TRT_TO_TORCH = {
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def make_optimization_profile(
    builder: "trt.Builder",
    network: "trt.INetworkDefinition",
    opt_shapes: dict[str, tuple[int, ...]],
    num_layers: int,
    max_batch: int = 8,
    min_seq: int = 64,
    max_seq: int = 4096,
) -> "trt.IOptimizationProfile":
    """Build a TRT optimization profile for the expert's dynamic axes.

    The exported ONNX has dynamic ``batch`` (dim 0), ``prompt_seq_len`` (KV
    dim 2) and ``total_seq_len`` (attention-mask dim 3) — see
    ``common.build_dynamic_axes``. ``opt_shapes`` are the representative
    (calibration-sample) shapes used as the profile's optimum.
    """
    from alpamayo1_5.trt.common import build_dynamic_axes

    axes = build_dynamic_axes(num_layers)
    profile = builder.create_optimization_profile()

    # total_seq_len = prompt_seq_len + n_diffusion_tokens, so the two seq axes
    # must move together or TRT flags the profile as "not self-consistent".
    prompt_opt = total_opt = None
    for name, dims in axes.items():
        for dim, axis_name in dims.items():
            if axis_name == "prompt_seq_len":
                prompt_opt = opt_shapes[name][dim]
            elif axis_name == "total_seq_len":
                total_opt = opt_shapes[name][dim]
    n_diff = (total_opt - prompt_opt) if (prompt_opt is not None and total_opt is not None) else 0

    for i in range(network.num_inputs):
        name = network.get_input(i).name
        opt = list(opt_shapes[name])
        mn, mx = list(opt), list(opt)
        for dim, axis_name in axes.get(name, {}).items():
            if axis_name == "batch":
                mn[dim], mx[dim] = 1, max(max_batch, opt[dim])
            elif axis_name == "prompt_seq_len":
                mn[dim], mx[dim] = min(min_seq, opt[dim]), max(max_seq, opt[dim])
            elif axis_name == "total_seq_len":
                mn[dim], mx[dim] = min(min_seq, opt[dim]) + n_diff, max(max_seq, opt[dim]) + n_diff
        profile.set_shape(name, tuple(mn), tuple(opt), tuple(mx))
    return profile


def build_fp8_engine(
    qdq_onnx_path: str | Path,
    engine_path: str | Path,
    opt_shapes: dict[str, tuple[int, ...]] | None = None,
    num_layers: int = 0,
    workspace_gb: int = 4,
    max_batch: int = 8,
    max_seq: int = 4096,
) -> Path:
    """Build a STRONGLY_TYPED TRT engine from an FP8 Q/DQ ONNX and serialize it.

    Precision is fully determined by the ONNX dtypes + Q/DQ nodes — TRT 10/11
    has no FP16/FP8 ``BuilderFlag`` to set. The exported graph has dynamic
    batch / prompt-length axes, so an optimization profile is required;
    ``opt_shapes`` (representative input shapes, e.g. a calibration sample)
    supplies the profile's optimum.
    """
    qdq_onnx_path = str(Path(qdq_onnx_path).expanduser().resolve())
    engine_path = Path(engine_path).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(qdq_onnx_path):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if opt_shapes is not None:
        config.add_optimization_profile(
            make_optimization_profile(
                builder, network, opt_shapes, num_layers,
                max_batch=max_batch, max_seq=max_seq,
            )
        )
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    engine_path.write_bytes(serialized)
    print(
        f"[fp8] TRT engine built in {time.perf_counter() - t0:.1f}s -> {engine_path} "
        f"({bytes(serialized).__sizeof__() / 1e6:.0f} MB)"
    )
    return engine_path


@dataclass
class TrtFp8Context:
    inputs: dict[str, torch.Tensor]  # static prompt-side inputs (KV, pos, mask) at prompt batch
    batched: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)  # batch-expanded cache


class TrtFp8ExpertEngine:
    """Runtime wrapper around a serialized FP8 expert engine.

    Interface mirrors ``expert_runtime.TrtExpertEngine`` so it can be passed to
    ``Alpamayo1_5.set_expert_step_runner``.
    """

    def __init__(self, engine_path: str | Path, device: str = "cuda") -> None:
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(logger)
        with open(Path(engine_path).expanduser().resolve(), "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()

        self._names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        self._dtypes = {n: _TRT_TO_TORCH[self._engine.get_tensor_dtype(n)] for n in self._names}
        self._is_input = {
            n: self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self._names
        }
        self._output_name = next(n for n in self._names if not self._is_input[n])
        self._stream = torch.cuda.Stream()

    def prepare_context(self, prompt_cache, position_ids, attention_mask) -> TrtFp8Context:
        """Convert the static prompt-side tensors (KV cache, position ids, mask)
        into device tensors of the engine's expected dtypes (done once per
        trajectory, reused across denoiser steps)."""
        legacy = legacy_cache_from_prompt_cache(prompt_cache)
        kv_names = past_key_value_input_names(len(legacy))
        inputs: dict[str, torch.Tensor] = {
            "position_ids": position_ids.detach().to(self.device, self._dtypes["position_ids"]).contiguous(),
            "attention_mask": attention_mask.detach().to(self.device, self._dtypes["attention_mask"]).contiguous(),
        }
        for name, tensor in zip(kv_names, flatten_past_key_values(legacy)):
            inputs[name] = tensor.detach().to(self.device, self._dtypes[name]).contiguous()
        return TrtFp8Context(inputs=inputs)

    def _batched_inputs(self, context: TrtFp8Context, batch: int) -> dict[str, torch.Tensor]:
        """Broadcast the prompt-side inputs (prepared at the VLM batch, usually 1)
        to the diffusion batch (B * num_traj_samples), matching how the native
        expert broadcasts the KV cache. Cached per batch (built once per rollout)."""
        cached = context.batched.get(batch)
        if cached is not None:
            return cached
        out: dict[str, torch.Tensor] = {}
        for name, tn in context.inputs.items():
            bdim = 1 if name == "position_ids" else 0  # KV/mask: dim0; position_ids: dim1
            if tn.shape[bdim] == batch:
                out[name] = tn
            elif tn.shape[bdim] == 1:
                shape = list(tn.shape)
                shape[bdim] = batch
                out[name] = tn.expand(*shape).contiguous()
            else:
                raise RuntimeError(
                    f"cannot broadcast {name} batch {tn.shape[bdim]} -> {batch}"
                )
        context.batched[batch] = out
        return out

    def step(self, x: torch.Tensor, t: torch.Tensor, context: TrtFp8Context) -> torch.Tensor:
        """Run a single denoiser step on the FP8 engine."""
        ctx = self._context
        bound: dict[str, torch.Tensor] = dict(self._batched_inputs(context, x.shape[0]))
        bound["x"] = x.detach().to(self.device, self._dtypes["x"]).contiguous()
        bound["t"] = t.detach().to(self.device, self._dtypes["t"]).contiguous()

        for name, tensor in bound.items():
            ctx.set_input_shape(name, tuple(tensor.shape))
            ctx.set_tensor_address(name, tensor.data_ptr())

        out_shape = tuple(ctx.get_tensor_shape(self._output_name))
        output = torch.empty(out_shape, dtype=self._dtypes[self._output_name], device=self.device)
        ctx.set_tensor_address(self._output_name, output.data_ptr())

        with torch.cuda.stream(self._stream):
            ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return output
