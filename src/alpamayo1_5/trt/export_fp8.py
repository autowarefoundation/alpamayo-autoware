"""FP8 (E4M3) quantization for the Alpamayo expert denoiser — the NVIDIA
ModelOpt "official method" adapted for real TensorRT deployment.

Why this exists alongside the INT8 path in ``export.py``:

* NVIDIA's Alpamayo quant recipe (``alpamayo-recipes``) calibrates the model in
  **FP8** with ModelOpt — the right precision for Blackwell (sm_120) tensor
  cores. On an RTX PRO 6000 Blackwell, an FP8 expert-step engine runs ~2.0x
  faster than PyTorch (vs ~1.6x for FP16) at half the engine size, with
  negligible trajectory error (max|Δ|≈0.02).
* ModelOpt's own ``torch.onnx`` FP8 export is **broken on torch 2.12** (the
  TorchScript exporter rejects the activation ``amax`` as a non-constant, and
  the dynamo exporter has no ONNX function for ``tensorrt.quantize_op``). So we
  use ModelOpt only to *calibrate* per-Linear amax, then insert FP8 Q/DQ nodes
  into the plain exported ONNX ourselves (``insert_fp8_qdq``).

Pipeline: export plain ONNX (``export.export_expert_denoiser_onnx``) →
``calibrate_fp8_amax`` (ModelOpt) → ``insert_fp8_qdq`` → build a STRONGLY_TYPED
TRT engine (``trt_fp8_runtime.build_fp8_engine``).

Requires the FP8 build env (not needed at ROS runtime): ``nvidia-modelopt``,
``onnx>=1.21`` (FLOAT8E4M3FN + ml_dtypes), ``tensorrt>=10``, torch 2.12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# FP8 E4M3 has a maximum representable magnitude of 448.0; scale = amax / 448.
E4M3_MAX = 448.0

# Quantize only the expert transformer's Linear layers (q/k/v/o_proj, MLP) —
# the bulk of the FLOPs. Attention BMM/softmax and the small projection heads
# (action_in_proj / action_out_proj) stay in FP16/BF16 for fidelity, matching
# the INT8 path's ``discover_nodes_to_exclude`` policy.
FP8_LINEAR_CFG: dict[str, Any] = {
    "quant_cfg": {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*action_in_proj*": {"enable": False},
        "*action_out_proj*": {"enable": False},
    },
    "algorithm": "max",
}


def calibrate_fp8_amax(
    export_module: torch.nn.Module,
    calibration_forward_loop: Any,
) -> dict[str, dict[str, float]]:
    """ModelOpt-calibrate ``export_module`` in FP8 and return per-Linear amax.

    ``calibration_forward_loop(module)`` must run a few forward passes on
    ``export_module`` over representative denoiser-step inputs.

    Returns ``{linear_fqn: {"input_quantizer": amax, "weight_quantizer": amax}}``
    for the (per-tensor) quantized Linears only.
    """
    import modelopt.torch.quantization as mtq

    mtq.quantize(export_module, FP8_LINEAR_CFG, forward_loop=calibration_forward_loop)

    amax: dict[str, dict[str, float]] = {}
    for name, mod in export_module.named_modules():
        entry: dict[str, float] = {}
        for qn in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qn, None)
            amax_t = getattr(q, "amax", None) if q is not None else None
            if amax_t is not None and amax_t.numel() == 1:
                entry[qn] = float(amax_t.detach().flatten()[0].item())
        if entry:
            amax[name] = entry
    return amax


def _matmul_node_to_linear_fqn(node_name: str) -> str:
    """``/expert/layers.0/self_attn/q_proj/MatMul`` -> ``expert.layers.0.self_attn.q_proj``.

    Only ``/`` becomes ``.`` — the ``.`` in ``layers.0`` (ModuleList index) is
    already present in the ONNX node name.
    """
    return node_name.strip("/").removesuffix("/MatMul").replace("/", ".")


def insert_fp8_qdq(
    source_onnx_path: str | Path,
    amax: dict[str, dict[str, float]],
    output_onnx_path: str | Path,
) -> Path:
    """Insert FP8 (E4M3) QuantizeLinear/DequantizeLinear pairs into the plain
    expert ONNX, on the activation and weight inputs of each calibrated Linear.

    The Q/DQ scale tensors are emitted as **BFloat16** to match the expert
    graph's dtype — otherwise STRONGLY_TYPED TensorRT fails with a Concat type
    mismatch (the DequantizeLinear output dtype follows the scale dtype).
    """
    import ml_dtypes
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source_onnx_path = Path(source_onnx_path)
    output_onnx_path = Path(output_onnx_path)

    model = onnx.load(str(source_onnx_path), load_external_data=False)
    g = model.graph

    targets: dict[str, dict[str, float]] = {}
    for n in g.node:
        if n.op_type == "MatMul":
            fqn = _matmul_node_to_linear_fqn(n.name)
            if fqn in amax:
                targets[n.name] = amax[fqn]

    new_nodes: list[Any] = []
    new_inits: list[Any] = []

    def add_qdq(tensor_name: str, amax_val: float, tag: str) -> str:
        scale = float(amax_val) / E4M3_MAX
        s_name, z_name = f"{tag}_scale", f"{tag}_zp"
        new_inits.append(numpy_helper.from_array(np.array(scale, dtype=ml_dtypes.bfloat16), s_name))
        new_inits.append(helper.make_tensor(z_name, TensorProto.FLOAT8E4M3FN, [], [0.0]))
        q_out, dq_out = f"{tag}_q", f"{tag}_dq"
        new_nodes.append(helper.make_node("QuantizeLinear", [tensor_name, s_name, z_name], [q_out], name=f"{tag}_QL"))
        new_nodes.append(helper.make_node("DequantizeLinear", [q_out, s_name, z_name], [dq_out], name=f"{tag}_DQL"))
        return dq_out

    by_name = {n.name: n for n in g.node}
    n_act = n_wt = 0
    for mm, entry in targets.items():
        node = by_name[mm]
        if "input_quantizer" in entry:
            node.input[0] = add_qdq(node.input[0], entry["input_quantizer"], f"{mm}/act")
            n_act += 1
        if "weight_quantizer" in entry:
            node.input[1] = add_qdq(node.input[1], entry["weight_quantizer"], f"{mm}/wt")
            n_wt += 1

    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)

    # Topologically re-sort (the Q/DQ nodes were appended at the end).
    available = {init.name for init in g.initializer} | {i.name for i in g.input}
    pending, ordered = list(g.node), []
    while pending:
        still, progressed = [], False
        for nd in pending:
            if all(inp == "" or inp in available for inp in nd.input):
                ordered.append(nd)
                available.update(nd.output)
                progressed = True
            else:
                still.append(nd)
        pending = still
        if not progressed:
            raise RuntimeError(f"FP8 Q/DQ toposort stuck: {len(pending)} nodes unresolved")
    del g.node[:]
    g.node.extend(ordered)

    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = max(op.version, 19)  # FLOAT8E4M3FN needs opset >= 19

    onnx.save(model, str(output_onnx_path))  # external data refs (same dir) stay valid
    onnx.checker.check_model(str(output_onnx_path), full_check=False)
    print(f"[fp8] matched {len(targets)} linears; inserted Q/DQ on {n_act} acts + {n_wt} weights")
    return output_onnx_path


def dump_amax(amax: dict[str, dict[str, float]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(amax, indent=1), encoding="utf-8")
