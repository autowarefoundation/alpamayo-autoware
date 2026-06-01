#!/usr/bin/env python3
"""Build an FP8 TensorRT engine for Alpamayo's expert denoiser step.

This is the NVIDIA ModelOpt "official method" (FP8) adapted for real TensorRT
deployment, replacing the previous ORT/INT8 SmoothQuant path. FP8 is the right
precision for Blackwell (sm_120): on an RTX PRO 6000 Blackwell the expert step
runs ~2.0x faster than PyTorch (7.3ms vs 14.8ms) at half the engine size, with
negligible trajectory error.

Flow: capture denoiser-step calibration inputs (expert-step observer) → export
plain ONNX → ModelOpt FP8 calibration for per-Linear amax → insert FP8 Q/DQ →
build a STRONGLY_TYPED TRT engine → validate native-PyTorch vs FP8-engine.

Build-time deps (NOT needed at ROS runtime): nvidia-modelopt, onnx>=1.21,
tensorrt>=10, torch 2.12. See scripts/requirements-trt-build.txt.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.trt.common import (
    flatten_past_key_values,
    legacy_cache_from_prompt_cache,
    past_key_value_input_names,
    resolve_artifact_dir,
)
from alpamayo1_5.trt.export import (
    ExpertDenoiserExportModule,
    export_expert_denoiser_onnx,
    load_calibration_sample,
)
from alpamayo1_5.trt.export_fp8 import calibrate_fp8_amax, dump_amax, insert_fp8_qdq
from alpamayo1_5.trt.trt_fp8_runtime import TrtFp8ExpertEngine, build_fp8_engine


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000
DEFAULT_OUTPUT_DIR = "~/autoware_data/alpamayo/v0.1"


class CalibrationCollector:
    """Capture denoiser step inputs and write them as torch tensors to disk."""

    def __init__(self, calibration_dir: Path, max_samples: int):
        self.calibration_dir = calibration_dir
        self.max_samples = max_samples
        self.sample_paths: list[Path] = []

    def __call__(self, step_inputs: dict[str, Any]) -> None:
        if len(self.sample_paths) >= self.max_samples:
            return

        prompt_cache = legacy_cache_from_prompt_cache(step_inputs["prompt_cache"])
        sample: dict[str, torch.Tensor] = {
            "x": step_inputs["x"].detach().to(dtype=torch.float16).cpu(),
            "t": step_inputs["t"].detach().to(dtype=torch.float16).cpu(),
            "position_ids": step_inputs["position_ids"].detach().cpu(),
            "attention_mask": step_inputs["attention_mask"].detach().cpu(),
        }
        for layer_idx, tensor in enumerate(flatten_past_key_values(prompt_cache)):
            name = (
                f"past_key_{layer_idx // 2:02d}"
                if layer_idx % 2 == 0
                else f"past_value_{layer_idx // 2:02d}"
            )
            sample[name] = tensor.detach().to(dtype=torch.float16).cpu()

        output_path = self.calibration_dir / f"sample_{len(self.sample_paths):03d}.pt"
        torch.save(sample, output_path)
        self.sample_paths.append(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an FP8 TensorRT engine for Alpamayo's expert denoiser."
    )
    parser.add_argument("--model-id", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    parser.add_argument("--max-generation-length", type=int, default=64)
    parser.add_argument("--num-calibration-samples", type=int, default=8)
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def fp8_artifact_layout(output_dir: str) -> dict[str, Path]:
    artifact_dir = resolve_artifact_dir(output_dir)
    layout = {
        "artifact_dir": artifact_dir,
        "calibration_dir": artifact_dir / "calibration",
        "plain_onnx": artifact_dir / "expert_step.fp16.onnx",
        "qdq_onnx": artifact_dir / "expert_step.fp8.qdq.onnx",
        "engine": artifact_dir / "expert_step.fp8.engine",
        "amax_json": artifact_dir / "expert_step.fp8.amax.json",
        "manifest": artifact_dir / "manifest.json",
    }
    layout["calibration_dir"].mkdir(parents=True, exist_ok=True)
    return layout


def load_model(model_id: str, device: torch.device) -> Alpamayo1_5:
    """Load Alpamayo with sdpa (VLM) + eager (expert) attention so the build
    needs NO flash-attn — its source build is unnecessary here and is a known
    machine-freeze risk; eager is also required for the expert's ONNX export.
    """
    cfg = Alpamayo1_5Config.from_pretrained(model_id)
    cfg.attn_implementation = "sdpa"
    model = Alpamayo1_5.from_pretrained(model_id, config=cfg, dtype=torch.bfloat16).to(device)
    model.eval()
    model.expert.config._attn_implementation = "eager"
    return model


def prepare_model_inputs(model: Alpamayo1_5, clip_id: str, t0_us: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, "cuda"), data


def run_full_inference(
    model: Alpamayo1_5,
    model_inputs: dict[str, Any],
    max_generation_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inference_inputs = copy.deepcopy(model_inputs)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inference_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            num_traj_sets=1,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    return pred_xyz.detach().cpu(), pred_rot.detach().cpu(), extra


def _sample_to_module_args(sample: dict[str, torch.Tensor], num_layers: int, device) -> tuple:
    flat = []
    for i in range(num_layers):
        flat.append(sample[f"past_key_{i:02d}"].to(device))
        flat.append(sample[f"past_value_{i:02d}"].to(device))
    return (
        sample["x"].to(device),
        sample["t"].to(device),
        sample["position_ids"].to(device),
        sample["attention_mask"].to(device),
        *flat,
    )


def main() -> None:
    args = parse_args()
    layout = fp8_artifact_layout(args.output_dir)
    device = torch.device("cuda")

    model = load_model(args.model_id, device)
    model_inputs, _ = prepare_model_inputs(model, args.clip_id, args.t0_us)

    collector = CalibrationCollector(layout["calibration_dir"], args.num_calibration_samples)
    model.set_expert_step_observer(collector)
    run_full_inference(model, model_inputs, args.max_generation_length, args.seed)
    model.set_expert_step_observer(None)
    if not collector.sample_paths:
        raise RuntimeError("No denoiser steps were captured for calibration.")

    num_layers = int(model.expert.config.num_hidden_layers)
    export_sample = load_calibration_sample(collector.sample_paths[0], device=device)

    # 1) Export the plain (FP16/BF16) expert ONNX BEFORE quantization mutates the model.
    export_expert_denoiser_onnx(model, export_sample, layout["plain_onnx"])

    # 2) ModelOpt FP8 calibration -> per-Linear amax (mutates model's expert in place).
    calib_module = ExpertDenoiserExportModule(model).eval().to(device)

    def calibration_forward_loop(m: torch.nn.Module) -> None:
        with torch.no_grad():
            for sample_path in collector.sample_paths:
                sample = load_calibration_sample(sample_path, device=device)
                m(*_sample_to_module_args(sample, num_layers, device))

    amax = calibrate_fp8_amax(calib_module, calibration_forward_loop)
    dump_amax(amax, layout["amax_json"])

    # 3) Insert FP8 Q/DQ into the plain ONNX, then 4) build the STRONGLY_TYPED
    #    engine. The graph has dynamic batch/prompt-length axes, so pass the
    #    calibration sample's shapes as the optimization-profile optimum.
    insert_fp8_qdq(layout["plain_onnx"], amax, layout["qdq_onnx"])
    input_names = ["x", "t", "position_ids", "attention_mask"] + past_key_value_input_names(num_layers)
    opt_shapes = {name: tuple(export_sample[name].shape) for name in input_names}
    build_fp8_engine(
        layout["qdq_onnx"],
        layout["engine"],
        opt_shapes=opt_shapes,
        num_layers=num_layers,
        workspace_gb=args.workspace_gb,
    )

    manifest = {
        "model_id": args.model_id,
        "clip_id": args.clip_id,
        "t0_us": args.t0_us,
        "seed": args.seed,
        "precision": "fp8_e4m3",
        "plain_onnx": str(layout["plain_onnx"]),
        "qdq_onnx": str(layout["qdq_onnx"]),
        "engine": str(layout["engine"]),
        "amax_json": str(layout["amax_json"]),
        "num_calibration_samples": len(collector.sample_paths),
        "num_quantized_linears": len(amax),
    }

    if not args.skip_validation:
        del model, calib_module
        torch.cuda.empty_cache()
        model = load_model(args.model_id, device)
        model_inputs, _ = prepare_model_inputs(model, args.clip_id, args.t0_us)

        native_pred_xyz, native_pred_rot, native_extra = run_full_inference(
            model=model,
            model_inputs=model_inputs,
            max_generation_length=args.max_generation_length,
            seed=args.seed,
        )
        model.set_expert_step_runner(TrtFp8ExpertEngine(layout["engine"]))
        trt_pred_xyz, trt_pred_rot, trt_extra = run_full_inference(
            model=model,
            model_inputs=model_inputs,
            max_generation_length=args.max_generation_length,
            seed=args.seed,
        )

        traj_abs_diff = (native_pred_xyz - trt_pred_xyz).abs()
        rot_abs_diff = (native_pred_rot - trt_pred_rot).abs()
        manifest["validation"] = {
            "trajectory_max_abs_diff": float(traj_abs_diff.max().item()),
            "trajectory_mean_abs_diff": float(traj_abs_diff.mean().item()),
            "rotation_max_abs_diff": float(rot_abs_diff.max().item()),
            "rotation_mean_abs_diff": float(rot_abs_diff.mean().item()),
            "native_cot": str(native_extra["cot"][0, 0, 0]),
            "trt_cot": str(trt_extra["cot"][0, 0, 0]),
        }

    layout["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
