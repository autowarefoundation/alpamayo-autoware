# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and order in-memory Edge inference inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

CAMERA_FRAME_COUNTS = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
CAMERA_PIXEL_BOUNDS = {
    camera: (163840, 196608) if camera in (1, 6) else (81920, 98304)
    for camera in CAMERA_FRAME_COUNTS
}
VQA_CAMERA_FRAME_COUNTS = {camera: 4 for camera in CAMERA_FRAME_COUNTS}
VQA_CAMERA_PIXEL_BOUNDS = {camera: (163840, 196608) for camera in CAMERA_FRAME_COUNTS}
INPUT_TENSOR_KEYS = (
    "image_frames",
    "camera_indices",
    "absolute_timestamps",
    "ego_history_xyz",
    "ego_history_rot",
)


def normalize_camera_pixel_bounds(bounds: Mapping[Any, Any]) -> dict[int, tuple[int, int]]:
    """Validate per-camera pixel budgets and normalize camera IDs."""
    if not isinstance(bounds, Mapping):
        raise TypeError("camera_pixel_bounds must be a camera-to-pixel-bounds mapping")
    normalized = {}
    for camera, pair in bounds.items():
        camera = int(camera)
        if camera not in CAMERA_FRAME_COUNTS:
            raise ValueError(f"Unsupported camera ID in camera_pixel_bounds: {camera}")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each camera_pixel_bounds value must be [min_pixels, max_pixels]")
        lower, upper = pair
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, int)
            or not isinstance(upper, int)
            or not 0 < lower <= upper
        ):
            raise ValueError("camera_pixel_bounds must contain positive integer min <= max")
        normalized[camera] = (lower, upper)
    return normalized


def _validate_camera_inputs(
    data: Mapping[str, Any], frame_counts: Mapping[int, int] = CAMERA_FRAME_COUNTS
) -> None:
    """Check camera images and timestamps against the required frame counts."""
    num_images = sum(frame_counts.values())
    for key in ("image_frames", "camera_indices", "absolute_timestamps"):
        if not isinstance(data.get(key), torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
    images = data["image_frames"]
    if (
        images.dtype != torch.uint8
        or images.device.type != "cpu"
        or images.ndim != 5
        or images.shape[:3] != (num_images, 1, 3)
        or min(images.shape[-2:]) <= 0
        or not images.is_contiguous()
    ):
        raise ValueError(f"image_frames must be contiguous CPU uint8 [{num_images}, 1, 3, H, W]")
    cameras = data["camera_indices"]
    if cameras.dtype != torch.int64 or cameras.shape != (num_images,):
        raise ValueError(f"camera_indices must be int64 [{num_images}]")
    ids, counts = cameras.unique(return_counts=True)
    if dict(zip(ids.tolist(), counts.tolist(), strict=True)) != frame_counts:
        expected_ids = "/".join(map(str, frame_counts))
        expected_counts = "/".join(map(str, frame_counts.values()))
        raise ValueError(
            f"Edge camera-major profile requires IDs {expected_ids} with frame counts {expected_counts}"
        )
    timestamps = data["absolute_timestamps"]
    if timestamps.dtype != torch.int64:
        raise TypeError("absolute_timestamps must be int64 microseconds")
    if timestamps.shape not in ((num_images,), (num_images, 1)):
        raise ValueError(
            f"absolute_timestamps must have shape [{num_images}] or [{num_images}, 1]"
        )
    if "camera_pixel_bounds" in data:
        normalize_camera_pixel_bounds(data["camera_pixel_bounds"])


def validate_vqa_sample(data: Mapping[str, Any]) -> None:
    """Check six cameras with four frames each and the VQA pixel bounds."""
    _validate_camera_inputs(data, VQA_CAMERA_FRAME_COUNTS)
    bounds = normalize_camera_pixel_bounds(data.get("camera_pixel_bounds", {}))
    if any(pair != VQA_CAMERA_PIXEL_BOUNDS[camera] for camera, pair in bounds.items()):
        raise ValueError("VQA camera_pixel_bounds must be (163840, 196608) for every camera")


def validate_sample(data: Mapping[str, Any], history_num_steps: int = 20) -> None:
    """Check camera inputs, ego history, and navigation before driving inference."""
    _validate_camera_inputs(data)
    for key, tail in (
        ("ego_history_xyz", (history_num_steps, 3)),
        ("ego_history_rot", (history_num_steps, 3, 3)),
    ):
        value = data.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if tuple(value.shape) not in ((1, *tail), (1, 1, *tail)):
            raise ValueError(
                f"{key} must have shape [1, {', '.join(map(str, tail))}] (optionally batched)"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain finite floating-point poses")
    nav = data.get("nav_text")
    if not isinstance(nav, (list, tuple)) or len(nav) != 1 or not isinstance(nav[0], str):
        raise ValueError("nav_text must be a one-element list containing the navigation text")
    if not nav[0].strip():
        raise ValueError("nav_text must contain nonempty navigation text")


def canonical_frame_order(camera_indices: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Frame indices sorted by camera, then capture timestamp."""
    timestamps = timestamps.reshape(-1)
    # Two stable sorts avoid subtracting large epoch timestamps or assuming a
    # fixed camera cadence. Sorting order is identical to Python (camera, time).
    by_time = torch.argsort(timestamps, stable=True)
    return by_time[torch.argsort(camera_indices[by_time], stable=True)]


def canonicalize_sample(data: Mapping[str, Any], history_num_steps: int = 20) -> dict[str, Any]:
    """Sort all frame-aligned fields together by camera, then capture timestamp."""
    validate_sample(data, history_num_steps=history_num_steps)
    return canonicalize_camera_inputs(data)


def canonicalize_camera_inputs(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and sort the 22 driving frames without navigation or ego history."""
    _validate_camera_inputs(data)
    return _sort_camera_inputs(data, CAMERA_PIXEL_BOUNDS)


def canonicalize_vqa_sample(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and sort the 24 VQA frames with uniform pixel bounds."""
    validate_vqa_sample(data)
    return _sort_camera_inputs(data, VQA_CAMERA_PIXEL_BOUNDS)


def _sort_camera_inputs(
    data: Mapping[str, Any], pixel_bounds: Mapping[int, tuple[int, int]]
) -> dict[str, Any]:
    result = dict(data)
    order = canonical_frame_order(data["camera_indices"], data["absolute_timestamps"])
    for key in ("image_frames", "camera_indices", "absolute_timestamps"):
        result[key] = data[key][order]
    result["camera_pixel_bounds"] = {
        **pixel_bounds,
        **normalize_camera_pixel_bounds(data.get("camera_pixel_bounds", {})),
    }
    return result
