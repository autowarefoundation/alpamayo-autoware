# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Edge navigation and VQA conversations with relative image times."""

from typing import Any

import torch

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
DEFAULT_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_TASK = (
    "Your task: output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)
# Driving answers follow an empty think block, which the prompt supplies.
CLOSED_THINKING = "<think></think>"


def relative_time_labels(timestamps: torch.Tensor) -> list[str]:
    """Label capture times relative to the final camera-major input timestamp."""
    timestamps = timestamps.reshape(-1)
    # The z format flag canonicalizes tiny negative deltas to 0.0, matching
    # the original template when cameras differ by only microseconds.
    return [f"<t={delta / 1_000_000:z.1f}s>" for delta in (timestamps - timestamps[-1]).tolist()]


def _image_content(data: dict[str, Any], include_camera_ids: bool) -> list[dict[str, Any]]:
    """Add camera names and capture times to sorted image inputs."""
    content: list[dict[str, Any]] = []
    previous_camera = None
    labels = relative_time_labels(data["absolute_timestamps"])
    for image, camera, label in zip(
        data["image_frames"], data["camera_indices"].tolist(), labels, strict=True
    ):
        if include_camera_ids and camera != previous_camera:
            content.append({"type": "text", "text": CAMERA_DISPLAY_NAMES[camera] + ": "})
        content.append({"type": "text", "text": label})
        lower, upper = data["camera_pixel_bounds"][camera]
        content.append({"type": "image", "image": image, "min_pixels": lower, "max_pixels": upper})
        previous_camera = camera
    return content


def _conversation(
    data: dict[str, Any], content: list[dict[str, Any]], system_prompt: str | None
) -> list[dict[str, Any]]:
    effective_system = data.get("system_prompt")
    if effective_system is None:
        effective_system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": [{"type": "text", "text": str(effective_system)}]},
        {"role": "user", "content": content},
    ]


def build_conversation(
    data: dict[str, Any],
    num_tokens_per_history_traj: int = 57,
    include_camera_ids: bool = True,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Compose a sorted raw sample into the checkpoint's generation prompt.

    Trajectory placeholders are replaced with discretized history tokens by the
    model. Neither an answer nor a future trajectory is included in this prompt.
    """
    content = _image_content(data, include_camera_ids)
    content.extend(
        [
            {"type": "text", "text": "\n"},
            {
                "type": "text",
                "text": "<|traj_history_start|>"
                + "<|traj_history|>" * num_tokens_per_history_traj
                + "<|traj_history_end|>",
            },
            {"type": "text", "text": "\n"},
            {"type": "text", "text": "Navigation input is: " + data["nav_text"][0]},
            {"type": "text", "text": "\n"},
            {"type": "text", "text": DRIVING_TASK},
        ]
    )
    return _conversation(data, content, system_prompt)


def build_vqa_conversation(
    data: dict[str, Any],
    question: str,
    include_camera_ids: bool = True,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build a conversation from camera images and a freeform question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("VQA question must be a nonempty string")
    content = _image_content(data, include_camera_ids)
    content.extend([{"type": "text", "text": "\n"}, {"type": "text", "text": question.strip()}])
    return _conversation(data, content, system_prompt)
