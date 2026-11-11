# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve self-contained HF checkpoints and refuse partial weight loads."""

import json
from pathlib import Path
from typing import Any


def validate_checkpoint(path: Path) -> None:
    """Check local config, processor, tokenizer and all indexed weight shards."""
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if not (path / "preprocessor_config.json").is_file():
        # Transformers 5 saves image settings inside the unified processor config.
        processor_path = path / "processor_config.json"
        processor_config = (
            json.loads(processor_path.read_text()) if processor_path.is_file() else {}
        )
        image_config = (
            processor_config.get("image_processor") if isinstance(processor_config, dict) else None
        )
        if not isinstance(image_config, dict) or not image_config:
            missing.append("preprocessor_config.json (or processor_config.json image_processor)")
    if not (path / "chat_template.jinja").is_file():
        tokenizer_config = path / "tokenizer_config.json"
        if not tokenizer_config.is_file() or not json.loads(tokenizer_config.read_text()).get(
            "chat_template"
        ):
            missing.append("chat_template.jinja (or tokenizer_config.json chat_template)")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing.extend(name for name in sorted(shards) if not (path / name).is_file())
    elif not (path / "model.safetensors").is_file():
        missing.append("model.safetensors (or model.safetensors.index.json and shards)")
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint at {path}: {', '.join(missing)}")


def resolve_checkpoint(
    model_id: str,
    *,
    subfolder: str = "",
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Accept a model directory, its parent containing model/, or a HF repository."""
    path = Path(model_id).expanduser()
    if path.exists():
        path = path / subfolder
        if (path / "model" / "config.json").is_file():
            path = path / "model"
    elif path.is_absolute() or model_id.startswith((".", "~")):
        raise FileNotFoundError(f"Local checkpoint does not exist: {path}")
    else:
        from huggingface_hub import snapshot_download

        prefix = subfolder.strip("/")
        if ".." in Path(prefix).parts:
            raise ValueError("subfolder must stay within the HF repository")
        patterns = (
            [f"{prefix}/*"]
            if prefix
            else [
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.txt",
                "*.model",
            ]
        )
        snapshot = snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        path = Path(snapshot) / prefix
        if (path / "model" / "config.json").is_file():
            path = path / "model"
    validate_checkpoint(path)
    return path.resolve()


def load_model(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    **resolve_kwargs: Any,
):
    """Load the full trained Edge checkpoint with no random missing parameters.

    Legacy config normalization happens in memory. The saved checkpoint and its
    tensor files are never edited. Returns ``(model, loading_info)``.
    """
    import torch

    from alpamayo2_edge.config import Alpamayo2EdgeConfig, normalize_checkpoint_config
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge

    path = resolve_checkpoint(model_id, **resolve_kwargs)
    config = Alpamayo2EdgeConfig(
        **normalize_checkpoint_config(json.loads((path / "config.json").read_text()))
    )
    config._name_or_path = str(path)
    config.set_dtype(dtype)
    model, loading_info = Alpamayo2Edge.from_pretrained(
        str(path),
        config=config,
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {
        name: loading_info.get(name)
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(name)
    }
    if problems:
        raise RuntimeError(f"Checkpoint did not load completely: {problems}")
    model.eval().to(device)
    return model, loading_info
