# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static-shape compilation and CUDA-graph capture shared by the accelerated parts."""

import functools
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch


def bucket(length: int, size: int) -> int:
    """Round ``length`` up to a multiple of ``size``."""
    return -(-length // size) * size


def freeze(value: Any) -> Any:
    """Hashable stand-in for a logits-processor attribute, used in graph keys."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, freeze(item)) for key, item in value.items()))
    return value


def autocast_state(device_type: str) -> tuple[bool, torch.dtype]:
    """Autocast setting that graphs captured on ``device_type`` depend on."""
    return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)


def override_forward(original: Callable, implementation: Callable) -> Callable:
    """Instance-level override keeping ``original``'s signature (``generate`` inspects it)."""

    @functools.wraps(original)
    def override(*args: Any, **kwargs: Any) -> Any:
        return implementation(*args, **kwargs)

    return override


def maybe_compile(fn: Callable, enabled: bool, mode: str) -> Callable:
    """``torch.compile`` with static shapes; CUDA graphs are captured separately."""
    if not enabled:
        return fn
    return torch.compile(fn, fullgraph=True, dynamic=False, mode=mode)


class AccelStats:
    """Graph builds and eager fallbacks, reported per accelerated model."""

    def __init__(self) -> None:
        self.builds: list[dict[str, Any]] = []
        self.fallbacks: Counter[str] = Counter()

    def fallback(self, reason: str) -> None:
        self.fallbacks[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_builds": len(self.builds),
            "build_seconds": round(sum(item["seconds"] for item in self.builds), 2),
            "builds": list(self.builds),
            "fallbacks": dict(sorted(self.fallbacks.items())),
        }


class GraphedCall:
    """A fixed-shape callable, warmed up and captured into one CUDA graph.

    ``args`` are static input tensors: callers copy new values into them before
    each call. ``restore`` undoes side effects of the warm-up runs on persistent
    state. Warm-up and capture leave the CUDA RNG state unchanged, so replays
    draw the same random numbers that eager execution would at that state.
    Without CUDA graphs (CPU, or ``cuda_graph=False``), calls run ``fn`` directly.
    """

    def __init__(
        self,
        name: str,
        key: Any,
        fn: Callable,
        args: tuple,
        *,
        stats: AccelStats,
        warmup: int,
        cuda_graph: bool,
        device: torch.device,
        restore: Callable[[], None] | None = None,
    ) -> None:
        self.fn, self.args, self.graph, self.device = fn, args, None, device
        start = time.perf_counter()
        if cuda_graph:
            with torch.cuda.device(device):
                rng_state = torch.cuda.get_rng_state(device)
                stream = torch.cuda.Stream(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        fn(*args)
                        if restore is not None:
                            restore()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = fn(*args)
                torch.cuda.synchronize(device)
                torch.cuda.set_rng_state(rng_state, device)
        stats.builds.append(
            {
                "part": name,
                "key": str(key),
                "cuda_graph": cuda_graph,
                "seconds": round(time.perf_counter() - start, 2),
            }
        )

    def __call__(self) -> Any:
        if self.graph is None:
            return self.fn(*self.args)
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.out
