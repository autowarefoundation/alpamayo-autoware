# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test for the FlashDrive sidecar (PR2).

Exercises the HTTP contract end-to-end against a RUNNING sidecar, using
synthetic frames of the right shape. It proves the pipeline is live:
window 0 must be a streaming prefill (status="prefill"), and windows 1+ must
return trajectory batches. CoT/ADE are meaningless on random pixels — this
checks liveness + shapes + the streaming state machine, not quality.

Run (on the GPU host, after starting server.py):

    # terminal 1
    python server.py
    # terminal 2 (any interpreter with numpy)
    python smoke_test.py --url http://127.0.0.1:8710 --windows 4

Exit code 0 = pass. Non-zero = a gate failed (details printed).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request

import numpy as np

import wire


def _post(url: str, path: str, meta: dict, arrays: dict, timeout: float):
    body = wire.encode(meta, arrays)
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=body,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return wire.decode(resp.read())
    finally:
        try:
            wire.release_shm(meta, unlink=True)
        except Exception:  # noqa: BLE001
            pass


def _wait_health(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3.0)
    raise SystemExit(f"sidecar not healthy within {timeout:.0f}s at {url}")


def _synthetic(n_cams=4, n_frames=4, h=560, w=1008):
    meta = {"num_frames_per_camera": n_frames, "nav_text": None}
    rng = np.random.default_rng(0)  # match server warmup seed
    arrays = {
        "image_frames": rng.integers(0, 256, (n_cams, n_frames, 3, h, w), dtype=np.uint8),
        "camera_indices": np.array([0, 1, 2, 6][:n_cams], dtype=np.int64),
        "ego_history_xyz": np.zeros((1, 1, 16, 3), dtype=np.float32),
        "ego_history_rot": np.tile(np.eye(3, dtype=np.float32), (1, 1, 16, 1, 1)),
    }
    return meta, arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8710")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--health-timeout", type=float, default=1800.0)
    args = ap.parse_args()

    print(f"waiting for {args.url}/health ...")
    _wait_health(args.url, args.health_timeout)
    print("healthy; resetting stream")
    _post(args.url, "/reset", {}, {}, args.timeout)

    ok = True
    for w in range(args.windows):
        meta, arrays = _synthetic()
        t0 = time.perf_counter()
        rmeta, rarr = _post(args.url, "/predict", meta, arrays, args.timeout)
        dt = (time.perf_counter() - t0) * 1000.0
        status = rmeta.get("status")
        if w == 0:
            if status != "prefill":
                print(f"FAIL window0: expected prefill, got {status} ({rmeta})")
                ok = False
            else:
                print(f"window0: {dt:8.1f} ms  prefill OK")
        else:
            if status != "ok":
                print(f"FAIL window{w}: status={status} ({rmeta})")
                ok = False
            else:
                traj = rarr.get("traj_batch")
                rot = rarr.get("rot_batch")
                shapes = (None if traj is None else traj.shape, None if rot is None else rot.shape)
                print(f"window{w}: {dt:8.1f} ms  ok traj/rot shapes={shapes}")
                if traj is None or traj.ndim != 3 or traj.shape[-1] != 3:
                    print(f"FAIL window{w}: bad traj_batch shape {None if traj is None else traj.shape}")
                    ok = False

    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
