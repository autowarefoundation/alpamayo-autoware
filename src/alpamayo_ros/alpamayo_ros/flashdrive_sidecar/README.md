# FlashDrive sidecar (optional)

Python 3.12 process that runs [FlashDrive](https://github.com/z-lab/flashdrive)
optimized Alpamayo 1.5 checkpoints. The ROS 2 Humble node stays on Python 3.10
and talks to this sidecar over HTTP using the numpy-only `wire` protocol.

## Why a sidecar?

FlashDrive requires Python ≥ 3.12, torch 2.9.x, and vLLM. ROS 2 Humble pins
the node venv to Python 3.10, so FlashDrive cannot run in-process in the node.

## Contract

| Method | Path | Behavior |
|--------|------|----------|
| `GET` | `/health` | 200 once the model is loaded (and warmed up) |
| `POST` | `/reset` | Drop streaming KV cache; next `/predict` is window 0 |
| `POST` | `/predict` | One streaming window (see below) |

`/predict` request arrays: `image_frames` uint8 `[n_cams, n_frames, 3, H, W]`,
`camera_indices` int64 `[n_cams]`, `ego_history_xyz` float32 `[1,1,16,3]`,
`ego_history_rot` float32 `[1,1,16,3,3]`.

Responses:

- Window 0 → `status=prefill` (no trajectory; fills the KV cache)
- Window 1+ → `status=ok` with `traj_batch` / `rot_batch` and CoT text

## Requirements

- NVIDIA GPU with sufficient VRAM (about 16 GB for `num_traj_samples=1`)
- Hugging Face access to gated z-lab Alpamayo-1.5 / DFlash weights (`HF_TOKEN`)
- Docker with GPU support, **or** a local FlashDrive Python 3.12 environment

## Build and run

```bash
# Build image (pins FlashDrive via FLASHDRIVE_REF)
docker build -t flashdrive_sidecar:py312 \
  -f src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar/Dockerfile \
  src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar

# Run (HF token required for gated z-lab checkpoints).
# Prefer FD_TORCH_COMPILE=max-autotune (~several minutes warmup).
# Use none only for fast bring-up / debugging.
docker run --gpus all --network host --ipc host \
  -e HF_TOKEN \
  -e FD_MODEL_PATH=z-lab/Alpamayo-1.5-10B \
  -e FD_TORCH_COMPILE=max-autotune \
  -e FD_WARMUP=1 \
  -e FD_NUM_TRAJ_SAMPLES=1 \
  -e FD_MAX_NEW_TOKENS=16 \
  -v "$PWD/src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar:/workspace/sidecar:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  flashdrive_sidecar:py312
```

Wait until `GET /health` returns 200, then enable the ROS node.

Or without Docker: create a Python 3.12 FlashDrive env per upstream FlashDrive
docs, then run `python server.py` from this directory.

## Smoke test

```bash
# Against a running sidecar (numpy only; no torch required on the client)
python3 src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar/smoke_test.py \
  --url http://127.0.0.1:8710 --windows 4
```

Expects window 0 = prefill and windows 1+ = trajectory tensors of shape
`[num_traj_samples, 64, 3]`.

## Enable from the ROS node

```bash
ros2 launch alpamayo_ros alpamayo.launch.py \
  use_flashdrive:=true \
  flashdrive_url:=http://127.0.0.1:8710
```

Default remains `use_flashdrive:=false` (in-process baseline / TRT path).

## Preliminary measurements

Hardware: NVIDIA RTX PRO 5000 Blackwell (48 GB). Config:
`FD_TORCH_COMPILE=max-autotune`, `FD_WARMUP=1`, greedy decode, 8-step
`euler_with_cache` + DFlash. Physical AI clip
`030c760c-ae38-49aa-9ad8-f5650a545d26 @ t0_us=5_100_000` (GT path length
46.64 m), same clip as the main README Trajectory Deviation column.

Report **steady-state** latency after warmup (smoke windows 2+). The first
trajectory window after `/reset` may recompile and is not representative.

| Setting | Latency | Trajectory Deviation | GPU mem |
|---------|---------|----------------------|---------|
| `FD_NUM_TRAJ_SAMPLES=1`, `FD_MAX_NEW_TOKENS=16` | ~0.22 s | — (single-sample; not the README N=6 method) | ~16 GB |
| `FD_NUM_TRAJ_SAMPLES=6`, `FD_MAX_NEW_TOKENS=64` | ~0.37 s | ~0.6% | ~24 GB |

## Caveats

- Streaming assumes a roughly uniform ~0.1 s window stride.
- FlashDrive diffusion/decode defaults differ from the in-process 5-step
  Euler + TRT expert path; treat quality as a measured delta.
- Sidecar latency is not identical to Tier IV rosbag
  `Alpamayo inference completed` medians in the main README table.
- Large camera tensors may use POSIX shared memory (`FD_WIRE_SHM=1`); prefer
  `--ipc host` between node and sidecar.
- Model weights remain under the licenses of the respective Hugging Face
  model cards.
