# Alpamayo 1.5 ROS 2 Node

![Alpamayo Autoware Demo](images/alpamayo-autoware.gif)

ROS 2 node for [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) end-to-end trajectory planning in [Autoware](https://autoware.org/).

## Architecture

```text
Camera Topics (CompressedImage × 4)     Odometry Topic
        │                                      │
        ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│                 Alpamayo 1.5 ROS Node                │
│                                                     │
│  GPU JPEG Decode ──► Tokenizer ──► VLM (BF16)      │
│  (torchvision)                        │             │
│                                  KV Cache           │
│                                       │             │
│                          Expert Denoiser            │
│                    (native PyTorch or TRT FP8)      │
│                                       │             │
│                          Trajectory Decode          │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Trajectory    CoT Text     Markers
```

Image preprocessing runs entirely on GPU: `torchvision.io.decode_jpeg` →
`F.interpolate` → `Qwen2VLImageProcessorFast(device="cuda")`. Decoded
pixels stay on GPU through normalize + patchify, eliminating the
~20 MB/inference round-trip to host memory that the CPU fast-path
incurred.

### Modes

| Mode | Expert | Decode | Diffusion | Use case |
|------|--------|--------|-----------|----------|
| **Baseline** | PyTorch native | Nucleus (top_p=0.98) | 10 steps | Reference quality |
| **Optimized** (default) | TRT FP8 engine | Nucleus (top_p=0.98) | 5–10 steps | Low-latency deployment |

Defaults are tuned for the optimized mode (5-step diffusion + nucleus
sampling); set `num_diffusion_steps:=10` for the baseline quality profile.

### Performance

Benchmarked on NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120).

**Expert denoiser step** (the diffusion inner loop, run `num_diffusion_steps`
times per inference) — single-step latency, verified against the native
PyTorch expert on the calibration clip:

| Expert runtime | per-step latency | speedup | engine size | max&#124;Δ&#124; trajectory vs PyTorch |
|---|---|---|---|---|
| PyTorch (bf16) | 14.8 ms | 1.0× | — | — |
| TRT FP16 | 9.1 ms | 1.6× | 4.57 GB | 0.016 |
| **TRT FP8** (default) | **7.3 ms** | **2.0×** | **2.29 GB** | 0.023 |

Chain-of-thought text is byte-identical between native and FP8 runs.

**End-to-end** (full `sample_trajectories_from_data_with_vlm_rollout` = VLM
rollout + diffusion), matching the ROS-node config: 4 cameras × 4 frames @
560×1008, nucleus sampling, `max_generation_length=16`, `num_traj_samples=1`
(latency) / `=6` for the minADE-deviation column; calibration clip, GT path
46.64 m; latency = median of 8 model-inference runs (excludes ROS messaging):

| Configuration | Latency | FPS | Trajectory Deviation |
|---------------|---------|-----|----------------------|
| native expert + 10-step | 0.717s | 1.40 | ~0.9% |
| FP8 expert + 10-step | 0.628s | 1.59 | ~0.9% |
| native expert + 5-step | 0.635s | 1.57 | ~0.7% |
| **FP8 expert + 5-step** (default) | **0.595s** | **1.68** | **~1.0%** |

FP8 saves ~40–90 ms (6–12%) end-to-end; the rollout is VLM-dominated, so the
gain is smaller than the 2× per-step speedup. Deviation is statistically equal
between native and FP8.

## Prerequisites

| Requirement | Specification |
|-------------|----------------------------------------------|
| **Python** | 3.10.x (ROS 2 Humble compatibility) |
| **ROS 2** | Humble |
| **GPU** | NVIDIA GPU with 24 GB+ VRAM |
| **CUDA** | 12.x+ |

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create Virtual Environment

```bash
uv venv a1_5_venv --python python3.10
source a1_5_venv/bin/activate
uv sync --active
```

### 3. HuggingFace Authentication

```bash
huggingface-cli login
```

Request access: [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

## Running

### Baseline Mode

```bash
source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

# Direct execution
python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
  '/sensing/camera/camera1/image_raw/compressed', \
  '/sensing/camera/camera4/image_raw/compressed', \
  '/sensing/camera/camera2/image_raw/compressed']" \
  -p camera_indices:="[0, 1, 2, 6]"

# Or via launch file
ros2 launch alpamayo_ros alpamayo.launch.py
```

### Optimized Mode (TRT Expert)

The FP8 expert engine is **built in a container** (one-time), then loaded by the
**native** ROS 2 node. Docker is used *only* to build the `.engine` — the node
runs outside Docker as usual. The container pins the finicky build stack (CUDA-13
torch, ModelOpt, TensorRT) and loads the model with sdpa/eager attention so it
needs **no flash-attn** (no source compile) — a fresh machine reproduces it with
one command.

**Step 1: Build the engine** (Docker; the `docker run` needs the GPU +
nvidia-container-toolkit):

```bash
docker build -f scripts/Dockerfile.trt-build -t alpamayo-trt-build .
docker run --gpus all -e HF_TOKEN=$HF_TOKEN -v ~/autoware_data:/data \
  alpamayo-trt-build \
  python scripts/build_trt_expert_engine.py --output-dir /data/alpamayo/v0.1
# -> ~/autoware_data/alpamayo/v0.1/expert_step.fp8.engine
```

**Step 2: Run the node** (native, ROS 2 Humble / Python 3.10 — outside Docker):

```bash
ros2 launch alpamayo_ros alpamayo.launch.py \
  expert_engine_path:=~/autoware_data/alpamayo/v0.1/expert_step.fp8.engine \
  num_diffusion_steps:=5
```

On top of the usual model deps, the node's runtime needs a **compatible
`tensorrt` (+ torch)** to deserialize the engine — but not the build-only quant
tooling (ModelOpt, onnx). The VLM still runs `flash_attention_2`, so the ROS env
also needs flash-attn (install a **prebuilt wheel** — do not source-compile it).
A TRT engine is not portable across TensorRT major versions, so build and runtime
must use the same TensorRT major.

### Rosbag Replay Evaluation

```bash
# Terminal 1: launch with sim time
ros2 launch alpamayo_ros alpamayo.launch.py use_sim_time:=true

# Terminal 2: play bag
ros2 bag play <bag_path> --clock --rate 0.5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_topics` | (required) | Camera image topics (CompressedImage) |
| `camera_indices` | (required) | Camera index for each topic (0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto) |
| `odometry_topic` | `/localization/kinematic_state` | Odometry topic |
| `route_topic` | `/planning/mission_planning/route` | Route topic (for navigation text) |
| `trajectory_topic` | `/alpamayo/predicted_trajectory` | Output trajectory topic |
| `cot_topic` | `/alpamayo/reasoning` | Output CoT reasoning topic |
| `cot_with_stamped_topic` | `/alpamayo/reasoning_stamped` | Timestamped reasoning topic |
| `nav_text_topic` | `/alpamayo/nav_text` | Navigation text topic |
| `inference_period_sec` | `0.1` | Inference trigger period |
| `expert_engine_path` | `""` | FP8 TRT expert `.engine` path (empty = native PyTorch) |
| `num_diffusion_steps` | `5` | Diffusion steps (10 = quality, 5 = speed) |
| `top_p` | `0.98` | Nucleus sampling threshold |
| `temperature` | `0.6` | Sampling temperature |
| `max_generation_length` | `64` | VLM token budget per tick |
| `use_sim_time` | `false` | Use ROS simulation time |

## Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/alpamayo/predicted_trajectory` | `autoware_planning_msgs/Trajectory` | 64-waypoint trajectory |
| `/alpamayo/reasoning` | `std_msgs/String` | Chain-of-thought reasoning |
| `/alpamayo/reasoning_stamped` | `autoware_internal_debug_msgs/StringStamped` | Timestamped reasoning |
| `/alpamayo/nav_text` | `std_msgs/String` | Derived navigation instruction |
| `{trajectory_topic}_markers` | `visualization_msgs/MarkerArray` | RViz visualization |

## TRT Expert Engine Build

`scripts/build_trt_expert_engine.py` exports the expert denoiser to ONNX, applies
NVIDIA ModelOpt **FP8** calibration + Q/DQ insertion, and compiles a
STRONGLY_TYPED TensorRT engine (`expert_step.fp8.engine`). Key options:
`--num-calibration-samples`, `--max-generation-length`, `--workspace-gb`,
`--skip-validation`.

The build environment (build-time only — separate from the ROS runtime) is the
container `scripts/Dockerfile.trt-build`, pinned via `scripts/trt-build.lock.txt`
(it pulls a fixed CUDA-13 base and installs torch from the CUDA-13 index; the
build uses sdpa attention so it needs no flash-attn). See
[Optimized Mode](#optimized-mode-trt-expert) above for the build + run commands.

## Troubleshooting

**`ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`** — Recreate venv with Python 3.10.

**CUDA OOM** — Use GPU with 24 GB+ VRAM. Increase `inference_period_sec`.

**Flash Attention issues** — Set `config.attn_implementation = "sdpa"` as fallback.

## References

- [Alpamayo](https://github.com/NVlabs/alpamayo) — Model weights, training, evaluation
- [alpamayo-autoware](https://github.com/autowarefoundation/alpamayo-autoware) — This repository

## License

- Inference code: Apache License 2.0
- Model weights: Non-commercial license ([HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B))

Alpamayo 1.5 is a pre-trained reasoning model for research purposes and is not a complete autonomous driving stack. It is not intended for use in production environments.
