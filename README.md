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
│                    (native PyTorch or TRT FP16)     │
│                                       │             │
│                          Trajectory Decode          │
│                                                     │
│  optional: use_flashdrive:=true ─────────────────┐  │
│     (HTTP client; no in-process model load)      │  │
└──────────────────────────┬───────────────────────┼──┘
                           │                       │
              ┌────────────┼────────────┐          ▼
              ▼            ▼            ▼   FlashDrive sidecar
         Trajectory    CoT Text     Markers  (Python 3.12 / z-lab)
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
| **Optimized** (default) | TRT FP16 engine | Greedy | 5–10 steps | Low-latency deployment |
| **Optional FlashDrive** | z-lab FlashDrive sidecar (Py 3.12) | Greedy + DFlash | 8-step `euler_with_cache` (sidecar default) | Experimental lower-latency path; **default off** |

Defaults are tuned for the optimized mode: 5-step diffusion +
greedy decode + `output_logits = False` on the VLM rollout. Flip
`num_diffusion_steps:=10` / `use_greedy_decode:=false` to reproduce
the baseline quality profile.

### Performance

Benchmarked on NVIDIA RTX PRO 6000 (96 GB, SM120) with 4 cameras × 4 temporal frames at 1080×1920.

Latency is measured end-to-end by replaying a Tier IV rosbag through
the ROS 2 node (`rate=0.5`, `max_generation_length=16`, 120 s warmup);
medians are taken over 15+ per-inference samples from the node's
`Alpamayo inference completed in X.XXs` log lines. Trajectory
Deviation is `minADE / ground-truth path length` measured over
`num_traj_samples=6` on the Physical AI AV clip used for TRT
calibration (`030c760c-ae38-49aa-9ad8-f5650a545d26 @ t0_us=5_100_000`,
GT path length = 46.64 m), same methodology as
`src/alpamayo1_5/test_inference.py`.

| Configuration | Latency | FPS | Trajectory Deviation |
|---------------|---------|-----|----------------------|
| Original (CPU preproc, sampling, native, 10-step) | 0.820s | 1.22 | Reference |
| GPU preproc + greedy + native expert + 10-step | 0.820s | 1.22 | ~0.4% |
| GPU preproc + greedy + native expert + 5-step | 0.720s | 1.39 | ~0.4% |
| GPU preproc + greedy + TRT expert + 10-step | 0.700s | 1.43 | ~1.3% |
| GPU preproc + greedy + TRT expert + 5-step | 0.660s | 1.52 | ~1.8% |
| **Full optimized** (GPU-resident preproc + greedy + TRT + 5-step) | **0.600s** | **1.67** | **~1.8%** |

> **Note on `num_traj_samples`:** the ROS 2 node path used for the latency
> column hardcodes `num_traj_samples=1`. The Trajectory Deviation column above
> is reported with `num_traj_samples=6` on the Physical AI clip (README
> methodology). Those two settings are independent.

### Optional FlashDrive (default off)

Preliminary numbers on **NVIDIA RTX PRO 5000 Blackwell (48 GB)** with
`use_flashdrive:=true` and `FD_TORCH_COMPILE=max-autotune`. This path does
**not** change the baseline / TRT rows above.

Latency is steady-state sidecar HTTP round-trip after warmup (not Tier IV
rosbag E2E). Trajectory Deviation uses the same Physical AI clip and
`minADE / 46.64 m` definition as above. Prefer `num_traj_samples=1` when
comparing latency to the in-process node; use `num_traj_samples=6` when
matching the README deviation methodology.

| Configuration | Latency | Trajectory Deviation |
|---------------|---------|----------------------|
| FlashDrive sidecar, `num_traj_samples=1`, `max_new_tokens=16` | **~0.22 s** | — |
| FlashDrive sidecar, `num_traj_samples=6`, `max_new_tokens=64` | ~0.37 s | **~0.6%** |

Build, run, and smoke-test details:
[`src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar/README.md`](src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar/README.md).

## Prerequisites

| Requirement | Specification |
|-------------|----------------------------------------------|
| **Python** | 3.10.x (ROS 2 Humble compatibility); FlashDrive sidecar needs 3.12 |
| **ROS 2** | Humble |
| **GPU** | NVIDIA GPU with 24 GB+ VRAM |
| **CUDA** | 12.x+ |
| **FlashDrive (optional)** | Hugging Face access to gated z-lab Alpamayo / DFlash weights + Docker (or a local FlashDrive 3.12 env) |

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

The TRT engine build requires `physical_ai_av` for calibration data, which needs Python >= 3.11. ROS 2 Humble ships Python 3.10 and cannot install this package. Use a **separate Python 3.12 venv** for building the engine, then use the exported ONNX file in the ROS 2 (3.10) runtime environment.

**Step 1: Build engine** (Python 3.12 venv, one-time):

```bash
uv venv .venv-trt --python python3.12
source .venv-trt/bin/activate
uv pip install -r scripts/requirements-trt-build.txt

python3 scripts/build_trt_expert_engine.py --output-dir /path/to/your/engines
```

The script exports `expert_step.int8.qdq.onnx` and caches the compiled TRT engine under `engine_cache/` in the same directory.

**Step 2: Run node** (Python 3.10, ROS 2 Humble):

```bash
ros2 launch alpamayo_ros alpamayo.launch.py \
  expert_onnx_path:=/path/to/your/engines/expert_step.int8.qdq.onnx \
  num_diffusion_steps:=5 \
  use_greedy_decode:=true
```

### Optional FlashDrive Mode (sidecar)

FlashDrive cannot run inside the ROS 2 Humble (Python 3.10) process. Enable the
default-off backend to forward payloads to a separate Python 3.12 sidecar. See
[`flashdrive_sidecar/README.md`](src/alpamayo_ros/alpamayo_ros/flashdrive_sidecar/README.md)
for build, run, and smoke-test details.

```bash
# Terminal 1: start the sidecar (Docker or FlashDrive py3.12 env)
# Terminal 2: ROS node
ros2 launch alpamayo_ros alpamayo.launch.py \
  use_flashdrive:=true \
  flashdrive_url:=http://127.0.0.1:8710
```

Notes:

- Baseline / TRT path is unchanged when `use_flashdrive:=false` (default).
- Streaming window 0 is a KV prefill and publishes no trajectory that cycle.
- FlashDrive decode/diffusion defaults differ from the optimized TRT path
  (8-step `euler_with_cache` + DFlash vs 5-step TRT Euler); treat quality as a
  measured delta. See **Optional FlashDrive** above for preliminary numbers.

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
| `expert_onnx_path` | `""` | TRT expert ONNX path (empty = native PyTorch) |
| `use_flashdrive` | `false` | Use FlashDrive Python 3.12 sidecar instead of in-process model |
| `flashdrive_url` | `http://127.0.0.1:8710` | FlashDrive sidecar base URL |
| `num_diffusion_steps` | `5` | Diffusion steps (10 = quality, 5 = speed) |
| `use_greedy_decode` | `true` | Greedy decode (faster, deterministic) |
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

The `scripts/build_trt_expert_engine.py` script exports the expert denoiser to ONNX, applies SmoothQuant + INT8 quantization, and compiles a TensorRT engine:

```bash
python3 scripts/build_trt_expert_engine.py --help
```

Key options: `--num-calibration-samples`, `--calibration-method`, `--smoothquant-alpha`, `--skip-validation`.

Requires the `trt` dependency group: `uv sync --active --group trt`

## Troubleshooting

**`ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`** — Recreate venv with Python 3.10.

**CUDA OOM** — Use GPU with 24 GB+ VRAM. Increase `inference_period_sec`.

**Flash Attention issues** — Set `config.attn_implementation = "sdpa"` as fallback.

**FlashDrive sidecar not healthy** — Confirm the Python 3.12 process is up,
`/health` returns 200, and `flashdrive_url` matches host/port. First load +
`torch.compile` warmup can take many minutes.

## References

- [Alpamayo](https://github.com/NVlabs/alpamayo) — Model weights, training, evaluation
- [alpamayo-autoware](https://github.com/autowarefoundation/alpamayo-autoware) — This repository
- [FlashDrive](https://github.com/z-lab/flashdrive) — Optional accelerated Alpamayo runtime

## License

- Inference code: Apache License 2.0
- Model weights: Non-commercial license ([HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B))

Alpamayo 1.5 is a pre-trained reasoning model for research purposes and is not a complete autonomous driving stack. It is not intended for use in production environments.
