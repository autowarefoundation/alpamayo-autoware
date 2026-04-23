# Alpamayo ROS 2 Node

![Alpamayo Autoware Demo](images/alpamayo-autoware.gif)

ROS 2 node for [Alpamayo](https://github.com/NVlabs/alpamayo) end-to-end trajectory planning in [Autoware](https://autoware.org/).

## Architecture

```text
Camera Topics (CompressedImage × 4)     Odometry Topic
        │                                      │
        ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│                 Alpamayo ROS Node                    │
│                                                     │
│  GPU JPEG Decode ──► Tokenizer ──► VLM (BF16)      │
│  (torchvision)                        │             │
│                                  KV Cache           │
│                                       │             │
│                          Expert Denoiser            │
│                    (native PyTorch or TRT FP16)     │
│                                       │             │
│                          Trajectory Decode          │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Trajectory    CoT Text     Markers
```

Image preprocessing runs entirely on GPU: `torchvision.io.decode_jpeg` →
`F.interpolate` → `Qwen2VLImageProcessorFast(device="cuda")`. The
GPU-resident path keeps uint8 pixels on device through normalize +
patchify, eliminating a ~20 MB/inference round-trip to host memory and
running the image-processor's normalize+patchify on GPU.

### Performance

Benchmarked on NVIDIA RTX PRO 6000 (96 GB, SM120) with 4 cameras × 4 temporal frames at 1080×1920.

| Configuration | Latency | FPS | Trajectory Deviation |
|---------------|---------|-----|----------------------|
| Original (CPU preproc, sampling, native, 10-step) | 1.018s | 0.98 | Reference |
| GPU preproc + greedy + native expert + 10-step | 0.862s | 1.16 | ~0% |
| GPU preproc + greedy + native expert + 5-step | 0.768s | 1.30 | ~0.9% |
| GPU preproc + greedy + TRT expert + 10-step | 0.751s | 1.33 | ~0.3% |
| GPU preproc + greedy + TRT expert + 5-step | 0.714s | 1.40 | ~1.2% |
| **Full optimized** (GPU-resident preproc + greedy + TRT + 5-step) | **0.644s** | **1.55** | **~1.2%** |

The last row adds two free-meal changes on top of the previous row:
(a) the image processor's normalize+patchify runs on GPU via
`apply_chat_template(device="cuda")` instead of a CPU round-trip, and
(b) `generation_config.output_logits = False` drops ~20 MB/token of
unused host-pinned VLM logits. Same VLM/expert/diffusion math, so
trajectory deviation is unchanged from the previous row.

### Modes

| Mode | Expert | Decode | Diffusion | Use case |
|------|--------|--------|-----------|----------|
| **Baseline** (default) | PyTorch native | Nucleus (top_p=0.98) | 10 steps | Reference quality |
| **Optimized** | TRT FP16 engine | Greedy or nucleus | 5–10 steps | Low-latency deployment |

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
uv venv ar1_venv --python python3.10
source ar1_venv/bin/activate
uv sync --active
```

### 3. HuggingFace Authentication

```bash
huggingface-cli login
```

Request access: [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

## Running

### Baseline Mode

```bash
source /opt/ros/humble/setup.bash
source ar1_venv/bin/activate

# Direct execution
python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
  -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
  '/sensing/camera/camera1/image_raw/compressed', \
  '/sensing/camera/camera4/image_raw/compressed', \
  '/sensing/camera/camera2/image_raw/compressed']"

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

Then launch with the exported ONNX path:

```bash
ros2 launch alpamayo_ros alpamayo.launch.py \
  expert_onnx_path:=/path/to/your/engines/expert_step.int8.qdq.onnx \
  num_diffusion_steps:=5 \
  use_greedy_decode:=true
```

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
| `odometry_topic` | `/localization/kinematic_state` | Odometry topic |
| `trajectory_topic` | `/alpamayo/predicted_trajectory` | Output trajectory topic |
| `cot_topic` | `/alpamayo/reasoning` | Output CoT reasoning topic |
| `inference_period_sec` | `1.0` | Inference trigger period |
| `expert_onnx_path` | `""` | TRT expert ONNX path (empty = native PyTorch) |
| `num_diffusion_steps` | `10` | Diffusion steps (10 = quality, 5 = speed) |
| `use_greedy_decode` | `false` | Greedy decode (faster, deterministic) |
| `top_p` | `0.98` | Nucleus sampling threshold |
| `temperature` | `0.6` | Sampling temperature |
| `frame_id` | `base_link` | Trajectory coordinate frame |
| `num_frames` | `4` | Temporal frames per camera |
| `num_history_steps` | `16` | Odometry history steps |

## Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/alpamayo/predicted_trajectory` | `autoware_planning_msgs/Trajectory` | 64-waypoint trajectory |
| `/alpamayo/reasoning` | `std_msgs/String` | Chain-of-thought reasoning |
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

## References

- [Alpamayo](https://github.com/NVlabs/alpamayo) — Model weights, training, evaluation
- [alpamayo-autoware](https://github.com/autowarefoundation/alpamayo-autoware) — This repository

## License

- Inference code: Apache License 2.0
- Model weights: Non-commercial license ([HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B))
