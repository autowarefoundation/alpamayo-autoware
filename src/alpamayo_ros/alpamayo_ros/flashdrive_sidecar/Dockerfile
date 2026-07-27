# FlashDrive sidecar image (Python 3.12), separate from the ROS Humble (3.10)
# node because FlashDrive requires torch 2.9.x / vLLM / Python 3.12.
#
# This image provides the FlashDrive environment. Mount server.py / wire.py at
# /workspace/sidecar at runtime (see README.md). nvcc is present (devel base)
# because paroquant JIT-builds a CUDA kernel on first import.
#
# Build:
#   docker build -t flashdrive_sidecar:py312 -f Dockerfile .
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 22.04 has no python3.12 in default apt; uv fetches CPython 3.12
# (same as FlashDrive README: `uv venv --python 3.12`).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# FLASHDRIVE_REF pins the upstream commit/tag for reproducibility.
ARG FLASHDRIVE_REF=main
RUN git clone https://github.com/z-lab/flashdrive.git flashdrive \
    && cd flashdrive && git checkout "${FLASHDRIVE_REF}"

RUN cd /workspace/flashdrive \
    && uv venv --python 3.12 \
    && bash -lc "source .venv/bin/activate && uv sync"

ENV VIRTUAL_ENV="/workspace/flashdrive/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Sidecar sources are mounted at /workspace/sidecar. HF cache + token are
# provided via env/volume at runtime for the gated z-lab checkpoints.
ENV FD_HOST=0.0.0.0 \
    FD_PORT=8710

CMD ["bash", "-lc", "source /workspace/flashdrive/.venv/bin/activate && python /workspace/sidecar/server.py"]
