#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
#
# Run the Alpamayo 2 Super node directly from the source tree, without a colcon install.
#
# The model package needs Python 3.10 with torch and flash-attn, which the virtual environment
# from the README's setup step provides; alpamayo2_super imports cleanly under 3.10 despite
# upstream's 3.12 pin (see src/alpamayo2_super/UPSTREAM.md). Set VENV to use a different one.
#
# Camera IDs are matched to the reference Autoware sensor set by /tf_static yaw and
# /camera_info FOV:
#   0 cross_left  <- camera5   1 front_wide  <- camera1   2 cross_right <- camera6
#   3 rear_left   <- camera9   5 rear_right  <- camera10  6 front_tele  <- camera2

# No `set -u`: ROS 2's setup.bash reads unset variables.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash
# Autoware message packages (autoware_planning_msgs, autoware_internal_debug_msgs) are
# not in the base ROS install. Point AUTOWARE_WS at your Autoware workspace.
AUTOWARE_WS="${AUTOWARE_WS:-$HOME/autoware}"
if [ -f "$AUTOWARE_WS/install/setup.bash" ]; then
    source "$AUTOWARE_WS/install/setup.bash"
else
    echo "warning: no Autoware workspace at $AUTOWARE_WS; autoware_*_msgs may be missing" >&2
fi
source "${VENV:-a1_5_venv}/bin/activate"

# src/ exposes alpamayo2_super; src/alpamayo_ros exposes the alpamayo_ros package.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/src/alpamayo_ros:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo2_node.py --ros-args \
    -p use_sim_time:=true \
    -p camera_topics:="['/sensing/camera/camera5/image_raw/compressed', \
    '/sensing/camera/camera1/image_raw/compressed', \
    '/sensing/camera/camera6/image_raw/compressed', \
    '/sensing/camera/camera9/image_raw/compressed', \
    '/sensing/camera/camera10/image_raw/compressed', \
    '/sensing/camera/camera2/image_raw/compressed']" \
    -p camera_indices:="[0, 1, 2, 3, 5, 6]" \
    "$@"
