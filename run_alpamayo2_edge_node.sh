#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
#
# Run the Alpamayo 2 Edge node directly from the source tree, without a colcon install.
#
# The node needs its own Python 3.10 environment (Transformers 5.17, see the README's
# "Alpamayo 2 Edge" setup); set VENV to use one other than a2e_venv. Extra arguments are
# passed to the node, e.g. -p lanelet2_map_path:=<map dir>/lanelet2_map.osm.
#
# Camera IDs are matched to the reference Autoware sensor set as in run_alpamayo2_node.sh:
#   0 cross_left  <- camera5   1 front_wide  <- camera1   2 cross_right <- camera6
#   3 rear_left   <- camera9   5 rear_right  <- camera10  6 front_tele  <- camera2

# No `set -u`: ROS 2's setup.bash reads unset variables.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash
# Autoware message packages (autoware_planning_msgs, autoware_internal_debug_msgs) and the
# lanelet2 bindings are not in the base ROS install. Point AUTOWARE_WS at your workspace.
AUTOWARE_WS="${AUTOWARE_WS:-$HOME/autoware}"
if [ -f "$AUTOWARE_WS/install/setup.bash" ]; then
    source "$AUTOWARE_WS/install/setup.bash"
else
    echo "warning: no Autoware workspace at $AUTOWARE_WS; autoware_*_msgs may be missing" >&2
fi
source "${VENV:-a2e_venv}/bin/activate"

# src/ exposes alpamayo2_edge; src/alpamayo_ros exposes the alpamayo_ros package.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/src/alpamayo_ros:${PYTHONPATH:-}"
# Keep the compiled graphs across runs (accel), which makes later starts much faster.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/alpamayo2_edge/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/alpamayo2_edge/triton}"

# exec, so that a signal to this script's PID reaches the node.
exec python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo2_edge_node.py --ros-args \
    -p use_sim_time:=true \
    -p camera_topics:="['/sensing/camera/camera5/image_raw/compressed', \
    '/sensing/camera/camera1/image_raw/compressed', \
    '/sensing/camera/camera6/image_raw/compressed', \
    '/sensing/camera/camera9/image_raw/compressed', \
    '/sensing/camera/camera10/image_raw/compressed', \
    '/sensing/camera/camera2/image_raw/compressed']" \
    -p camera_indices:="[0, 1, 2, 3, 5, 6]" \
    "$@"
