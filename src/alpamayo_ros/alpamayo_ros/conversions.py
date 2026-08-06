#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Model-version-agnostic conversions shared by the Alpamayo 1.5 and 2 Super nodes.

Both model generations emit the same trajectory representation — waypoints at a fixed
0.1 s cadence in the ego frame at t0, plus an optional per-waypoint rotation matrix — so
the Autoware/RViz message builders and the odometry history transform are identical. Only
the horizon length differs (1.5: 20 points / 2.0 s, 2 Super: 64 points / 6.4 s), which
these functions derive from the input tensor.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

#: Waypoint spacing of every Alpamayo trajectory head, in seconds.
WAYPOINT_DT = 0.1

#: Anything ``numpy.asarray`` accepts — a torch tensor, a numpy array, or nested lists. The
#: message builders take this rather than a tensor so the tools that rebuild markers from
#: recorded predictions can pass plain lists.
ArrayLike = Any


def build_ego_history(
    odom_history: Sequence[Odometry],
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Re-express an odometry history in the ego frame at t0.

    Implements the transform the models are trained on::

        xyz_local = R_t0^-1 @ (xyz_world - xyz_t0)
        rot_local = R_t0^-1 @ R_t

    so the last entry is always the origin with identity rotation. The action space
    differentiates this history to estimate the t0 velocity, which is *not* a model
    input — a jittery or wrongly-framed history silently corrupts the whole rollout.

    Args:
        odom_history: Poses sampled at the model's input rate (10 Hz), oldest first,
            with the last element at t0.

    Returns:
        ``(ego_history_xyz, ego_history_rot, positions_map)`` where the tensors are
        shaped ``(1, 1, T, 3)`` and ``(1, 1, T, 3, 3)``, and ``positions_map`` is the
        raw ``(T, 3)`` map-frame positions (used for navigation lookups).
    """
    positions = []
    rotations = []
    for msg in odom_history:
        pose = msg.pose.pose
        positions.append([pose.position.x, pose.position.y, pose.position.z])
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rotations.append(Rotation.from_quat(quat).as_matrix())

    positions_np = np.asarray(positions, dtype=np.float32)
    rotations_np = np.asarray(rotations, dtype=np.float32)

    t0_rot_inv = np.linalg.inv(rotations_np[-1])
    centered = positions_np - positions_np[-1]
    history_xyz_local = centered @ t0_rot_inv.T
    history_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)

    ego_history_xyz = torch.from_numpy(history_xyz_local).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(history_rot_local).unsqueeze(0).unsqueeze(0)
    return ego_history_xyz, ego_history_rot, positions_np


def check_ego_history(
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    reference_speed_mps: Optional[float] = None,
) -> tuple[bool, str]:
    """Sanity-check a history built by :func:`build_ego_history`.

    The first two invariants hold by construction *if* the frame convention is right. When
    they don't, every predicted trajectory is garbage in a way that is hard to spot
    downstream. A standing start inside the window is also reported, but does not fail the
    check — see the comment on ``standing_start`` below.

    Args:
        ego_history_xyz: ``(1, 1, T, 3)`` history positions.
        ego_history_rot: ``(1, 1, T, 3, 3)`` history rotations.
        reference_speed_mps: Independently measured speed at t0 (e.g. the odometry
            twist), compared against the speed implied by the last two positions.

    Returns:
        ``(ok, message)`` — the message is a single log line either way.
    """
    xyz = ego_history_xyz[0, 0].numpy()
    rot = ego_history_rot[0, 0].numpy()

    origin_err = float(np.abs(xyz[-1]).max())
    identity_err = float(np.abs(rot[-1] - np.eye(3)).max())
    steps = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    implied_speed = float(steps[-1] / WAYPOINT_DT)

    # A history straddling a standing start is where the action space breaks down:
    # estimate_t0_states differentiates all 16 poses through a Tikhonov-regularized solve, and
    # when the first half is parked (millimetre steps that are just localization noise) while
    # the second half accelerates, the solve is ill-conditioned. One such history produced a
    # rollout starting 10.6 m behind the rear axle and sweeping 27 m backwards.
    #
    # It is reported but deliberately does *not* fail the check: measured over 93 predictions
    # this condition fired three times and only one of the three rollouts was actually bad,
    # and the ratio does not separate them (the worst ratio belonged to a good one). It is a
    # risk factor, not a predictor — check_trajectory is what rejects the broken output.
    early_speed = float(steps[: len(steps) // 2].mean() / WAYPOINT_DT)
    standing_start = implied_speed > 0.5 and early_speed < 0.1 * implied_speed

    checks = [origin_err < 1e-3, identity_err < 1e-3]
    parts = [
        f"t0_at_origin={origin_err:.2e}",
        f"t0_rot_identity={identity_err:.2e}",
        f"implied_v0={implied_speed:.2f}m/s",
        f"early_v={early_speed:.2f}m/s",
    ]
    if standing_start:
        parts.append("standing start in window (initial-state estimate may be unreliable)")
    if reference_speed_mps is not None:
        speed_err = abs(implied_speed - reference_speed_mps)
        # 0.5 m/s absorbs the difference between a finite difference over one 0.1 s
        # step and the filtered twist estimate; a frame bug is off by far more.
        checks.append(speed_err < 0.5 + 0.1 * abs(reference_speed_mps))
        parts.append(f"odom_v0={reference_speed_mps:.2f}m/s (diff={speed_err:.2f})")

    ok = all(checks)
    return ok, ("ego history OK: " if ok else "ego history SUSPECT: ") + ", ".join(parts)


def check_trajectory(
    trajectory: ArrayLike,
    reference_speed_mps: float,
    max_start_error_m: float = 2.0,
    max_reverse_m: float = 2.0,
) -> tuple[bool, str]:
    """Sanity-check a predicted trajectory against the measured speed.

    The action space integrates a unicycle model from the origin, so the first waypoint —
    the pose at t0 + 0.1 s — must land ``v0 * 0.1`` ahead of ``base_link`` and essentially on
    the vehicle's axis. When the estimated initial state is wrong the whole rollout is
    displaced: one prediction in a 93-sample recording started 10.6 m *behind* the rear axle
    and swept 27 m backwards, which in RViz reads as a path detached from the vehicle.

    Checking the output as well as the input matters because the two failures are
    independent: :func:`check_ego_history` catches a malformed history, but a plausible
    history can still yield a displaced rollout.

    Args:
        trajectory: ``(T, 3)`` waypoints in the ego frame at t0.
        reference_speed_mps: Speed at t0 from odometry.
        max_start_error_m: Tolerance on the first waypoint's distance from ``v0 * dt``.
        max_reverse_m: How far behind ``base_link`` any waypoint may fall. Some backwards
            drift is legitimate — a stopped vehicle's rollout creeps back by centimetres.

    Returns:
        ``(ok, message)``.
    """
    traj = np.asarray(trajectory)
    expected = abs(reference_speed_mps) * WAYPOINT_DT
    start_distance = float(np.linalg.norm(traj[0, :2]))
    start_error = abs(start_distance - expected)
    min_x = float(traj[:, 0].min())

    ok = start_error <= max_start_error_m and min_x >= -max_reverse_m
    return ok, (
        ("trajectory OK: " if ok else "trajectory SUSPECT: ")
        + f"start={start_distance:.2f}m (expected {expected:.2f}m, "
        f"error {start_error:.2f}m), min_x={min_x:.2f}m"
    )


def to_autoware_trajectory(
    trajectory: ArrayLike,
    rotations: Optional[ArrayLike],
    stamp: Time,
    frame_id: str,
) -> Trajectory:
    """Convert predicted waypoints into an Autoware ``Trajectory``.

    Accepts torch tensors or numpy arrays: the tools that rebuild markers from recorded
    predictions have no reason to pull in torch.

    Args:
        trajectory: ``(T, 3)`` waypoints in the ego frame at t0, x forward / y left.
        rotations: ``(T, 3, 3)`` per-waypoint rotations, or ``None`` for identity.
        stamp: Header stamp; pass the *input* t0 stamp so consumers can see staleness.
        frame_id: Header frame, normally ``base_link``.
    """
    traj_np = np.asarray(trajectory)
    rot_np = np.asarray(rotations) if rotations is not None else None

    traj_msg = Trajectory()
    traj_msg.header.stamp = stamp
    traj_msg.header.frame_id = frame_id

    prev_xy = None
    for idx, point in enumerate(traj_np):
        traj_point = TrajectoryPoint()
        traj_point.pose.position.x = float(point[0])
        traj_point.pose.position.y = float(point[1])
        traj_point.pose.position.z = float(point[2])

        if rot_np is not None:
            quat = Rotation.from_matrix(rot_np[idx]).as_quat()
            traj_point.pose.orientation.x = float(quat[0])
            traj_point.pose.orientation.y = float(quat[1])
            traj_point.pose.orientation.z = float(quat[2])
            traj_point.pose.orientation.w = float(quat[3])
        else:
            traj_point.pose.orientation.w = 1.0

        if prev_xy is None:
            speed = 0.0
        else:
            dx = float(point[0] - prev_xy[0])
            dy = float(point[1] - prev_xy[1])
            speed = math.hypot(dx, dy) / WAYPOINT_DT

        traj_point.longitudinal_velocity_mps = float(speed)
        traj_point.lateral_velocity_mps = 0.0
        traj_point.acceleration_mps2 = 0.0
        traj_point.heading_rate_rps = 0.0

        # Waypoint idx is the pose at t0 + (idx + 1) * dt: both model generations predict
        # the future starting one step after t0, never t0 itself.
        seconds_float = (idx + 1) * WAYPOINT_DT
        seconds_int = int(seconds_float)
        nanosec = int((seconds_float - seconds_int) * 1e9)
        traj_point.time_from_start = Duration(sec=seconds_int, nanosec=nanosec)

        traj_msg.points.append(traj_point)
        prev_xy = (point[0], point[1])

    return traj_msg


def wrap_text(text: str, width: int = 56, max_lines: int = 8) -> str:
    """Word-wrap reasoning text for a fixed-width RViz text marker."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) >= max_lines and len(" ".join(words)) > sum(len(line) for line in lines):
        lines[-1] = lines[-1] + " ..."
    return "\n".join(lines)


def _ego_footprint_marker(
    stamp: Time,
    frame_id: str,
    length_m: float,
    width_m: float,
    rear_overhang_m: float,
    z_offset_m: float,
) -> Marker:
    """Outline the ego vehicle so the start of the trajectory has something to sit against.

    The prediction begins at ``base_link``, which is the rear axle — roughly 3.8 m behind the
    front of a JPN TAXI — and the ground-removed point cloud has no returns near the vehicle.
    With nothing drawn there, the trajectory looks like it starts in mid-air behind wherever
    the viewer assumes the car is. A thin footprint is enough to anchor it; a full vehicle
    model would bury the first waypoints.
    """
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = "ego_footprint"
    marker.id = 2
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.25
    marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
    marker.pose.orientation.w = 1.0
    marker.frame_locked = True

    front = length_m - rear_overhang_m
    rear = -rear_overhang_m
    half = width_m / 2.0
    corners = [
        (front, half),
        (front, -half),
        (rear, -half),
        (rear, half),
        (front, half),  # close the loop
    ]
    for x, y in corners:
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z_offset_m)
        marker.points.append(point)
    return marker


def trajectory_to_markers(
    trajectory: ArrayLike,
    stamp: Time,
    frame_id: str,
    line_width: float = 1.0,
    cot_text: Optional[str] = None,
    cot_height_m: float = 4.0,
    z_offset_m: float = 0.0,
    ego_footprint: Optional[tuple[float, float, float]] = None,
) -> MarkerArray:
    """Build the RViz markers for one prediction.

    Emits the green trajectory line strip, plus — when ``cot_text`` is given — a
    ``TEXT_VIEW_FACING`` marker floating above the ego vehicle. RViz has no built-in
    text overlay display, so the marker is how the chain-of-thought reasoning ends up
    in a screen recording without extra plugins.

    ``z_offset_m`` lifts the line strip off the ground plane. The model holds z constant
    at the t0 height, which puts the trajectory in the same plane as an Autoware vector
    map's road polygons — the resulting z-fighting chops the line into fragments.

    ``ego_footprint`` is ``(length, width, rear_overhang)`` in metres; pass it to outline the
    vehicle at the origin so the first waypoint reads as attached to the car.
    """
    traj_np = np.asarray(trajectory)
    marker_array = MarkerArray()

    line_marker = Marker()
    line_marker.header.stamp = stamp
    line_marker.header.frame_id = frame_id
    line_marker.ns = "trajectory"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = line_width
    line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
    line_marker.pose.orientation.w = 1.0
    # Without this RViz resolves base_link -> fixed frame once, when the marker arrives, and
    # pins the result in the fixed frame. The trajectory then falls behind the vehicle until
    # the next prediction lands — at 11 m/s and a 0.5 s replan that is over 5 m of visible
    # lag, which reads as the path starting somewhere behind the car.
    line_marker.frame_locked = True

    for point in traj_np:
        p = Point()
        p.x = float(point[0])
        p.y = float(point[1])
        p.z = float(point[2]) + z_offset_m
        line_marker.points.append(p)

    marker_array.markers.append(line_marker)

    if ego_footprint is not None:
        length, width, rear_overhang = ego_footprint
        marker_array.markers.append(
            _ego_footprint_marker(stamp, frame_id, length, width, rear_overhang, z_offset_m)
        )

    if cot_text:
        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = frame_id
        text_marker.ns = "reasoning"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = cot_height_m
        text_marker.pose.orientation.w = 1.0
        text_marker.frame_locked = True
        text_marker.scale.z = 0.8  # Glyph height in metres
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = wrap_text(cot_text)
        marker_array.markers.append(text_marker)

    return marker_array


def extract_text(extra: Optional[dict], key: str) -> Optional[str]:
    """Pull one text field out of a model's ``extra`` dict.

    Both generations return ``extra[key]`` as an object-dtype array with a leading
    batch dim plus trajectory-set/sample dims, so the first element is the text for
    the single sample we request.
    """
    if not extra or key not in extra:
        return None
    text_array = extra[key]
    try:
        text = text_array[0, 0, 0]
    except Exception:
        try:
            text = np.asarray(text_array).reshape(-1)[0]
        except Exception:
            return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = str(text).strip()
    return text or None
