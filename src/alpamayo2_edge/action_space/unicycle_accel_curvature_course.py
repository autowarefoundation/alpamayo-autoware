# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

from alpamayo2_edge.action_space.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
)
from alpamayo2_edge.action_space.utils import unwrap_angle
from alpamayo2_edge.common import logging
from alpamayo2_edge.geometry.rotation import so3_to_yaw_torch

logger = logging.RankedLogger(__name__, rank_zero_only=True)

# The course offset fades in linearly between these chord speeds. Below the lower one a 0.1 s chord
# is a few centimetres long and its direction is egomotion noise, so the heading falls back to yaw.
_COURSE_FADE_START_MPS = 0.5
_COURSE_FADE_END_MPS = 1.0

# Reject raw course offsets above 50 mrad to limit plan rotation from pose errors.
# This bound is a robustness limit. Physical sideslip depends on the vehicle and lateral
# acceleration.
_MAX_COURSE_OFFSET_RAD = 0.05

# Curvature is kept down to 0.1 m/s (the yaw-heading space zeroes it below 0.6 m/s). Turns pulled
# away from a stop at full lock change heading by a few degrees below 0.6 m/s, and zeroing that
# rotates the rest of the decoded path. GT yaw jitter and standstill drift give |curvature| below
# 0.13 1/m, far inside the token range (0.26 1/m). Below 0.1 m/s a step covers under 1 cm, so
# curvature is barely observable and keeping it only adds clamped tokens.
# This encoding threshold is part of the action-space version, not a checkpoint config field.
# Retuning it requires a new action-space version to preserve existing training targets.
_COURSE_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.1


def _course_offset(chord_xy: torch.Tensor, yaw: torch.Tensor, chord_dt: float) -> torch.Tensor:
    """Course minus body yaw along a position chord, faded to 0 at low speed.

    The chord direction is taken modulo pi, so reversing yields the same small offset as driving
    forward. A raw offset beyond ``_MAX_COURSE_OFFSET_RAD`` is set to 0 before the speed fade.

    Args:
        chord_xy: (..., 2) displacement [m] over ``chord_dt``.
        yaw: (...,) body yaw [rad] at the chord midpoint.
        chord_dt: chord duration [s].

    Returns:
        (...,) offset [rad] in [-0.05, 0.05], i.e. minus the sideslip.
    """
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    along = cos_yaw * chord_xy[..., 0] + sin_yaw * chord_xy[..., 1]
    across = cos_yaw * chord_xy[..., 1] - sin_yaw * chord_xy[..., 0]
    offset = torch.atan2(across * torch.sign(along), along.abs())
    offset = torch.where(offset.abs() > _MAX_COURSE_OFFSET_RAD, torch.zeros_like(offset), offset)
    speed = torch.linalg.vector_norm(chord_xy, dim=-1) / chord_dt
    fade = (speed - _COURSE_FADE_START_MPS) / (_COURSE_FADE_END_MPS - _COURSE_FADE_START_MPS)
    return offset * fade.clamp(min=0.0, max=1.0)


class UnicycleAccelCurvatureCourseActionSpace(UnicycleAccelCurvatureActionSpace):
    """Unicycle accel/curvature action space whose heading is the course, not the body yaw.

    The rig origin (rear-axle centre) travels along its course, which differs from the body yaw by
    the rear-axle sideslip (~0.4 mrad + ~2.7 mrad per m/s^2 of lateral acceleration, yaw pointing
    into the turn). Integrating along the yaw puts decoded paths inside the driven one by roughly
    the integral of v * sideslip, so this space uses the course throughout:

    - Encoding fits the heading to course targets: body yaw plus the direction of the position
      chords relative to it (central chords, a backward chord at the last step).
    - Encoding and decoding both start from ``t0_states["theta"]``, the course at t0 relative to
      the t0 rig yaw, estimated from the history alone so ``decode(encode(traj))`` is consistent.
    - The course offset is 0 (heading = body yaw) below 0.5 m/s and fades in linearly to its full
      value at 1 m/s.
    - A raw course offset beyond 50 mrad is set to 0 before the speed fade.
      This bound limits plan rotation from pose errors. Physical sideslip depends on the vehicle
      and lateral acceleration.
    - Reversing measures chord directions modulo pi.
    - Encoded curvature is zeroed only below 0.1 m/s, not 0.6 m/s, so turns pulled away from a
      stop keep most of their heading change.

    Decoded positions follow the course. Decoded rotations estimate the body yaw that heading
    controllers track: the course plus a sideslip anchored to the measured one at t0,
    ``yaw_k = course_k - theta0 + slip_gradient * (a_lat_k - a_lat_0)`` with
    ``a_lat = v * d(course)/dt``, so the yaw is 0 at t0 and continuous with the history.
    ``slip_gradient`` [rad per m/s^2] is the rear-axle sideslip gradient of the rig the model was
    trained for.
    The GT egomotion dead-reckons positions with the rig's sideslip angle gradient (1.4e-3 to
    5.3e-3 across platforms); the default 2.5e-3 is the Hyperion 8 Lincoln rig's. It changes only
    the rotations, never the positions or the encoding.

    This space is for future trajectories and requires at least two history poses; it cannot
    encode history tokens that supply only one anchor pose. ``theta0`` rotates the whole decoded
    plan, so the history must be egomotion in the t0 rig frame, not reconstructed from tokens or
    bins. Float32 inputs and bfloat16 storage cast back to float32 are supported; this does not
    imply support for a native bfloat16 solver.
    """

    low_speed_curvature_threshold_mps = _COURSE_LOW_SPEED_CURVATURE_THRESHOLD_MPS

    def __init__(self, *args: Any, slip_gradient: float = 2.5e-3, **kwargs: Any) -> None:
        """Build the parent space.

        Args:
            *args: parent arguments.
            slip_gradient: rear-axle sideslip [rad per m/s^2] of the training rig; sets only the
                decoded yaw.
            **kwargs: parent keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.slip_gradient = slip_gradient
        logger.info(
            f"Course-heading action space: decoded yaw uses slip_gradient={slip_gradient} rad per "
            "m/s^2 (the training rig's)"
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Estimate speed and course heading at t0 from the history.

        The course at t0 is the last history chord's direction advanced by half the last yaw step,
        i.e. the sideslip is assumed constant over that half step.

        Args:
            traj_history_xyz: (..., T_hist, 3) in the t0 rig frame, with T_hist >= 2.
            traj_history_rot: (..., T_hist, 3, 3), with T_hist >= 2.

        Returns:
            Dict with ``v``: (...,) speed [m/s] at t0, as in the parent, and ``theta``: (...,)
            course [rad] at t0 relative to the t0 rig yaw; 0 below 0.5 m/s, faded in up to 1 m/s.

        Raises:
            ValueError: the history has fewer than 2 poses.
        """
        if traj_history_xyz.shape[-2] < 2:
            raise ValueError(
                "The course-heading action space is for future trajectories and needs at least "
                f"2 history poses, got {traj_history_xyz.shape[-2]}."
            )
        t0_states = super().estimate_t0_states(traj_history_xyz, traj_history_rot)
        last_yaw = unwrap_angle(so3_to_yaw_torch(traj_history_rot[..., -2:, :, :]))  # (..., 2)
        theta0 = _course_offset(
            chord_xy=traj_history_xyz[..., -1, :2] - traj_history_xyz[..., -2, :2],
            yaw=last_yaw.mean(dim=-1),
            chord_dt=self.dt,
        )
        return {**t0_states, "theta": theta0}

    def _initial_heading(self, t0_states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Course [rad] at t0 relative to the t0 rig yaw: (...,) ``t0_states["theta"]``.

        Raises:
            ValueError: ``t0_states`` lacks ``theta`` (e.g. built by the yaw-heading space).
        """
        if "theta" not in t0_states:
            raise ValueError(
                "The course-heading action space needs t0_states['theta'] (the course at t0); "
                "build t0_states with this space's estimate_t0_states()."
            )
        return t0_states["theta"]

    def _output_heading(
        self, theta: torch.Tensor, velocity: torch.Tensor, t0_states: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Body yaw [rad] for the decoded rotations: the course plus a sideslip anchored at t0.

        The sideslip at t0 is the measured ``-theta0``. After t0 it changes with the lateral
        acceleration ``v * d(theta)/dt`` at ``slip_gradient``.

        Args:
            theta: (..., N+1) integrated course at steps 0..N.
            velocity: (..., N+1) signed speed [m/s] at steps 0..N (negative when reversing).
            t0_states: output of ``estimate_t0_states``.

        Returns:
            (..., N+1) yaw relative to the t0 rig yaw; 0 at step 0.
        """
        lateral_accel = velocity * torch.gradient(theta, spacing=self.dt, dim=-1)[0]
        slip_change = self.slip_gradient * (lateral_accel - lateral_accel[..., :1])
        return theta - self._initial_heading(t0_states).unsqueeze(-1) + slip_change

    def _heading_targets(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Course targets [rad] at future steps 1..N: body yaw plus the chord course offset.

        Step k < N uses the central chord p[k+1] - p[k-1] with yaw[k]; step N uses the backward
        chord with the yaw at its midpoint. Accelerating through a curve, a central chord leads the
        tangent by curvature * accel * dt^2 / 2 (0.4 mrad at 0.05 1/m and 1.5 m/s^2).

        Args:
            traj_history_xyz: (..., T_hist, 3) in the t0 rig frame; the last pose is t0.
            traj_history_rot: (..., T_hist, 3, 3)
            traj_future_xyz: (..., N, 3)
            traj_future_rot: (..., N, 3, 3)

        Returns:
            (..., N) course targets, unwrapped from the t0 yaw.
        """
        xy = torch.cat([traj_history_xyz[..., -1:, :2], traj_future_xyz[..., :2]], dim=-2)
        rot = torch.cat([traj_history_rot[..., -1:, :, :], traj_future_rot], dim=-3)
        yaw = unwrap_angle(so3_to_yaw_torch(rot))  # (..., N+1), index 0 = t0
        interior_offset = _course_offset(
            chord_xy=xy[..., 2:, :] - xy[..., :-2, :], yaw=yaw[..., 1:-1], chord_dt=2 * self.dt
        )  # (..., N-1)
        last_offset = _course_offset(
            chord_xy=xy[..., -1, :] - xy[..., -2, :],
            yaw=yaw[..., -2:].mean(dim=-1),
            chord_dt=self.dt,
        )  # (...,)
        return yaw[..., 1:] + torch.cat([interior_offset, last_offset.unsqueeze(-1)], dim=-1)
