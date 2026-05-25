#!/usr/bin/env python3
"""
Adapt an Alpamayo predicted trajectory for the diffusion-planner stack.

Subscribes to an input ``Trajectory`` (default ``/alpamayo/predicted_trajectory``),
drops the first abnormal point, transforms the remaining points into the map frame
using the latest ``/localization/kinematic_state``, wraps the result as a single
``CandidateTrajectory``, and publishes ``CandidateTrajectories`` on the diffusion
planner input topic (default
``/planning/generator/diffusion_planner/modified_candidate_trajectories``).
"""

from __future__ import annotations

import copy

import numpy as np
import rclpy
from tf_transformations import concatenate_matrices, quaternion_matrix, quaternion_multiply, translation_matrix
from typing import List
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_internal_planning_msgs.msg import CandidateTrajectories, CandidateTrajectory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from geometry_msgs.msg import Transform


class TrajectoryUpdateNode(Node):
    """Bridge Alpamayo trajectories into the diffusion-planner candidate format.

    Processing pipeline for each incoming trajectory:

    1. Drop the first point, which is often abnormal relative to the ego state.
    2. If the input frame differs from the map frame in ``kinematic_state``, rigidly
       transform all remaining points from ``base_link`` to map.
    3. Package the trajectory as one ``CandidateTrajectory`` inside
       ``CandidateTrajectories`` for downstream diffusion-planner modules.
    """

    def __init__(self) -> None:
        super().__init__("trajectory_update_node")

        self.declare_parameter("input_trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("output_trajectories_topic", "/planning/generator/diffusion_planner/modified_candidate_trajectories")
        self.declare_parameter("kinematic_state_topic", "/localization/kinematic_state")

        input_topic = self.get_parameter("input_trajectory_topic").value
        output_topic = self.get_parameter("output_trajectories_topic").value
        kinematic_topic = self.get_parameter("kinematic_state_topic").value

        self._kinematic_state: Odometry | None = None

        self._kinematic_sub = self.create_subscription(
            Odometry,
            kinematic_topic,
            self._on_kinematic_state,
            10,
        )
        self._trajectory_sub = self.create_subscription(
            Trajectory,
            input_topic,
            self._on_input_trajectory,
            10,
        )
        self._trajectories_pub = self.create_publisher(CandidateTrajectories, output_topic, 10)

        self.get_logger().info(
            f"trajectory_update_node: {input_topic} -> {output_topic}, "
            f"kinematic_state={kinematic_topic}"
        )

    def _on_kinematic_state(self, msg: Odometry) -> None:
        self._kinematic_state = msg

    def _kinematic_state_to_transform(self) -> Transform:
        """Build T_map_base from /localization/kinematic_state (Odometry)."""
        if self._kinematic_state is None:
            raise RuntimeError("kinematic_state is not available")

        pose = self._kinematic_state.pose.pose
        transform = Transform()
        transform.translation.x = pose.position.x
        transform.translation.y = pose.position.y
        transform.translation.z = pose.position.z
        transform.rotation = pose.orientation
        return transform

    def transform_point(self, point: TrajectoryPoint, transform: Transform) -> TrajectoryPoint:
        """Apply rigid transform (e.g. base_link -> map) to pose; copy motion scalars."""
        new_point = copy.deepcopy(point)

        q_transform = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        q_point = [
            point.pose.orientation.x,
            point.pose.orientation.y,
            point.pose.orientation.z,
            point.pose.orientation.w,
        ]
        q_out = quaternion_multiply(q_transform, q_point)
        new_point.pose.orientation.x = q_out[0]
        new_point.pose.orientation.y = q_out[1]
        new_point.pose.orientation.z = q_out[2]
        new_point.pose.orientation.w = q_out[3]

        mat = concatenate_matrices(
            translation_matrix(
                [transform.translation.x, transform.translation.y, transform.translation.z]
            ),
            quaternion_matrix(q_transform),
        )
        p_in = np.array(
            [point.pose.position.x, point.pose.position.y, point.pose.position.z, 1.0]
        )
        p_out = mat @ p_in
        new_point.pose.position.x = float(p_out[0])
        new_point.pose.position.y = float(p_out[1])
        new_point.pose.position.z = float(p_out[2])

        return new_point

    def convert_trajectory_to_map_frame(
        self, points: List[TrajectoryPoint]
    ) -> List[TrajectoryPoint]:
        """Transform trajectory points from base_link to map using latest kinematic_state."""
        transform = self._kinematic_state_to_transform()
        return [self.transform_point(point, transform) for point in points]

    def _on_input_trajectory(self, msg: Trajectory) -> None:
        """Drop the first point, convert to map frame, and publish candidate trajectories."""
        if not msg.points:
            self.get_logger().warn("Received empty trajectory; skipping publish")
            return

        if self._kinematic_state is None:
            self.get_logger().warn(
                "No kinematic_state received yet; publishing trajectory unchanged"
            )
            return
        else:

            target_header = self._kinematic_state.header

            out = CandidateTrajectories() 
            out.candidate_trajectories = []

            msg.points.pop(0)
            single_trajectory = CandidateTrajectory()
            single_trajectory.header = target_header

            if msg.header.frame_id != target_header.frame_id:
                # Input is in base_link; use kinematic_state to express points in map frame.
                single_trajectory.points = self.convert_trajectory_to_map_frame(msg.points)
            else:
                single_trajectory.points = msg.points
            out.candidate_trajectories.append(single_trajectory)
        self._trajectories_pub.publish(out)
        
        self.get_logger().info(f"Published {len(out.candidate_trajectories)} trajectories")

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrajectoryUpdateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
