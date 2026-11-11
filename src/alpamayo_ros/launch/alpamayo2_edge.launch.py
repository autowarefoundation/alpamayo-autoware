# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Launch the Alpamayo 2 Edge node against a six-camera Autoware sensor set."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # The same camera assignment as alpamayo2.launch.py (Alpamayo 2 Super), whose comments
    # give the field of view and yaw each topic was matched by.
    default_camera_topics = [
        "/sensing/camera/camera5/image_raw/compressed",  # cross_left   -> 0
        "/sensing/camera/camera1/image_raw/compressed",  # front_wide   -> 1
        "/sensing/camera/camera6/image_raw/compressed",  # cross_right  -> 2
        "/sensing/camera/camera9/image_raw/compressed",  # rear_left    -> 3
        "/sensing/camera/camera10/image_raw/compressed",  # rear_right  -> 5
        "/sensing/camera/camera2/image_raw/compressed",  # front_tele   -> 6
    ]
    default_camera_indices = [0, 1, 2, 3, 5, 6]

    arguments = [
        DeclareLaunchArgument("model_name", default_value="nvidia/Alpamayo2-Edge"),
        DeclareLaunchArgument(
            "model_revision", default_value="f8951d095de3589b3face1dec3f4394c8ecedb38"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        # tuned / compile need flash-attn; off runs eager PyTorch.
        DeclareLaunchArgument("accel", default_value="tuned"),
        # Camera (height, width) to build the accelerated graphs for at startup.
        DeclareLaunchArgument("warm_up_image_size", default_value="[0, 0]"),
        DeclareLaunchArgument("inference_period_sec", default_value="0.05"),
        DeclareLaunchArgument("max_generation_length", default_value="256"),
        DeclareLaunchArgument("num_diffusion_steps", default_value="10"),
        DeclareLaunchArgument("top_p", default_value="0.98"),
        DeclareLaunchArgument("temperature", default_value="0.6"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("frame_match_tolerance_sec", default_value="0.06"),
        DeclareLaunchArgument("odometry_topic", default_value="/localization/kinematic_state"),
        DeclareLaunchArgument("trajectory_topic", default_value="/alpamayo/predicted_trajectory"),
        DeclareLaunchArgument("marker_line_width", default_value="1.2"),
        DeclareLaunchArgument("marker_z_offset", default_value="0.4"),
        DeclareLaunchArgument("max_frame_age_sec", default_value="3.0"),
        DeclareLaunchArgument("skip_on_bad_history", default_value="true"),
        DeclareLaunchArgument("drop_bad_trajectory", default_value="true"),
        # (length, width, rear_overhang) in metres; "[0.0, 0.0, 0.0]" draws nothing.
        DeclareLaunchArgument("ego_footprint", default_value="[4.77, 1.73, 1.03]"),
        # With a lanelet2 map the instruction follows the mission route; without one, or
        # until a route arrives, it is default_nav_text.
        DeclareLaunchArgument("lanelet2_map_path", default_value=""),
        DeclareLaunchArgument("route_topic", default_value="/planning/mission_planning/route"),
        DeclareLaunchArgument("default_nav_text", default_value="Continue straight"),
    ]

    node = Node(
        package="alpamayo_ros",
        executable="alpamayo2_edge_node",
        name="alpamayo2_edge_node",
        # colcon writes the system interpreter into the entry point; run it with the python3
        # on PATH instead, which is a2e_venv's once that environment is activated.
        prefix="python3",
        output="screen",
        parameters=[
            {
                "model_name": LaunchConfiguration("model_name"),
                "model_revision": LaunchConfiguration("model_revision"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "camera_topics": default_camera_topics,
                "camera_indices": default_camera_indices,
                "accel": LaunchConfiguration("accel"),
                "warm_up_image_size": LaunchConfiguration("warm_up_image_size"),
                "inference_period_sec": LaunchConfiguration("inference_period_sec"),
                "max_generation_length": LaunchConfiguration("max_generation_length"),
                "num_diffusion_steps": LaunchConfiguration("num_diffusion_steps"),
                "top_p": LaunchConfiguration("top_p"),
                "temperature": LaunchConfiguration("temperature"),
                "seed": LaunchConfiguration("seed"),
                "frame_match_tolerance_sec": LaunchConfiguration("frame_match_tolerance_sec"),
                "odometry_topic": LaunchConfiguration("odometry_topic"),
                "trajectory_topic": LaunchConfiguration("trajectory_topic"),
                "cot_topic": "/alpamayo/reasoning",
                "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                "nav_text_topic": "/alpamayo/nav_text",
                "marker_line_width": LaunchConfiguration("marker_line_width"),
                "marker_z_offset": LaunchConfiguration("marker_z_offset"),
                "max_frame_age_sec": LaunchConfiguration("max_frame_age_sec"),
                "skip_on_bad_history": LaunchConfiguration("skip_on_bad_history"),
                "drop_bad_trajectory": LaunchConfiguration("drop_bad_trajectory"),
                "ego_footprint": LaunchConfiguration("ego_footprint"),
                "lanelet2_map_path": LaunchConfiguration("lanelet2_map_path"),
                "route_topic": LaunchConfiguration("route_topic"),
                "default_nav_text": LaunchConfiguration("default_nav_text"),
            }
        ],
    )

    return LaunchDescription(arguments + [node])
