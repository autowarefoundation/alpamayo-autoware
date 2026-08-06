# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Launch the Alpamayo 2 Super node against a six-camera Autoware sensor set."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Alpamayo 2 Super's trajectory task consumes exactly six cameras, IDs (0, 1, 2, 3, 5, 6)
    # in ascending order. The topics below were matched to the Alpamayo camera definitions
    # using /tf_static yaw and the horizontal FOV from each /camera_info, on the reference
    # Autoware sensor set this was developed against:
    #
    #   idx 0  cross_left_120fov   <- camera5   (HFOV 88.3, yaw +60.7)
    #   idx 1  front_wide_120fov   <- camera1   (HFOV 88.1, yaw  -0.2)
    #   idx 2  cross_right_120fov  <- camera6   (HFOV 88.7, yaw -60.2)
    #   idx 3  rear_left_70fov     <- camera9   (HFOV 57.3, yaw +161.6)
    #   idx 5  rear_right_70fov    <- camera10  (HFOV 56.8, yaw -159.0)
    #   idx 6  front_tele_30fov    <- camera2   (HFOV 30.2, yaw  +0.6)
    #
    # The narrower cross cameras (camera3 at 57.2/+49.8 and camera4 at 57.0/-50.5, which
    # the 1.5 launch file uses) are the alternative for IDs 0 and 2; camera7/camera8
    # (88 deg, yaw +-155) are the alternative for the rear pair. Override camera_topics to
    # compare.
    default_camera_topics = [
        "/sensing/camera/camera5/image_raw/compressed",   # cross_left   -> 0
        "/sensing/camera/camera1/image_raw/compressed",   # front_wide   -> 1
        "/sensing/camera/camera6/image_raw/compressed",   # cross_right  -> 2
        "/sensing/camera/camera9/image_raw/compressed",   # rear_left    -> 3
        "/sensing/camera/camera10/image_raw/compressed",  # rear_right   -> 5
        "/sensing/camera/camera2/image_raw/compressed",   # front_tele   -> 6
    ]
    default_camera_indices = [0, 1, 2, 3, 5, 6]

    arguments = [
        DeclareLaunchArgument("model_name", default_value="nvidia/Alpamayo2-Super"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("inference_period_sec", default_value="2.0"),
        DeclareLaunchArgument("max_generation_length", default_value="256"),
        DeclareLaunchArgument("num_diffusion_steps", default_value="10"),
        DeclareLaunchArgument("max_image_long_side", default_value="1280"),
        DeclareLaunchArgument("odometry_topic", default_value="/localization/kinematic_state"),
        DeclareLaunchArgument("trajectory_topic", default_value="/alpamayo/predicted_trajectory"),
        DeclareLaunchArgument("marker_line_width", default_value="1.2"),
        DeclareLaunchArgument("marker_z_offset", default_value="0.4"),
        DeclareLaunchArgument("max_frame_age_sec", default_value="3.0"),
        DeclareLaunchArgument("skip_on_bad_history", default_value="true"),
        # (length, width, rear_overhang) in metres; "[0.0, 0.0, 0.0]" draws nothing.
        DeclareLaunchArgument("ego_footprint", default_value="[4.77, 1.73, 1.03]"),
        # Navigation classifier-free guidance. Off by default and experimental -- see the
        # README. Enabling it also needs lanelet2_map_path and a mission route.
        DeclareLaunchArgument("nav_cfg_enabled", default_value="false"),
        DeclareLaunchArgument("lanelet2_map_path", default_value=""),
        DeclareLaunchArgument("route_topic", default_value="/planning/mission_planning/route"),
        # Negative means the checkpoint's own inference_guidance_weight (3.0).
        DeclareLaunchArgument("nav_guidance_weight", default_value="-1.0"),
    ]

    node = Node(
        package="alpamayo_ros",
        executable="alpamayo2_node",
        name="alpamayo2_node",
        output="screen",
        parameters=[
            {
                "model_name": LaunchConfiguration("model_name"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "camera_topics": default_camera_topics,
                "camera_indices": default_camera_indices,
                "odometry_topic": LaunchConfiguration("odometry_topic"),
                "trajectory_topic": LaunchConfiguration("trajectory_topic"),
                "cot_topic": "/alpamayo/reasoning",
                "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                "inference_period_sec": LaunchConfiguration("inference_period_sec"),
                "max_generation_length": LaunchConfiguration("max_generation_length"),
                "num_diffusion_steps": LaunchConfiguration("num_diffusion_steps"),
                "max_image_long_side": LaunchConfiguration("max_image_long_side"),
                "marker_line_width": LaunchConfiguration("marker_line_width"),
                "marker_z_offset": LaunchConfiguration("marker_z_offset"),
                "max_frame_age_sec": LaunchConfiguration("max_frame_age_sec"),
                "skip_on_bad_history": LaunchConfiguration("skip_on_bad_history"),
                "ego_footprint": LaunchConfiguration("ego_footprint"),
                "nav_cfg_enabled": LaunchConfiguration("nav_cfg_enabled"),
                "lanelet2_map_path": LaunchConfiguration("lanelet2_map_path"),
                "route_topic": LaunchConfiguration("route_topic"),
                "nav_guidance_weight": LaunchConfiguration("nav_guidance_weight"),
            }
        ],
    )

    return LaunchDescription(arguments + [node])
