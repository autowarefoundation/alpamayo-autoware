from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo inference and the diffusion-planner trajectory porter."""
    # Camera topics and their corresponding camera indices:
    # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
    default_camera_topics = [
        "/sensing/camera/camera3/image_raw/compressed",   # cross_left  -> index 0
        "/sensing/camera/camera1/image_raw/compressed",   # front_wide  -> index 1
        "/sensing/camera/camera4/image_raw/compressed",   # cross_right -> index 2
        "/sensing/camera/camera2/image_raw/compressed",   # front_tele  -> index 6
    ]
    default_camera_indices = [0, 1, 2, 6]
    return LaunchDescription(
        [
            Node(
                package="alpamayo_ros",
                executable="alpamayo_node",
                name="alpamayo_node",
                output="screen",
                parameters=[
                    {
                        "auto_run": True,
                        "camera_topics": default_camera_topics,
                        "camera_indices": default_camera_indices,
                        "odometry_topic": "/localization/kinematic_state",
                        "trajectory_topic": "/alpamayo/predicted_trajectory",
                        "cot_topic": "/alpamayo/reasoning",
                        "cot_with_stamped_topic": "/alpamayo/reasoning_stamped",
                        "inference_period_sec": 1.0,
                    }
                ],
            ),
            Node(
                package="alpamayo_ros",
                executable="dp_stack_trajectory_porter",
                name="dp_stack_trajectory_porter",
                output="screen",
                parameters=[
                    {
                        "input_trajectory_topic": "/alpamayo/predicted_trajectory",
                        "output_trajectories_topic": (
                            "/planning/generator/diffusion_planner/modified_candidate_trajectories"
                        ),
                        "kinematic_state_topic": "/localization/kinematic_state",
                    }
                ],
            ),
        ]
    )
