from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo node that listens to live camera + odometry topics."""
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
            DeclareLaunchArgument("use_flashdrive", default_value="false"),
            DeclareLaunchArgument(
                "flashdrive_url", default_value="http://127.0.0.1:8710"
            ),
            DeclareLaunchArgument("expert_onnx_path", default_value=""),
            DeclareLaunchArgument("num_diffusion_steps", default_value="5"),
            DeclareLaunchArgument("use_greedy_decode", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
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
                        "use_flashdrive": LaunchConfiguration("use_flashdrive"),
                        "flashdrive_url": LaunchConfiguration("flashdrive_url"),
                        "expert_onnx_path": LaunchConfiguration("expert_onnx_path"),
                        "num_diffusion_steps": LaunchConfiguration("num_diffusion_steps"),
                        "use_greedy_decode": LaunchConfiguration("use_greedy_decode"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            )
        ]
    )
