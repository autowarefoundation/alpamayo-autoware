#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo and publishes Autoware trajectories.

Image preprocessing runs entirely on GPU (torchvision JPEG decode + F.interpolate).
Two expert modes are available:

  baseline (default)  – native PyTorch expert, nucleus sampling, 10-step diffusion.
  optimized           – TRT FP16 expert engine (via ``expert_onnx_path``),
                        configurable greedy / nucleus decode, tunable diffusion steps.
"""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
import torch
import torchvision
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoRosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) to run Alpamayo inference."""

    def __init__(self) -> None:
        super().__init__("alpamayo_node")

        self.model_name: str = "nvidia/Alpamayo-R1-10B"

        # ── Topic parameters ──
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("inference_period_sec", 1.0)
        # ROS2 Jazzy: Use non-empty default for string array parameters to properly infer type
        self.declare_parameter("camera_topics", [""])

        # ── Optimization parameters ──
        self.declare_parameter("expert_onnx_path", "")
        self.declare_parameter("num_diffusion_steps", 10)
        self.declare_parameter("use_greedy_decode", False)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)

        # ── Frame / history ──
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("num_frames", 4)
        self.declare_parameter("num_history_steps", 16)

        self._frame_id: str = self.get_parameter("frame_id").value
        self._num_frames: int = self.get_parameter("num_frames").value
        self._num_history_steps: int = self.get_parameter("num_history_steps").value

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 1.0
        self._skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)

        inference_period = float(self.get_parameter("inference_period_sec").value)

        # ── Publishers ──
        queue_size = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, queue_size)
        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, queue_size)
        self._marker_pub = self.create_publisher(MarkerArray, traj_topic + "_markers", queue_size)

        # ── Camera subscriptions ──
        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        self._camera_topics = [t for t in camera_topics if t]
        if not self._camera_topics:
            raise ValueError("camera_topics parameter must list at least one image topic.")
        self._camera_buffers: Dict[str, deque] = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._image_callback(t, msg),
                camera_qos,
            )
            self.get_logger().info(f"Subscribed to camera: {topic}")

        # ── Odometry subscription ──
        odom_topic = self.get_parameter("odometry_topic").value
        self._odometry_buffer: deque[Odometry] = deque(
            maxlen=self._num_history_steps * self._skip_num + 10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)

        # ── Inference executor ──
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        # ── Load model ──
        self.get_logger().info(f"Loading Alpamayo model: {self.model_name}")
        self._model = AlpamayoR1.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()

        # TRT Expert (optional — set expert_onnx_path to enable)
        expert_onnx = self.get_parameter("expert_onnx_path").value
        if expert_onnx and Path(expert_onnx).exists():
            from alpamayo_r1.trt.expert_runtime import TrtExpertEngine

            engine = TrtExpertEngine(
                onnx_model_path=expert_onnx,
                engine_cache_dir=str(Path(expert_onnx).parent / "engine_cache"),
                enable_int8=False,
                enable_fp16=True,
            )
            self._model.set_expert_step_runner(engine)
            self.get_logger().info(f"TRT Expert loaded: {expert_onnx}")

        # Diffusion steps
        num_steps: int = self.get_parameter("num_diffusion_steps").value
        self._model.diffusion.num_inference_steps = num_steps

        self._processor = helper.get_processor(self._model.tokenizer)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # ── Timer ──
        self._timer = self.create_timer(inference_period, self._timer_callback)
        mode = "TRT Expert" if (expert_onnx and Path(expert_onnx).exists()) else "native Expert"
        self.get_logger().info(
            f"Alpamayo ready — {mode}, {num_steps}-step diffusion, period={inference_period}s"
        )

    def destroy_node(self) -> None:
        """Cleanup resources before shutting down."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_payload()
        if payload is None:
            return
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        # Store raw JPEG bytes — GPU decode happens in inference thread
        jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg_bytes))

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _prepare_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None

        # Collect raw JPEG bytes per camera for GPU decode
        jpeg_buffers = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            jpeg_buffers.extend([f for _, f in frames])

        if len(self._odometry_buffer) < self._num_history_steps * self._skip_num:
            return None
        odom_history = list(self._odometry_buffer)[
            -self._num_history_steps * self._skip_num :: self._skip_num
        ]

        positions, rotations = [], []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rotations.append(Rotation.from_quat(quat).as_matrix())

        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        history_xyz = (positions_np - positions_np[-1]) @ t0_rot_inv.T
        history_rot = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)

        return {
            "jpeg_buffers": jpeg_buffers,
            "ego_history_xyz": torch.from_numpy(history_xyz).unsqueeze(0).unsqueeze(0),
            "ego_history_rot": torch.from_numpy(history_rot).unsqueeze(0).unsqueeze(0),
        }

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()

        # GPU JPEG decode + GPU batch resize. Keep uint8 on GPU all the way
        # through the processor — ``.cpu()`` here would round-trip ~20 MB per
        # inference and push normalize+patchify onto the CPU fast-path
        # (~58 ms/frame instead of ~0.7 ms/frame on GPU).
        decoded = []
        for jpeg_buf in payload["jpeg_buffers"]:
            decoded.append(torchvision.io.decode_jpeg(jpeg_buf, device="cuda"))
        stacked = torch.stack(decoded)  # [N, 3, H, W] uint8 on GPU
        if stacked.shape[-2:] != (560, 1008):
            stacked = torch.nn.functional.interpolate(
                stacked.float(), size=(560, 1008), mode="bicubic", align_corners=False,
            ).clamp(0, 255).to(torch.uint8)

        n_cams = len(self._camera_topics)
        camera_per_cam = [
            stacked[i * self._num_frames : (i + 1) * self._num_frames]
            for i in range(n_cams)
        ]
        image_frames = torch.stack(camera_per_cam)  # [n_cams, n_frames, 3, H, W] on GPU

        messages = helper.create_message(image_frames.flatten(0, 1))
        # device="cuda" makes Qwen2VLImageProcessorFast run normalize + patchify
        # on GPU (~0.7 ms/frame vs ~58 ms/frame on the CPU fast-path).
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            device="cuda",
        )
        # apply_chat_template(device=cuda) only puts pixel_values on GPU; text
        # ids / attention_mask / image_grid_thw still come back on CPU. Move
        # them once so the forward does not hit per-tensor sync waits.
        processor_inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v
            for k, v in processor_inputs.items()
        }
        model_inputs = {
            "tokenized_data": processor_inputs,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device=self._device)

        use_greedy = self.get_parameter("use_greedy_decode").value
        top_p = 1.0 if use_greedy else float(self.get_parameter("top_p").value)
        temperature = 1.0 if use_greedy else float(self.get_parameter("temperature").value)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self._dtype):
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=1,
                num_traj_sets=1,
                max_generation_length=64,
                return_extra=True,
            )

        trajectory = pred_xyz[0, 0, 0].detach().cpu()
        rotation = pred_rot[0, 0, 0].detach().cpu()

        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)
        self._marker_pub.publish(self._trajectory_to_markers(trajectory))

        cot_text = self._extract_text(extra, "cot")
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

        return {"duration_sec": time.time() - start, "num_poses": len(traj_msg.points)}

    def _to_autoware_trajectory(
        self,
        trajectory: torch.Tensor,
        rotations: torch.Tensor | None,
    ) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None
        now = self.get_clock().now().to_msg()

        traj_msg = Trajectory()
        traj_msg.header.stamp = now
        traj_msg.header.frame_id = self._frame_id

        dt = 0.1
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
                dist = math.hypot(dx, dy)
                speed = dist / dt if dt > 0 else 0.0

            traj_point.longitudinal_velocity_mps = float(speed)
            traj_point.lateral_velocity_mps = 0.0
            traj_point.acceleration_mps2 = 0.0
            traj_point.heading_rate_rps = 0.0

            seconds_float = idx * dt
            seconds_int = int(seconds_float)
            nanosec = int((seconds_float - seconds_int) * 1e9)
            traj_point.time_from_start = Duration(sec=seconds_int, nanosec=nanosec)

            traj_msg.points.append(traj_point)
            prev_xy = (point[0], point[1])

        return traj_msg

    def _trajectory_to_markers(self, trajectory: torch.Tensor) -> MarkerArray:
        traj_np = trajectory.numpy()
        now = self.get_clock().now().to_msg()

        marker_array = MarkerArray()

        # LINE_STRIP marker for trajectory path
        line_marker = Marker()
        line_marker.header.stamp = now
        line_marker.header.frame_id = self._frame_id
        line_marker.ns = "trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 1.0  # Line width
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        line_marker.pose.orientation.w = 1.0

        for point in traj_np:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2])
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)
        return marker_array

    def _extract_text(self, extra: dict, key: str) -> Optional[str]:
        if not extra or key not in extra:
            return None
        text_array = extra[key]
        try:
            text = text_array[0, 0, 0]
        except Exception:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = str(text).strip()
        return text or None

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo inference failed: {exc}")
            return
        if not metrics:
            return
        self.get_logger().info(
            f"Alpamayo inference completed in {metrics['duration_sec']:.2f}s "
            f"(points={metrics['num_poses']})."
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AlpamayoRosNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
