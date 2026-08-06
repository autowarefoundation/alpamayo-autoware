#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo 2 Super and publishes trajectories.

Alpamayo 2 Super is a 34B model (32B Qwen3-VL backbone + 2B flow-matching action expert).
It differs from the 1.5 node in five ways that shape this file:

* Six cameras, fixed IDs ``(0, 1, 2, 3, 5, 6)`` in ascending order, four frames each at
  10 Hz — the model's ``trajectory`` task input profile.
* Inputs are assembled as a plain ``data`` dict and tokenized by the upstream chat
  template, rather than 1.5's ``helper.create_message`` path.
* ``sample_trajectories_from_data`` returns four values; the third (``logprob``) is a
  zero placeholder upstream, not a confidence, so it is discarded.
* The horizon is 64 waypoints at 0.1 s = 6.4 s (1.5 emits 20 / 2.0 s).
* Inference takes seconds, not milliseconds. This is a visualization/research node: the
  trajectory it publishes is far too stale to close the loop on. Trajectory headers carry
  the *input* t0 stamp so consumers can measure that staleness.

The TensorRT expert engine is unavailable for this generation.
"""

from __future__ import annotations

import os

# Must be set before the first CUDA allocation: the 72 GB of weights plus a ~6k-token
# prefill fragments the allocator badly without it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time  # noqa: E402
from collections import deque  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import rclpy  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from autoware_internal_debug_msgs.msg import StringStamped  # noqa: E402
from autoware_planning_msgs.msg import Trajectory  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CompressedImage  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

from alpamayo2_super import helper  # noqa: E402
from alpamayo2_super.input_profiles import (  # noqa: E402
    DRIVING_SIX_CAMERA_FOUR_FRAME,
    input_profile_record,
)
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo_ros import conversions  # noqa: E402

#: Camera IDs of the model's ``trajectory`` task profile. Ascending order is asserted
#: inside the chat template, so the node validates it at startup instead.
TRAJECTORY_CAMERA_IDS = list(DRIVING_SIX_CAMERA_FOUR_FRAME.camera_ids)


class Alpamayo2RosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) for Alpamayo 2 Super."""

    def __init__(self) -> None:
        super().__init__("alpamayo2_node")

        self.declare_parameter("model_name", "nvidia/Alpamayo2-Super")
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        # A single inference takes seconds on a 34B model; 0.1 s (the 1.5 default) would
        # just mean every tick is dropped by the in-flight guard below.
        self.declare_parameter("inference_period_sec", 2.0)

        # ROS 2 Jazzy: non-empty defaults so string/int array parameter types are inferred.
        self.declare_parameter("camera_topics", [""])
        # One Alpamayo camera ID per entry of camera_topics, same order. Must be exactly
        # (0, 1, 2, 3, 5, 6): 0=Front left, 1=Front, 2=Front right, 3=Rear left,
        # 5=Rear right, 6=Front telephoto.
        self.declare_parameter("camera_indices", TRAJECTORY_CAMERA_IDS)

        # Upstream default is max(256, tokens_per_future_traj) = 256. Chain-of-causation
        # decode dominates latency, so this is the biggest lever — but cutting it too far
        # truncates generation before the trajectory-start token and breaks CoT parsing.
        self.declare_parameter("max_generation_length", 256)
        # Flow-matching Euler steps. Checkpoint default is 10.
        self.declare_parameter("num_diffusion_steps", 10)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        # Native frames are 2880x1860; the processor downsamples to ~196k px anyway, so
        # shrinking before the processor cuts decode/normalize cost with no real loss.
        self.declare_parameter("max_image_long_side", 1280)
        self.declare_parameter("marker_line_width", 1.2)
        # Lifts the trajectory line off the ground plane so it does not z-fight with an
        # Autoware vector map's road polygons in RViz.
        self.declare_parameter("marker_z_offset", 0.4)
        # Ego outline drawn at base_link: (length, width, rear_overhang) in metres. Defaults
        # to a JPN TAXI. Set length to 0 to draw nothing.
        self.declare_parameter("ego_footprint", [4.77, 1.73, 1.03])
        # Guards against inferring on leftover buffered frames after a rosbag replay
        # ends. 0 disables the check.
        self.declare_parameter("max_frame_age_sec", 3.0)
        # Skip the tick when the ego history fails its invariants instead of spending 3 s of
        # GPU on it. A history with a discontinuity — which is what a replay's first ticks
        # see, as /clock jumps to the start offset — makes the action space estimate a wild
        # initial speed, and the trajectory comes back starting metres from the vehicle.
        self.declare_parameter("skip_on_bad_history", True)
        # Drop a prediction whose waypoints do not start at the vehicle. This is a separate
        # failure from a malformed history: a plausible history can still produce a displaced
        # rollout, and the result looks like a path detached from the car.
        self.declare_parameter("drop_bad_trajectory", True)
        self.declare_parameter("seed", 0)

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 10.0  # Model expects 10 Hz (time_step = 0.1 s)
        self.skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)

        self._num_history_steps = 16
        self._num_frames = len(DRIVING_SIX_CAMERA_FOUR_FRAME.frame_indices)
        self._max_long_side = int(self.get_parameter("max_image_long_side").value)
        self._marker_line_width = float(self.get_parameter("marker_line_width").value)
        self._marker_z_offset = float(self.get_parameter("marker_z_offset").value)
        self._max_frame_age_sec = float(self.get_parameter("max_frame_age_sec").value)
        footprint = list(
            self.get_parameter("ego_footprint").get_parameter_value().double_array_value
        )
        self._ego_footprint = (
            tuple(footprint) if len(footprint) == 3 and footprint[0] > 0.0 else None
        )
        self._skip_on_bad_history = bool(self.get_parameter("skip_on_bad_history").value)
        self._drop_bad_trajectory = bool(self.get_parameter("drop_bad_trajectory").value)
        self._frame_id = "base_link"
        # The ego-history invariants are checked and logged once, on the first inference.
        self._history_checked = False

        inference_period = float(self.get_parameter("inference_period_sec").value)

        queue_size = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, queue_size)
        self.get_logger().info(f"Publishing Autoware trajectories on {traj_topic}")

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, queue_size)
        self.get_logger().info(f"Publishing reasoning traces on {cot_topic}")

        cot_stamped_topic = self.get_parameter("cot_with_stamped_topic").value
        self._cot_stamped_pub = self.create_publisher(StringStamped, cot_stamped_topic, queue_size)
        self.get_logger().info(f"Publishing reasoning traces (stamped) on {cot_stamped_topic}")

        marker_topic = traj_topic + "_markers"
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, queue_size)
        self.get_logger().info(f"Publishing trajectory markers on {marker_topic}")

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        camera_indices = list(
            self.get_parameter("camera_indices").get_parameter_value().integer_array_value
        )
        self._validate_camera_config(camera_topics, camera_indices)
        self._camera_topics = camera_topics
        self._camera_indices = torch.tensor(camera_indices, dtype=torch.int64)
        self._camera_names = list(DRIVING_SIX_CAMERA_FOUR_FRAME.camera_names)
        self._input_profile = input_profile_record(DRIVING_SIX_CAMERA_FOUR_FRAME)
        for topic, cam_id, name in zip(camera_topics, camera_indices, self._camera_names):
            self.get_logger().info(f"Camera {cam_id} ({name}) <- {topic}")

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
            self.get_logger().info(f"Subscribed to camera topic: {topic}")

        self._odometry_buffer: deque = deque(
            maxlen=self._num_history_steps * self.skip_num + 10
        )
        odom_topic = self.get_parameter("odometry_topic").value
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        self._auto_timer = self.create_timer(inference_period, self._timer_callback)

        self._load_model()

        self._top_p = float(self.get_parameter("top_p").value)
        self._temperature = float(self.get_parameter("temperature").value)
        self._max_gen_len = int(self.get_parameter("max_generation_length").value)
        self._num_diffusion_steps = int(self.get_parameter("num_diffusion_steps").value)
        self.get_logger().info(
            f"Generation: top_p={self._top_p} temperature={self._temperature} "
            f"max_generation_length={self._max_gen_len} "
            f"diffusion_steps={self._num_diffusion_steps}"
        )

        seed = int(self.get_parameter("seed").value)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.get_logger().info("Alpamayo 2 Super model loaded and ready.")

    # --- Setup helpers ---

    def _validate_camera_config(self, topics: List[str], indices: List[int]) -> None:
        """Reject camera configurations the model cannot consume.

        The chat template asserts ascending camera IDs deep inside tokenization, which
        would surface as an opaque mid-run failure; and the trajectory task profile is a
        fixed six-camera set, not a subset. Both are cheap to check here.
        """
        if not topics or topics == [""]:
            raise ValueError(
                "camera_topics must list six image topics, one per Alpamayo camera ID "
                f"{TRAJECTORY_CAMERA_IDS}."
            )
        if len(indices) != len(topics):
            raise ValueError(
                f"camera_indices length ({len(indices)}) must match "
                f"camera_topics length ({len(topics)})."
            )
        if indices != TRAJECTORY_CAMERA_IDS:
            raise ValueError(
                "Alpamayo 2 Super's trajectory task requires exactly camera_indices="
                f"{TRAJECTORY_CAMERA_IDS} in ascending order, got {indices}."
            )

    def _load_model(self) -> None:
        """Load the checkpoint onto the GPU with SDPA attention."""
        model_name = str(self.get_parameter("model_name").value)
        self.get_logger().info(
            f"Loading Alpamayo 2 Super model {model_name} "
            f"on device={self._device} dtype={self._dtype} (~72 GB, this takes minutes)"
        )
        started = time.time()
        # flash-attn 2.8.3 ships no sm_120 (Blackwell) kernels, and both Alpamayo 2 model
        # classes declare SDPA support. Decode is memory-bound anyway, so SDPA costs
        # little. device_map places the weights directly — calling .to() afterwards would
        # try to materialize a second 72 GB copy.
        self._model = Alpamayo2Super.from_pretrained(
            model_name,
            dtype=self._dtype,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        self._model.eval()

        attn_outer = getattr(self._model.config, "_attn_implementation", "?")
        attn_vlm = getattr(getattr(self._model.vlm, "config", None), "_attn_implementation", "?")
        self.get_logger().info(
            f"Model loaded in {time.time() - started:.1f}s "
            f"(attn: outer={attn_outer} vlm={attn_vlm}, "
            f"cuda_alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB)"
        )

        # get_processor() rebuilds an AutoProcessor from disk on every call, and
        # prepare_model_inputs() calls it each time — fine for a one-shot script, a large
        # per-tick cost here. Build it once and tokenize inline in _run_inference.
        self._processor = helper.get_processor(self._model.tokenizer, self._model.config)

    # --- Core node logic ---

    def destroy_node(self) -> None:
        """Cleanup resources before shutting down."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_inference_payload()
        if payload is None:
            return
        self.get_logger().info("Starting Alpamayo 2 Super inference from streaming data.")
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        # Stash raw JPEG bytes; decode happens on the GPU in _prepare_inference_payload.
        jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg_bytes))

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _prepare_inference_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None
        if len(self._odometry_buffer) < self._num_history_steps * self.skip_num:
            return None

        # Every check runs before the JPEG decode: decoding 24 frames on the GPU is the
        # expensive part of assembling a payload, and there is no point paying for it on a
        # tick that is about to be dropped.
        jpeg_buffers: List[torch.Tensor] = []
        newest_stamps = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            jpeg_buffers.extend([f for _, f in frames])
            newest_stamps.append(Time.from_msg(frames[-1][0]))

        # The cameras are not hardware synchronized and each topic's buffer is read
        # independently, so t0 is the oldest of the per-topic newest frames — the most recent
        # instant every camera actually covers.
        t0_time = min(newest_stamps)

        # Without this the node keeps predicting from whatever is left in the buffers after
        # a rosbag replay ends, publishing fresh-looking trajectories for a scene that
        # stopped advancing.
        if self._max_frame_age_sec > 0.0:
            age = (self.get_clock().now() - t0_time).nanoseconds * 1e-9
            if age > self._max_frame_age_sec:
                self.get_logger().warn(
                    f"Skipping inference: newest camera frame is {age:.1f}s old "
                    f"(max_frame_age_sec={self._max_frame_age_sec}).",
                    throttle_duration_sec=10.0,
                )
                return None

        odom_history = list(self._odometry_buffer)[
            -self._num_history_steps * self.skip_num :: self.skip_num
        ]
        ego_history_xyz, ego_history_rot, _ = conversions.build_ego_history(odom_history)
        reference_speed = float(odom_history[-1].twist.twist.linear.x)

        history_ok, history_message = conversions.check_ego_history(
            ego_history_xyz, ego_history_rot, reference_speed_mps=reference_speed
        )
        if not self._history_checked:
            self._history_checked = True
            (self.get_logger().info if history_ok else self.get_logger().warn)(history_message)
        if not history_ok and self._skip_on_bad_history:
            self.get_logger().warn(
                f"Skipping inference: {history_message}", throttle_duration_sec=5.0
            )
            return None

        decoded = [torchvision.io.decode_jpeg(buf, device="cuda") for buf in jpeg_buffers]
        stacked = torch.stack(decoded)  # [n_cams * n_frames, 3, H, W] uint8 on GPU
        stacked = self._downscale(stacked)

        n_cams = len(self._camera_topics)
        image_frames = torch.stack(
            [
                stacked[i * self._num_frames : (i + 1) * self._num_frames]
                for i in range(n_cams)
            ],
            dim=0,
        )  # [6, 4, 3, H, W]

        return {
            "image_frames": image_frames,
            "camera_indices": self._camera_indices,
            "camera_names": self._camera_names,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_available": torch.tensor(True),
            "input_profile": self._input_profile,
            "_t0_stamp": t0_time.to_msg(),
            "_reference_speed_mps": reference_speed,
        }

    def _downscale(self, images: torch.Tensor) -> torch.Tensor:
        """Shrink decoded frames so the long side is at most ``max_image_long_side``."""
        height, width = images.shape[-2:]
        long_side = max(height, width)
        if self._max_long_side <= 0 or long_side <= self._max_long_side:
            return images
        scale = self._max_long_side / long_side
        target = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
        return (
            torch.nn.functional.interpolate(
                images.float(), size=target, mode="bicubic", align_corners=False
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )

    def _tokenize(self, payload: dict) -> dict:
        """Build model inputs using the cached processor.

        Mirrors ``helper.prepare_model_inputs`` but reuses ``self._processor``. The image
        preprocessing runs on the GPU (``device=``), which matters because the six-camera
        batch is 24 frames.
        """
        messages = helper.create_messages(payload, self._model.config)
        has_assistant_content = messages[-1]["role"] == "assistant" and bool(
            messages[-1]["content"]
        )
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not has_assistant_content,
            add_vision_id=False,
            continue_final_message=has_assistant_content,
        )
        images = payload["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            self._processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
                device=self._device,
            )
        )
        if tokenized_data["input_ids"].shape[0] != 1:
            raise ValueError("Expected exactly one sample per inference")
        return {
            "tokenized_data": tokenized_data,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        }

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()

        model_inputs = helper.to_device(self._tokenize(payload), device=self._device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # The third return value is a zeros placeholder upstream, not a confidence.
            pred_xyz, pred_rot, _logprob, extra = self._model.sample_trajectories_from_data(
                data=model_inputs,
                top_p=self._top_p,
                temperature=self._temperature,
                num_traj_samples=1,
                num_traj_sets=1,
                max_generation_length=self._max_gen_len,
                diffusion_kwargs={"inference_step": self._num_diffusion_steps},
                return_extra=True,
            )

        trajectory = pred_xyz.detach().cpu()[0, 0, 0]  # (64, 3)
        rotation = pred_rot.detach().cpu()[0, 0, 0]  # (64, 3, 3)
        cot_text = conversions.extract_text(extra, "cot")

        traj_ok, traj_message = conversions.check_trajectory(
            trajectory, payload["_reference_speed_mps"]
        )
        if not traj_ok:
            self.get_logger().warn(traj_message)
            if self._drop_bad_trajectory:
                return {
                    "duration_sec": time.time() - start,
                    "num_poses": 0,
                    "cot_chars": len(cot_text) if cot_text else 0,
                    "dropped": traj_message,
                }

        # Stamp with the input t0, not "now": at multi-second latency the difference is
        # the whole point, and downstream consumers need to see it.
        stamp = payload["_t0_stamp"] or self.get_clock().now().to_msg()

        traj_msg = conversions.to_autoware_trajectory(
            trajectory, rotation, stamp=stamp, frame_id=self._frame_id
        )
        self._trajectory_pub.publish(traj_msg)

        self._marker_pub.publish(
            conversions.trajectory_to_markers(
                trajectory,
                stamp=stamp,
                frame_id=self._frame_id,
                line_width=self._marker_line_width,
                cot_text=cot_text,
                z_offset_m=self._marker_z_offset,
                ego_footprint=self._ego_footprint,
            )
        )

        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

            cot_stamped_msg = StringStamped()
            cot_stamped_msg.stamp = stamp
            cot_stamped_msg.data = cot_text
            self._cot_stamped_pub.publish(cot_stamped_msg)

        return {
            "duration_sec": time.time() - start,
            "num_poses": len(traj_msg.points),
            "cot_chars": len(cot_text) if cot_text else 0,
        }

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo 2 Super inference failed: {exc}")
            return
        if not metrics:
            return
        if metrics.get("dropped"):
            self.get_logger().warn(
                f"Alpamayo 2 Super inference completed in {metrics['duration_sec']:.2f}s "
                f"but was not published: {metrics['dropped']}"
            )
            return
        self.get_logger().info(
            f"Alpamayo 2 Super inference completed in {metrics['duration_sec']:.2f}s "
            f"(points={metrics['num_poses']}, cot_chars={metrics['cot_chars']}, "
            f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)."
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = Alpamayo2RosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl-C rclpy may already have torn the context down, in which case both of
        # these raise and bury the real shutdown reason in a traceback.
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
