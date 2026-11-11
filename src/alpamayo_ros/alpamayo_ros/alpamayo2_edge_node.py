#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo 2 Edge and publishes trajectories.

Alpamayo 2 Edge is a 2B reasoning VLA (Cosmos3-Edge backbone + flow-matching action expert).
Compared with the Alpamayo 2 Super node, which this one follows:

* The input is the six cameras ``(0, 1, 2, 3, 5, 6)`` with 22 frames between them: the front
  and telephoto cameras five each, at -2, -1, -0.5, -0.1 and 0 s, the other four three each,
  at -1, -0.1 and 0 s. The ego history is 20 poses, -1.9 to 0 s at 10 Hz.
* A navigation instruction is part of every prompt ("Turn right in 30m", "Continue
  straight"). It is derived from a lanelet2 map and the mission route when both are
  available, and is ``default_nav_text`` otherwise.
* The model is called only through the upstream package's public API
  (``checkpoint.load_model``, ``helper.prepare_model_inputs``,
  ``sample_trajectories_from_data``), with upstream's optional ``torch.compile`` + CUDA-graph
  acceleration (``accel``).
* The output is the same as the Super node's, on the same topics: 64 waypoints at 0.1 s in
  the ego frame at t0, stamped with t0, plus RViz markers and the reasoning text.

This is a visualization/research node, not a closed-loop planner.
"""

from __future__ import annotations

import bisect
import os
import signal
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

import rclpy  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from autoware_internal_debug_msgs.msg import StringStamped  # noqa: E402
from autoware_planning_msgs.msg import LaneletRoute, Trajectory  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rcl_interfaces.msg import ParameterDescriptor  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CompressedImage  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

from alpamayo2_edge import chat_template, helper  # noqa: E402
from alpamayo2_edge.checkpoint import load_model  # noqa: E402
from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS, CAMERA_PIXEL_BOUNDS  # noqa: E402
from alpamayo_ros import conversions, nav_text  # noqa: E402

#: Camera IDs Edge reads, ascending: 0=Front left, 1=Front, 2=Front right, 3=Rear left,
#: 5=Rear right, 6=Front telephoto.
EDGE_CAMERA_IDS = sorted(CAMERA_FRAME_COUNTS)
#: When each camera's frames are taken, in seconds from t0, oldest first: the schedule of the
#: upstream PhysicalAI-AV loader (its README's "Dataset input profile").
FRAME_OFFSETS_SEC: Dict[int, Sequence[float]] = {
    camera: (-2.0, -1.0, -0.5, -0.1, 0.0) if camera in (1, 6) else (-1.0, -0.1, 0.0)
    for camera in EDGE_CAMERA_IDS
}
assert {c: len(o) for c, o in FRAME_OFFSETS_SEC.items()} == CAMERA_FRAME_COUNTS
HISTORY_STEPS = 20
STEP_SEC = 0.1
#: How long camera and odometry messages are kept: the oldest frame is t0 - 2 s.
BUFFER_SEC = 2.5
#: How far a history pose's odometry stamp may be from its scheduled time. Kinematic state
#: normally runs at 50 Hz, so the nearest message is within 10 ms; more means a gap.
ODOM_MATCH_TOLERANCE_SEC = 0.05
#: The checkpoint's pinned revision (the upstream README's).
DEFAULT_REVISION = "f8951d095de3589b3face1dec3f4394c8ecedb38"


def _relative_time_labels(timestamps: torch.Tensor) -> List[str]:
    """``chat_template.relative_time_labels`` without the Python 3.11 ``z`` format flag.

    ``z`` turns a negative zero left by rounding into a positive one; at one decimal place
    that is exactly the string ``-0.0``. The labels, and so the prompt, are identical.
    """
    labels = []
    for delta in (timestamps - timestamps[-1]).tolist():
        text = f"{delta / 1_000_000:.1f}"
        labels.append(f"<t={'0.0' if text == '-0.0' else text}s>")
    return labels


if sys.version_info < (3, 11):
    # Upstream asks for Python 3.12; ROS 2 Humble runs 3.10, and this format flag is the only
    # newer feature the package uses. build_conversation looks it up at call time.
    chat_template.relative_time_labels = _relative_time_labels


def _nearest(stamps_ns: List[int], target_ns: int, lower: int = 0) -> int:
    """Index of the stamp nearest ``target_ns`` among ``stamps_ns[lower:]`` (ascending)."""
    i = bisect.bisect_left(stamps_ns, target_ns, lo=lower)
    if i == len(stamps_ns):
        return i - 1
    if i > lower and target_ns - stamps_ns[i - 1] <= stamps_ns[i] - target_ns:
        return i - 1
    return i


class Alpamayo2EdgeRosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) for Alpamayo 2 Edge."""

    def __init__(self) -> None:
        super().__init__("alpamayo2_edge_node")

        self.declare_parameter("model_name", "nvidia/Alpamayo2-Edge")
        # Ignored for a local model directory. Empty means the repository's main branch.
        self.declare_parameter("model_revision", DEFAULT_REVISION)
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("nav_text_topic", "/alpamayo/nav_text")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        # How often to check for a new input. A tick is dropped while an inference is in
        # flight or when no camera has a new frame since the last one.
        self.declare_parameter("inference_period_sec", 0.05)

        # ROS 2 Jazzy: non-empty defaults so string/int array parameter types are inferred.
        self.declare_parameter("camera_topics", [""])
        # One Alpamayo camera ID per entry of camera_topics. Must be exactly (0, 1, 2, 3, 5, 6).
        self.declare_parameter("camera_indices", EDGE_CAMERA_IDS)
        # A scheduled frame takes the camera's nearest frame within this many seconds.
        self.declare_parameter("frame_match_tolerance_sec", 0.06)

        # "tuned" / "compile": upstream's torch.compile + CUDA graphs for the prefill, the
        # reasoning decode and the expert (needs flash-attn). "off": eager PyTorch. Typed
        # dynamically because a command line or launch file reads `accel:=off` as false.
        self.declare_parameter("accel", "tuned", ParameterDescriptor(dynamic_typing=True))
        # Camera resolution (height, width) to compile the graphs for at startup. [0, 0]
        # compiles them on the first real input instead, which then takes a minute or two.
        self.declare_parameter("warm_up_image_size", [0, 0])
        self.declare_parameter("max_generation_length", 256)
        self.declare_parameter("num_diffusion_steps", 10)  # the checkpoint's
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("seed", 0)

        self.declare_parameter("lanelet2_map_path", "")
        self.declare_parameter("route_topic", "/planning/mission_planning/route")
        # Used until a map and a route give an instruction. The model needs one every call.
        self.declare_parameter("default_nav_text", "Continue straight")

        self.declare_parameter("marker_line_width", 1.2)
        self.declare_parameter("marker_z_offset", 0.4)
        # Ego outline drawn at base_link: (length, width, rear_overhang) in metres. Set length
        # to 0 to draw nothing.
        self.declare_parameter("ego_footprint", [4.77, 1.73, 1.03])
        # Do not infer on frames this much older than now, e.g. after a rosbag replay ends.
        # 0 disables the check.
        self.declare_parameter("max_frame_age_sec", 3.0)
        self.declare_parameter("skip_on_bad_history", True)
        self.declare_parameter("drop_bad_trajectory", True)

        self._device = torch.device("cuda")
        self._frame_id = "base_link"
        self._frame_tolerance_ns = int(
            float(self.get_parameter("frame_match_tolerance_sec").value) * 1e9
        )
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
        self._default_nav_text = str(self.get_parameter("default_nav_text").value).strip()
        if not self._default_nav_text:
            raise ValueError("default_nav_text must not be empty: the model needs an instruction")
        self._top_p = float(self.get_parameter("top_p").value)
        self._temperature = float(self.get_parameter("temperature").value)
        self._max_gen_len = int(self.get_parameter("max_generation_length").value)
        self._num_diffusion_steps = int(self.get_parameter("num_diffusion_steps").value)
        self._history_checked = False
        self._last_t0_ns: Optional[int] = None
        self._nav_text: Optional[str] = None

        queue_size = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, queue_size)
        self._marker_pub = self.create_publisher(MarkerArray, traj_topic + "_markers", queue_size)
        self._cot_pub = self.create_publisher(
            String, self.get_parameter("cot_topic").value, queue_size
        )
        self._cot_stamped_pub = self.create_publisher(
            StringStamped, self.get_parameter("cot_with_stamped_topic").value, queue_size
        )
        self._nav_text_pub = self.create_publisher(
            String, self.get_parameter("nav_text_topic").value, queue_size
        )
        self.get_logger().info(f"Publishing trajectories on {traj_topic} (+ _markers)")

        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        camera_indices = list(
            self.get_parameter("camera_indices").get_parameter_value().integer_array_value
        )
        self._validate_camera_config(camera_topics, camera_indices)
        self._camera_topics = camera_topics
        self._camera_ids = camera_indices
        names = chat_template.CAMERA_DISPLAY_NAMES
        for topic, camera in zip(camera_topics, camera_indices):
            self.get_logger().info(f"Camera {camera} ({names.get(camera, '?')}) <- {topic}")

        # Per topic: (stamp ns, JPEG bytes), oldest first. Decoding waits until a frame is
        # picked for an input, which is 22 of the ~150 frames the cameras send in 2.5 s.
        self._camera_buffers: Dict[str, List[tuple]] = {topic: [] for topic in camera_topics}
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        for topic in camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._image_callback(t, msg),
                camera_qos,
            )

        self._odometry_buffer: List[tuple] = []  # (stamp ns, Odometry), oldest first
        odom_topic = self.get_parameter("odometry_topic").value
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        self._lanelet_map: Optional[dict] = None
        self._route_lanelet_ids: List[int] = []
        map_path = str(self.get_parameter("lanelet2_map_path").value)
        if map_path:
            if not nav_text.HAS_LANELET2:
                raise ValueError(
                    "lanelet2_map_path needs the lanelet2 Python bindings; source an Autoware "
                    "workspace that provides autoware_lanelet2_extension_python."
                )
            self._lanelet_map = nav_text.load_lanelet_map(map_path)
            self.get_logger().info(
                f"Loaded {len(self._lanelet_map)} drivable lanelets from {map_path}"
            )
            # The route is latched, so a late subscriber still needs transient-local.
            route_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            route_topic = self.get_parameter("route_topic").value
            self.create_subscription(LaneletRoute, route_topic, self._route_callback, route_qos)
        else:
            self.get_logger().info(
                f"No lanelet2_map_path: every instruction is {self._default_nav_text!r}"
            )

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None
        self._load_model()
        seed = int(self.get_parameter("seed").value)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._warm_up()

        period = float(self.get_parameter("inference_period_sec").value)
        self._timer = self.create_timer(period, self._timer_callback)
        self.get_logger().info("Alpamayo 2 Edge model loaded and ready.")

    # --- Setup ---

    def _validate_camera_config(self, topics: List[str], indices: List[int]) -> None:
        if not topics or topics == [""]:
            raise ValueError(
                "camera_topics must list six image topics, one per Alpamayo camera ID "
                f"{EDGE_CAMERA_IDS}."
            )
        if len(indices) != len(topics):
            raise ValueError(
                f"camera_indices length ({len(indices)}) must match "
                f"camera_topics length ({len(topics)})."
            )
        if sorted(indices) != EDGE_CAMERA_IDS:
            raise ValueError(
                f"Alpamayo 2 Edge needs exactly the cameras {EDGE_CAMERA_IDS}, got {indices}."
            )

    def _load_model(self) -> None:
        model_name = str(self.get_parameter("model_name").value)
        revision = str(self.get_parameter("model_revision").value) or None
        self.get_logger().info(f"Loading {model_name} (revision {revision or 'main'}) ...")
        started = time.time()
        self._model, _info = load_model(model_name, revision=revision, device="cuda")
        self._model.eval()
        # prepare_model_inputs builds the processor from the checkpoint files on every call,
        # 3-5 s each time. Build it once and hand the same one back.
        processor = helper.get_processor(self._model.tokenizer, self._model.config)
        helper.get_processor = lambda tokenizer, model_config: processor
        self._resize_factor = (
            processor.image_processor.patch_size * processor.image_processor.merge_size
        )
        self.get_logger().info(
            f"Model loaded in {time.time() - started:.1f}s "
            f"({torch.cuda.memory_allocated() / 2**30:.1f} GiB)"
        )

        self._accel = None
        value = self.get_parameter("accel").value
        mode = "off" if value is False else str(value)
        if mode == "off":
            return
        if mode not in ("tuned", "compile"):
            raise ValueError(f"accel must be tuned, compile or off, got {value!r}")
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration, resolve_kernels

        if resolve_kernels("auto", self._device)["decode"] is None:
            self.get_logger().warn(
                f"accel={mode} needs flash-attn (FlashAttention 2), which is not installed; "
                "running eager. See the README to install it."
            )
            return
        self._accel = enable_acceleration(self._model, AccelConfig(mode=mode))
        self.get_logger().info(f"Acceleration: {self._accel.describe()['parts']} ({mode})")

    def _warm_up(self) -> None:
        """Compile and capture the graphs for ``warm_up_image_size`` before going live."""
        size = list(
            self.get_parameter("warm_up_image_size").get_parameter_value().integer_array_value
        )
        if self._accel is None or len(size) != 2 or min(size) <= 0:
            return
        self.get_logger().info(f"Building the graphs for {size[0]}x{size[1]} cameras ...")
        started = time.time()
        frames = torch.zeros((22, 3, *size), dtype=torch.uint8, device=self._device)
        cameras, stamps_ns = [], []
        for camera in EDGE_CAMERA_IDS:
            for offset in FRAME_OFFSETS_SEC[camera]:
                cameras.append(camera)
                stamps_ns.append(int(round(offset * 1e9)))
        history_xyz = torch.zeros((1, 1, HISTORY_STEPS, 3))
        history_rot = torch.eye(3).expand(1, 1, HISTORY_STEPS, 3, 3).contiguous()
        sample = self._sample(frames, cameras, stamps_ns, history_xyz, history_rot)
        inputs = helper.to_device(
            helper.prepare_model_inputs(sample, self._model.config, self._model.tokenizer),
            device=self._device,
        )
        self._accel.warmup(inputs, **self._sample_kwargs())
        self.get_logger().info(f"Graphs built in {time.time() - started:.0f}s")

    # --- Callbacks ---

    def destroy_node(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _route_callback(self, msg: LaneletRoute) -> None:
        self._route_lanelet_ids = [seg.preferred_primitive.id for seg in msg.segments]
        self.get_logger().info(f"Received route with {len(self._route_lanelet_ids)} segments.")

    @staticmethod
    def _append(buffer: List[tuple], stamp: int, item) -> None:
        """Add to a stamp-ordered buffer and drop what is older than BUFFER_SEC."""
        if buffer and stamp <= buffer[-1][0]:
            if stamp == buffer[-1][0]:
                return
            buffer.clear()  # time went back, e.g. a replayed bag started over
        buffer.append((stamp, item))
        while buffer[0][0] < stamp - BUFFER_SEC * 1e9:
            buffer.pop(0)

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        stamp = Time.from_msg(msg.header.stamp).nanoseconds
        jpeg = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._append(self._camera_buffers[topic], stamp, jpeg)

    def _odometry_callback(self, msg: Odometry) -> None:
        self._append(self._odometry_buffer, Time.from_msg(msg.header.stamp).nanoseconds, msg)

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_inference_payload()
        if payload is None:
            return
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    # --- Input ---

    def _prepare_inference_payload(self) -> Optional[dict]:
        if not all(self._camera_buffers.values()) or not self._odometry_buffer:
            return None
        # The cameras are not hardware synchronized, so t0 is the oldest of the per-topic
        # newest frames: the most recent instant every camera covers.
        t0_ns = min(buf[-1][0] for buf in self._camera_buffers.values())
        if t0_ns == self._last_t0_ns:
            return None  # this input has been run already

        if self._max_frame_age_sec > 0.0:
            age = (self.get_clock().now().nanoseconds - t0_ns) * 1e-9
            if age > self._max_frame_age_sec:
                self.get_logger().warn(
                    f"Skipping inference: newest camera frame is {age:.1f}s old "
                    f"(max_frame_age_sec={self._max_frame_age_sec}).",
                    throttle_duration_sec=10.0,
                )
                return None

        frames = self._select_frames(t0_ns)
        if frames is None:
            return None
        odom_history = self._select_history(t0_ns)
        if odom_history is None:
            return None
        ego_history_xyz, ego_history_rot, positions_map = conversions.build_ego_history(
            odom_history
        )
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

        # Recorded only now: a t0 skipped above, e.g. because its odometry had not arrived
        # yet, is tried again on the next tick.
        self._last_t0_ns = t0_ns
        jpegs, cameras, stamps_ns = frames
        return {
            "jpegs": jpegs,
            "cameras": cameras,
            "stamps_ns": stamps_ns,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "nav_text": self._resolve_nav_text(positions_map[-1]),
            "_t0_stamp": Time(nanoseconds=t0_ns).to_msg(),
            "_reference_speed_mps": reference_speed,
        }

    def _select_frames(self, t0_ns: int) -> Optional[tuple]:
        """Each camera's frames nearest its scheduled times, or None if one is missing.

        The returned times are the schedule's, not the frames' stamps: the prompt labels each
        frame with its time relative to the newest one, to 0.1 s, and unsynchronized cameras
        would otherwise give labels such as -0.1 s for a camera's newest frame.
        """
        jpegs, cameras, stamps_ns = [], [], []
        for topic, camera in zip(self._camera_topics, self._camera_ids):
            buffered = self._camera_buffers[topic]
            buffer_stamps = [stamp for stamp, _ in buffered]
            lower = 0
            for offset in FRAME_OFFSETS_SEC[camera]:
                target = t0_ns + int(round(offset * 1e9))
                # Searching only past the previous pick keeps one frame from filling two slots.
                i = _nearest(buffer_stamps, target, lower) if lower < len(buffered) else -1
                if i < 0 or abs(buffer_stamps[i] - target) > self._frame_tolerance_ns:
                    self.get_logger().warn(
                        f"Skipping inference: camera {camera} ({topic}) has no frame within "
                        f"{self._frame_tolerance_ns * 1e-9:.2f}s of t0{offset:+.1f}s.",
                        throttle_duration_sec=5.0,
                    )
                    return None
                jpegs.append(buffered[i][1])
                cameras.append(camera)
                stamps_ns.append(target)
                lower = i + 1
        return jpegs, cameras, stamps_ns

    def _select_history(self, t0_ns: int) -> Optional[List[Odometry]]:
        """The odometry messages nearest each of the 20 history times, oldest first."""
        stamps = [stamp for stamp, _ in self._odometry_buffer]
        picked = []
        for k in range(HISTORY_STEPS - 1, -1, -1):
            target = t0_ns - int(round(k * STEP_SEC * 1e9))
            i = _nearest(stamps, target)
            if abs(stamps[i] - target) > ODOM_MATCH_TOLERANCE_SEC * 1e9:
                self.get_logger().warn(
                    f"Skipping inference: no odometry within {ODOM_MATCH_TOLERANCE_SEC}s of "
                    f"t0-{k * STEP_SEC:.1f}s.",
                    throttle_duration_sec=5.0,
                )
                return None
            picked.append(self._odometry_buffer[i][1])
        return picked

    def _resolve_nav_text(self, ego_pos_map) -> str:
        instruction = None
        if self._lanelet_map is not None:
            instruction = nav_text.compute_nav_text(
                self._lanelet_map, self._route_lanelet_ids, ego_pos_map
            )
        instruction = instruction or self._default_nav_text
        # Logged when the maneuver changes, not each time its distance does.
        maneuver = instruction.split(" in ")[0]
        if maneuver != self._nav_text:
            self._nav_text = maneuver
            self.get_logger().info(f"Navigation instruction: {instruction!r}")
        return instruction

    def _sample(self, frames, cameras, stamps_ns, history_xyz, history_rot, nav=None) -> dict:
        """The raw sample ``prepare_model_inputs`` takes (upstream's dataset input profile)."""
        frames = self._resize(frames)
        # Upstream takes the frames on the CPU; through pinned memory the copy takes ~1 ms.
        host = torch.empty(frames.shape, dtype=torch.uint8, pin_memory=frames.is_cuda)
        host.copy_(frames)
        return {
            "image_frames": host.unsqueeze(1),  # [22, 1, 3, H, W]
            "camera_indices": torch.tensor(cameras, dtype=torch.int64),
            "absolute_timestamps": torch.tensor(stamps_ns, dtype=torch.int64) // 1000,
            "ego_history_xyz": history_xyz,
            "ego_history_rot": history_rot,
            "nav_text": [nav or self._default_nav_text],
            "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        }

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        """All frames at the size upstream gives the front and telephoto cameras, on the GPU.

        Upstream resizes each frame on the CPU to its camera's pixel budget (front and
        telephoto ~192k px, the others ~96k). From the cameras' own resolution that took 0.66 s
        a call for 2880x1860 frames. Resized here first, the front and telephoto frames are
        left as they are, and only the others are shrunk further on the CPU.
        """
        height, width = frames.shape[-2:]
        lower, upper = CAMERA_PIXEL_BOUNDS[1]
        size = helper.smart_resize(
            height, width, factor=self._resize_factor, min_pixels=lower, max_pixels=upper
        )
        if tuple(size) == (height, width):
            return frames
        return (
            torch.nn.functional.interpolate(
                frames.float(), size=size, mode="bicubic", antialias=True, align_corners=False
            )
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
        )

    def _sample_kwargs(self) -> dict:
        return dict(
            num_traj_sets=1,
            num_traj_samples=1,
            top_p=self._top_p,
            temperature=self._temperature,
            max_generation_length=self._max_gen_len,
            diffusion_kwargs={"inference_step": self._num_diffusion_steps},
            return_extra=True,
        )

    # --- Inference ---

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()
        decoded = torchvision.io.decode_jpeg(payload["jpegs"], device=self._device)
        sizes = {tuple(image.shape[-2:]) for image in decoded}
        if len(sizes) != 1:
            raise ValueError(f"All cameras must have the same resolution, got {sorted(sizes)}")
        sample = self._sample(
            torch.stack(decoded),
            payload["cameras"],
            payload["stamps_ns"],
            payload["ego_history_xyz"],
            payload["ego_history_rot"],
            payload["nav_text"],
        )
        inputs = helper.to_device(
            helper.prepare_model_inputs(sample, self._model.config, self._model.tokenizer),
            device=self._device,
        )
        prepared = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # The third return value is a zeros placeholder upstream, not a confidence.
            pred_xyz, pred_rot, _logprob, extra = self._model.sample_trajectories_from_data(
                inputs, **self._sample_kwargs()
            )
        trajectory = pred_xyz.float().cpu()[0, 0, 0]  # (64, 3)
        rotation = pred_rot.float().cpu()[0, 0, 0]  # (64, 3, 3)
        inferred = time.time()
        cot_text = conversions.extract_text(extra, "cot")
        metrics = {
            "prepare_sec": prepared - start,
            "model_sec": inferred - prepared,
            "cot": cot_text or "",
        }

        traj_ok, traj_message = conversions.check_trajectory(
            trajectory, payload["_reference_speed_mps"]
        )
        if not traj_ok:
            self.get_logger().warn(traj_message)
            if self._drop_bad_trajectory:
                return {**metrics, "dropped": traj_message}

        # Stamped with the input t0, not "now", so consumers can measure the latency.
        stamp = payload["_t0_stamp"]
        self._trajectory_pub.publish(
            conversions.to_autoware_trajectory(
                trajectory, rotation, stamp=stamp, frame_id=self._frame_id
            )
        )
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
        nav_msg = String()
        nav_msg.data = payload["nav_text"]
        self._nav_text_pub.publish(nav_msg)
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)
            cot_stamped_msg = StringStamped()
            cot_stamped_msg.stamp = stamp
            cot_stamped_msg.data = cot_text
            self._cot_stamped_pub.publish(cot_stamped_msg)
        return metrics

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo 2 Edge inference failed: {exc!r}")
            return
        timing = (
            f"prepare {metrics['prepare_sec'] * 1e3:.0f} ms, "
            f"model {metrics['model_sec'] * 1e3:.0f} ms"
        )
        if metrics.get("dropped"):
            self.get_logger().warn(f"Not published ({timing}): {metrics['dropped']}")
            return
        self.get_logger().info(
            f"{timing}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB: {metrics['cot']}",
            throttle_duration_sec=1.0,
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = Alpamayo2EdgeRosNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # On SIGTERM rclpy shuts the context down under the executor, which can then fail
        # to build its wait set instead of raising ExternalShutdownException.
        if rclpy.ok():
            raise
    finally:
        # Ctrl-C under ros2 launch arrives twice, from the terminal and forwarded by launch.
        # The second would interrupt the teardown below with a KeyboardInterrupt traceback.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
