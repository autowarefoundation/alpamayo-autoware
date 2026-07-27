#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo and publishes Autoware trajectories."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

# cv2 is no longer needed for the image preproc path (decode runs on
# GPU via torchvision.io.decode_jpeg) but kept here as a soft dep — some
# downstream lanelet helpers still import it. If you remove it, also
# remove the lanelet path that imports cv2 transitively.
import numpy as np
import rclpy
import torch
from autoware_internal_debug_msgs.msg import StringStamped
from autoware_planning_msgs.msg import LaneletRoute, Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

try:
    from alpamayo_ros.flashdrive_sidecar import wire
except ImportError:  # pragma: no cover - direct script execution
    from flashdrive_sidecar import wire  # type: ignore

try:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    _HAS_LANELET2 = True
except ImportError:
    _HAS_LANELET2 = False


class FlashDriveBackend:
    """Thin HTTP client for the optional Python 3.12 FlashDrive sidecar.

    Used only when ``use_flashdrive`` is true. The baseline in-process path is
    unchanged. Window 0 (streaming prefill) returns ``None`` (no trajectory).
    """

    def __init__(self, node: "AlpamayoRosNode", url: str) -> None:
        self._node = node
        self._url = url.rstrip("/")
        self._num_frames = node._num_frames
        self._timeout = float(os.environ.get("FLASHDRIVE_TIMEOUT_SEC", "600"))
        self._http = None
        log = node.get_logger()
        ready_wait = float(os.environ.get("FLASHDRIVE_READY_WAIT_SEC", "1800"))
        log.info(f"FlashDrive sidecar: waiting for {self._url}/health (<= {ready_wait:.0f}s)...")
        self._wait_ready(ready_wait)
        try:
            self._post("/reset", {}, {})
            log.info("FlashDrive sidecar: stream reset OK.")
        except Exception as exc:  # noqa: BLE001
            log.warn(f"FlashDrive sidecar reset failed (continuing): {exc}")

    def _wait_ready(self, timeout_sec: float) -> None:
        import urllib.request

        deadline = time.time() + timeout_sec
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self._url + "/health", timeout=10) as resp:
                    if resp.status == 200:
                        self._node.get_logger().info("FlashDrive sidecar: healthy.")
                        return
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            time.sleep(5.0)
        raise RuntimeError(
            f"FlashDrive sidecar not healthy within {timeout_sec:.0f}s "
            f"(url={self._url}, last error={last_err})"
        )

    def _post(self, path: str, meta: dict, arrays: dict):
        import http.client
        from urllib.parse import urlparse

        body = wire.encode(meta, arrays)
        parsed = urlparse(self._url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        try:
            if self._http is None:
                self._http = http.client.HTTPConnection(host, port, timeout=self._timeout)
            self._http.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(body)),
                    "Connection": "keep-alive",
                },
            )
            resp = self._http.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"FlashDrive HTTP {resp.status}")
            return wire.decode(raw)
        except Exception:
            try:
                if self._http is not None:
                    self._http.close()
            except Exception:  # noqa: BLE001
                pass
            self._http = http.client.HTTPConnection(host, port, timeout=self._timeout)
            self._http.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(body)),
                    "Connection": "keep-alive",
                },
            )
            resp = self._http.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"FlashDrive HTTP {resp.status}")
            return wire.decode(raw)
        finally:
            try:
                wire.release_shm(meta, unlink=True)
            except Exception:  # noqa: BLE001
                pass

    def infer(self, payload: dict):
        """Return ``(traj[T,3], rot[T,3,3], cot_text|None)`` or ``None`` on prefill."""
        frames = payload["image_frames"]
        meta = {
            "num_frames_per_camera": int(self._num_frames),
            "nav_text": payload.get("nav_text"),
        }
        img = frames.detach()
        if img.device.type != "cpu":
            img = img.to("cpu", non_blocking=False)
        arrays = {
            "image_frames": np.ascontiguousarray(img.numpy(), dtype=np.uint8),
            "camera_indices": np.ascontiguousarray(
                payload["camera_indices"].detach().cpu().numpy(), dtype=np.int64
            ),
            "ego_history_xyz": np.ascontiguousarray(
                payload["ego_history_xyz"].detach().cpu().numpy(), dtype=np.float32
            ),
            "ego_history_rot": np.ascontiguousarray(
                payload["ego_history_rot"].detach().cpu().numpy(), dtype=np.float32
            ),
        }
        t0 = time.time()
        resp_meta, resp_arrays = self._post("/predict", meta, arrays)
        roundtrip_sec = time.time() - t0

        status = resp_meta.get("status")
        if status == "prefill":
            return None
        if status != "ok":
            raise RuntimeError(f"FlashDrive sidecar error: {resp_meta.get('message')}")

        tok = float(resp_meta.get("tokenize_sec", resp_meta.get("prep_sec", 0.0)))
        model = float(resp_meta.get("model_sec", 0.0))
        post = float(resp_meta.get("post_sec", 0.0))
        server = float(resp_meta.get("server_sec", 0.0))
        self._node.get_logger().info(
            f"[PROFILER][flashdrive] roundtrip={roundtrip_sec * 1000:.0f}ms "
            f"server={server * 1000:.0f}ms tokenize={tok * 1000:.0f}ms "
            f"model={model * 1000:.0f}ms post={post * 1000:.0f}ms "
            f"wire≈{max(0.0, roundtrip_sec - server) * 1000:.0f}ms"
        )
        traj_batch = np.ascontiguousarray(resp_arrays["traj_batch"], dtype=np.float32)
        rot_batch = np.ascontiguousarray(resp_arrays["rot_batch"], dtype=np.float32)
        return traj_batch[0], rot_batch[0], (resp_meta.get("cot") or None)


class AlpamayoRosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) to run Alpamayo inference."""

    def __init__(self) -> None:
        super().__init__("alpamayo_node")

        self.model_name: str = "nvidia/Alpamayo-1.5-10B"
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("nav_text_topic", "/alpamayo/nav_text")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("route_topic", "/planning/mission_planning/route")
        self.declare_parameter("inference_period_sec", 0.1)

        # ROS2 Jazzy: Use non-empty default for string array parameters to properly infer type
        self.declare_parameter("camera_topics", [""])
        # Camera indices matching CAMERA_DISPLAY_NAMES in helper.py:
        # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
        # Each index corresponds to the camera topic at the same position in camera_topics.
        self.declare_parameter("camera_indices", [0])

        # Lanelet2 map file for navigation instruction generation
        self.declare_parameter("lanelet2_map_path", "")

        # 5-step Euler keeps trajectory ADE within ~1% of 10-step but cuts
        # ~94 ms / inference (3 saved step_fn calls); adaptive_flow caches
        # the middle steps on top of that.
        self.declare_parameter("num_diffusion_steps", 5)
        # Greedy decode yields a single deterministic trajectory with ~0%
        # deviation vs nucleus on a fixed bag. Set False to use the nucleus
        # preset (top_p=0.98 / temperature=0.6) below.
        self.declare_parameter("use_greedy_decode", True)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("max_generation_length", 64)
        # Optional TRT FP16 expert engine. Empty string → use the native
        # PyTorch denoiser step. Set to an ONNX file produced by
        # ``scripts/build_trt_expert_engine.py`` to swap in a TrtExpertEngine
        # runtime for the 5-step diffusion inner loop.
        self.declare_parameter("expert_onnx_path", "")
        # Optional FlashDrive sidecar (Python 3.12). Default OFF — baseline/TRT
        # in-process path below is unchanged unless explicitly enabled.
        self.declare_parameter("use_flashdrive", False)
        self.declare_parameter("flashdrive_url", "http://127.0.0.1:8710")

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 10.0  # Model expects 10Hz (time_step=0.1s)
        self.skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)

        self._num_history_steps = 16
        self._num_frames = 4
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

        nav_text_topic = self.get_parameter("nav_text_topic").value
        self._nav_text_pub = self.create_publisher(String, nav_text_topic, queue_size)
        self.get_logger().info(f"Publishing navigation text on {nav_text_topic}")

        marker_topic = traj_topic + "_markers"
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, queue_size)
        self.get_logger().info(f"Publishing trajectory markers on {marker_topic}")

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        self._frame_id = "base_link"

        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        if not camera_topics:
            raise ValueError("camera_topics parameter must list at least one image topic.")
        self._camera_topics = camera_topics

        camera_indices = list(
            self.get_parameter("camera_indices").get_parameter_value().integer_array_value
        )
        if len(camera_indices) != len(camera_topics):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"camera_topics length ({len(camera_topics)})."
            )
        self._camera_indices = torch.tensor(camera_indices, dtype=torch.int64)
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

        self._odometry_buffer: deque[Odometry] = deque(
            maxlen=self._num_history_steps * self.skip_num + 10
        )
        odom_topic = self.get_parameter("odometry_topic").value

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        # --- Lanelet2 map & route for navigation instructions ---
        self._lanelet_map = None  # dict[int, dict] mapping lanelet_id -> info
        self._route_lanelet_ids: List[int] = []
        self._nav_text: Optional[str] = None

        lanelet2_map_path = self.get_parameter("lanelet2_map_path").value
        if lanelet2_map_path:
            self._load_lanelet2_map(lanelet2_map_path)

        route_topic = self.get_parameter("route_topic").value
        route_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaneletRoute, route_topic, self._route_callback, route_qos)
        self.get_logger().info(f"Subscribed to route topic: {route_topic}")

        self._auto_timer = self.create_timer(inference_period, self._timer_callback)

        # Optional FlashDrive sidecar: skip in-process model load entirely.
        self._flashdrive: Optional[FlashDriveBackend] = None
        if bool(self.get_parameter("use_flashdrive").value):
            url = str(self.get_parameter("flashdrive_url").value)
            self.get_logger().info(f"Backend: FLASHDRIVE sidecar @ {url}")
            self._flashdrive = FlashDriveBackend(self, url)
            self.get_logger().info("Alpamayo node ready (FlashDrive backend).")
            return

        self.get_logger().info(
            f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
        )
        self._model = Alpamayo1_5.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()

        # Optional TRT FP16 expert engine — swap in if expert_onnx_path set
        # and the file exists. Falls back silently to native PyTorch otherwise.
        expert_onnx = str(self.get_parameter("expert_onnx_path").value or "")
        if expert_onnx and Path(expert_onnx).exists():
            from alpamayo1_5.trt.expert_runtime import TrtExpertEngine

            engine = TrtExpertEngine(
                onnx_model_path=expert_onnx,
                engine_cache_dir=str(Path(expert_onnx).parent / "engine_cache"),
                enable_int8=False,
                enable_fp16=True,
            )
            self._model.set_expert_step_runner(engine)
            self.get_logger().info(f"TRT Expert loaded: {expert_onnx}")
        else:
            self.get_logger().info("TRT Expert: off (native PyTorch denoiser).")

        # Apply diffusion-step override (R1's 5-step preset is the
        # optimized default; native model config is 10).
        num_steps = int(self.get_parameter("num_diffusion_steps").value)
        self._model.diffusion.num_inference_steps = num_steps
        self.get_logger().info(f"Diffusion inference steps: {num_steps}")

        # Cache greedy/sampling config for the per-call generation.
        self._use_greedy = bool(self.get_parameter("use_greedy_decode").value)
        if self._use_greedy:
            self._top_p, self._temperature = 1.0, 1.0
            self.get_logger().info("Generation: GREEDY (top_p=1.0, temperature=1.0)")
        else:
            self._top_p = float(self.get_parameter("top_p").value)
            self._temperature = float(self.get_parameter("temperature").value)
            self.get_logger().info(
                f"Generation: NUCLEUS (top_p={self._top_p}, temperature={self._temperature})"
            )
        self._max_gen_len = int(self.get_parameter("max_generation_length").value)

        self._processor = helper.get_processor(self._model.tokenizer)

        # Set random seed once during initialization
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.get_logger().info("Alpamayo model loaded and ready.")

    # --- Lanelet2 map loading ---

    def _load_lanelet2_map(self, map_path: str) -> None:
        """Load lanelet2 map and extract lanelet centerlines and turn directions."""
        if not _HAS_LANELET2:
            self.get_logger().warn(
                "lanelet2 / autoware_lanelet2_extension_python not available. "
                "Navigation instructions disabled."
            )
            return

        self.get_logger().info(f"Loading lanelet2 map from {map_path}...")
        projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        ll2_map = lanelet2.io.load(map_path, projection)

        lanelet_info: dict[int, dict] = {}
        for ll in ll2_map.laneletLayer:
            subtype = ll.attributes.get("subtype", "") if hasattr(ll.attributes, "get") else ""
            if not subtype:
                subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            if subtype not in ("road", "highway", "road_shoulder", "bicycle_lane"):
                continue

            centerline = np.array([(p.x, p.y, p.z) for p in ll.centerline])
            turn_dir_str = ""
            if "turn_direction" in ll.attributes:
                turn_dir_str = ll.attributes["turn_direction"]

            lanelet_info[ll.id] = {
                "centerline": centerline,
                "turn_direction": turn_dir_str,
                "center": np.mean(centerline[:, :2], axis=0),
            }

        self._lanelet_map = lanelet_info
        self.get_logger().info(f"Loaded {len(lanelet_info)} road lanelets from map.")

    def _route_callback(self, msg: LaneletRoute) -> None:
        """Store ordered lanelet IDs from the route."""
        self._route_lanelet_ids = [seg.preferred_primitive.id for seg in msg.segments]
        self.get_logger().info(
            f"Received route with {len(self._route_lanelet_ids)} segments."
        )

    def _compute_nav_text(self, ego_pos_map: np.ndarray) -> Optional[str]:
        """Compute navigation instruction from ego position, route and lanelet map.

        Finds the closest lanelet on the route using the first point of each
        lanelet's centerline. If the ego is on a turning lanelet, returns the
        turn direction immediately. Otherwise looks ahead for the next turn
        and reports the distance.
        """
        if self._lanelet_map is None or not self._route_lanelet_ids:
            return None

        ego_xy = ego_pos_map[:2]

        # Find which route lanelet the ego is closest to (using first centerline point)
        best_idx = 0
        best_dist = float("inf")
        for i, ll_id in enumerate(self._route_lanelet_ids):
            info = self._lanelet_map.get(ll_id)
            if info is None:
                continue
            first_pt = info["centerline"][0, :2]
            dist = np.linalg.norm(first_pt - ego_xy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Check if ego is currently on a turning lanelet
        current_info = self._lanelet_map.get(self._route_lanelet_ids[best_idx])
        if current_info is not None and current_info["turn_direction"] in ("left", "right"):
            return f"Turn {current_info['turn_direction']}"

        # Look ahead for next turn, accumulating distance via first centerline points
        cumulative_dist = 0.0
        prev_pt = ego_xy

        for i in range(best_idx + 1, len(self._route_lanelet_ids)):
            ll_id = self._route_lanelet_ids[i]
            info = self._lanelet_map.get(ll_id)
            if info is None:
                continue

            first_pt = info["centerline"][0, :2]
            cumulative_dist += np.linalg.norm(first_pt - prev_pt)
            prev_pt = first_pt

            turn_dir = info["turn_direction"]
            if turn_dir in ("left", "right"):
                dist_m = int(round(cumulative_dist))
                return f"Turn {turn_dir} in {dist_m}m"

        return "Continue straight"

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
        nav_text = payload.get("nav_text")
        self.get_logger().info(
            f"Starting Alpamayo inference from streaming data. nav_text={nav_text}"
        )
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        # Stash raw JPEG bytes as torch.uint8; GPU decode + resize happens
        # later in _prepare_inference_payload via torchvision.io.decode_jpeg
        # + F.interpolate. Saves ~150 ms/frame vs cv2.imdecode + CPU copy.
        jpeg_bytes = torch.frombuffer(bytearray(msg.data), dtype=torch.uint8)
        self._camera_buffers[topic].append((msg.header.stamp, jpeg_bytes))

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _prepare_inference_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None

        # GPU-decode path: collect raw JPEG bytes, batch-decode on GPU,
        # resize to 560x1008 on GPU. Saves ~150 ms / frame vs CPU cv2 path.
        import torchvision
        jpeg_buffers: list[torch.Tensor] = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            jpeg_buffers.extend([f for _, f in frames])

        decoded = [torchvision.io.decode_jpeg(buf, device="cuda") for buf in jpeg_buffers]
        stacked = torch.stack(decoded)  # [N_total, 3, H, W] uint8 on GPU
        if stacked.shape[-2:] != (560, 1008):
            stacked = torch.nn.functional.interpolate(
                stacked.float(), size=(560, 1008), mode="bicubic", align_corners=False,
            ).clamp(0, 255).to(torch.uint8)

        n_cams = len(self._camera_topics)
        camera_tensors = [
            stacked[i * self._num_frames : (i + 1) * self._num_frames]
            for i in range(n_cams)
        ]
        image_frames = torch.stack(camera_tensors, dim=0)  # [n_cams, n_frames, 3, H, W]

        if len(self._odometry_buffer) < self._num_history_steps * self.skip_num:
            return None
        odom_history = list(self._odometry_buffer)[
            -self._num_history_steps * self.skip_num :: self.skip_num
        ]

        # Build history tensors
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

        # Compute navigation text from current ego position (map frame)
        ego_pos_map = positions_np[-1]
        nav_text = self._compute_nav_text(ego_pos_map)
        if nav_text and nav_text != self._nav_text:
            self._nav_text = nav_text
            self.get_logger().info(f"Navigation instruction: {nav_text}")

        return {
            "image_frames": image_frames,
            "camera_indices": self._camera_indices,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "nav_text": self._nav_text,
        }

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()
        if self._flashdrive is not None:
            return self._run_inference_flashdrive(payload, start)

        frames = payload["image_frames"]  # already on GPU (uint8) from GPU preproc
        messages = helper.create_message(
            frames.flatten(0, 1),
            camera_indices=payload["camera_indices"],
            num_frames_per_camera=self._num_frames,
            nav_text=payload.get("nav_text"),
        )
        # device="cuda" makes Qwen2VLImageProcessorFast run the normalize +
        # patchify pass on GPU (~0.7 ms/frame vs ~58 ms/frame on the CPU
        # fast-path for 4-cam × 4-frame batches).
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            device="cuda",
        )
        # apply_chat_template(device=cuda) only puts pixel_values on GPU;
        # text ids / attention_mask / image_grid_thw still come back on CPU.
        # Move them to GPU once here so the model forward doesn't hit per-
        # tensor sync waits.
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

        generation_kwargs = {
            "top_p": self._top_p,
            "temperature": self._temperature,
            "num_traj_samples": 1,
            "num_traj_sets": 1,
            "max_generation_length": self._max_gen_len,
            "return_extra": True,
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if payload.get("nav_text"):
                pred_xyz, pred_rot, extra = (
                    self._model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                        data=model_inputs,
                        **generation_kwargs,
                    )
                )
            else:
                pred_xyz, pred_rot, extra = (
                    self._model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        **generation_kwargs,
                    )
                )

        pred_xyz_cpu = pred_xyz.detach().cpu()
        pred_rot_cpu = pred_rot.detach().cpu()
        trajectory = pred_xyz_cpu[0, 0, 0]
        rotation = pred_rot_cpu[0, 0, 0]
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)

        marker_array = self._trajectory_to_markers(trajectory)
        self._marker_pub.publish(marker_array)

        nav_text = payload.get("nav_text")
        if nav_text:
            nav_msg = String()
            nav_msg.data = nav_text
            self._nav_text_pub.publish(nav_msg)

        cot_text = self._extract_text(extra, "cot")
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

            cot_stamped_msg = StringStamped()
            cot_stamped_msg.stamp = traj_msg.header.stamp
            cot_stamped_msg.data = cot_text
            self._cot_stamped_pub.publish(cot_stamped_msg)

        duration = time.time() - start
        return {"duration_sec": duration, "num_poses": len(traj_msg.points)}

    def _run_inference_flashdrive(self, payload: dict, start: float) -> dict:
        """Publish path for the optional FlashDrive sidecar (default-off)."""
        assert self._flashdrive is not None
        out = self._flashdrive.infer(payload)
        if out is None:
            self._marker_pub.publish(MarkerArray())
            return {"duration_sec": time.time() - start, "num_poses": 0}

        traj_np, rot_np, cot_text = out
        trajectory = torch.from_numpy(traj_np)
        rotation = torch.from_numpy(rot_np)
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)

        marker_array = self._trajectory_to_markers(trajectory)
        self._marker_pub.publish(marker_array)

        nav_text = payload.get("nav_text")
        if nav_text:
            nav_msg = String()
            nav_msg.data = nav_text
            self._nav_text_pub.publish(nav_msg)

        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

            cot_stamped_msg = StringStamped()
            cot_stamped_msg.stamp = traj_msg.header.stamp
            cot_stamped_msg.data = cot_text
            self._cot_stamped_pub.publish(cot_stamped_msg)

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
