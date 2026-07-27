# SPDX-License-Identifier: Apache-2.0
"""FlashDrive sidecar server (Python 3.12).

Owns the FlashDrive model, the Qwen3-VL processor, and the streaming KV
lifecycle. The ROS node (Python 3.10) is a thin client that ships pre-tokenized
payloads here over the numpy-only ``wire`` protocol; this process does
tokenization + ``convert_to_streaming_window`` + ``sample_trajectories_streaming``
so the processor version that builds ``tokenized_data`` always matches the model.

Contract (matches ``alpamayo_node.FlashDriveBackend``):

  GET  /health   -> 200 once the model is loaded (and warmed up).
  POST /reset    -> drop the streaming KV cache; next /predict is window 0.
  POST /predict  -> one streaming window.
       request  meta   {num_frames_per_camera:int, nav_text:str|null}
                arrays image_frames  uint8   [n_cams, n_frames, 3, H, W]
                       camera_indices int64  [n_cams]
                       ego_history_xyz float32 [1,1,16,3]
                       ego_history_rot float32 [1,1,16,3,3]
       response window 0 -> meta {status:"prefill", tokenize_sec, model_sec, ...}
                window 1+ -> meta {status:"ok", cot:str, tokenize_sec, model_sec,
                                   post_sec, server_sec, prep_sec}
                            arrays traj_batch float32 [N,T,3]
                                   rot_batch  float32 [N,T,3,3]
                error   -> meta {status:"error", message:str}  (HTTP 200)

Sampling/streaming knobs are read ONCE from the environment at startup and held
constant, because FlashDrive freezes ``max_new_tokens`` / ``num_traj_samples`` /
``temperature`` / ``top_p`` into the static cache + cudagraph shapes on window 0.

Env:
  FD_MODEL_PATH        z-lab/Alpamayo-1.5-10B
  FD_HOST              0.0.0.0
  FD_PORT              8710
  FD_NUM_TRAJ_SAMPLES  1
  FD_MAX_NEW_TOKENS    64        (aligned with node max_generation_length default)
  FD_TEMPERATURE       0.0
  FD_TOP_P             1.0
  FD_DIFFUSION_STEPS   8
  FD_CACHE_STEPS       3,4,5,6
  FD_TORCH_COMPILE     max-autotune  ("" or "none" = eager)
  FD_NUM_VIEWS         4
  FD_NUM_FRAMES        4
  FD_WARMUP            1
  FD_WARMUP_HW         560x1008
  FD_WIRE_SHM          1        (large arrays via POSIX shm; needs ipc:host)
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

import wire

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sidecar] %(levelname)s %(message)s")
log = logging.getLogger("flashdrive_sidecar")


def _cot_from_extra(extra) -> str:
    if not extra or "cot" not in extra:
        return ""
    try:
        text = extra["cot"][0][0][0]
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return str(text).strip()


class FlashDriveEngine:
    """Loads FlashDrive once and runs one streaming window per call (serialized)."""

    def __init__(self) -> None:
        self.model_path = os.environ.get("FD_MODEL_PATH", "z-lab/Alpamayo-1.5-10B")
        self.device = os.environ.get("FD_DEVICE", "cuda")
        self.num_traj_samples = int(os.environ.get("FD_NUM_TRAJ_SAMPLES", "1"))
        self.max_new_tokens = int(os.environ.get("FD_MAX_NEW_TOKENS", "64"))
        self.temperature = float(os.environ.get("FD_TEMPERATURE", "0.0"))
        self.top_p = float(os.environ.get("FD_TOP_P", "1.0"))
        steps = int(os.environ.get("FD_DIFFUSION_STEPS", "8"))
        cache_steps = [int(s) for s in os.environ.get("FD_CACHE_STEPS", "3,4,5,6").split(",") if s != ""]
        self.diffusion_kwargs = {
            "inference_step": steps,
            "cache_steps": cache_steps,
            "int_method": "euler_with_cache",
        }
        compile_mode = os.environ.get("FD_TORCH_COMPILE", "max-autotune")
        self.torch_compile = None if compile_mode in ("", "none", "None") else compile_mode
        self.num_views = int(os.environ.get("FD_NUM_VIEWS", "4"))
        self.num_frames_default = int(os.environ.get("FD_NUM_FRAMES", "4"))

        self._lock = threading.Lock()
        self._window_index = 0
        self.ready = False

        torch.set_float32_matmul_precision("high")
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
        except Exception as exc:  # noqa: BLE001
            log.warning("could not set dynamo capture flags: %s", exc)
        self._load()

    def _load(self) -> None:
        import flashdrive
        from flashdrive.streaming import convert_to_streaming_window

        self._flashdrive = flashdrive
        self._convert = convert_to_streaming_window

        log.info("Loading %s (torch_compile=%s max_new_tokens=%d)...",
                 self.model_path, self.torch_compile, self.max_new_tokens)
        t0 = time.perf_counter()
        self.model = flashdrive.from_pretrained(
            self.model_path, device=self.device, torch_compile=self.torch_compile
        )
        if not isinstance(self.model, flashdrive.FlashDriveMixin):
            raise RuntimeError(
                f"{self.model_path!r} is not an optimized (z-lab) checkpoint; "
                "sample_trajectories_streaming is unavailable."
            )
        package = flashdrive.resolve_model_class(self.model_path).__module__.split(".")[0]
        self.helper = importlib.import_module(f"{package}.helper")
        self.processor = self.helper.get_processor(self.model.tokenizer)
        self.camera_conditioned = (
            "camera_indices" in inspect.signature(self.helper.create_message).parameters
        )
        self.vs_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.ve_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        log.info("Model loaded in %.1f s (camera_conditioned=%s).", time.perf_counter() - t0, self.camera_conditioned)

        if os.environ.get("FD_WARMUP", "1") not in ("0", "false", "False"):
            self._warmup()
        self.ready = True
        log.info("Sidecar READY.")

    def _synthetic_payload(self):
        hw = os.environ.get("FD_WARMUP_HW", "560x1008")
        h, w = (int(x) for x in hw.split("x"))
        n_cams = self.num_views
        n_frames = self.num_frames_default
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (n_cams, n_frames, 3, h, w), dtype=np.uint8)
        cam_idx = np.array([0, 1, 2, 6][:n_cams], dtype=np.int64)
        ego_xyz = np.zeros((1, 1, 16, 3), dtype=np.float32)
        ego_rot = np.tile(np.eye(3, dtype=np.float32), (1, 1, 16, 1, 1))
        return frames, cam_idx, ego_xyz, ego_rot

    def _warmup(self) -> None:
        log.info("Warmup: triggering torch.compile via synthetic prefill + streaming window...")
        t0 = time.perf_counter()
        frames, cam_idx, ego_xyz, ego_rot = self._synthetic_payload()
        try:
            self._infer(frames, cam_idx, ego_xyz, ego_rot, self.num_frames_default, None)
            self._infer(frames, cam_idx, ego_xyz, ego_rot, self.num_frames_default, None)
        finally:
            self._clear_stream_state()
        log.info("Warmup complete in %.1f s; stream reset.", time.perf_counter() - t0)

    def _invalidate_compiled_stream_steps(self) -> None:
        reg = getattr(self.model, "_compiled_step_registry", None)
        if not isinstance(reg, dict):
            return
        drop = [k for k in reg if k == "encode" or k.startswith("dflash")]
        for k in drop:
            reg.pop(k, None)
        if drop:
            log.info("Invalidated compiled steps %s (recompile on next streaming window).", drop)

    def _clear_stream_state(self) -> None:
        self.model._past_key_values = None
        self.model._frame_ranges = None
        self.model._traj_text_range = None
        self.model._shift_src_index = None
        self.model._shift_dst_index = None
        self.model.streaming_position_ids = None
        self.model.streaming_cache_position = None
        self.model._cached_streaming_attention_mask = None
        try:
            self.model.vlm.model.visual.reset_shape_caches()
            self.model.vlm.model.language_model.reset_shape_caches()
        except Exception as exc:  # noqa: BLE001
            log.warning("reset_shape_caches failed: %s", exc)
        if self.torch_compile is not None:
            self._invalidate_compiled_stream_steps()
        self._window_index = 0

    def reset(self) -> None:
        self._clear_stream_state()
        rewarm_env = os.environ.get("FD_RESET_REWARM", "0")
        want_rewarm = (
            self.torch_compile is not None
            and rewarm_env not in ("0", "false", "False", "")
        )
        if not want_rewarm:
            return
        log.info("Reset rewarm: synthetic w0+w1 (FD_RESET_REWARM=1)...")
        t0 = time.perf_counter()
        frames, cam_idx, ego_xyz, ego_rot = self._synthetic_payload()
        try:
            self._infer(frames, cam_idx, ego_xyz, ego_rot, self.num_frames_default, None)
            self._infer(frames, cam_idx, ego_xyz, ego_rot, self.num_frames_default, None)
        finally:
            self._clear_stream_state()
        log.info("Reset rewarm complete in %.1f s.", time.perf_counter() - t0)

    def _build_inputs(self, frames, cam_idx, ego_xyz, ego_rot, num_frames_per_camera, nav_text, streaming):
        # Streaming windows only keep the last frame per camera after
        # convert_to_streaming_window (see flashdrive.streaming). Tokenizing all
        # 16 frames then discarding 12 is pure waste (~60–80 ms). Feed only the
        # last frame/view so the processor work matches what the model sees.
        n_frames = int(num_frames_per_camera)
        frames_use = frames
        if streaming and n_frames > 1:
            frames_use = frames[:, -1:, ...]
            n_frames = 1
        frames_t = torch.from_numpy(np.ascontiguousarray(frames_use)).flatten(0, 1)
        kwargs = {}
        if self.camera_conditioned:
            kwargs["camera_indices"] = torch.from_numpy(np.ascontiguousarray(cam_idx))
            kwargs["num_frames_per_camera"] = n_frames
        if nav_text:
            kwargs["nav_text"] = nav_text
        messages = self.helper.create_message(frames_t, **kwargs)
        tok = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": tok,
            "ego_history_xyz": torch.from_numpy(np.ascontiguousarray(ego_xyz)),
            "ego_history_rot": torch.from_numpy(np.ascontiguousarray(ego_rot)),
        }
        if streaming:
            model_inputs = self._convert(
                model_inputs,
                self.vs_id,
                self.ve_id,
                num_views=self.num_views,
                num_frames_per_view=n_frames,
            )
        return self.helper.to_device(model_inputs, self.device)

    def _infer(self, frames, cam_idx, ego_xyz, ego_rot, num_frames_per_camera, nav_text):
        streaming = self._window_index > 0
        t_tok0 = time.perf_counter()
        data = self._build_inputs(
            frames, cam_idx, ego_xyz, ego_rot, num_frames_per_camera, nav_text, streaming
        )
        tokenize_sec = time.perf_counter() - t_tok0
        torch.cuda.synchronize()
        t_m0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.sample_trajectories_streaming(
                data=data,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_traj_samples=self.num_traj_samples,
                num_traj_sets=1,
                diffusion_kwargs=self.diffusion_kwargs,
                return_extra=True,
            )
        torch.cuda.synchronize()
        model_sec = time.perf_counter() - t_m0
        self._window_index += 1
        return out, tokenize_sec, model_sec

    def predict(self, meta, arrays):
        """Serialized single-window predict. Returns (resp_meta, resp_arrays)."""
        with self._lock:
            t0 = time.perf_counter()
            out, tokenize_sec, model_sec = self._infer(
                arrays["image_frames"],
                arrays["camera_indices"],
                arrays["ego_history_xyz"],
                arrays["ego_history_rot"],
                meta.get("num_frames_per_camera", self.num_frames_default),
                meta.get("nav_text"),
            )
            pred_xyz = out[0]
            if pred_xyz is None:
                return {
                    "status": "prefill",
                    "tokenize_sec": tokenize_sec,
                    "model_sec": model_sec,
                    "post_sec": 0.0,
                    "prep_sec": tokenize_sec,
                    "server_sec": time.perf_counter() - t0,
                }, {}
            pred_rot, extra = out[1], out[2]
            t_p0 = time.perf_counter()
            traj_batch = pred_xyz.detach().float().cpu().numpy()[0, 0]
            rot_batch = pred_rot.detach().float().cpu().numpy()[0, 0]
            post_sec = time.perf_counter() - t_p0
            resp_meta = {
                "status": "ok",
                "cot": _cot_from_extra(extra),
                "prep_sec": tokenize_sec,
                "tokenize_sec": tokenize_sec,
                "model_sec": model_sec,
                "post_sec": post_sec,
                "server_sec": time.perf_counter() - t0,
            }
            return resp_meta, {
                "traj_batch": np.ascontiguousarray(traj_batch, dtype=np.float32),
                "rot_batch": np.ascontiguousarray(rot_batch, dtype=np.float32),
            }


def make_handler(engine: FlashDriveEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            return

        def _send(self, code, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length) if length else b""

        def do_GET(self):
            if self.path == "/health":
                self._send(200 if engine.ready else 503, wire.encode({"ready": engine.ready}, {}))
            else:
                self._send(404, wire.encode({"status": "error", "message": "not found"}, {}))

        def do_POST(self):
            req_meta: dict = {}
            try:
                if self.path == "/predict":
                    req_meta, arrays = wire.decode(self._read_body())
                    resp_meta, resp_arrays = engine.predict(req_meta, arrays)
                    self._send(200, wire.encode(resp_meta, resp_arrays))
                elif self.path == "/reset":
                    engine.reset()
                    self._send(200, wire.encode({"status": "ok"}, {}))
                else:
                    self._send(404, wire.encode({"status": "error", "message": "not found"}, {}))
            except Exception as exc:  # noqa: BLE001
                log.exception("predict failed")
                self._send(200, wire.encode({"status": "error", "message": str(exc)}, {}))
            finally:
                # Close attached views only; client owns unlink of _shm_owned.
                try:
                    wire.release_shm(req_meta, unlink=False)
                except Exception:  # noqa: BLE001
                    pass

    return Handler


def main() -> None:
    host = os.environ.get("FD_HOST", "0.0.0.0")
    port = int(os.environ.get("FD_PORT", "8710"))
    engine = FlashDriveEngine()
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    log.info("Serving on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
