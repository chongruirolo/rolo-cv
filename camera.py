"""
Camera interface for the Orbbec Gemini 336L RGB-D sensor.
Requires pyorbbecsdk built from source:
  git clone https://github.com/orbbec/pyorbbecsdk
  cd pyorbbecsdk && pip install -e .
"""

import cv2
import numpy as np
from geometry import CameraIntrinsics


class Camera:
    def __init__(
        self,
        color_width: int = 1280,
        color_height: int = 720,
        fps: int = 30,
    ):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode

        self._pipeline = Pipeline()
        cfg = Config()
        color_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_video_stream_profile(color_width, color_height, OBFormat.RGB, fps)
        depth_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()
        cfg.enable_stream(color_profile)
        cfg.enable_stream(depth_profile)
        cfg.set_align_mode(OBAlignMode.SW_MODE)
        self._pipeline.start(cfg)
        self._intrinsics = None
        self._warmup()
        print("[camera] Orbbec Gemini 336L ready")

    def _warmup(self, max_frames: int = 30) -> None:
        # Discard frames until both colour and depth streams are delivering data.
        # The camera hardware needs a moment to initialise after startup —
        # early frames often have one or both streams missing.
        for _ in range(max_frames):
            frames = self._pipeline.wait_for_frames(2000)
            if frames is None:
                continue
            if frames.get_color_frame() is not None and frames.get_depth_frame() is not None:
                return
        raise RuntimeError("Orbbec camera: streams did not become ready after warmup")

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            param = self._pipeline.get_camera_param()
            ci = param.rgb_intrinsic
            self._intrinsics = CameraIntrinsics(fx=ci.fx, fy=ci.fy, cx=ci.cx, cy=ci.cy)
        return self._intrinsics

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        # Returns (rgb, depth_m) where depth_m is float32 in metres.
        # Gemini 336L always outputs uint16 millimetres, scale = 0.001.
        # If you swap to a different Orbbec model, verify the depth unit and
        # update the 0.001 multiplier accordingly.
        for _ in range(10):
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
                color_frame.get_height(), color_frame.get_width(), 3
            )
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            depth_m = depth_raw.astype(np.float32) * 0.001
            # Resize depth to match colour resolution if SW alignment left them different sizes
            h, w = rgb.shape[:2]
            if depth_m.shape != (h, w):
                depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
            return rgb, depth_m
        raise RuntimeError("Camera: failed to get a complete frameset after 10 attempts")

    def stop(self):
        self._pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
