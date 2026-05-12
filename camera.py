"""
Orbbec Gemini 336L RGB-D camera interface.

Requires pyorbbecsdk — build from: https://github.com/orbbec/pyorbbecsdk

Returns aligned RGB (uint8 HxWx3) and depth (float32 HxW, metres).
"""

import numpy as np


class Camera:
    def __init__(self, color_width: int = 1280, color_height: int = 720, fps: int = 30):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode

        self._pipeline = Pipeline()
        cfg = Config()

        color_profile = self._pipeline.get_stream_profile_list(OBSensorType.COLOR).get_video_stream_profile(
            color_width, color_height, OBFormat.RGB, fps
        )
        depth_profile = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH).get_default_video_stream_profile()

        cfg.enable_stream(color_profile)
        cfg.enable_stream(depth_profile)
        cfg.set_align_mode(OBAlignMode.HW_MODE)   # align depth to color in hardware

        self._pipeline.start(cfg)
        self._intrinsics = None   # populated on first frame

    def _get_intrinsics(self):
        from pyorbbecsdk import OBSensorType
        param = self._pipeline.get_camera_param()
        ci = param.rgb_intrinsic
        from geometry import CameraIntrinsics
        return CameraIntrinsics(fx=ci.fx, fy=ci.fy, cx=ci.cx, cy=ci.cy)

    @property
    def intrinsics(self):
        if self._intrinsics is None:
            self._intrinsics = self._get_intrinsics()
        return self._intrinsics

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            rgb   — uint8 array (H, W, 3)
            depth — float32 array (H, W) in metres
        """
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        if frames is None:
            raise RuntimeError("Camera timeout: no frames received")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            raise RuntimeError("Incomplete frameset from camera")

        rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
            color_frame.get_height(), color_frame.get_width(), 3
        )

        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth_scale = depth_frame.get_depth_scale()     # mm -> metres factor
        depth_m = depth_raw.astype(np.float32) * depth_scale

        return rgb, depth_m

    def stop(self):
        self._pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
