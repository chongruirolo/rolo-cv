"""
Coordinate transforms:
  pixel + depth  ->  3D camera frame
  camera frame   ->  robot base frame
"""

"""
convert to robot coordinates
"""

import numpy as np


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def pixel_to_camera(u: int, v: int, depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return [X, Y, Z] in camera frame (metres)."""
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    return np.array([x, y, depth_m], dtype=np.float64)


def camera_to_robot(cam_point: np.ndarray, T_cam_to_robot: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform and return [X, Y, Z] in robot base frame (metres)."""
    p_hom = np.append(cam_point, 1.0)          # homogeneous
    robot_hom = T_cam_to_robot @ p_hom
    return robot_hom[:3]


def pixel_to_robot(
    u: int,
    v: int,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    T_cam_to_robot: np.ndarray,
) -> np.ndarray:
    """Convenience: pixel + depth directly to robot frame."""
    cam = pixel_to_camera(u, v, depth_m, intrinsics)
    return camera_to_robot(cam, T_cam_to_robot)
