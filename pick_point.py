"""
Pick-point selection and detection scoring.

Scoring factors (normalised to [0,1], then weighted):
  confidence       : blob fill ratio — compact blob = better grip
  isolation        : distance to nearest neighbour — further = gripper can close
  depth_flatness   : low depth variance inside mask = flat, stable surface
  height_priority  : prefer wings flat on the board (low height_above_table)
                     over stacked wings — flat wings are stable to grasp

The pick order produced by this scorer directly drives the robot: it will
always attempt the most accessible wing first.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import center_of_mass

from vision import Detection


@dataclass
class PickPoint:
    u: int
    v: int
    depth_m: float
    score: float
    detection: Detection
    yaw_rad: float = 0.0         # commanded gripper Rz: wing axis + pi/2 (jaws close across the wing)
    yaw_image_rad: float = 0.0   # wing long axis in image space (for visualization only)


@dataclass
class BoxPoint:
    """Drop-zone (black box) target, mirrors PickPoint."""
    u: int
    v: int
    depth_m: float
    robot_xyz: np.ndarray        # [x, y, z] in robot base frame, metres
    yaw_rad: float = 0.0         # commanded gripper Rz over the box: box axis + pi/2
    yaw_image_rad: float = 0.0   # box long axis in image space (visualization only)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    cy, cx = center_of_mass(mask)
    return float(cx), float(cy)


def _isolation(idx: int, centroids: list[tuple[float, float]]) -> float:
    if len(centroids) == 1:
        return 1.0
    cx, cy = centroids[idx]
    return min(np.hypot(cx - ox, cy - oy) for j, (ox, oy) in enumerate(centroids) if j != idx)


def _depth_flatness(mask: np.ndarray, depth: np.ndarray) -> float:
    """Lower std = flatter surface. Returns raw std in metres."""
    pixels = depth[mask > 0]
    valid = pixels[(pixels > 0.05) & (pixels < 5.0)]
    return float(np.std(valid)) if len(valid) >= 10 else float("inf")


def wrap_half_turn(a: float) -> float:
    """Wrap an undirected-axis angle to (-pi/2, pi/2]."""
    if a > math.pi / 2:
        a -= math.pi
    elif a <= -math.pi / 2:
        a += math.pi
    return a


def principal_axis_yaw_robot(
    mask: np.ndarray,
    R_cam_to_robot: np.ndarray,
) -> tuple[float, float]:
    """
    Returns (yaw_robot_rad, yaw_image_rad), each wrapped to (-pi/2, pi/2].

    PCA on the mask gives a unit direction (du, dv) in image space, which
    coincides with the camera's XY plane: image +u = camera +X, image +v =
    camera +Y. Rotating (du, dv, 0) by the camera->robot rotation R and
    taking atan2 of the robot XY components gives the yaw the gripper needs.
    The image-frame angle is returned alongside so the visualizer can draw
    the axis on the camera feed.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 0.0, 0.0

    cu = xs.mean()
    cv = ys.mean()
    pts = np.stack([xs - cu, ys - cv], axis=0).astype(np.float64)   # 2 x N
    cov = pts @ pts.T / pts.shape[1]
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending
    if eigvals[-1] < 1e-6:                        # degenerate / single-pixel blob
        return 0.0, 0.0
    du, dv = eigvecs[:, -1]

    yaw_image_rad = wrap_half_turn(math.atan2(float(dv), float(du)))

    axis_cam = np.array([du, dv, 0.0], dtype=np.float64)
    axis_robot = R_cam_to_robot @ axis_cam
    yaw_axis_robot = math.atan2(float(axis_robot[1]), float(axis_robot[0]))
    # Gripper jaws close across the wing, so command axis + pi/2.
    yaw_robot_rad = wrap_half_turn(yaw_axis_robot + math.pi / 2)

    return yaw_robot_rad, yaw_image_rad


def score_all_detections(
    detections: list[Detection],
    depth: np.ndarray,
    w_confidence: float = 0.20,
    w_isolation: float = 0.35,
    w_flatness:  float = 0.15,
    w_height:    float = 0.30,
) -> list[tuple[Detection, float, int, int]]:
    """
    Score every detection.  Returns list of (detection, score, u, v)
    sorted best-first.

    Isolation weighting (w_isolation=0.35): gripper fingers need clearance on
    both sides of the wing.  Isolated wings score highest; wings touching
    neighbours score lowest.

    Height weighting (w_height=0.30): a wing flat on the board scores 1.0;
    a wing stacked above scores proportionally less.
    """
    if not detections:
        return []

    centroids   = [_centroid(d.mask)            for d in detections]
    isolations  = [_isolation(i, centroids)      for i in range(len(detections))]
    flatness    = [_depth_flatness(d.mask, depth) for d in detections]
    heights     = [d.height_above_table_m        for d in detections]

    # Normalise isolation (larger = better)
    max_iso = max(isolations) or 1.0
    iso_norm = [v / max_iso for v in isolations]

    # Normalise flatness (smaller std = better, invert)
    finite = [v for v in flatness if v < float("inf")]
    max_flat = max(finite) if finite else 1.0
    flat_norm = [1.0 - min(v, max_flat) / max_flat for v in flatness]

    # Normalise height (smaller = better, invert)
    # Cap at 0.10 m so extreme outliers don't compress the range
    max_h = max(min(h, 0.10) for h in heights) or 1e-6
    height_norm = [1.0 - min(h, 0.10) / max_h for h in heights]

    scored = []
    for i, det in enumerate(detections):
        score = (
            w_confidence * det.confidence
            + w_isolation * iso_norm[i]
            + w_flatness  * flat_norm[i]
            + w_height    * height_norm[i]
        )
        cx, cy = centroids[i]
        scored.append((det, score, int(round(cx)), int(round(cy))))

    scored.sort(key=lambda t: -t[1])
    return scored


def select_pick_point(
    detections: list[Detection],
    depth: np.ndarray,
    T_cam_to_robot: np.ndarray,
) -> PickPoint | None:
    scored = score_all_detections(detections, depth)
    if not scored:
        return None

    det, score, u, v = scored[0]
    patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    valid = patch[(patch > 0.05) & (patch < 5.0)]
    depth_m = float(np.median(valid)) if len(valid) > 0 else 0.0

    yaw_rad, yaw_image_rad = principal_axis_yaw_robot(det.mask, T_cam_to_robot[:3, :3])

    return PickPoint(
        u=u, v=v, depth_m=depth_m, score=score, detection=det,
        yaw_rad=yaw_rad, yaw_image_rad=yaw_image_rad,
    )
