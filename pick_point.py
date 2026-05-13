"""
Pick-point selection and detection scoring.

Scoring factors (normalised to [0,1], then weighted):
  confidence       : blob fill ratio — compact blob = better suction grip
  isolation        : distance to nearest neighbour — further = easier pick
  depth_flatness   : low depth variance inside mask = flat surface
  height_priority  : prefer wings flat on the board (low height_above_table)
                     over stacked wings — flat wings are stable for suction

The pick order produced by this scorer directly drives the robot: it will
always attempt the most accessible wing first.
"""

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


def score_all_detections(
    detections: list[Detection],
    depth: np.ndarray,
    w_confidence: float = 0.20,
    w_isolation: float = 0.15,
    w_flatness:  float = 0.35,
    w_height:    float = 0.30,
) -> list[tuple[Detection, float, int, int]]:
    """
    Score every detection.  Returns list of (detection, score, u, v)
    sorted best-first.

    Height weighting (w_height=0.30): a wing flat on the board scores 1.0;
    a wing stacked N cm above scores proportionally less.  This makes the
    robot preferentially pick stable, accessible wings.
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
) -> PickPoint | None:
    scored = score_all_detections(detections, depth)
    if not scored:
        return None

    det, score, u, v = scored[0]
    patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    valid = patch[(patch > 0.05) & (patch < 5.0)]
    depth_m = float(np.median(valid)) if len(valid) > 0 else 0.0

    return PickPoint(u=u, v=v, depth_m=depth_m, score=score, detection=det)
