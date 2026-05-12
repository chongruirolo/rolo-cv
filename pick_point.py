"""
Pick-point selection.

Scores each detected food item and returns the (u, v, depth) of the best one.

Scoring factors (all normalised to [0, 1], then weighted):
  - confidence     : model certainty
  - isolation      : distance to nearest neighbour centroid (farther = easier pick)
  - depth_flatness : low depth variance inside mask = flat surface = better suction
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
    return float(cx), float(cy)   # (u, v)


def _isolation(idx: int, centroids: list[tuple[float, float]]) -> float:
    if len(centroids) == 1:
        return 1.0
    cx, cy = centroids[idx]
    dists = [
        np.hypot(cx - ox, cy - oy)
        for j, (ox, oy) in enumerate(centroids)
        if j != idx
    ]
    return min(dists)


def _depth_flatness(mask: np.ndarray, depth: np.ndarray) -> float:
    """Low variance = flat surface. Returns raw std in metres."""
    pixels = depth[mask > 0]
    valid = pixels[(pixels > 0.05) & (pixels < 5.0)]   # ignore missing/invalid depth
    if len(valid) < 10:
        return float("inf")
    return float(np.std(valid))


def select_pick_point(
    detections: list[Detection],
    depth: np.ndarray,
    w_confidence: float = 0.4,
    w_isolation: float = 0.4,
    w_flatness: float = 0.2,
) -> PickPoint | None:
    """
    Score each detection and return the best PickPoint, or None if no detections.

    depth: float32 (H, W) in metres
    """
    if not detections:
        return None

    centroids = [_centroid(d.mask) for d in detections]

    # raw scores
    isolations = [_isolation(i, centroids) for i in range(len(detections))]
    flatness_stds = [_depth_flatness(d.mask, depth) for d in detections]

    # normalise isolation (larger = better)
    max_iso = max(isolations) if max(isolations) > 0 else 1.0
    iso_norm = [v / max_iso for v in isolations]

    # normalise flatness (smaller std = better, invert)
    max_flat = max(flatness_stds) if max(flatness_stds) < float("inf") else 1.0
    flat_norm = [1.0 - min(v, max_flat) / max_flat for v in flatness_stds]

    best_score = -1.0
    best_idx = 0

    for i, det in enumerate(detections):
        score = (
            w_confidence * det.confidence
            + w_isolation * iso_norm[i]
            + w_flatness * flat_norm[i]
        )
        if score > best_score:
            best_score = score
            best_idx = i

    cx, cy = centroids[best_idx]
    u, v = int(round(cx)), int(round(cy))

    # sample depth at centroid (average small patch for robustness)
    patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    valid_patch = patch[(patch > 0.05) & (patch < 5.0)]
    depth_m = float(np.median(valid_patch)) if len(valid_patch) > 0 else 0.0

    return PickPoint(u=u, v=v, depth_m=depth_m, score=best_score, detection=detections[best_idx])
