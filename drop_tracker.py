"""
Minimal slot tracker — Hungarian assignment + slot decay + optional 2D dedup.

Per frame:
  1. Filter detections by `min_confidence`.
  2. (Optional) Dedup: collapse detections that are likely duplicate masks
     of the same physical wing — close in XY AND similar in depth.
  3. Compute mask centroids of the survivors.
  4. Optimally match observations to existing slots via
     `scipy.optimize.linear_sum_assignment`, capped at `max_match_dist_px`.
  5. Matched slots snap to the new centroid and refresh their last-seen frame.
  6. Unmatched observations become brand new slots.
  7. Slots not matched for more than `decay_frames` are dropped.

count = number of currently-tracked slots (can both increase and decrease).

Dedup is a separate module-level function and can be toggled on/off via
`dedup_enabled` for A/B debugging.

Tuning parameters
-----------------
max_match_dist_px        Hungarian cap. Max pixels a wing can move between
                         frames and still snap to its existing slot.
decay_frames             Slot dropped after this many consecutive frames
                         without a match.
min_confidence           Minimum detection confidence considered.
slot_radius_px           Render-only footprint used by visualize.py.
dedup_enabled            Master switch — turn off to see raw detector
                         behaviour through the tracker.
dedup_radius_px          XY threshold below which two detections might be
                         duplicates of the same wing.
dedup_depth_tolerance_m  Depth difference below which two XY-close
                         detections are confirmed duplicates. Above this
                         they are treated as distinct objects (e.g. stacked
                         wings, where the top wing reads ~one wing
                         thickness closer to the camera).
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment

from vision import Detection


class DropTracker:
    def __init__(
        self,
        max_match_dist_px:       float = 80.0,
        decay_frames:            int   = 60,
        min_confidence:          float = 0.3,
        slot_radius_px:          float = 30.0,
        dedup_enabled:           bool  = True,
        dedup_radius_px:         float = 25.0,
        dedup_depth_tolerance_m: float = 0.012,
    ):
        self._max_match    = max_match_dist_px
        self._decay_frames = decay_frames
        self._min_conf     = min_confidence
        self._slot_r       = slot_radius_px

        self._dedup_enabled  = dedup_enabled
        self._dedup_radius   = dedup_radius_px
        self._dedup_depth_tol = dedup_depth_tolerance_m

        # Each slot is a (cx, cy, last_seen_frame) tuple
        self._slots: list[tuple[float, float, int]] = []
        self._frame_count = 0

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, in_box: list[Detection], drop_zone: Detection | None = None) -> int:
        """Call once per frame. Returns current slot count (may go up or down).

        `drop_zone` is ignored — present only for backward compatibility."""
        _ = drop_zone
        self._frame_count += 1

        # 1. Confidence filter
        strong = [d for d in in_box if d.confidence >= self._min_conf]

        # 2. Optional 2D dedup (XY + depth)
        if self._dedup_enabled:
            strong = dedup_detections(
                strong,
                radius_px=self._dedup_radius,
                depth_tolerance_m=self._dedup_depth_tol,
            )

        # 3. Centroids
        observed: list[tuple[float, float]] = [_mask_centroid(d) for d in strong]

        # 4. Hungarian assignment
        matched_obs: set[int] = set()
        if self._slots and observed:
            n_slots, n_obs = len(self._slots), len(observed)
            BIG = self._max_match * 10.0
            cost = np.full((n_slots, n_obs), BIG, dtype=np.float64)
            for i, slot in enumerate(self._slots):
                slot_pos = (slot[0], slot[1])
                for j, obs_pos in enumerate(observed):
                    d = _dist(slot_pos, obs_pos)
                    if d <= self._max_match:
                        cost[i, j] = d
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= BIG:
                    continue
                cx, cy = observed[j]
                self._slots[i] = (cx, cy, self._frame_count)
                matched_obs.add(j)

        # 5. Unmatched observations → new slots
        for j, obs_pos in enumerate(observed):
            if j not in matched_obs:
                self._slots.append((obs_pos[0], obs_pos[1], self._frame_count))

        # 6. Decay
        self._slots = [
            s for s in self._slots
            if self._frame_count - s[2] <= self._decay_frames
        ]

        return self.confirmed_count

    def reset(self):
        self._slots.clear()
        self._frame_count = 0

    def set_dedup_enabled(self, enabled: bool) -> None:
        """Runtime toggle — useful for A/B comparison via key binding."""
        self._dedup_enabled = bool(enabled)

    # ── properties (compatible with visualize.py + main.py) ──────────────────

    @property
    def confirmed_count(self) -> int:
        return len(self._slots)

    @property
    def confirmed_slots(self) -> list[tuple[float, float, float, int]]:
        """(cx, cy, radius_px, stack_count) — stack_count is always 1."""
        return [(s[0], s[1], self._slot_r, 1) for s in self._slots]

    @property
    def active_candidates(self) -> list[tuple[float, float, int]]:
        return []

    @property
    def slot_radius(self) -> float:
        return self._slot_r

    @property
    def dedup_enabled(self) -> bool:
        return self._dedup_enabled


# ── dedup (module-level, standalone, testable) ────────────────────────────────

def dedup_detections(
    detections: list[Detection],
    radius_px: float,
    depth_tolerance_m: float,
) -> list[Detection]:
    """Suppress detections that are likely duplicate masks of the same wing.

    Two detections are considered duplicates when BOTH:
      - their mask centroids are within `radius_px`, AND
      - their median depths are within `depth_tolerance_m` (treated as
        unknown/matching when either depth is invalid: depth <= 0).

    The list is processed strongest-first (confidence descending); the
    highest-confidence detection in each duplicate cluster survives.

    XY-close detections with depths differing by MORE than the tolerance are
    kept as separate objects — this is the stacked-wings case (top wing
    reads ~one wing-thickness closer to the camera).
    """
    if not detections:
        return []

    # Sort strongest first so the highest-confidence member of each cluster
    # is the one chosen as the survivor.
    sorted_dets = sorted(detections, key=lambda d: -d.confidence)

    # Pre-compute centroids once
    centroids = [_mask_centroid(d) for d in sorted_dets]

    kept_idx: list[int] = []
    for i, det in enumerate(sorted_dets):
        is_dup = False
        for k in kept_idx:
            if _dist(centroids[i], centroids[k]) >= radius_px:
                continue                                  # far in XY, can't be a duplicate
            # XY-close — now check depth
            di, dk = det.median_depth_m, sorted_dets[k].median_depth_m
            if di > 0 and dk > 0 and abs(di - dk) >= depth_tolerance_m:
                continue                                  # depths differ → distinct objects
            is_dup = True
            break
        if not is_dup:
            kept_idx.append(i)
    return [sorted_dets[i] for i in kept_idx]


# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_centroid(det: Detection) -> tuple[float, float]:
    if det.mask is not None and det.mask.any():
        cy, cx = center_of_mass(det.mask)
        return float(cx), float(cy)
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
