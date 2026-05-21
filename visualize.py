"""
Live frame visualization for the pick-and-place pipeline.

Draws segmentation masks, bounding boxes, and the chosen pick point
onto the RGB frame, then shows it in an OpenCV window.

Returns True while the window is open, False when the user presses 'q'.
"""

import cv2
import numpy as np

from vision import Detection
from pick_point import PickPoint

# Distinct BGR colours for up to 10 simultaneous detections.
_PALETTE = [
    (0, 255, 127),
    (0, 165, 255),
    (255, 0, 127),
    (127, 0, 255),
    (0, 255, 255),
    (255, 127, 0),
    (0, 200, 255),
    (180, 0, 255),
    (255, 0, 200),
    (0, 255, 50),
]

_WINDOW = "rolo-cv  |  press q to quit"


def draw_frame(
    rgb: np.ndarray,
    detections: list[Detection],
    pick: PickPoint | None,
    roi: list | None = None,
    box_count: int = 0,
    tracker_count: int = 0,
    tracker_count_no_dedup: int = 0,
    raw_count: int = 0,
    tracker=None,   # DropTracker instance — draws slots and candidates if provided
    filter_enabled: bool = False,
) -> int:
    """
    Overlay detections and pick point on `rgb`, display in a named window.

    Returns the raw key code from cv2.waitKey (& 0xFF). -1 means no key pressed.
    Caller should stop the loop when the returned key is ord('q').
    """
    # OpenCV expects BGR
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw ROI boundary so you can see exactly what area the model processes
    if roi is not None:
        h, w = frame.shape[:2]
        x1f, y1f, x2f, y2f = roi
        rx1, ry1 = int(x1f * w), int(y1f * h)
        rx2, ry2 = int(x2f * w), int(y2f * h)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # All detection/tracker overlays only render when the filter is ON.
    # Filter OFF keeps the window clean — just the raw RGB feed + ROI + button.
    if filter_enabled:
        if tracker is not None:
            default_r = int(tracker.slot_radius)
            for cx, cy, radius, stack in tracker.confirmed_slots:
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)  # green = confirmed
                if stack > 1:
                    label = f"x{stack}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    tx, ty = int(cx) - tw // 2, int(cy) + th // 2
                    cv2.rectangle(frame, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3),
                                  (0, 120, 0), -1)
                    cv2.putText(frame, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            for cx, cy, frames in tracker.active_candidates:
                cv2.circle(frame, (int(cx), int(cy)), default_r, (0, 255, 255), 1)  # yellow = pending

        for idx, det in enumerate(detections):
            color = _PALETTE[idx % len(_PALETTE)]
            _draw_mask(frame, det.mask, color)
            _draw_box(frame, det, color)

        counter = (
            f"Box(raw): {raw_count}  "
            f"Box(debounce): {box_count}  "
            f"Box(no-dedup): {tracker_count_no_dedup}  "
            f"Box(dedup): {tracker_count}"
        )
        (tw, th), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, 8), (18 + tw, 28 + th), (30, 30, 30), -1)
        cv2.putText(frame, counter, (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if pick is not None:
            _draw_pick(frame, pick)

    # Filter toggle button — top-right corner
    btn_label = "Filter [F]: ON" if filter_enabled else "Filter [F]: OFF"
    btn_color = (0, 200, 0) if filter_enabled else (0, 0, 180)
    (bw, bh), _ = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    fw = frame.shape[1]
    bx1, by1 = fw - bw - 20, 8
    bx2, by2 = fw - 8, bh + 24
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_color, -1)
    cv2.putText(frame, btn_label, (bx1 + 6, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(_WINDOW, frame)
    return cv2.waitKey(1) & 0xFF


def close():
    cv2.destroyAllWindows()


# ---------- helpers ----------

def _draw_mask(frame: np.ndarray, mask: np.ndarray, color: tuple) -> None:
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def _draw_box(frame: np.ndarray, det: Detection, color: tuple) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    depth_str = f" {det.median_depth_m:.2f}m" if det.median_depth_m > 0 else ""
    label = f"{det.label}{depth_str} {det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_pick(frame: np.ndarray, pick: PickPoint) -> None:
    u, v = pick.u, pick.v
    cv2.circle(frame, (u, v), 10, (0, 0, 255), 2)
    cv2.circle(frame, (u, v), 2, (0, 0, 255), -1)
    info = f"PICK  depth={pick.depth_m:.3f}m  score={pick.score:.2f}"
    cv2.putText(frame, info, (u + 14, v + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
