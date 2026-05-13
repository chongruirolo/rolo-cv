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
) -> bool:
    """
    Overlay detections and pick point on `rgb`, display in a named window.

    Returns False when the user presses 'q' (caller should stop the loop).
    """
    # OpenCV expects BGR
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for idx, det in enumerate(detections):
        color = _PALETTE[idx % len(_PALETTE)]
        _draw_mask(frame, det.mask, color)
        _draw_box(frame, det, color)

    if pick is not None:
        _draw_pick(frame, pick)

    cv2.imshow(_WINDOW, frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def close():
    cv2.destroyAllWindows()


# ---------- helpers ----------

def _draw_mask(frame: np.ndarray, mask: np.ndarray, color: tuple) -> None:
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def _draw_box(frame: np.ndarray, det: Detection, color: tuple) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    depth_str  = f" {det.median_depth_m:.2f}m" if det.median_depth_m > 0 else ""
    height_str = f" h={det.height_above_table_m * 100:.1f}cm"
    label = f"{det.label}{depth_str}{height_str} fill={det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_pick(frame: np.ndarray, pick: PickPoint) -> None:
    u, v = pick.u, pick.v
    cv2.circle(frame, (u, v), 14, (0, 0, 255), 3)
    cv2.circle(frame, (u, v), 3, (0, 0, 255), -1)
    info = f"PICK  {pick.detection.label}  depth={pick.depth_m:.3f}m  score={pick.score:.2f}"
    cv2.putText(frame, info, (u + 18, v + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
