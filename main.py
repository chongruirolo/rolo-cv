"""
Main orchestration loop.

Run: python main.py
     python main.py --dry-run          (skip robot, print coordinates only)
     python main.py --dry-run --show   (live OpenCV window, press q to quit)
"""

import argparse
import datetime
import time
import numpy as np
import yaml

from camera import Camera
from vision import FoodDetector, Detection
from pick_point import select_pick_point, score_all_detections
from geometry import pixel_to_robot
from robot_controller import RobotController
from drop_tracker import DropTracker

# deals with classification problem between wings in box and on board, dont know how to differentiate
def _box_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _overlaps_any(det: Detection, others: list[Detection], threshold: float = 0.3) -> bool:
    return any(_box_iou(det, o) >= threshold for o in others)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(cfg: dict) -> np.ndarray:
    T = np.array(cfg["camera"]["T_cam_to_robot"], dtype=np.float64)
    assert T.shape == (4, 4), "T_cam_to_robot must be a 4x4 matrix"
    return T


def _log_detections(
    detections: list,
    depth: np.ndarray,
    pick_u: int | None,
    pick_v: int | None,
) -> None:
    """Print a scored table of every detected wing, once per call."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    scored = score_all_detections(detections, depth)
    print(f"[{ts}] {len(scored)} wing(s) detected:")
    for rank, (det, score, u, v) in enumerate(scored, 1):
        x1, y1, x2, y2 = det.bbox
        pick_marker = "  ← PICK" if (u == pick_u and v == pick_v) else ""
        h_cm = det.height_above_table_m * 100
        depth_str = f"{det.median_depth_m:.3f}m" if det.median_depth_m > 0 else "n/a "
        # Stacking indicator: flat=on board, stacked=>3cm above
        stack_tag = " [STACKED]" if h_cm > 3.0 else " [flat]   "
        print(
            f"  #{rank}{stack_tag}"
            f"  height={h_cm:5.1f}cm"
            f"  depth={depth_str}"
            f"  fill={det.confidence:.2f}"
            f"  score={score:.3f}"
            f"  bbox=({x1},{y1},{x2},{y2})"
            f"{pick_marker}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print robot coords without moving")
    parser.add_argument("--show", action="store_true", help="Open a live OpenCV window with detections overlaid")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.show:
        from visualize import draw_frame, close as close_window

    cfg = load_config(args.config)
    T = build_transform(cfg)

    if args.dry_run:
        print("[DRY RUN] Robot will not move.")
        robot = None
    else:
        robot_cfg = cfg.get("robot", {})
        robot = RobotController(
            can_interface=robot_cfg.get("can_interface", "can0"),
            grip_z_offset_m=robot_cfg.get("grip_z_offset_m", 0.02),
        )

    pick_count = 0
    max_picks = cfg.get("max_picks", 50)

    last_log = 0.0   # timestamp of last per-second detection log

    confirmed_box_count = 0
    _candidate_count    = 0
    _candidate_frames   = 0
    CONFIRM_FRAMES      = 3

    tracker_cfg = cfg.get("drop_tracker", {})
    # Two trackers, identical except for the dedup switch — running them
    # side-by-side lets the HUD show "with dedup" vs "without dedup" so the
    # effect of dedup is observable live.
    _tracker_kwargs = dict(
        max_match_dist_px=tracker_cfg.get("max_match_dist_px", 80.0),
        decay_frames=tracker_cfg.get("decay_frames", 60),
        min_confidence=tracker_cfg.get("min_confidence", 0.3),
        slot_radius_px=tracker_cfg.get("slot_radius_px", 30.0),
        dedup_radius_px=tracker_cfg.get("dedup_radius_px", 25.0),
        dedup_depth_tolerance_m=tracker_cfg.get("dedup_depth_tolerance_m", 0.012),
    )
    drop_tracker          = DropTracker(dedup_enabled=True,  **_tracker_kwargs)
    drop_tracker_no_dedup = DropTracker(dedup_enabled=False, **_tracker_kwargs)

    try:
        with Camera(
            color_width=cfg.get("camera", {}).get("width", 1280),
            color_height=cfg.get("camera", {}).get("height", 720),
            fps=cfg.get("camera", {}).get("fps", 30),
        ) as cam:
            intrinsics = cam.intrinsics

            vision_cfg = cfg.get("vision", {})
            detector = FoodDetector(
                model_path=vision_cfg.get("model_path", "FastSAM-s.pt"),
                backend=vision_cfg.get("backend", "sam"),
                intrinsics=intrinsics,
                confidence=vision_cfg.get("confidence", 0.3),
                iou=vision_cfg.get("iou", 0.45),
                device=vision_cfg.get("device", "cuda"),
                elevation_min_h=vision_cfg.get("elevation_min_cm", 1.0) / 100.0,
                elevation_max_h=vision_cfg.get("elevation_max_cm", 10.0) / 100.0,
                roi=vision_cfg.get("roi"),
            )

            print("Starting. Press q to quit, f to toggle overlays." if args.show else "Starting pick loop. Ctrl-C to stop.")

            _prev_key = 0xFF
            show_overlays = True

            while True:
                if not args.show and pick_count >= max_picks:
                    break

                rgb, depth = cam.capture()
                detections = detector.detect(rgb, depth)
                in_box = [d for d in detections if d.label == "wing-in-box"]
                drop_zone_det = next((d for d in detections if d.label == "drop-zone"), None)

                tracker_count          = drop_tracker.update(in_box, drop_zone=drop_zone_det)
                tracker_count_no_dedup = drop_tracker_no_dedup.update(in_box, drop_zone=drop_zone_det)

                detected = len(in_box)
                if detected > confirmed_box_count:
                    if detected == _candidate_count:
                        _candidate_frames += 1
                    else:
                        _candidate_count = detected
                        _candidate_frames = 1
                    if _candidate_frames >= CONFIRM_FRAMES:
                        confirmed_box_count = _candidate_count
                        _candidate_frames = 0
                else:
                    _candidate_count = 0
                    _candidate_frames = 0

                pick_candidates = [
                    d for d in detections
                    if d.label == "chicken-wings" and not _overlaps_any(d, in_box)
                ]

                pick = select_pick_point(pick_candidates, depth) if pick_candidates else None
                if pick is not None and robot is not None and pick.depth_m <= 0:
                    pick = None

                robot_xyz = None
                if pick is not None:
                    robot_xyz = pixel_to_robot(pick.u, pick.v, pick.depth_m, intrinsics, T)

                # Per-second detection log
                now = time.monotonic()
                if now - last_log >= 1.0:
                    if pick is not None and robot_xyz is not None:
                        _log_detections(pick_candidates, depth, pick.u, pick.v)
                        x, y, z = robot_xyz.tolist()
                        print(f"  → pick_drop(pick=[{x:.4f}, {y:.4f}, {z:.4f}])")
                    else:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] 0 wing(s) detected")
                    last_log = now

                if args.show:
                    key = draw_frame(
                        rgb, detections, pick,
                        roi=vision_cfg.get("roi"),
                        box_count=confirmed_box_count,
                        tracker_count=tracker_count,
                        tracker_count_no_dedup=tracker_count_no_dedup,
                        raw_count=len(in_box),
                        tracker=drop_tracker,
                        filter_enabled=show_overlays,
                    )
                    if key == ord("q"):
                        print("Window closed by user.")
                        break
                    if key in (ord("f"), ord("F")) and _prev_key not in (ord("f"), ord("F")):
                        show_overlays = not show_overlays
                        print(f"[overlays] {'ON' if show_overlays else 'OFF'}")
                    _prev_key = key

                if robot is not None and robot_xyz is not None:
                    robot.pick_and_drop(robot_xyz)
                    pick_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if args.show:
            close_window()
        if robot is not None:
            robot.stop()

    print(f"Done. Total picks: {pick_count}")


if __name__ == "__main__":
    main()
