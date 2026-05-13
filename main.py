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
from vision import FoodDetector
from pick_point import select_pick_point, score_all_detections
from geometry import pixel_to_robot
from robot_controller import RobotController


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
        robot = RobotController(can_interface=cfg.get("robot", {}).get("can_interface", "can0"))

    pick_count = 0
    max_picks = cfg.get("max_picks", 50)

    last_log = 0.0   # timestamp of last per-second detection log

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
                device=vision_cfg.get("device", "cpu"),
                elevation_min_h=vision_cfg.get("elevation_min_cm", 1.0) / 100.0,
                elevation_max_h=vision_cfg.get("elevation_max_cm", 10.0) / 100.0,
                roi=vision_cfg.get("roi"),
            )

            print("Starting. Press q to quit." if args.show else "Starting pick loop. Ctrl-C to stop.")

            while True:
                if not args.show and pick_count >= max_picks:
                    break

                rgb, depth = cam.capture()
                detections = detector.detect(rgb, depth)

                if not detections:
                    now = time.monotonic()
                    if now - last_log >= 1.0:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] 0 wing(s) detected")
                        last_log = now
                    if args.show:
                        if not draw_frame(rgb, [], None):
                            break
                    continue

                pick = select_pick_point(detections, depth)

                # Only require valid depth when the robot needs to move.
                if pick is None or (robot is not None and pick.depth_m <= 0):
                    if args.show:
                        if not draw_frame(rgb, detections, None):
                            break
                    continue

                # Per-second detection log with scores for every wing
                now = time.monotonic()
                if now - last_log >= 1.0:
                    _log_detections(detections, depth, pick.u, pick.v)
                    last_log = now

                if args.show:
                    if not draw_frame(rgb, detections, pick):
                        print("Window closed by user.")
                        break

                if robot is not None:
                    robot_xyz = pixel_to_robot(pick.u, pick.v, pick.depth_m, intrinsics, T)
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
