"""
Main orchestration loop.

Run: python main.py
     python main.py --dry-run   (skip robot, print coordinates only)
"""

import argparse
import sys
import numpy as np
import yaml

from camera import Camera
from vision import FoodDetector
from pick_point import select_pick_point
from geometry import CameraIntrinsics, pixel_to_robot
from robot_controller import RobotController


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(cfg: dict) -> np.ndarray:
    T = np.array(cfg["camera"]["T_cam_to_robot"], dtype=np.float64)
    assert T.shape == (4, 4), "T_cam_to_robot must be a 4x4 matrix"
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print robot coords without moving")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="models/food_seg.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    T = build_transform(cfg)

    detector = FoodDetector(
        model_path=args.model,
        confidence_threshold=cfg.get("vision", {}).get("confidence_threshold", 0.4),
    )

    if args.dry_run:
        print("[DRY RUN] Robot will not move.")
        robot = None
    else:
        robot = RobotController(can_interface=cfg.get("robot", {}).get("can_interface", "can0"))

    pick_count = 0
    max_picks = cfg.get("max_picks", 50)

    try:
        with Camera(
            color_width=cfg.get("camera", {}).get("width", 1280),
            color_height=cfg.get("camera", {}).get("height", 720),
            fps=cfg.get("camera", {}).get("fps", 30),
        ) as cam:
            intrinsics = cam.intrinsics

            print("Starting pick loop. Ctrl-C to stop.")
            while pick_count < max_picks:
                rgb, depth = cam.capture()
                detections = detector.detect(rgb)

                if not detections:
                    print("No food detected — tray empty or model needs fine-tuning.")
                    break

                pick = select_pick_point(detections, depth)
                if pick is None or pick.depth_m <= 0:
                    print("Could not determine pick point, skipping frame.")
                    continue

                robot_xyz = pixel_to_robot(pick.u, pick.v, pick.depth_m, intrinsics, T)
                print(
                    f"Pick #{pick_count + 1}: pixel=({pick.u},{pick.v}) "
                    f"depth={pick.depth_m:.3f}m  robot={robot_xyz.round(4)}  "
                    f"label={pick.detection.label}  score={pick.score:.2f}"
                )

                if robot is not None:
                    robot.pick_and_drop(robot_xyz)

                pick_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if robot is not None:
            robot.stop()

    print(f"Done. Total picks: {pick_count}")


if __name__ == "__main__":
    main()
