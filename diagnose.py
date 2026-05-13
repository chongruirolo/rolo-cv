"""
Quick diagnostic: captures one frame and runs YOLO on it, printing every
detection regardless of confidence threshold so you can see what the model
is finding and at what scores.

Run:  python diagnose.py
"""

import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from camera import Camera


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg.get("camera", {})
    vis_cfg = cfg.get("vision", {})
    model_path = vis_cfg.get("model_path", "yolov8n-seg.pt")

    print(f"Model: {model_path}")

    model = YOLO(model_path)

    with Camera(
        color_width=cam_cfg.get("width", 1280),
        color_height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    ) as cam:
        print("Capturing frame...")
        rgb, depth = cam.capture()

    # Run at very low confidence to see everything the model considers
    results = model(rgb, conf=0.01, iou=0.45, verbose=False)

    print(f"\n{'─'*60}")
    print(f"Detections at conf >= 0.01 (no threshold filtering):")
    print(f"{'─'*60}")

    total = 0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {label:20s}  conf={conf:.3f}  bbox=({x1},{y1},{x2},{y2})")
            total += 1

    if total == 0:
        print("  No detections at all — model may not recognise this scene.")
    print(f"{'─'*60}")
    print(f"Total: {total} detection(s)")
    print(f"\nYour current confidence threshold: {vis_cfg.get('confidence', 0.3)}")

    # Save annotated frame for inspection
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results_vis = model(rgb, conf=0.01, iou=0.45, verbose=False)
    annotated = results_vis[0].plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite("diagnose_output.jpg", annotated_bgr)
    print("Annotated frame saved to diagnose_output.jpg")


if __name__ == "__main__":
    main()
