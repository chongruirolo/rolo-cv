"""
Training data collection for YOLOv8 fine-tuning.

Run:   python collect_data.py

Controls:
  SPACE  — save current frame
  q      — quit

Frames are saved to data/images/  as JPGs.
After collecting ~200-300 images, label them at https://roboflow.com or
using CVAT, export as YOLOv8 format, then train with:

  yolo train model=yolov8n-seg.pt data=wings.yaml epochs=100 imgsz=640

Tips:
  - Vary the number of wings per frame (1 wing, 3 wings, pile of wings)
  - Include touching and overlapping wings
  - Capture at different times of day to get lighting variation
  - Include partial wings at the edge of frame
  - Aim for at least 200 frames before labelling
"""

import argparse
import os
import time

import cv2
import yaml

from camera import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default="data/images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cam_cfg = cfg.get("camera", {})
    roi = cfg.get("vision", {}).get("roi")

    existing = [f for f in os.listdir(args.out_dir) if f.endswith(".jpg")]
    count = len(existing)

    roi_msg = f"ROI crop {roi}" if roi else "full frame (no ROI set)"
    print(f"Saving frames to {args.out_dir}/  [{roi_msg}]")
    print(f"{count} frames already collected.")
    print("Press SPACE to save a frame, q to quit.")

    window = "collect_data  |  SPACE=save  q=quit"

    with Camera(
        color_width=cam_cfg.get("width", 1280),
        color_height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    ) as cam:
        cv2.namedWindow(window)

        while True:
            rgb, _ = cam.capture()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Crop to ROI if set — saved images match exactly what the model sees
            if roi is not None:
                h, w = bgr.shape[:2]
                x1f, y1f, x2f, y2f = roi
                rx1, ry1 = int(x1f * w), int(y1f * h)
                rx2, ry2 = int(x2f * w), int(y2f * h)
                save_frame = bgr[ry1:ry2, rx1:rx2]
                # Draw ROI boundary on the preview only
                preview = bgr.copy()
                cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            else:
                save_frame = bgr
                preview = bgr

            # Status overlay on preview
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 32), (30, 30, 30), -1)
            cv2.putText(
                preview, f"Saved: {count}  |  SPACE=save  q=quit  [{roi_msg}]",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
            )

            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                ts = int(time.time() * 1000)
                path = os.path.join(args.out_dir, f"frame_{ts}.jpg")
                cv2.imwrite(path, save_frame)
                count += 1
                print(f"  [{count}] saved {path}")

    cv2.destroyAllWindows()
    print(f"\nDone. {count} total frames in {args.out_dir}/")
    print("Next step: label them at https://roboflow.com, export as YOLOv8 format.")


if __name__ == "__main__":
    main()
