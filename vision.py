"""
Food instance segmentation using YOLOv8-seg.

Usage:
    detector = FoodDetector("models/food_seg.pt")
    detections = detector.detect(rgb_frame)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    mask: np.ndarray        # binary mask, same HxW as input frame
    confidence: float
    bbox: tuple             # (x1, y1, x2, y2) in pixels
    label: str


class FoodDetector:
    def __init__(self, model_path: str = "yolov8n-seg.pt", confidence_threshold: float = 0.4):
        from ultralytics import YOLO
        path = Path(model_path)
        # fall back to downloading the base COCO model if custom weights don't exist yet
        self._model = YOLO(str(path) if path.exists() else "yolov8n-seg.pt")
        self._conf = confidence_threshold

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run inference on an HxWx3 uint8 RGB frame. Returns a list of Detection."""
        results = self._model(rgb, conf=self._conf, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.masks is None:
                continue
            masks_data = result.masks.data.cpu().numpy()   # (N, H, W) float32 0-1
            boxes = result.boxes

            for i in range(len(boxes)):
                mask_bin = (masks_data[i] > 0.5).astype(np.uint8)

                # YOLOv8 masks may be at a different resolution; resize to match input
                if mask_bin.shape[:2] != rgb.shape[:2]:
                    import cv2
                    mask_bin = cv2.resize(
                        mask_bin, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i])
                label = self._model.names.get(cls_id, str(cls_id))

                detections.append(Detection(
                    mask=mask_bin,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    label=label,
                ))

        return detections
