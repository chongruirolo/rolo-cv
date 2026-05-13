"""
Food detection with two swappable backends.  Switch in config.yaml:

  backend: "yolo"
    model_path: "yolov8n-seg.pt"   — detects 80 COCO classes (needs training for wings)
    model_path: "wings.pt"         — your fine-tuned model (best accuracy)

  backend: "sam"
    model_path: "FastSAM-s.pt"     — segments everything, no class labels needed
    model_path: "FastSAM-x.pt"     — larger/slower, more accurate

SAM backend:  segments every distinct object in the frame automatically, assigns
              label="object" to all of them.  Elevation filter (depth sensor) then
              keeps only things physically sitting on the cutting board.  No training
              needed — works immediately.

YOLO backend: detects + classifies objects.  Needs a model trained on your wings
              for reliable results.  Use collect_data.py + Roboflow to build that.

After segmentation, both backends compute depth and height above the cutting board
from the Orbbec depth sensor using SVD plane fitting.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from geometry import CameraIntrinsics


@dataclass
class Detection:
    mask: np.ndarray        # binary uint8, same HxW as input frame
    confidence: float       # model confidence [0–1]
    bbox: tuple             # (x1, y1, x2, y2) in pixels
    label: str              # class name ("object" for SAM backend)
    median_depth_m: float   # median depth of blob in metres (0 = no data)
    height_above_table_m: float  # perpendicular distance above table plane (metres)


# ── 3D geometry helpers ───────────────────────────────────────────────────────

def _backproject(us, vs, depths, intr):
    X = (us.astype(np.float64) - intr.cx) * depths / intr.fx
    Y = (vs.astype(np.float64) - intr.cy) * depths / intr.fy
    return np.stack([X, Y, depths.astype(np.float64)], axis=1)


def _sample_mask_pts(mask, depth, intr, max_pts=400):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    if len(xs) > max_pts:
        idx = np.random.choice(len(xs), max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    ds = depth[ys, xs]
    ok = (ds > 0.05) & (ds < 5.0)
    if ok.sum() < 5:
        return None
    return _backproject(xs[ok], ys[ok], ds[ok], intr)


def _ring_mask(blob, ring_px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    return ((cv2.dilate(blob, k) > 0) & (blob == 0)).astype(np.uint8) * 255


def _fit_plane(pts):
    if len(pts) < 6:
        return None
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1].astype(np.float64)
    normal /= np.linalg.norm(normal)
    if np.dot(normal, centroid) > 0:
        normal = -normal
    return normal, centroid


def _perp_dist(pt, normal, centroid):
    return float(np.dot(normal, pt.astype(np.float64) - centroid))


# ── Segmentation backends ─────────────────────────────────────────────────────

class _YoloBackend:
    """
    YOLOv8 instance segmentation.  Returns labelled detections.
    Needs a model trained on your specific objects for good results.
    """
    def __init__(self, model_path, confidence, iou, device):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf = confidence
        self._iou = iou
        self._device = device
        print(f"[vision] backend=yolo  model={model_path}")

    def segment(self, rgb):
        """Returns list of (mask_uint8, confidence, label, bbox)."""
        h, w = rgb.shape[:2]
        results = self._model(rgb, conf=self._conf, iou=self._iou,
                              device=self._device, verbose=False)
        out = []
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = self._model.names[int(box.cls[0])]
                if r.masks is not None:
                    blob = cv2.resize(
                        r.masks.data[i].cpu().numpy().astype(np.uint8) * 255,
                        (w, h), interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    blob = np.zeros((h, w), dtype=np.uint8)
                    blob[y1:y2, x1:x2] = 255
                out.append((blob, conf, label, (x1, y1, x2, y2)))
        return out


class _SamBackend:
    """
    FastSAM — segments every distinct object automatically, no class labels.
    Works immediately with no training.  Label is always 'object'.
    Elevation filter in FoodDetector then keeps only board-level objects.
    Download weights:  wget https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-s.pt
    """
    def __init__(self, model_path, confidence, iou, device):
        from ultralytics import FastSAM
        self._model = FastSAM(model_path)
        self._conf = confidence
        self._iou = iou
        self._device = device
        print(f"[vision] backend=sam  model={model_path}")

    def segment(self, rgb):
        """Returns list of (mask_uint8, confidence, 'object', bbox)."""
        h, w = rgb.shape[:2]
        results = self._model(
            rgb,
            device=self._device,
            retina_masks=True,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        out = []
        for r in results:
            if r.masks is None:
                continue
            for i, mask_tensor in enumerate(r.masks.data):
                blob = cv2.resize(
                    mask_tensor.cpu().numpy().astype(np.uint8) * 255,
                    (w, h), interpolation=cv2.INTER_NEAREST,
                )
                if not blob.any():
                    continue
                ys, xs = np.where(blob > 0)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                conf = float(r.boxes.conf[i]) if r.boxes is not None else 1.0
                out.append((blob, conf, "object", (x1, y1, x2, y2)))
        return out


# ── Detector ──────────────────────────────────────────────────────────────────

class FoodDetector:
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        backend: str = "yolo",
        intrinsics: CameraIntrinsics | None = None,
        confidence: float = 0.3,
        iou: float = 0.45,
        device: str = "cpu",
        elevation_min_h: float = 0.01,
        elevation_max_h: float = 0.10,
        roi: list | None = None,
    ):
        self._intr = intrinsics
        self._elev_min_h = elevation_min_h
        self._elev_max_h = elevation_max_h
        self._roi = roi  # [x1f, y1f, x2f, y2f] as fractions, or None
        self._table_plane_cache: tuple[np.ndarray, np.ndarray] | None = None

        if backend == "sam":
            self._backend = _SamBackend(model_path, confidence, iou, device)
        else:
            self._backend = _YoloBackend(model_path, confidence, iou, device)

    # ── table plane ───────────────────────────────────────────────────────────

    def _update_table_plane(self, masks, depth):
        if not masks or self._intr is None:
            return
        combined = np.zeros(depth.shape, dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, m)
        if not combined.any():
            return
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        border = ((cv2.dilate(combined, kernel) > 0) & (combined == 0)).astype(np.uint8) * 255
        pts = _sample_mask_pts(border, depth, self._intr, max_pts=800)
        if pts is None:
            return
        plane = _fit_plane(pts)
        if plane is not None:
            self._table_plane_cache = plane

    # ── per-object height ─────────────────────────────────────────────────────

    def _measure_height(self, blob, depth):
        if self._table_plane_cache is None or self._intr is None:
            return 0.0
        normal, centroid = self._table_plane_cache
        pts = _sample_mask_pts(blob, depth, self._intr, max_pts=200)
        if pts is not None:
            return max(0.0, _perp_dist(np.median(pts, axis=0), normal, centroid))
        for ring_px in (14, 30, 50):
            ring = _ring_mask(blob, ring_px)
            pts = _sample_mask_pts(ring, depth, self._intr, max_pts=200)
            if pts is not None:
                return max(0.0, _perp_dist(np.median(pts, axis=0), normal, centroid))
        return 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> list[Detection]:
        h, w = rgb.shape[:2]

        # Crop to ROI before inference — model only sees the chopping board area.
        if self._roi is not None:
            x1f, y1f, x2f, y2f = self._roi
            rx1, ry1 = int(x1f * w), int(y1f * h)
            rx2, ry2 = int(x2f * w), int(y2f * h)
            rgb_in = rgb[ry1:ry2, rx1:rx2]
        else:
            rx1, ry1 = 0, 0
            rgb_in = rgb

        segments = self._backend.segment(rgb_in)

        detections: list[Detection] = []
        blobs: list[np.ndarray] = []

        for blob_crop, conf, label, (bx1, by1, bx2, by2) in segments:
            # Place cropped mask back into a full-frame canvas
            blob = np.zeros((h, w), dtype=np.uint8)
            blob[ry1:ry1 + blob_crop.shape[0], rx1:rx1 + blob_crop.shape[1]] = blob_crop
            bbox = (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1)

            blob_depth = 0.0
            if depth is not None:
                bv = depth[blob > 0]
                bv = bv[(bv > 0.05) & (bv < 5.0)]
                blob_depth = float(np.median(bv)) if len(bv) >= 5 else 0.0

            blobs.append(blob)
            detections.append(Detection(
                mask=blob,
                confidence=conf,
                bbox=bbox,
                label=label,
                median_depth_m=blob_depth,
                height_above_table_m=0.0,
            ))

        # Update table plane then compute height per object
        if depth is not None and blobs:
            self._update_table_plane(blobs, depth)
            for det, blob in zip(detections, blobs):
                det.height_above_table_m = self._measure_height(blob, depth)

        # Keep only objects sitting on the cutting board at the right height.
        # For SAM this is the primary filter replacing class labels.
        # For YOLO this removes background detections at wrong depth.
        if self._table_plane_cache is not None:
            detections = [
                d for d in detections
                if self._elev_min_h <= d.height_above_table_m <= self._elev_max_h
            ]

        return detections
