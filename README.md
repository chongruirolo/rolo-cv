# Food Picker — Robotic Food Pick-and-Place

A computer vision pipeline that detects food items on a tray using an RGB-D camera and commands a robot arm to pick and drop them.

## Hardware

| Component | Model |
|-----------|-------|
| Camera | Orbbec Gemini 336L (USB RGB-D) |
| Robot | AgileX Piper 6-DOF arm |
| Gripper | Suction cup |

## Pipeline

```
RGB-D Camera (camera.py)
       ↓
YOLOv8-seg food detection (vision.py)
       ↓
Pick-point scoring — confidence, isolation, depth flatness (pick_point.py)
       ↓
Pixel + depth → 3D camera coord → robot base coord (geometry.py)
       ↓
Pick → lift → drop zone → release (robot_controller.py)
       ↓
Repeat until tray empty
```

## Project Structure

```
food_picker/
├── camera.py           Orbbec Gemini 336L wrapper (pyorbbecsdk)
├── vision.py           YOLOv8-seg instance segmentation
├── pick_point.py       Scoring and pick-point selection
├── geometry.py         Coordinate transforms (pixel → camera → robot)
├── robot_controller.py AgileX Piper SDK wrapper
├── main.py             Orchestration loop
├── config.yaml         Camera intrinsics, transform matrix, thresholds
├── requirements.txt
└── models/
    └── food_seg.pt     Fine-tuned YOLOv8 weights (add after training)
```

## Setup

### 1. Install Python dependencies

```bash
pip install ultralytics opencv-python numpy scipy pyyaml
pip install piper-sdk
```

For the Orbbec camera SDK (not on PyPI):
```bash
git clone https://github.com/orbbec/pyorbbecsdk
cd pyorbbecsdk && pip install -e .
```

### 2. Configure

Edit `config.yaml`:
- Set `T_cam_to_robot` to your hand-eye calibration result (currently identity placeholder)
- Set `drop_zone_xyz` to your actual drop zone in robot base frame
- Adjust `confidence_threshold` if detections are noisy

### 3. Test vision without hardware

```python
from vision import FoodDetector
import cv2

det = FoodDetector()   # downloads yolov8n-seg.pt automatically
results = det.detect(cv2.imread("your_food.jpg"))
for r in results:
    print(r.label, r.confidence, r.bbox)
```

### 4. Run dry-run (no robot)

```bash
cd food_picker
python main.py --dry-run
```

### 5. Full run (robot connected)

```bash
python main.py
```

## Model Training (if base COCO model is insufficient)

1. Collect 200–500 images of your food on the tray
2. Label polygon masks using [Roboflow](https://roboflow.com) — export as YOLOv8 segmentation format
3. Fine-tune:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n-seg.pt")
   model.train(data="food.yaml", epochs=50, imgsz=640)
   ```
4. Copy best weights to `models/food_seg.pt`

## Hand-Eye Calibration

The camera is mounted fixed above the tray ("eye-to-hand"). Before full operation:

1. Place a checkerboard at 4+ known robot positions
2. Run OpenCV's `calibrateHandEye` to solve for `T_cam_to_robot`
3. Paste the 4×4 result into `config.yaml`

## Pick-Point Scoring

Each detected food item is scored on three weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Confidence | 0.4 | Model certainty for this detection |
| Isolation | 0.4 | Distance to nearest neighbour — isolated pieces are easier to pick |
| Depth flatness | 0.2 | Low depth variance inside mask = flat surface = better suction |

Weights are tunable in `pick_point.py`.
