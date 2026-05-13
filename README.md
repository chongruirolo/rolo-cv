# rolo-cv — Chicken Wing Pick-and-Place Pipeline

A computer vision pipeline that detects individual chicken wings on a cutting board using an RGB-D camera and commands a robot arm with a finger gripper to pick and drop them one at a time.

---

## Hardware

| Component | Model |
|---|---|
| Camera | Orbbec Gemini 336L (USB RGB-D) |
| Robot | AgileX Piper 6-DOF arm |
| Gripper | Finger gripper end-effector |

---

## How It Works

Every frame the pipeline does the following:

```
Orbbec Gemini 336L
        │
        ├── RGB image (1280×720)
        └── Depth image (float32, metres)
        │
        ▼
vision.py — segment each wing
  • Crop frame to ROI (cutting board area only)
  • Run YOLO or FastSAM to get per-wing masks
  • For each wing: read median depth from sensor
  • Fit a 3D plane to the cutting board surface (SVD)
  • Compute each wing's height above the board
  • Filter: keep only objects 1–10 cm above the board
        │
        ▼
pick_point.py — score every wing, pick the best
  • Isolation   (35%) — clearance for gripper fingers
  • Height      (30%) — prefer flat/low wings over stacked
  • Confidence  (20%) — how certain the model is
  • Flatness    (15%) — reasonably flat surface for grip
        │
        ▼
geometry.py — convert pixel to robot coordinate
  • pixel + depth → 3D camera frame (using intrinsics)
  • camera frame → robot base frame (using T_cam_to_robot)
        │
        ▼
robot_controller.py — physical pick sequence
  1. Open gripper
  2. Hover 10 cm above target
  3. Descend to wing
  4. Close gripper, dwell 0.3s
  5. Lift back up
  6. Move to drop zone
  7. Open gripper to release
  8. Return home
        │
        └── Repeat until board is empty
```

---

## Project Structure

```
rolo-cv/
├── main.py             Orchestration loop — ties everything together
├── camera.py           Orbbec Gemini 336L driver (pyorbbecsdk)
├── vision.py           YOLO/SAM segmentation + depth height measurement
├── pick_point.py       Wing scoring and best-pick selection
├── geometry.py         Coordinate transforms (pixel → camera → robot)
├── robot_controller.py AgileX Piper SDK wrapper (pick-and-drop sequence)
├── visualize.py        Live OpenCV overlay (--show mode)
├── calibrate.py        Hand-eye calibration utility (run once)
├── collect_data.py     Training data capture utility
├── diagnose.py         Model diagnostic — shows raw detections at low confidence
├── config.yaml         All tuning parameters
└── requirements.txt
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/chongruirolo/rolo-cv.git
cd rolo-cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Orbbec camera SDK

Not on PyPI — must be built from source:

```bash
git clone https://github.com/orbbec/pyorbbecsdk
cd pyorbbecsdk && pip install -e .
```

### 3. Install robot SDK

```bash
pip install piper-sdk
```

### 4. Download model weights

```bash
# FastSAM (works immediately, no training needed)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-x.pt

# YOLOv8 base model (needed for fine-tuning)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
```

---

## Running

### Test vision only (no robot)

```bash
python main.py --dry-run --show
```

`--dry-run` skips all robot movement. `--show` opens a live OpenCV window with detections overlaid. Press `q` to quit.

### Full run (robot connected)

```bash
python main.py
```

### Diagnose what the model is detecting

```bash
python diagnose.py
```

Captures one frame, runs the model at near-zero confidence threshold, prints every detection with its score, and saves an annotated image to `diagnose_output.jpg`.

---

## Configuration — config.yaml

All tuning parameters are in `config.yaml`. No code changes needed to adjust behaviour.

```yaml
camera:
  width: 1280
  height: 720
  fps: 30
  T_cam_to_robot:       # 4x4 matrix from calibrate.py — MUST be set before robot use

vision:
  backend: "sam"        # "sam" = no training needed | "yolo" = trained model
  model_path: "FastSAM-x.pt"   # or "wings.pt" after training
  confidence: 0.1       # minimum detection confidence [0–1]
  iou: 0.45             # NMS overlap threshold [0–1]
  device: "cpu"         # "cpu" | "cuda" | "mps"
  roi: [0.2, 0.3, 0.7, 0.6]   # crop region [x1, y1, x2, y2] as frame fractions
  elevation_min_cm: 1.0        # minimum height above board to count as a wing
  elevation_max_cm: 10.0       # maximum height (filters out background objects)
```

### Switching between backends

```yaml
# FastSAM — works immediately, no training, segments everything in ROI
backend: "sam"
model_path: "FastSAM-x.pt"

# Trained YOLO — best accuracy, requires wings.pt from training step
backend: "yolo"
model_path: "wings.pt"
```

---

## Detection Backends

### FastSAM (current, no training needed)

FastSAM segments every distinct object in the frame automatically with no class labels. The elevation filter then keeps only objects sitting 1–10 cm above the cutting board surface. Works immediately but will also segment non-wing objects inside the ROI.

### YOLOv8-seg (recommended for production)

A fine-tuned YOLOv8 segmentation model trained specifically on your chicken wings. Gives labelled, reliable detections in any lighting condition. Requires training data — see Training section below.

---

## Height Measurement

The pipeline computes how far above the cutting board each wing is sitting. This is used to distinguish flat wings (on the board) from stacked wings (on top of another wing).

Method:
1. Dilate the combined wing mask outward — this reveals the cutting board surface around the wings
2. Back-project those border pixels into 3D using camera intrinsics
3. Fit a plane to those 3D points using SVD (handles camera tilt automatically)
4. For each wing, compute the perpendicular distance from the wing to the fitted plane

For specular wings (shiny skin reflects IR → depth = 0), the pipeline reads depth from a ring of pixels around the wing blob instead of the wing surface itself.

Output in logs:
- `[flat]` — height 0–3 cm, wing is on the board
- `[STACKED]` — height > 3 cm, wing is on top of another wing

---

## Pick Scoring

Every detected wing is scored on four factors. The highest-scoring wing is sent to the robot.

| Factor | Weight | Why |
|---|---|---|
| Isolation | 35% | Gripper fingers need clearance on both sides |
| Height | 30% | Prefer flat/low wings — more stable to grasp |
| Confidence | 20% | Model certainty |
| Flatness | 15% | Reasonably flat surface helps consistent grip |

Isolation is weighted highest because gripper fingers need clear space on both sides of the wing to close. Weights are in `pick_point.py` and will be tuned over time using pick outcome data.

---

## Training a Custom Model

The trained model will significantly outperform FastSAM for your specific setup.

### Step 1 — Collect images

```bash
python collect_data.py
```

Press **space** to save a frame. Aim for 200–300 images. Vary wing count and arrangement — full board, small groups, touching wings, stacked wings.

### Step 2 — Label on Roboflow

1. Create a free account at [roboflow.com](https://roboflow.com)
2. New project → **Instance Segmentation** → name it `wings`
3. Upload all images from `data/images/`
4. Use the **Smart Polygon tool** to trace around each wing → label `wing`
5. After 20 manual labels, use **Auto-Label** to pre-annotate the rest
6. Export → **YOLOv8 format** → download zip → extract into `data/`

### Step 3 — Train

```bash
yolo train \
  model=yolov8n-seg.pt \
  data=data/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=wings
```

Training takes ~1 hour on a laptop GPU or 20-40 minutes on Google Colab (free).

### Step 4 — Deploy

```bash
cp runs/segment/wings/weights/best.pt wings.pt
```

Update `config.yaml`:
```yaml
backend: "yolo"
model_path: "wings.pt"
confidence: 0.5
```

---

## Hand-Eye Calibration

Required before the robot can move to the correct position. The `T_cam_to_robot` matrix in `config.yaml` is currently an identity placeholder and must be replaced with a real calibration result.

```bash
python calibrate.py
```

**Procedure:**
1. A window opens showing the live camera feed with depth overlay
2. Move the robot end-effector to a position on the cutting board
3. Left-click the tip in the camera window
4. Type the robot XYZ in metres at the terminal prompt (read from teach pendant)
5. Repeat for at least 4 positions spread across the board at 2 different heights
6. Press `s` to solve — aim for RMS residual < 10 mm
7. Type `y` to save the result to `config.yaml`

Press `v` at any time to enter verify mode — click any pixel to see its depth and 3D camera coordinates.

**Re-run calibration whenever:**
- The camera is physically moved or knocked
- The robot is remounted or repositioned

---

## Camera Notes

The Orbbec Gemini 336L outputs depth as uint16 millimetres. The pipeline converts to metres by multiplying by 0.001 (hardcoded — do not change for this sensor model).

Depth returns zero (no reading) for:
- Shiny/specular surfaces (raw chicken skin reflects IR)
- Very dark surfaces (absorb IR)
- Transparent surfaces
- Surfaces at steep angles to the sensor
- Objects outside the 0.15–3.0 m range

The pipeline handles specular wings by reading depth from the cutting board surface immediately around each wing instead of the wing surface itself.

---

## Camera Mounting

- Mount **top-down**, pointing straight at the board
- Height: **60–80 cm** above the board surface
- Fix rigidly — any movement after calibration requires recalibration
- Ensure the full cutting board is visible within the ROI

---

## Roadmap

- [x] RGB-D camera driver with depth scale fix
- [x] FastSAM zero-shot segmentation backend
- [x] YOLOv8-seg backend (awaiting training data)
- [x] SVD plane fitting for height measurement
- [x] Pick scoring (isolation-prioritised for gripper)
- [x] ROI crop to restrict detection area
- [x] Hand-eye calibration utility
- [x] Training data collection utility
- [ ] Collect and label training dataset
- [ ] Train wings.pt on labelled data
- [ ] Run hand-eye calibration
- [ ] Pick outcome logging for RL weight tuning
