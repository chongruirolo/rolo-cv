# rolo-cv — Chicken Wing Pick-and-Place Pipeline

Detects individual chicken wings on a cutting board with an RGB-D camera and commands a robot arm to pick and drop them one at a time.

## Hardware

| Component | Model |
|---|---|
| Camera | Orbbec Gemini 336L (USB RGB-D) |
| Robot | AgileX Piper 6-DOF arm |
| Gripper | Finger gripper end-effector |

## How It Works

```
Orbbec Gemini 336L  →  RGB + depth frame
        │
        ▼  vision.py        segment wings, measure height above board via SVD plane fit
        ▼  pick_point.py    score each wing (isolation, height, confidence, flatness)
        ▼  geometry.py      best-wing pixel → robot XYZ via T_cam_to_robot
        ▼  robot_controller pick-and-drop sequence (approach → grip → drop)
        └─ loop until board is empty
```

## Project Structure

```
main.py             Orchestration loop
camera.py           Orbbec Gemini 336L driver
vision.py           YOLO/SAM segmentation + depth height measurement
pick_point.py       Wing scoring and best-pick selection
geometry.py         Pixel → camera → robot coordinate transforms
robot_controller.py AgileX Piper SDK wrapper
visualize.py        Live OpenCV overlay (--show)
drop_tracker.py     Slot-based counter for wings in the drop zone
calibrate.py        Hand-eye calibration utility
collect_data.py     Training data capture
diagnose.py         Single-frame detection diagnostic
config.yaml         All tuning parameters
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Orbbec SDK (not on PyPI)
git clone https://github.com/orbbec/pyorbbecsdk && (cd pyorbbecsdk && pip install -e .)

# Robot SDK
pip install piper-sdk

# Model weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
```

## Running

```bash
python main.py --dry-run --show   # vision only, live OpenCV window, press q to quit
python main.py                    # full run with robot connected
python diagnose.py                # one-shot detection diagnostic → diagnose_output.jpg
```

## Configuration

All tuning lives in `config.yaml`. The most-touched keys:

```yaml
vision:
  backend: "yolo"       # "yolo" = trained wings.pt | "sam" = zero-shot FastSAM
  model_path: "models/wings.pt"
  confidence: 0.1       # detector threshold
  iou: 0.45             # NMS overlap threshold
  roi: [0.35, 0.1, 1.0, 0.85]   # crop region, fractions of frame
  elevation_min_cm: 1.0
  elevation_max_cm: 10.0

drop_tracker:
  max_match_dist_px: 80.0          # Hungarian cap — max wing motion per frame
  decay_frames: 30                 # slot dropped after this many frames with no match
  dedup_enabled: true              # collapse duplicate masks of one wing
  dedup_radius_px: 25.0
  dedup_depth_tolerance_m: 0.012
```

## Detection

Two interchangeable backends in `vision.py`. **FastSAM** is zero-shot and segments everything; an elevation filter (1–10 cm above the board) keeps only board-level objects. **YOLOv8-seg** is the trained model (`models/wings.pt`) — more accurate, requires labelled data. Switch via the `backend` key.

Height per wing is computed by fitting an SVD plane to a dilated ring around each mask and measuring the perpendicular distance. Specular wings (depth = 0 on raw skin) fall back to reading depth from the ring of pixels around the wing.

## Pick Scoring

Every wing is scored on four factors; the highest scorer is sent to the robot.

| Factor | Weight | Why |
|---|---|---|
| Isolation | 35% | Gripper fingers need clearance on both sides |
| Height | 30% | Prefer flat/low wings — more stable to grasp |
| Confidence | 20% | Model certainty |
| Flatness | 15% | Consistent grip |

---

## Counting Wings in the Drop Zone

Counting wings in a box from a top-down camera looks trivial but is the hardest signal-processing problem in the pipeline. Wings stack, occlude each other, the detector flickers, chicken skin scatters IR depth, and mask boundaries wobble. A raw count is noisy; a monotonic count drifts upward and never recovers; and nothing visual can see through a wing to count what's underneath.

### The three-signal philosophy

Instead of asking one detector to do everything, the design uses **three independent measurement systems** with different blind spots. When all agree, the count is trustworthy. When they disagree, the *kind* of disagreement is itself useful information.

| Signal | Measures | Blind to |
|---|---|---|
| **Vision tracker** ✅ | Wings currently visible, tracked over time | Stacked / occluded wings |
| **Robot ledger** 🔜 | Wings the robot has placed | Wings added by hand |
| **Depth volume** 🔜 | Total mass above the box floor | Exact integer counts |

Divergence examples:

> `robot=5, visible=3, volume=5` → two stacked. Robot is right.
> `robot=5, visible=4, volume=4` → a wing fell out.
> `robot=5, visible=7, volume=7` → wings added by hand → operator alert.

Industrial-inspection pattern: **agreement = trustworthy, disagreement = informative**.

### Vision tracker (current)

```
detections → confidence filter → DEDUP (close in XY AND depth)
           → Hungarian assignment → slot snap / append / decay
           → count = number of currently-tracked slots
```

Dedup is the key step. The detector often emits multiple overlapping masks for one wing; without dedup these become duplicate slots. Requiring both image-plane proximity AND depth proximity before collapsing keeps stacked wings (close in XY but different depths) correctly separate.

Hungarian assignment handles wings that move (settling, bumps). Slots that go a full second without a match are dropped — count comes back down when wings are removed.

The HUD shows four diagnostic counters: `Box(raw)` (no memory), `Box(debounce)` (monotonic), `Box(no-dedup)` (tracker with dedup off), `Box(dedup)` (canonical visible count). The gap between the last two is the live value dedup is adding.

### Why no vision-based stack detection

The natural question is "can we just detect stacks from depth?" — and the honest answer is no, for a **physical** reason. The signal you'd need is a ~2 cm depth gap (one wing thickness). But the depth sensor's noise on chicken skin is also 1–2 cm — specular reflection scatters the IR pattern exactly where we need precision. Signal-to-noise is ~2:1.

Every threshold tested either flickered (overcount) or missed real stacks (undercount). No tuning saves this. A top-down view of a perfectly aligned stack contains no information distinguishing 2 wings from 5 — no algorithm can recover information that isn't in the pixels.

The right move is to source the count from a signal that **does** carry it: the robot already knows when it released a wing. That's the upcoming robot ledger.

---

## Training a Custom Model

```bash
python collect_data.py            # space to save a frame, aim for 200–300
```

Label on [roboflow.com](https://roboflow.com) → Instance Segmentation project → upload from `data/images/` → Smart Polygon → export YOLOv8 format → extract into `data/`.

```bash
yolo train model=yolov8n-seg.pt data=data/data.yaml epochs=100 imgsz=640 batch=16 name=wings
cp runs/segment/wings/weights/best.pt wings.pt
```

Then in `config.yaml`: `backend: "yolo"`, `model_path: "wings.pt"`, `confidence: 0.5`.

## Hand-Eye Calibration

`T_cam_to_robot` in `config.yaml` must be set before robot use. Re-run after the camera or robot is moved.

```bash
python calibrate.py
```

Procedure: place a marker on the board, click it in the camera window (captures depth), move the robot end-effector to touch the marker, type its XYZ in metres. Repeat ≥ 4 points across the board at 2 heights. Press `s` to solve (aim for RMS < 10 mm), `y` to save. Press `v` for verify mode.

## Camera Setup

Mount **top-down**, 60–80 cm above the board, fixed rigidly. The full board must fall within the ROI.

The Orbbec returns depth as uint16 mm (pipeline converts ×0.001 to metres). Depth returns zero for specular, very dark, transparent, or steep-angle surfaces, and outside 0.15–3.0 m.

## Roadmap

- [x] RGB-D camera driver + depth scale fix
- [x] FastSAM + YOLOv8-seg backends
- [x] SVD plane fit for wing height
- [x] Pick scoring (isolation-prioritised)
- [x] Hand-eye calibration utility
- [x] Drop-zone vision tracker (Hungarian + slot decay + 2D dedup)
- [x] Live four-counter HUD for tracker debugging
- [ ] Train wings.pt on labelled data
- [ ] Run hand-eye calibration on production rig
- [ ] Robot ledger — primary count from `pick_and_drop()` events
- [ ] Depth volume estimator — mass-based cross-check
- [ ] Divergence detector — alert on robot/visible/volume disagreement
- [ ] Pick outcome logging for RL weight tuning
