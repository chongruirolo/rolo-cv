"""
Hand-eye calibration for the Orbbec Gemini 336L → robot base frame transform.

Procedure
---------
1. Run:   python calibrate.py
2. The camera window opens.  Blue = near, red = far in the depth overlay.
3. Move the robot end-effector (or a calibration pin) to N ≥ 4 positions
   spread across the work area (different X, Y, and ideally two heights).
4. At each position:
     a. Left-click the pixel where the tip appears in the camera window.
     b. At the terminal prompt, enter the robot XYZ in metres
        (read from the teach pendant or robot controller).
5. Press 's' to solve and write T_cam_to_robot to config.yaml.
   Press 'u' to undo the last correspondence.
   Press 'v' to enter verify mode (click any pixel to read live depth).
   Press 'q' to quit without saving.

The solver uses Horn's SVD method (closed-form rigid-body fit) and prints the
RMS residual in mm.  Aim for < 10 mm before trusting the result.

Tips
----
- Spread points across the full XY work area, not just one corner.
- Include at least two distinct Z heights for a well-conditioned solution.
- If RMS > 20 mm, check that you are reading robot XYZ in metres, not mm.
"""

import sys
import yaml
import numpy as np
import cv2

from camera import Camera
from geometry import CameraIntrinsics, pixel_to_camera


# ── Rigid-body fit ──────────────────────────────────────────────────────────

def solve_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find R, t minimising ||dst - (R @ src + t)||  (Horn/SVD method).
    src, dst: (N, 3) arrays of 3-D correspondences.
    Returns R (3×3) and t (3,).
    """
    src_c, dst_c = src.mean(0), dst.mean(0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:       # flip to avoid reflection
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    return R, t


def rms_mm(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    pred = (R @ src.T).T + t
    return float(np.sqrt(((pred - dst) ** 2).sum(1).mean()) * 1000)


# ── Depth helpers ────────────────────────────────────────────────────────────

def sample_depth(depth: np.ndarray, u: int, v: int, r: int = 6) -> float:
    """Median of valid pixels in a (2r+1)² patch — robust to single-pixel dropouts."""
    h, w = depth.shape
    patch = depth[max(0, v - r):min(h, v + r + 1),
                  max(0, u - r):min(w, u + r + 1)]
    valid = patch[(patch > 0.05) & (patch < 5.0)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def colorise_depth(depth: np.ndarray) -> np.ndarray:
    """Return a uint8 BGR depth-colourmap (blue=near, red=far, black=invalid)."""
    valid = depth > 0.05
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    d_min, d_max = depth[valid].min(), depth[valid].max()
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    coloured = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)  # near=blue, far=red
    coloured[~valid] = 0
    return coloured


# ── Visualisation ────────────────────────────────────────────────────────────

_WINDOW = "calibrate  |  click=mark  s=solve  u=undo  v=verify  q=quit"


def _render(
    rgb: np.ndarray,
    depth: np.ndarray,
    pixels: list[tuple[int, int]],
    verify_mode: bool,
    verify_info: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_col = colorise_depth(depth)
    valid = (depth > 0.05)[..., None]
    frame = np.where(valid, cv2.addWeighted(bgr, 0.6, depth_col, 0.4, 0), bgr)

    # Draw stored correspondences
    for i, (u, v) in enumerate(pixels):
        cv2.circle(frame, (u, v), 9, (0, 255, 0), 2)
        cv2.putText(frame, str(i + 1), (u + 12, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Status bar
    n = len(pixels)
    if verify_mode:
        status = f"VERIFY MODE  |  {verify_info}  |  v=exit verify  q=quit"
    else:
        status = (f"{n} point(s) collected  |  need >= 4"
                  if n < 4 else
                  f"{n} point(s) collected  |  s=solve & save  u=undo  q=quit")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(frame, status, (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cam_cfg = cfg.get("camera", {})

    # Shared mutable state accessed from the mouse callback
    state = {
        "click": None,          # pending (u, v) from mouse
        "depth": None,          # current depth frame
        "verify_mode": False,
        "verify_info": "",
    }

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    pixels: list[tuple[int, int]] = []
    pts_cam: list[np.ndarray] = []
    pts_robot: list[np.ndarray] = []

    with Camera(
        color_width=cam_cfg.get("width", 1280),
        color_height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    ) as cam:
        intr: CameraIntrinsics = cam.intrinsics
        print(f"\nCamera intrinsics  fx={intr.fx:.2f}  fy={intr.fy:.2f}"
              f"  cx={intr.cx:.2f}  cy={intr.cy:.2f}")
        print(__doc__)

        cv2.namedWindow(_WINDOW)
        cv2.setMouseCallback(_WINDOW, on_mouse)

        while True:
            rgb, depth = cam.capture()
            state["depth"] = depth

            frame = _render(rgb, depth, pixels,
                            state["verify_mode"], state["verify_info"])
            cv2.imshow(_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ────────────────────────────────────────────────
            if key == ord('q'):
                print("Quit — config.yaml unchanged.")
                break

            if key == ord('v'):
                state["verify_mode"] = not state["verify_mode"]
                state["verify_info"] = ""
                print("Verify mode ON — click any pixel to read depth." if state["verify_mode"]
                      else "Verify mode OFF.")

            if key == ord('u'):
                if pixels:
                    pixels.pop(); pts_cam.pop(); pts_robot.pop()
                    print(f"Undone. {len(pixels)} correspondence(s) remain.")
                else:
                    print("Nothing to undo.")

            if key == ord('s'):
                n = len(pts_cam)
                if n < 4:
                    print(f"Need >= 4 correspondences (have {n}). Keep collecting.")
                else:
                    src = np.array(pts_cam)
                    dst = np.array(pts_robot)
                    R, t = solve_rigid(src, dst)
                    err = rms_mm(src, dst, R, t)

                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t

                    print("\n" + "=" * 60)
                    print(f"T_cam_to_robot  (RMS residual = {err:.1f} mm)")
                    print(np.round(T, 6))
                    if err > 20:
                        print("WARNING: residual > 20 mm — check units (must be metres).")
                    print("=" * 60)

                    # Per-point residuals for diagnosis
                    pred = (R @ src.T).T + t
                    for i, (p, q, r_) in enumerate(zip(src, dst, pred)):
                        e_mm = np.linalg.norm(r_ - q) * 1000
                        print(f"  point {i+1}: robot={q.round(4)}  pred={r_.round(4)}"
                              f"  err={e_mm:.1f} mm")

                    answer = input("\nSave to config.yaml? [y/N] ").strip().lower()
                    if answer == "y":
                        cfg["camera"]["T_cam_to_robot"] = T.tolist()
                        with open(args.config, "w") as f:
                            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                        print(f"Saved to {args.config}.")
                        break
                    else:
                        print("Not saved. You can collect more points or press 'q'.")

            # ── Mouse click ─────────────────────────────────────────────────
            if state["click"] is not None:
                u, v = state["click"]
                state["click"] = None
                d = sample_depth(state["depth"], u, v)

                if state["verify_mode"]:
                    p_cam = pixel_to_camera(u, v, d, intr) if d > 0 else None
                    if d > 0:
                        state["verify_info"] = (f"({u},{v})  depth={d:.4f} m"
                                                f"  cam_xyz={p_cam.round(3)}")
                        print(f"  pixel ({u},{v})  depth={d:.4f} m"
                              f"  cam_xyz={p_cam.round(4)}")
                    else:
                        state["verify_info"] = f"({u},{v})  NO DEPTH"
                        print(f"  pixel ({u},{v}): no valid depth reading.")
                    continue

                # Normal correspondence collection
                if d <= 0:
                    print(f"  pixel ({u},{v}): no valid depth — try a nearby spot.")
                    continue

                p_cam = pixel_to_camera(u, v, d, intr)
                print(f"\nPoint {len(pts_cam)+1}  pixel=({u},{v})  "
                      f"depth={d:.4f} m  cam_xyz={p_cam.round(4)}")
                raw = input("  Robot XYZ in metres [X Y Z]: ").strip()
                try:
                    vals = [float(x) for x in raw.split()]
                    assert len(vals) == 3
                except (ValueError, AssertionError):
                    print("  Bad input — need exactly 3 numbers. Try again.")
                    continue

                p_robot = np.array(vals, dtype=np.float64)
                pts_cam.append(p_cam)
                pts_robot.append(p_robot)
                pixels.append((u, v))
                print(f"  Stored. {len(pts_cam)} correspondence(s) so far.")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
