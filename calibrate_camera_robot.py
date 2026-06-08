"""
Interactive camera-to-robot extrinsic calibration tool.

Usage
-----
1. Place the robot in several (≥4) known poses.
2. Run this script while the RealSense camera is connected.
3. For each frame the live RGB image is shown.
   - Click on a clearly visible robot feature (joint center, flange, TCP …)
   - In the terminal enter the corresponding robot-base-frame coordinates
     in millimetres: X Y Z  (e.g.  234.5 -120.0 450.0)
   - Press ENTER.  The point pair is saved.
4. After ≥4 point pairs press 'c' to solve PnP and save extrinsic.json.
5. Press 'q' to quit without saving.

The resulting extrinsic.json contains:
  "T_cr": 4×4 matrix  (camera-from-robot, transforms robot points → camera frame)
  "R_vec": 3-element Rodrigues rotation vector
  "t_vec": 3-element translation vector (metres)
  "rms_px": re-projection RMS in pixels

References
----------
• Z. Zhang, "A flexible new technique for camera calibration," IEEE TPAMI 2000.
• OpenCV docs: cv2.solvePnP, cv2.calibrateCamera.
"""

import json
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — using webcam fallback (no depth).")

# ---------------------------------------------------------------------------
# Point storage
# ---------------------------------------------------------------------------
_img_pts: List[np.ndarray] = []   # pixel coords (u, v)
_obj_pts: List[np.ndarray] = []   # robot-base coords (X, Y, Z) in metres
_last_click: Optional[Tuple[int, int]] = None
_WINDOW = "Calibration  [click→point | c=solve | r=reset | q=quit]"
_input_active: bool = False
_input_result: Optional[np.ndarray] = None
_input_error: Optional[str] = None


def _mouse_cb(event, x, y, _flags, _param):
    global _last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _last_click = (x, y)
        print(f"[CLICK] pixel ({x}, {y}) — enter robot XYZ in mm, then press ENTER:")


def _read_xyz_input(click_pos: Tuple[int, int]) -> None:
    """Read XYZ from terminal in a background thread."""
    global _input_active, _input_result, _input_error
    try:
        raw = input(f"  XYZ (mm) for click {click_pos} [X Y Z]: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            _input_error = "Expected 3 numbers"
            _input_result = None
        else:
            _input_result = np.array([float(p) / 1000.0 for p in parts], dtype=np.float64)
            _input_error = None
    except Exception as e:
        _input_result = None
        _input_error = str(e)
    finally:
        _input_active = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_points(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for i, (uv, xyz) in enumerate(zip(_img_pts, _obj_pts)):
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(out, (u, v), 6, (0, 255, 0), 2)
        cv2.putText(out, f"P{i+1}", (u + 8, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if _last_click is not None:
        cv2.circle(out, _last_click, 8, (0, 120, 255), 2)
    n = len(_img_pts)
    status = f"{n} point(s) collected — press c to solve (need >=4)" if n < 4 else \
             f"{n} point(s) — press c to solve PnP or keep adding"
    cv2.putText(out, status, (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return out


def _draw_cross(img: np.ndarray, pt: Tuple[int, int], color=(0, 120, 255), size: int = 10) -> None:
    x, y = pt
    cv2.line(img, (x - size, y), (x + size, y), color, 2)
    cv2.line(img, (x, y - size), (x, y + size), color, 2)


def _solve_and_save(
    intrinsics: dict,
    out_path: Path,
    img_size: Tuple[int, int],
) -> bool:
    if len(_img_pts) < 4:
        print("[ERROR] Need at least 4 point correspondences.")
        return False

    obj = np.array(_obj_pts, dtype=np.float64)          # (N,3) metres
    img = np.array(_img_pts, dtype=np.float64)          # (N,2)

    cam_mat = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]],
         [0, intrinsics["fy"], intrinsics["cy"]],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)

    ok, r_vec, t_vec, inl = cv2.solvePnPRansac(
        obj, img, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        print("[ERROR] solvePnPRansac failed — check your point correspondences.")
        return False

    # Refine with LM using inliers
    if inl is not None and len(inl) >= 4:
        inl_idx = inl.flatten()
        r_vec, t_vec = cv2.solvePnPRefineLM(
            obj[inl_idx], img[inl_idx], cam_mat, dist, r_vec, t_vec,
        )

    # Build 4×4 extrinsic matrix T_cr  (camera-from-robot)
    R, _ = cv2.Rodrigues(r_vec)
    T_cr = np.eye(4, dtype=np.float64)
    T_cr[:3, :3] = R
    T_cr[:3, 3] = t_vec.flatten()

    # Compute re-projection RMS
    proj, _ = cv2.projectPoints(obj, r_vec, t_vec, cam_mat, dist)
    proj = proj.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))

    print(f"\n[OK] Solved PnP  RMS = {rms:.2f} px  (inliers={len(inl) if inl is not None else 'N/A'})")
    print(f"     R_vec = {r_vec.flatten()}")
    print(f"     t_vec = {t_vec.flatten()}")
    print("     T_cr =")
    print(np.round(T_cr, 6))

    data = {
        "T_cr": T_cr.tolist(),
        "R_vec": r_vec.flatten().tolist(),
        "t_vec": t_vec.flatten().tolist(),
        "rms_px": rms,
        "n_points": len(_img_pts),
        "intrinsics": intrinsics,
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\n[SAVED] {out_path.resolve()}")
    return True


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _open_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    intr_raw = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .intrinsics
    )
    intrinsics = {
        "fx": intr_raw.fx,
        "fy": intr_raw.fy,
        "cx": intr_raw.ppx,
        "cy": intr_raw.ppy,
        "width": intr_raw.width,
        "height": intr_raw.height,
    }
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    return pipeline, align, intrinsics, depth_scale


def _get_frame_rs(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    return color, depth


def _open_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def _get_frame_webcam(cap):
    ok, frame = cap.read()
    return (frame, None) if ok else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _last_click, _img_pts, _obj_pts, _input_active, _input_result, _input_error

    out_path = Path("extrinsic.json")
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])

    pipeline = align = cap = None
    intrinsics = None
    depth_scale = 0.001

    if REALSENSE_AVAILABLE:
        try:
            pipeline, align, intrinsics, depth_scale = _open_realsense()
            print("[OK] RealSense camera opened.")
        except Exception as e:
            print(f"[WARN] RealSense failed ({e}), trying webcam.")
    if pipeline is None:
        cap = _open_webcam()
        if cap is None or not cap.isOpened():
            print("[ERROR] No camera available.")
            sys.exit(1)
        h, w = 480, 640
        intrinsics = {
            "fx": 600.0, "fy": 600.0,
            "cx": w / 2.0, "cy": h / 2.0,
            "width": w, "height": h,
        }
        print("[WARN] Using webcam with estimated intrinsics fx=fy=600.")
        print("       For accurate calibration use RealSense or provide real intrinsics.")

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WINDOW, _mouse_cb)

    print("\n=== Extrinsic Calibration ===")
    print("Click on a robot feature → enter robot XYZ (mm) in terminal → ENTER")
    print("Keys:  c = solve & save    r = reset points    q = quit\n")

    pending_click = None
    pending_depth_m: Optional[float] = None
    frozen_frame: Optional[np.ndarray] = None
    input_thread: Optional[threading.Thread] = None

    while True:
        if _input_active and frozen_frame is not None:
            frame = frozen_frame.copy()
            depth = None
        else:
            if pipeline is not None:
                rs_data = _get_frame_rs(pipeline, align)
                if rs_data is None:
                    continue
                frame, depth = rs_data
            else:
                wc_data = _get_frame_webcam(cap)
                if wc_data is None:
                    continue
                frame, depth = wc_data
        if frame is None:
            continue

        display = _draw_points(frame)

        # Start async input after click, while freezing image
        if _last_click is not None and _last_click != pending_click:
            pending_click = _last_click
            frozen_frame = frame.copy()
            pending_depth_m = None
            if depth is not None:
                u, v = pending_click
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    z_raw = int(depth[v, u])
                    if z_raw > 0:
                        pending_depth_m = z_raw * depth_scale
            _input_result = None
            _input_error = None
            _input_active = True
            input_thread = threading.Thread(
                target=_read_xyz_input,
                args=(pending_click,),
                daemon=True,
            )
            input_thread.start()

        if pending_click is not None:
            _draw_cross(display, pending_click)
            if pending_depth_m is not None:
                cv2.putText(
                    display,
                    f"click depth: {pending_depth_m:.3f} m",
                    (pending_click[0] + 10, max(20, pending_click[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Finalize point after async terminal input is done
        if pending_click is not None and not _input_active and input_thread is not None:
            input_thread = None
            if _input_result is not None:
                _img_pts.append(np.array(pending_click, dtype=np.float64))
                _obj_pts.append(_input_result.copy())
                print(f"  [STORED] P{len(_img_pts)}: pixel={pending_click}  robot={_input_result} m")
            else:
                print(f"  [SKIP] Input error: {_input_error or 'invalid XYZ'}")
            pending_click = None
            pending_depth_m = None
            _last_click = None
            frozen_frame = None

        if _input_active:
            cv2.putText(
                display,
                "Waiting for XYZ in terminal... window stays responsive",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _img_pts.clear()
            _obj_pts.clear()
            pending_click = None
            print("[RESET] All points cleared.")
        elif key == ord("c"):
            img_h, img_w = frame.shape[:2]
            _solve_and_save(intrinsics, out_path, (img_w, img_h))

    cv2.destroyAllWindows()
    if pipeline is not None:
        pipeline.stop()
    elif cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
