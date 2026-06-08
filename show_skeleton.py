"""
Live FK-skeleton overlay — verify extrinsic.json on the real camera frame.

Loads the camera-from-robot extrinsic (extrinsic.json), reads live joint angles
from the KUKA controller, projects the FK skeleton (base + J1..J6 + TCP) onto the
video frame and draws it. Use it to visually check calibration quality:

  • If the skeleton lands on the real robot at every height  → calibration good.
  • If the top joints (J2/J3) drift while TCP fits             → depth/tilt error,
    re-calibrate with points at different heights (option A: multi-joint clicks).

Usage
-----
  python show_skeleton.py
  python show_skeleton.py --extrinsic extrinsic.json --gripper-length-mm 137
  python show_skeleton.py --no-robot          # static demo with fixed angles

Keys:  q = quit    s = save current frame to skeleton_check.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from kuka_fk import joints_to_uvs

# reuse robot connection + angle parser from the calibration tool
from calibrate_click_fk import _connect_robot, _read_joint_angles

try:
    from gripper_detection import draw_fk_projections
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — using webcam.")

_WINDOW = "Skeleton Check  [q=quit | s=save frame]"

# Fallback pose if robot not connected (the one from the kuka_fk self-test)
_DEMO_ANGLES = np.array([-59.884, -26.499, 120.533, -0.854, -90.858, 2.948])

_JOINT_NAMES = ["Base", "J1", "J2", "J3", "J4", "J5(W)", "J6(G)", "TCP"]


# ── extrinsic ─────────────────────────────────────────────────────────────────

def _load_extrinsic(path: Path) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """Load T_cr (4×4) and intrinsics from extrinsic.json."""
    data = json.loads(path.read_text())
    T_cr = np.array(data["T_cr"], dtype=np.float64)
    assert T_cr.shape == (4, 4), "T_cr must be 4×4"
    intr = data.get("intrinsics")
    rms = data.get("rms_px", "?")
    print(f"[OK] Loaded {path}  (RMS was {rms} px)")
    return T_cr, intr


# ── camera ────────────────────────────────────────────────────────────────────

def _open_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .intrinsics
    )
    return pipeline, {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.ppx, "cy": intr.ppy,
        "width": intr.width, "height": intr.height,
    }


def _get_rs_frame(pipeline) -> Optional[np.ndarray]:
    frames = pipeline.wait_for_frames()
    cf = frames.get_color_frame()
    return np.asanyarray(cf.get_data()) if cf else None


def _get_webcam_frame(cap) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    return frame if ok else None


# ── simple skeleton drawing (fallback if gripper_detection unavailable) ─────────

def _draw_skeleton_basic(img, uvs, names):
    h, w = img.shape[:2]
    colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (255, 0, 255),
              (0, 255, 255), (255, 128, 0), (128, 255, 0), (0, 0, 255)]
    for i in range(len(uvs) - 1):
        a, b = uvs[i], uvs[i + 1]
        if a is None or b is None:
            continue
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (255, 200, 0), 2, cv2.LINE_AA)
    for i, uv in enumerate(uvs):
        if uv is None:
            continue
        u, v = int(uv[0]), int(uv[1])
        c = colors[i % len(colors)]
        cv2.circle(img, (u, v), 7, c, 2, cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, c, -1)
        lbl = names[i] if i < len(names) else f"J{i}"
        cv2.putText(img, lbl, (u + 9, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrinsic", default="extrinsic.json")
    ap.add_argument("--robot-ip", default="192.168.17.2")
    ap.add_argument("--robot-port", type=int, default=7000)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--gripper-length-mm", type=float, default=137.0,
                    help="Tool length flange→TCP tip in mm (match your calibration)")
    args = ap.parse_args()

    ext_path = Path(args.extrinsic)
    if not ext_path.exists():
        print(f"[ERROR] {ext_path} not found. Run calibration first.")
        sys.exit(1)

    T_cr, intr_json = _load_extrinsic(ext_path)
    gripper_m = args.gripper_length_mm / 1000.0

    # robot
    osv = None
    if not args.no_robot:
        osv = _connect_robot(args.robot_ip, args.robot_port, timeout=3.0)

    # camera + intrinsics
    pipeline = cap = None
    intrinsics: Dict[str, float] = {}

    if REALSENSE_AVAILABLE:
        try:
            pipeline, intrinsics = _open_realsense()
            print("[OK] RealSense opened.")
        except Exception as e:
            print(f"[WARN] RealSense failed ({e}), trying webcam.")

    if pipeline is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No camera found.")
            sys.exit(1)
        # prefer intrinsics stored in extrinsic.json (matches calibration)
        intrinsics = intr_json or {"fx": 600.0, "fy": 600.0,
                                    "cx": 320.0, "cy": 240.0,
                                    "width": 640, "height": 480}
        print("[WARN] Using webcam — intrinsics taken from extrinsic.json.")

    # If RealSense intrinsics differ from calibration intrinsics, warn (projection
    # must use the SAME intrinsics the extrinsic was solved with).
    if intr_json is not None:
        if abs(intrinsics.get("fx", 0) - intr_json.get("fx", 0)) > 1.0:
            print("[WARN] Live fx differs from calibration fx — using calibration intrinsics.")
        intrinsics = intr_json

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    print("\n=== Skeleton Check ===")
    print("► Skeleton is projected from extrinsic.json + live joint angles.")
    print("► Move the robot (A1) and watch if the skeleton stays on it.")
    print("► q = quit    s = save frame\n")

    last_angles = _DEMO_ANGLES.copy()
    robot_connected = osv is not None

    while True:
        if pipeline is not None:
            frame = _get_rs_frame(pipeline)
        else:
            frame = _get_webcam_frame(cap)
        if frame is None:
            continue

        # read live angles
        if osv is not None:
            a = _read_joint_angles(osv)
            if a is not None:
                last_angles = a
        angles = last_angles

        # project skeleton
        uvs = joints_to_uvs(tuple(angles), T_cr, intrinsics,
                            gripper_length_m=gripper_m)

        if DRAW_AVAILABLE:
            draw_fk_projections(frame, uvs, names=_JOINT_NAMES,
                                color_bgr=(255, 200, 0), draw_skeleton=True)
        else:
            _draw_skeleton_basic(frame, uvs, _JOINT_NAMES)

        # status bar
        conn = "Robot: CONNECTED" if robot_connected else "Robot: NOT connected (demo pose)"
        ccol = (0, 200, 0) if robot_connected else (0, 80, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(frame, conn, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ccol, 1, cv2.LINE_AA)
        cv2.putText(frame, f"A1={angles[0]:.1f} A2={angles[1]:.1f} A3={angles[2]:.1f}",
                    (frame.shape[1] - 250, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out = Path("skeleton_check.png")
            cv2.imwrite(str(out), frame)
            print(f"[SAVED] {out.resolve()}")

    cv2.destroyAllWindows()
    if pipeline is not None:
        pipeline.stop()
    elif cap is not None:
        cap.release()
    if osv is not None:
        try:
            osv.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
