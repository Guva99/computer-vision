"""
Extrinsic calibration: click TCP in image → FK → solvePnP.
No printed board required.

Workflow
--------
  1. Run:  python calibrate_click_fk.py
  2. Move robot to any pose.
  3. CLICK on the TCP tip in the camera window.
     Robot angles are read automatically → FK → 3D point stored.
  4. Repeat ≥10 times with different robot poses.
  5. Press  c  to compute and save extrinsic.json.

Keys: c=solve & save   d=delete last point   r=reset all   q=quit

If robot not connected, use:  python calibrate_click_fk.py --no-robot
  (you will be prompted to type joint angles in the terminal after each click)
"""

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import socket
import cv2
import numpy as np

try:
    from kuka_fk import fk_joints
    FK_AVAILABLE = True
except ImportError:
    FK_AVAILABLE = False
    print("[WARN] kuka_fk.py not found.")

try:
    from robot.drivers.openshowvar import OpenShowVar
    OSV_AVAILABLE = True
except ImportError:
    OSV_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — using webcam.")


def _connect_robot(ip: str, port: int, timeout: float = 3.0) -> Optional["OpenShowVar"]:
    """Try to connect to KUKA VarProxy with a short timeout. Returns None on failure."""
    if not OSV_AVAILABLE:
        return None
    print(f"Connecting to KUKA at {ip}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        sock.settimeout(None)
        osv = OpenShowVar(ip, port)
        osv.sock = sock          # reuse the already-open socket
        print(f"[OK] Robot connected: {ip}:{port}")
        return osv
    except (socket.timeout, socket.error, OSError) as e:
        sock.close()
        print(f"[WARN] Robot not reachable ({e}) — manual angle entry mode.")
        return None


def _read_joint_angles(osv: "OpenShowVar") -> Optional[np.ndarray]:
    """
    Read $AXIS_ACT from robot → (A1..A6) as float array, or None.

    Raw format (KUKA E6AXIS struct):
        E6AXIS: A1 -39.6693 A2 -26.4447 A3 120.5337 A4 -0.8537 A5 -90.8575
        A6 3.4721 E1 0.0 E2 0.0 ...
    We locate each axis label A1..A6 and take the token right after it.
    """
    try:
        raw = osv.read("$AXIS_ACT", False).decode(errors="ignore")
        # normalise separators: drop braces, struct name, commas, colons
        cleaned = raw.replace("{", " ").replace("}", " ") \
                     .replace(",", " ").replace(":", " ")
        parts = cleaned.split()

        angles = []
        for axis in ("A1", "A2", "A3", "A4", "A5", "A6"):
            if axis in parts:
                idx = parts.index(axis)
                if idx + 1 < len(parts):
                    angles.append(float(parts[idx + 1]))
        if len(angles) == 6:
            return np.array(angles, dtype=np.float64)
        print(f"[WARN] Parsed {len(angles)}/6 angles from: '{raw}'")
    except Exception as e:
        print(f"[WARN] Could not read angles: {e}")
    return None

_WINDOW = "Calibration  [CLICK = add point | c=solve | d=delete | r=reset | q=quit]"


# ── state ─────────────────────────────────────────────────────────────────────
_img_pts: List[np.ndarray] = []
_obj_pts: List[np.ndarray] = []
_pending_click: Optional[Tuple[int, int]] = None
_status: str = ""


def _mouse_cb(event, x, y, _flags, _param):
    global _pending_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _pending_click = (x, y)


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


# ── FK ────────────────────────────────────────────────────────────────────────

def _tcp_3d(angles_deg: np.ndarray, gripper_m: float) -> np.ndarray:
    positions = fk_joints(tuple(angles_deg), gripper_length_m=gripper_m)
    return positions[-1].copy()


# ── solve & save ──────────────────────────────────────────────────────────────

def _solve_and_save(out_path: Path, intrinsics: dict) -> bool:
    n = len(_img_pts)
    if n < 4:
        print(f"[ERROR] Need ≥4 points, have {n}.")
        return False

    obj = np.array(_obj_pts, dtype=np.float64)
    img = np.array(_img_pts, dtype=np.float64)
    cam_mat = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    ok, r_vec, t_vec, inl = cv2.solvePnPRansac(
        obj, img, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=2000,
        reprojectionError=4.0,
        confidence=0.999,
    )
    if not ok:
        print("[ERROR] solvePnPRansac failed — try more / better-spread points.")
        return False

    n_inl = len(inl) if inl is not None else n
    if inl is not None and len(inl) >= 4:
        r_vec, t_vec = cv2.solvePnPRefineLM(
            obj[inl.flatten()], img[inl.flatten()], cam_mat, dist, r_vec, t_vec,
        )

    R, _ = cv2.Rodrigues(r_vec)
    T_cr = np.eye(4, dtype=np.float64)
    T_cr[:3, :3] = R
    T_cr[:3, 3] = t_vec.flatten()

    proj, _ = cv2.projectPoints(obj, r_vec, t_vec, cam_mat, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    print(f"\n[OK] RMS = {rms:.2f} px   inliers = {n_inl}/{n}")
    print(f"     t_vec = {np.round(t_vec.flatten(), 4)}")
    print("     T_cr =\n", np.round(T_cr, 5))

    out_path.write_text(json.dumps({
        "T_cr": T_cr.tolist(),
        "R_vec": r_vec.flatten().tolist(),
        "t_vec": t_vec.flatten().tolist(),
        "rms_px": rms,
        "n_points": n,
        "n_inliers": n_inl,
        "method": "solvePnPRansac+RefineLM/TCP",
        "intrinsics": intrinsics,
    }, indent=2))
    print(f"[SAVED] {out_path.resolve()}")
    return True


# ── overlay ───────────────────────────────────────────────────────────────────

def _draw(frame: np.ndarray, waiting_input: bool,
          pending_uv: Optional[Tuple[int, int]],
          robot_connected: bool) -> np.ndarray:
    out = frame.copy()

    # robot connection status bar
    conn_txt = "Robot: CONNECTED" if robot_connected else "Robot: NOT connected (enter angles in terminal)"
    conn_color = (0, 200, 0) if robot_connected else (0, 80, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, conn_txt, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, conn_color, 1, cv2.LINE_AA)

    # stored points
    for i, uv in enumerate(_img_pts):
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(out, (u, v), 7, (0, 220, 0), 2)
        cv2.putText(out, f"P{i+1}", (u + 9, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

    # pending click (orange crosshair)
    if pending_uv is not None:
        x, y = pending_uv
        cv2.line(out, (x - 14, y), (x + 14, y), (0, 130, 255), 2)
        cv2.line(out, (x, y - 14), (x, y + 14), (0, 130, 255), 2)
        cv2.circle(out, (x, y), 8, (0, 130, 255), 2)

    # waiting banner
    if waiting_input:
        cv2.rectangle(out, (0, 30), (out.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(out, ">>> Type 6 joint angles in the TERMINAL below, then press Enter <<<",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)

    # bottom hint
    n = len(_img_pts)
    hint = (f"{n} point(s) — need >=4 (aim for >=12)"
            if n < 10 else f"{n} point(s) — press c to solve & save")
    cv2.rectangle(out, (0, out.shape[0] - 32), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, hint, (8, out.shape[0] - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
    if _status:
        cv2.putText(out, _status, (8, out.shape[0] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 210, 255), 1, cv2.LINE_AA)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global _pending_click, _status

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="extrinsic.json")
    ap.add_argument("--robot-ip", default="192.168.17.2")
    ap.add_argument("--robot-port", type=int, default=7000)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--gripper-length-mm", type=float, default=137.0,
                    help="Tool length flange→TCP tip in mm (default 137)")
    args = ap.parse_args()

    if not FK_AVAILABLE:
        print("[ERROR] kuka_fk.py required. Exiting.")
        sys.exit(1)

    gripper_m = args.gripper_length_mm / 1000.0
    out_path = Path(args.out)

    # robot — fast connect with 3-second timeout (same as main.py approach)
    osv = None
    if not args.no_robot:
        osv = _connect_robot(args.robot_ip, args.robot_port, timeout=3.0)

    # camera
    pipeline = cap = None
    intrinsics: dict = {}

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
        intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
                      "width": 640, "height": 480}
        print("[WARN] Using webcam with estimated intrinsics.")

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WINDOW, _mouse_cb)

    print("\n=== Click-TCP Extrinsic Calibration ===")
    print("► Move robot to any pose, then CLICK on the TCP tip in the window.")
    print("► Repeat ≥10 times with varied poses.")
    print("► Press  c = solve & save   d = delete last   r = reset   q = quit\n")

    robot_connected = osv is not None
    frozen_frame: Optional[np.ndarray] = None
    waiting_input = False
    queued_click: Optional[Tuple[int, int]] = None

    # thread for manual angle input
    input_thread: Optional[threading.Thread] = None
    input_result: List[Optional[np.ndarray]] = [None]   # mutable container

    def _read_angles_bg(click_pos, result_box):
        try:
            raw = input(f"  Joint angles A1..A6 for click {click_pos} [degrees, space-separated]: ").strip()
            parts = raw.replace(",", " ").split()
            if len(parts) == 6:
                result_box[0] = np.array([float(p) for p in parts])
            else:
                print("  [ERROR] Expected 6 numbers — click skipped.")
        except Exception:
            pass

    while True:
        # grab frame
        if not waiting_input:
            if pipeline is not None:
                frame = _get_rs_frame(pipeline)
            else:
                frame = _get_webcam_frame(cap)
            if frame is not None:
                frozen_frame = frame
        else:
            frame = frozen_frame

        if frame is None:
            continue

        # new click arrived (and we're not already waiting for input)
        if _pending_click is not None and not waiting_input:
            click = _pending_click
            _pending_click = None

            # try to get angles
            angles: Optional[np.ndarray] = None
            if osv is not None:
                angles = _read_joint_angles(osv)

            if angles is not None:
                # robot gave angles directly — store point immediately
                try:
                    p3d = _tcp_3d(angles, gripper_m)
                    _img_pts.append(np.array(click, dtype=np.float64))
                    _obj_pts.append(p3d)
                    _status = (f"P{len(_img_pts)}  A={np.round(angles, 1)}  "
                               f"TCP=({p3d[0]*1000:.1f}, {p3d[1]*1000:.1f}, {p3d[2]*1000:.1f}) mm")
                    print(f"[P{len(_img_pts)}] pixel={click}  "
                          f"$AXIS_ACT=A1..A6={np.round(angles, 2)}  "
                          f"TCP=({p3d[0]*1000:.1f},{p3d[1]*1000:.1f},{p3d[2]*1000:.1f}) mm")
                except Exception as e:
                    _status = f"FK error: {e}"
                    print(f"[FK ERROR] {e}")
            else:
                # no robot — ask in terminal via background thread
                waiting_input = True
                queued_click = click
                input_result[0] = None
                input_thread = threading.Thread(
                    target=_read_angles_bg,
                    args=(click, input_result),
                    daemon=True,
                )
                input_thread.start()

        # check if background input thread finished
        if waiting_input and input_thread is not None and not input_thread.is_alive():
            input_thread = None
            waiting_input = False
            if input_result[0] is not None and queued_click is not None:
                try:
                    p3d = _tcp_3d(input_result[0], gripper_m)
                    _img_pts.append(np.array(queued_click, dtype=np.float64))
                    _obj_pts.append(p3d)
                    _status = f"P{len(_img_pts)} stored"
                    print(f"[P{len(_img_pts)}] pixel={queued_click}  "
                          f"TCP=({p3d[0]*1000:.1f},{p3d[1]*1000:.1f},{p3d[2]*1000:.1f}) mm")
                except Exception as e:
                    _status = f"FK error: {e}"
            else:
                _status = "Point skipped."
            queued_click = None

        cv2.imshow(_WINDOW, _draw(frame, waiting_input, queued_click, robot_connected))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _img_pts.clear(); _obj_pts.clear()
            _status = "Reset — all points cleared."
            print("[RESET]")
        elif key == ord("d"):
            if _img_pts:
                _img_pts.pop(); _obj_pts.pop()
                _status = f"Deleted. {len(_img_pts)} remain."
        elif key == ord("c"):
            _solve_and_save(out_path, intrinsics)

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
