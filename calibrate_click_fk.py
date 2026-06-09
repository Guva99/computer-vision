"""
Extrinsic calibration: click a robot JOINT in the image → choose which joint →
FK → solvePnP. No printed board required.

Why click different joints (not only TCP)
------------------------------------------
PnP needs points spread in DEPTH. Clicking only the TCP puts every point at one
height → depth is weakly constrained → collision distances drift.
Instead click DIFFERENT joints (J2/J3 high, J5/J6 mid, TCP low). Even in a single
robot pose they span Z≈0..470 mm, which fixes the depth. The robot may keep
rotating only A1 — height variation comes from the joints, not from moving A2/A3.

Workflow
--------
  1. Run:  python calibrate_click_fk.py
  2. Put the robot in a pose.
  3. CLICK on a visible joint centre in the image.
  4. Press the joint key:  1=J1 2=J2 3=J3 4=J4 5=J5 6=J6  t=TCP
     → angles are read from the robot, that joint's 3D point is stored.
  5. Click several joints per pose (different heights!). Rotate A1, repeat.
  6. Collect ≥15 points across heights → press  c  to solve & save.

Keys: 1-6/t = assign joint   c = solve & save   d = delete last   r = reset   q = quit

If robot not connected:  python calibrate_click_fk.py --no-robot
  (after the joint key you type the 6 joint angles in the terminal)
"""

import argparse
import json
import sys
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


# fk_joints returns [base(0), J1(1), J2(2), J3(3), J4(4), J5(5), J6(6), TCP(last)]
# Key → (joint index in fk_joints output, label). TCP uses -1 = last point (tip).
_KEY_TO_JOINT = {
    ord("1"): (1, "J1"),
    ord("2"): (2, "J2"),
    ord("3"): (3, "J3"),
    ord("4"): (4, "J4"),
    ord("5"): (5, "J5"),
    ord("6"): (6, "J6"),
    ord("t"): (-1, "TCP"),
}

_WINDOW = "Calibration  [CLICK joint -> press 1-6/t | c=solve | d=del | r=reset | q=quit]"


# ── state ─────────────────────────────────────────────────────────────────────
_img_pts: List[np.ndarray] = []
_obj_pts: List[np.ndarray] = []
_labels: List[str] = []
_pending_click: Optional[Tuple[int, int]] = None


def _mouse_cb(event, x, y, _flags, _param):
    global _pending_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _pending_click = (x, y)


# ── robot ─────────────────────────────────────────────────────────────────────

def _connect_robot(ip: str, port: int, timeout: float = 3.0) -> Optional["OpenShowVar"]:
    """Connect to KUKA VarProxy with a short timeout. Returns None on failure."""
    if not OSV_AVAILABLE:
        return None
    print(f"Connecting to KUKA at {ip}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        sock.settimeout(None)
        osv = OpenShowVar(ip, port)
        osv.sock = sock
        print(f"[OK] Robot connected: {ip}:{port}")
        return osv
    except (socket.timeout, socket.error, OSError) as e:
        sock.close()
        print(f"[WARN] Robot not reachable ({e}) — manual angle entry mode.")
        return None


def _read_joint_angles(osv: "OpenShowVar") -> Optional[np.ndarray]:
    """Read $AXIS_ACT → (A1..A6) float array, or None. Finds each axis label."""
    try:
        raw = osv.read("$AXIS_ACT", False).decode(errors="ignore")
        cleaned = raw.replace("{", " ").replace("}", " ").replace(",", " ").replace(":", " ")
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


def _ask_angles_terminal() -> Optional[np.ndarray]:
    """Manual fallback: read 6 joint angles from the terminal."""
    raw = input("  Joint angles A1..A6 [degrees, space-separated]: ").strip()
    parts = raw.replace(",", " ").split()
    if len(parts) == 6:
        try:
            return np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError:
            pass
    print("  [ERROR] Expected 6 numbers — skipped.")
    return None


# ── camera ────────────────────────────────────────────────────────────────────

def _open_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    return pipeline, {
        "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
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

def _joint_3d(angles_deg: np.ndarray, joint_idx: int, gripper_m: float) -> np.ndarray:
    """3D position (m, robot base frame) of the requested joint. joint_idx=-1 → TCP tip."""
    positions = fk_joints(tuple(angles_deg), gripper_length_m=gripper_m)
    return positions[joint_idx].copy()


# ── solve & save ──────────────────────────────────────────────────────────────

def _solve_and_save(out_path: Path, intrinsics: dict) -> bool:
    n = len(_img_pts)
    if n < 4:
        print(f"[ERROR] Need >=4 points, have {n}.")
        return False

    obj = np.array(_obj_pts, dtype=np.float64)
    img = np.array(_img_pts, dtype=np.float64)
    cam_mat = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # Diagnostics: spread of object points in depth (Z range) — мера обусловленности
    z_span_mm = float(obj[:, 2].max() - obj[:, 2].min()) * 1000.0
    print(f"[INFO] Объекты по высоте/глубине: Z-разброс = {z_span_mm:.0f} мм "
          f"({'мало — добавь точки по высоте!' if z_span_mm < 150 else 'ок'})")

    ok, r_vec, t_vec, inl = cv2.solvePnPRansac(
        obj, img, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=2000, reprojectionError=4.0, confidence=0.999,
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
        "z_span_mm": z_span_mm,
        "method": "solvePnPRansac+RefineLM/multi-joint",
        "intrinsics": intrinsics,
    }, indent=2))
    print(f"[SAVED] {out_path.resolve()}")
    return True


# ── overlay ───────────────────────────────────────────────────────────────────

def _draw(frame, pending_uv, robot_connected) -> np.ndarray:
    out = frame.copy()

    conn_txt = "Robot: CONNECTED" if robot_connected else "Robot: NOT connected (type angles in terminal)"
    conn_color = (0, 200, 0) if robot_connected else (0, 80, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, conn_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conn_color, 1, cv2.LINE_AA)

    # stored points with their joint label
    for i, (uv, lbl) in enumerate(zip(_img_pts, _labels)):
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(out, (u, v), 6, (0, 220, 0), 2)
        cv2.putText(out, f"{i+1}:{lbl}", (u + 8, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)

    # pending click — waiting for a joint key
    if pending_uv is not None:
        x, y = pending_uv
        cv2.line(out, (x - 14, y), (x + 14, y), (0, 130, 255), 2)
        cv2.line(out, (x, y - 14), (x, y + 14), (0, 130, 255), 2)
        cv2.circle(out, (x, y), 8, (0, 130, 255), 2)
        cv2.rectangle(out, (0, 30), (out.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(out, "Press joint key:  1=J1 2=J2 3=J3 4=J4 5=J5 6=J6  t=TCP",
                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

    # legend + counter (с разбивкой по высоте)
    n = len(_img_pts)
    z_hint = ""
    if n >= 2:
        zs = np.array([p[2] for p in _obj_pts]) * 1000.0
        z_hint = f"  Z-разброс={zs.max()-zs.min():.0f}мм"
    hint = f"{n} pts{z_hint}  |  1-6/t=assign  c=solve  d=del  r=reset  q=quit"
    cv2.rectangle(out, (0, out.shape[0] - 24), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, hint, (8, out.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global _pending_click

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="extrinsic.json")
    ap.add_argument("--robot-ip", default="192.168.17.2")
    ap.add_argument("--robot-port", type=int, default=7000)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--gripper-length-mm", type=float, default=137.0,
                    help="Tool length flange->TCP tip in mm (default 137)")
    args = ap.parse_args()

    if not FK_AVAILABLE:
        print("[ERROR] kuka_fk.py required. Exiting.")
        sys.exit(1)

    gripper_m = args.gripper_length_mm / 1000.0
    out_path = Path(args.out)

    osv = None
    if not args.no_robot:
        osv = _connect_robot(args.robot_ip, args.robot_port, timeout=3.0)
    robot_connected = osv is not None

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

    print("\n=== Multi-Joint Extrinsic Calibration ===")
    print("► Кликни на сустав в кадре → нажми клавишу: 1=J1 2=J2 3=J3 4=J4 5=J5 6=J6 t=TCP")
    print("► Кликай РАЗНЫЕ суставы (разная высота!): J2/J3 верх, J5/J6 середина, TCP низ")
    print("► Набери >=15 точек по высотам → c = solve & save\n")

    while True:
        if pipeline is not None:
            frame = _get_rs_frame(pipeline)
        else:
            frame = _get_webcam_frame(cap)
        if frame is None:
            continue

        cv2.imshow(_WINDOW, _draw(frame, _pending_click, robot_connected))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _img_pts.clear(); _obj_pts.clear(); _labels.clear()
            _pending_click = None
            print("[RESET]")
        elif key == ord("d"):
            if _img_pts:
                _img_pts.pop(); _obj_pts.pop()
                lbl = _labels.pop()
                print(f"[DELETE] removed {lbl}; {len(_img_pts)} remain.")
        elif key == ord("c"):
            _solve_and_save(out_path, intrinsics)
        elif key in _KEY_TO_JOINT:
            # assign the pending click to the chosen joint
            if _pending_click is None:
                print("[SKIP] Сначала кликни на сустав, потом нажми клавишу.")
                continue
            joint_idx, joint_name = _KEY_TO_JOINT[key]

            # get angles (robot read or manual terminal)
            if osv is not None:
                angles = _read_joint_angles(osv)
            else:
                print(f"  Назначаю {joint_name} для клика {_pending_click}")
                angles = _ask_angles_terminal()
            if angles is None:
                print("[SKIP] Нет углов — точка пропущена.")
                _pending_click = None
                continue

            try:
                p3d = _joint_3d(angles, joint_idx, gripper_m)
            except Exception as e:
                print(f"[FK ERROR] {e}")
                _pending_click = None
                continue

            _img_pts.append(np.array(_pending_click, dtype=np.float64))
            _obj_pts.append(p3d)
            _labels.append(joint_name)
            print(f"[P{len(_img_pts)}] {joint_name}  pixel={_pending_click}  "
                  f"3D=({p3d[0]*1000:.1f}, {p3d[1]*1000:.1f}, {p3d[2]*1000:.1f}) mm")
            _pending_click = None

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
