"""
Hand-eye calibration (eye-to-hand setup) — KUKA KR4 R600 + RealSense D4xx.

Setup
-----
  • Camera is FIXED in the workspace (not mounted on the robot).
  • Print a checkerboard and attach it rigidly to the robot TCP / flange.
  • The script collects (robot FK pose, detected board pose in camera) pairs.
  • After ≥10 diverse poses cv2.calibrateHandEye solves for T_cr.

Algorithm
---------
  Eye-to-hand:  AX = XB  where
      A_i  = T_gripper_base[i]   — robot base ← gripper  (from FK)
      B_i  = T_board_cam[i]      — camera ← board        (from solvePnP)
      X    = T_base_cam          — camera ← robot base   = T_cr

Checkerboard
------------
  Default: 9×6 inner corners, 25 mm squares.
  Print calibration/checkerboard_9x6_25mm.png (or any standard OpenCV board).
  Adjust BOARD_COLS / BOARD_ROWS / SQUARE_SIZE_M below if you use a different board.

Usage
-----
  python calibrate_handeye.py [--out extrinsic.json] [--robot-ip 192.168.1.100]

  Controls:
    SPACE  — capture current pose (robot position + board detection)
    d      — delete last captured pose
    c      — compute calibration and save
    r      — reset all captured poses
    q      — quit without saving
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
try:
    from kuka_fk import fk_tcp
    FK_AVAILABLE = True
except ImportError:
    FK_AVAILABLE = False
    print("[WARN] kuka_fk not found — FK will be unavailable, enter angles manually.")

try:
    from robot.drivers.openshowvar import OpenShowVar

    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("[WARNING] Robot modules not found")

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not found — falling back to webcam (no depth).")

# ── board configuration ───────────────────────────────────────────────────────
BOARD_COLS = 9          # inner corners horizontally
BOARD_ROWS = 6          # inner corners vertically
SQUARE_SIZE_M = 0.025   # metres per square (adjust to your printed board)

# ── calibration state ─────────────────────────────────────────────────────────
_R_gripper2base: List[np.ndarray] = []   # (3,3) rotation from FK
_t_gripper2base: List[np.ndarray] = []   # (3,1) translation from FK
_R_target2cam: List[np.ndarray] = []     # (3,3) rotation from solvePnP
_t_target2cam: List[np.ndarray] = []     # (3,1) translation from solvePnP

_WINDOW = "Hand-Eye Calibration  [SPACE=capture | d=delete | c=solve | r=reset | q=quit]"


# ── checkerboard helpers ──────────────────────────────────────────────────────

def _board_object_points() -> np.ndarray:
    """3-D corners of the checkerboard in board frame (z=0 plane), metres."""
    pts = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float64)
    pts[:, :2] = (
        np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE_M
    )
    return pts


_OBJ_PTS = _board_object_points()


def _detect_board(
    gray: np.ndarray,
) -> Optional[np.ndarray]:
    """Return sub-pixel corners or None."""
    ok, corners = cv2.findChessboardCorners(
        gray,
        (BOARD_COLS, BOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners


def _solve_board_pose(
    corners: np.ndarray,
    cam_mat: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """PnP → (R_3x3, t_3x1) of board in camera frame."""
    ok, r_vec, t_vec = cv2.solvePnP(
        _OBJ_PTS, corners, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for board pose")
    R, _ = cv2.Rodrigues(r_vec)
    return R, t_vec.reshape(3, 1)


# ── FK helpers ────────────────────────────────────────────────────────────────

def _angles_to_T_gripper_base(angles_deg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use kuka_fk.fk_tcp to get T_base_gripper (4×4).
    Return (R_gripper2base 3×3, t_gripper2base 3×1).
    """
    _, T_base_gripper = fk_tcp(tuple(angles_deg))
    R = T_base_gripper[:3, :3].copy()
    t = T_base_gripper[:3, 3:4].copy()
    return R, t


def _parse_angles_from_string(s: str) -> Optional[np.ndarray]:
    """Parse '1 2 3 4 5 6' or '1,2,3,4,5,6' → float array of 6."""
    parts = s.replace(",", " ").split()
    if len(parts) == 6:
        try:
            return np.array([float(p) for p in parts])
        except ValueError:
            pass
    return None


# ── camera helpers ─────────────────────────────────────────────────────────────

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
    intrinsics = {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.ppx, "cy": intr.ppy,
        "width": intr.width, "height": intr.height,
    }
    return pipeline, intrinsics


def _get_frame_rs(pipeline) -> Optional[np.ndarray]:
    frames = pipeline.wait_for_frames()
    cf = frames.get_color_frame()
    return np.asanyarray(cf.get_data()) if cf else None


def _open_webcam():
    cap = cv2.VideoCapture(0)
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    return cap, intrinsics


def _get_frame_webcam(cap) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    return frame if ok else None


# ── solve + save ─────────────────────────────────────────────────────────────

def _solve_and_save(out_path: Path, intrinsics: dict) -> bool:
    n = len(_R_gripper2base)
    if n < 4:
        print(f"[ERROR] Need ≥4 pose pairs, have {n}.")
        return False

    methods = [
        ("TSAI",     cv2.CALIB_HAND_EYE_TSAI),
        ("PARK",     cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD",   cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF",  cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]

    # Build camera matrix for re-projection RMS check
    cam_mat = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]],
         [0, intrinsics["fy"], intrinsics["cy"]],
         [0, 0, 1]], dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)

    best_name = None
    best_T_cr = None
    best_rms = float("inf")

    print(f"\n[SOLVE] Trying {len(methods)} hand-eye methods on {n} poses…")
    for name, method in methods:
        try:
            R_b2c, t_b2c = cv2.calibrateHandEye(
                _R_gripper2base, _t_gripper2base,
                _R_target2cam,  _t_target2cam,
                method=method,
            )
        except cv2.error as e:
            print(f"  {name}: FAILED ({e})")
            continue

        # Assemble T_cr = T_base2cam
        T_cr = np.eye(4, dtype=np.float64)
        T_cr[:3, :3] = R_b2c
        T_cr[:3, 3] = t_b2c.flatten()

        # Compute re-projection RMS: project board 3-D corners through
        # T_gripper_base → T_base_cam → camera, compare with solvePnP corners
        errs = []
        for i in range(n):
            R_g2b = _R_gripper2base[i]
            t_g2b = _t_gripper2base[i].flatten()

            # board corners in robot base frame:
            # p_base = R_g2b @ p_board + t_g2b ... but board is rigidly on gripper
            # p_cam = T_cr @ T_g2b @ p_board_h
            T_g2b_4 = np.eye(4)
            T_g2b_4[:3, :3] = R_g2b
            T_g2b_4[:3, 3] = t_g2b

            p_board_h = np.hstack([_OBJ_PTS, np.ones((len(_OBJ_PTS), 1))]).T  # (4,N)
            p_cam = (T_cr @ T_g2b_4 @ p_board_h).T[:, :3]  # (N,3)

            r_vec_i, _ = cv2.Rodrigues(_R_target2cam[i])
            img_pts_solvepnp, _ = cv2.projectPoints(_OBJ_PTS, r_vec_i, _t_target2cam[i], cam_mat, dist)
            img_pts_solvepnp = img_pts_solvepnp.reshape(-1, 2)

            r_vec_he, _ = cv2.Rodrigues(T_cr[:3, :3] @ T_g2b_4[:3, :3])
            t_he = (T_cr[:3, :3] @ t_g2b + T_cr[:3, 3]).reshape(3, 1)
            img_pts_he, _ = cv2.projectPoints(_OBJ_PTS, r_vec_he, t_he, cam_mat, dist)
            img_pts_he = img_pts_he.reshape(-1, 2)

            diff = np.linalg.norm(img_pts_solvepnp - img_pts_he, axis=1)
            errs.extend(diff.tolist())

        rms = float(np.sqrt(np.mean(np.square(errs))))
        status = "  ← BEST" if rms < best_rms else ""
        print(f"  {name:12s}: RMS = {rms:.3f} px{status}")
        if rms < best_rms:
            best_rms = rms
            best_T_cr = T_cr
            best_name = name

    if best_T_cr is None:
        print("[ERROR] All methods failed.")
        return False

    r_vec_best, _ = cv2.Rodrigues(best_T_cr[:3, :3])
    t_vec_best = best_T_cr[:3, 3]

    print(f"\n[OK] Best method: {best_name}  RMS = {best_rms:.3f} px  n_poses = {n}")
    print(f"     R_vec = {r_vec_best.flatten()}")
    print(f"     t_vec = {t_vec_best}")
    print("     T_cr =")
    print(np.round(best_T_cr, 6))

    data = {
        "T_cr": best_T_cr.tolist(),
        "R_vec": r_vec_best.flatten().tolist(),
        "t_vec": t_vec_best.tolist(),
        "rms_px": best_rms,
        "n_points": n,
        "method": f"calibrateHandEye/{best_name}",
        "board": {
            "cols": BOARD_COLS,
            "rows": BOARD_ROWS,
            "square_size_m": SQUARE_SIZE_M,
        },
        "intrinsics": intrinsics,
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"[SAVED] {out_path.resolve()}")
    return True


# ── overlay drawing ────────────────────────────────────────────────────────────

def _draw_overlay(
    img: np.ndarray,
    corners: Optional[np.ndarray],
    board_ok: bool,
    n_captured: int,
    status_msg: str,
) -> np.ndarray:
    out = img.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, (BOARD_COLS, BOARD_ROWS), corners, board_ok)
    color = (0, 255, 0) if board_ok else (0, 80, 255)
    board_txt = "Board FOUND" if board_ok else "Board NOT found"
    cv2.putText(out, board_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"Captured: {n_captured}  (need >=10 for good result)",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1, cv2.LINE_AA,
    )
    if status_msg:
        cv2.putText(out, status_msg, (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Hand-eye calibration for KUKA + RealSense")
    ap.add_argument("--out", default="extrinsic.json", help="Output JSON path")
    ap.add_argument("--robot-ip", default="192.168.17.2")
    ap.add_argument("--robot-port", type=int, default=7000)
    ap.add_argument("--no-robot", action="store_true",
                    help="Skip robot connection (enter joint angles manually)")
    args = ap.parse_args()

    out_path = Path(args.out)

    # ── robot connection ──────────────────────────────────────────────────────
    # robot: Optional[RobotService] = None
    if not args.no_robot and FK_AVAILABLE:
        robot = OpenShowVar(ip=args.robot_ip, port=args.robot_port)
        robot.connect()
        if not robot.is_connected:
            print("[WARN] Robot not connected — will prompt for manual angle entry.")
            robot = None
    elif not FK_AVAILABLE:
        print("[WARN] kuka_fk not available — enter joint angles manually.")
        

    # ── camera open ───────────────────────────────────────────────────────────
    pipeline = cap = None
    intrinsics: dict = {}

    if REALSENSE_AVAILABLE:
        try:
            pipeline, intrinsics = _open_realsense()
            print("[OK] RealSense opened.")
        except Exception as e:
            print(f"[WARN] RealSense failed ({e}), trying webcam.")

    if pipeline is None:
        cap, intrinsics = _open_webcam()
        print("[WARN] Using webcam with estimated intrinsics. Results may be less accurate.")

    cam_mat = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]],
         [0, intrinsics["fy"], intrinsics["cy"]],
         [0, 0, 1]], dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    print("\n=== Hand-Eye Calibration (eye-to-hand) ===")
    print(f"Board: {BOARD_COLS}×{BOARD_ROWS} inner corners, {SQUARE_SIZE_M*1000:.0f} mm squares")
    print("Attach the checkerboard rigidly to the robot TCP/flange.")
    print("Move robot to diverse poses and press SPACE to capture each one.\n")
    print("Keys:  SPACE = capture    d = delete last    c = solve & save    r = reset    q = quit\n")

    status_msg = ""
    last_corners: Optional[np.ndarray] = None
    board_ok = False

    while True:
        # Grab frame
        if pipeline is not None:
            frame = _get_frame_rs(pipeline)
        else:
            frame = _get_frame_webcam(cap)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = _detect_board(gray)
        board_ok = corners is not None
        if corners is not None:
            last_corners = corners

        display = _draw_overlay(frame, corners, board_ok, len(_R_gripper2base), status_msg)
        cv2.imshow(_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            _R_gripper2base.clear(); _t_gripper2base.clear()
            _R_target2cam.clear();  _t_target2cam.clear()
            status_msg = "Reset — all poses cleared."
            print("[RESET] All poses cleared.")

        elif key == ord("d"):
            if _R_gripper2base:
                for lst in (_R_gripper2base, _t_gripper2base, _R_target2cam, _t_target2cam):
                    lst.pop()
                status_msg = f"Deleted last pose. {len(_R_gripper2base)} remain."
                print(f"[DELETE] Last pose removed. {len(_R_gripper2base)} remain.")
            else:
                status_msg = "Nothing to delete."

        elif key == ord("c"):
            _solve_and_save(out_path, intrinsics)

        elif key == ord(" "):
            # ── Capture current pose ──────────────────────────────────────
            if not board_ok or last_corners is None:
                status_msg = "Board not detected — cannot capture."
                print("[SKIP] Board not visible. Move board in front of camera.")
                continue

            # --- Get robot joint angles ---
            angles = None
            if robot is not None and robot.is_connected:
                raw = robot.get_joint_angles()
                if raw is not None:
                    angles = np.array(raw, dtype=np.float64)
            if angles is None:
                raw_str = input(
                    "  Enter joint angles A1..A6 in degrees (space-separated): "
                ).strip()
                angles = _parse_angles_from_string(raw_str)
                if angles is None:
                    status_msg = "Invalid angles — pose skipped."
                    print("[SKIP] Could not parse angles.")
                    continue

            if not FK_AVAILABLE:
                print("[ERROR] kuka_fk not available — cannot compute FK.")
                status_msg = "FK unavailable — install kuka_fk."
                continue

            try:
                R_g2b, t_g2b = _angles_to_T_gripper_base(angles)
            except Exception as e:
                status_msg = f"FK error: {e}"
                print(f"[FK ERROR] {e}")
                continue

            # --- solvePnP for board → camera ---
            try:
                R_t2c, t_t2c = _solve_board_pose(last_corners, cam_mat, dist)
            except RuntimeError as e:
                status_msg = f"PnP error: {e}"
                print(f"[PNP ERROR] {e}")
                continue

            _R_gripper2base.append(R_g2b)
            _t_gripper2base.append(t_g2b)
            _R_target2cam.append(R_t2c)
            _t_target2cam.append(t_t2c)

            n = len(_R_gripper2base)
            status_msg = f"Pose {n} captured. Angles: {np.round(angles, 1)}"
            print(f"[CAPTURE {n}] angles={np.round(angles, 1)}  t_board={t_t2c.flatten()}")

    # Cleanup
    cv2.destroyAllWindows()
    if pipeline is not None:
        pipeline.stop()
    elif cap is not None:
        cap.release()
    if robot is not None:
        robot.disconnect()


if __name__ == "__main__":
    main()
