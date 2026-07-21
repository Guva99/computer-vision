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

Keys: 1-6/t = assign joint   c = solve & save   v = validate saved extrinsic
      d = delete last   r = reset   q = quit

If robot not connected:  python calibrate_click_fk.py --no-robot
  (after the joint key you type the 6 joint angles in the terminal)

Quality control (замечание рецензента: «ошибка оценки расстояния»)
------------------------------------------------------------------
On `c` the script now:
  * iteratively rejects outlier points (residual > --outlier-px), re-solves;
  * prints a QUALITY REPORT: RMS, inliers, residual distribution, Z-span,
    image-coverage of the points;
  * runs HOLD-OUT validation: a fraction of points is excluded from the fit
    and the reprojection error on them is reported in px AND millimetres
    (err_mm ≈ err_px · z / fx) — direct answer to the distance-error question;
  * warns loudly and asks for confirmation before saving when
    rms_px > --rms-warn-px or inliers < --min-inliers;
  * appends every attempt to captures/calibration_log.json (history).
`v` validates the CURRENTLY SAVED extrinsic.json against the collected points
without refitting — collect fresh control points and press v to measure the
error of an existing calibration on points that never participated in the fit.
"""

import argparse
import json
import sys
from datetime import datetime
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

def _cam_matrix(intrinsics: dict) -> np.ndarray:
    return np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)


def _residuals_px(obj: np.ndarray, img: np.ndarray, r_vec, t_vec,
                  cam_mat: np.ndarray) -> np.ndarray:
    """Пиксельная ошибка репроекции каждой точки."""
    proj, _ = cv2.projectPoints(obj, r_vec, t_vec, cam_mat, np.zeros(5))
    return np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)


def _residuals_mm(obj: np.ndarray, res_px: np.ndarray, r_vec, t_vec,
                  intrinsics: dict) -> np.ndarray:
    """Перевод пиксельной ошибки в миллиметры на глубине точки:
       err_mm ≈ err_px · z_cam / fx  (метрическая ошибка оценки положения)."""
    R, _ = cv2.Rodrigues(r_vec)
    z_cam = (R @ obj.T).T[:, 2] + float(t_vec.flatten()[2])
    z_cam = np.maximum(z_cam, 1e-6)
    return res_px * z_cam / float(intrinsics["fx"]) * 1000.0


def _fit_pnp(obj: np.ndarray, img: np.ndarray, cam_mat: np.ndarray):
    """RANSAC + LM-refine. Возвращает (ok, r_vec, t_vec, inlier_idx)."""
    ok, r_vec, t_vec, inl = cv2.solvePnPRansac(
        obj, img, cam_mat, np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=2000, reprojectionError=4.0, confidence=0.999,
    )
    if not ok:
        return False, None, None, None
    inl_idx = inl.flatten() if inl is not None else np.arange(len(obj))
    if len(inl_idx) >= 4:
        r_vec, t_vec = cv2.solvePnPRefineLM(
            obj[inl_idx], img[inl_idx], cam_mat, np.zeros(5), r_vec, t_vec,
        )
    return True, r_vec, t_vec, inl_idx


def _coverage_frac(img: np.ndarray, intrinsics: dict) -> Tuple[float, float]:
    """Какую долю кадра покрывают точки по u и v (0..1) — контроль разнесения
    по полю кадра."""
    w = float(intrinsics.get("width", 640))
    h = float(intrinsics.get("height", 480))
    du = (img[:, 0].max() - img[:, 0].min()) / max(w, 1.0)
    dv = (img[:, 1].max() - img[:, 1].min()) / max(h, 1.0)
    return float(du), float(dv)


def _append_calibration_log(record: dict, log_path: Path) -> None:
    """История калибровок: captures/calibration_log.json (список записей)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if log_path.exists():
        try:
            history = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except Exception:
            history = []
    history.append(record)
    log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[LOG] Запись добавлена в {log_path} (всего {len(history)})")


def _holdout_validation(obj: np.ndarray, img: np.ndarray, cam_mat: np.ndarray,
                        intrinsics: dict, holdout_frac: float,
                        seed: int = 0) -> Optional[dict]:
    """Hold-out: часть точек исключается из подгонки, ошибка на них — честная
    оценка точности калибровки (px и мм). None, если точек мало."""
    n = len(obj)
    n_hold = max(3, int(round(n * holdout_frac)))
    if n - n_hold < 6:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    hold, train = idx[:n_hold], idx[n_hold:]
    ok, r_vec, t_vec, _ = _fit_pnp(obj[train], img[train], cam_mat)
    if not ok:
        return None
    res_px = _residuals_px(obj[hold], img[hold], r_vec, t_vec, cam_mat)
    res_mm = _residuals_mm(obj[hold], res_px, r_vec, t_vec, intrinsics)
    return {
        "n_holdout": int(n_hold),
        "n_train": int(n - n_hold),
        "err_px_mean": float(res_px.mean()),
        "err_px_median": float(np.median(res_px)),
        "err_px_max": float(res_px.max()),
        "err_mm_mean": float(res_mm.mean()),
        "err_mm_median": float(np.median(res_mm)),
        "err_mm_p95": float(np.percentile(res_mm, 95)),
        "err_mm_max": float(res_mm.max()),
    }


def _print_quality_report(rms: float, n_inl: int, n: int, res_px: np.ndarray,
                          res_mm: np.ndarray, z_span_mm: float,
                          cov_u: float, cov_v: float,
                          holdout: Optional[dict],
                          rms_warn_px: float, min_inliers: int) -> bool:
    """Печать отчёта о качестве. Возвращает True, если качество приемлемое."""
    print("\n" + "=" * 62)
    print("ОТЧЁТ О КАЧЕСТВЕ КАЛИБРОВКИ")
    print("=" * 62)
    print(f"  Точек: {n}   инлайеров: {n_inl}   RMS = {rms:.2f} px")
    print(f"  Остатки, px: min {res_px.min():.1f}  median {np.median(res_px):.1f}  "
          f"p95 {np.percentile(res_px, 95):.1f}  max {res_px.max():.1f}")
    print(f"  Остатки, мм (на глубине точки): median {np.median(res_mm):.1f}  "
          f"p95 {np.percentile(res_mm, 95):.1f}  max {res_mm.max():.1f}")
    print(f"  Z-разброс точек: {z_span_mm:.0f} мм "
          f"({'МАЛО — добавь точки по высоте!' if z_span_mm < 150 else 'ок'})")
    print(f"  Покрытие кадра: {cov_u * 100:.0f}% по ширине, {cov_v * 100:.0f}% по высоте "
          f"({'МАЛО — разнеси точки по кадру!' if min(cov_u, cov_v) < 0.4 else 'ок'})")
    if holdout is not None:
        print(f"  HOLD-OUT ({holdout['n_holdout']} контрольных точек вне подгонки):")
        print(f"    ошибка: {holdout['err_px_mean']:.1f} px (mean) | "
              f"{holdout['err_mm_mean']:.1f} мм (mean)  "
              f"{holdout['err_mm_median']:.1f} мм (median)  "
              f"{holdout['err_mm_p95']:.1f} мм (p95)  {holdout['err_mm_max']:.1f} мм (max)")
    else:
        print("  HOLD-OUT: недостаточно точек (нужно >=9 для валидации)")

    ok = True
    if rms > rms_warn_px:
        print(f"  [!!] ПРЕДУПРЕЖДЕНИЕ: RMS {rms:.1f} px > порога {rms_warn_px:.0f} px — "
              "калибровка НЕнадёжна, ПЕРЕСНИМИ набор точек.")
        ok = False
    if n_inl < min_inliers:
        print(f"  [!!] ПРЕДУПРЕЖДЕНИЕ: инлайеров {n_inl} < {min_inliers} — "
              "мало согласованных точек, добавь точки и повтори.")
        ok = False
    print("=" * 62)
    return ok


def _solve_and_save(out_path: Path, intrinsics: dict, args=None) -> bool:
    n = len(_img_pts)
    if n < 4:
        print(f"[ERROR] Need >=4 points, have {n}.")
        return False

    rms_warn_px = float(getattr(args, "rms_warn_px", 5.0)) if args else 5.0
    min_inliers = int(getattr(args, "min_inliers", 15)) if args else 15
    target_points = int(getattr(args, "target_points", 30)) if args else 30
    outlier_px = float(getattr(args, "outlier_px", 8.0)) if args else 8.0
    holdout_frac = float(getattr(args, "holdout_frac", 0.25)) if args else 0.25
    log_path = Path(getattr(args, "log", "captures/calibration_log.json")
                    if args else "captures/calibration_log.json")

    if n < target_points:
        print(f"[INFO] Точек {n} < рекомендуемых {target_points} — "
              "лучше добрать (разные высоты и разные зоны кадра).")

    obj_all = np.array(_obj_pts, dtype=np.float64)
    img_all = np.array(_img_pts, dtype=np.float64)
    labels_all = list(_labels)
    cam_mat = _cam_matrix(intrinsics)

    # ── итеративная отбраковка выбросов: fit → drop(res > outlier_px) → refit ──
    keep = np.arange(n)
    r_vec = t_vec = inl_idx = None
    for it in range(3):
        obj, img = obj_all[keep], img_all[keep]
        ok, r_vec, t_vec, inl_idx = _fit_pnp(obj, img, cam_mat)
        if not ok:
            print("[ERROR] solvePnPRansac failed — try more / better-spread points.")
            return False
        res = _residuals_px(obj, img, r_vec, t_vec, cam_mat)
        bad = res > outlier_px
        if not np.any(bad) or (len(keep) - int(bad.sum())) < 6:
            break
        dropped = [f"{labels_all[keep[i]]}({res[i]:.0f}px)"
                   for i in np.where(bad)[0]]
        print(f"[OUTLIER] Итерация {it + 1}: отброшено {int(bad.sum())} точек: "
              f"{', '.join(dropped)}")
        keep = keep[~bad]

    obj, img = obj_all[keep], img_all[keep]
    n_used = len(obj)
    n_inl = len(inl_idx) if inl_idx is not None else n_used

    R, _ = cv2.Rodrigues(r_vec)
    T_cr = np.eye(4, dtype=np.float64)
    T_cr[:3, :3] = R
    T_cr[:3, 3] = t_vec.flatten()

    res_px = _residuals_px(obj, img, r_vec, t_vec, cam_mat)
    res_mm = _residuals_mm(obj, res_px, r_vec, t_vec, intrinsics)
    rms = float(np.sqrt(np.mean(res_px ** 2)))
    z_span_mm = float(obj[:, 2].max() - obj[:, 2].min()) * 1000.0
    cov_u, cov_v = _coverage_frac(img, intrinsics)

    holdout = _holdout_validation(obj, img, cam_mat, intrinsics, holdout_frac)
    quality_ok = _print_quality_report(
        rms, n_inl, n_used, res_px, res_mm, z_span_mm, cov_u, cov_v,
        holdout, rms_warn_px, min_inliers,
    )

    print(f"     t_vec = {np.round(t_vec.flatten(), 4)}")
    print("     T_cr =\n", np.round(T_cr, 5))

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_file": str(out_path),
        "n_points_collected": n,
        "n_points_used": n_used,
        "n_outliers_dropped": n - n_used,
        "n_inliers": n_inl,
        "rms_px": rms,
        "residual_px": {
            "median": float(np.median(res_px)),
            "p95": float(np.percentile(res_px, 95)),
            "max": float(res_px.max()),
        },
        "residual_mm": {
            "median": float(np.median(res_mm)),
            "p95": float(np.percentile(res_mm, 95)),
            "max": float(res_mm.max()),
        },
        "z_span_mm": z_span_mm,
        "coverage_frac_uv": [cov_u, cov_v],
        "holdout": holdout,
        "quality_ok": quality_ok,
        "saved": False,
    }

    if not quality_ok:
        try:
            ans = input("\nКачество НИЖЕ порога. Всё равно сохранить extrinsic.json? "
                        "[y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans != "y":
            print("[SKIP] Не сохранено. Пересними точки и нажми c снова.")
            _append_calibration_log(record, log_path)
            return False

    out_path.write_text(json.dumps({
        "T_cr": T_cr.tolist(),
        "R_vec": r_vec.flatten().tolist(),
        "t_vec": t_vec.flatten().tolist(),
        "rms_px": rms,
        "n_points": n_used,
        "n_inliers": n_inl,
        "z_span_mm": z_span_mm,
        "residual_px_median": float(np.median(res_px)),
        "residual_mm_median": float(np.median(res_mm)),
        "holdout": holdout,
        "coverage_frac_uv": [cov_u, cov_v],
        "calibrated_at": record["timestamp"],
        "method": "solvePnPRansac+RefineLM/multi-joint+outlier-reject",
        "intrinsics": intrinsics,
    }, indent=2))
    print(f"[SAVED] {out_path.resolve()}")
    record["saved"] = True
    _append_calibration_log(record, log_path)
    return True


def _validate_against_saved(out_path: Path, intrinsics: dict) -> None:
    """Проверка ТЕКУЩЕГО extrinsic.json на собранных точках БЕЗ подгонки.

    Использование: собери свежие контрольные точки (они не участвовали в
    подгонке сохранённой калибровки) и нажми v — получишь ошибку в px и мм.
    """
    n = len(_img_pts)
    if n < 3:
        print(f"[ERROR] Для валидации нужно >=3 точек, есть {n}.")
        return
    if not out_path.exists():
        print(f"[ERROR] {out_path} не найден — нечего валидировать.")
        return
    data = json.loads(out_path.read_text())
    r_vec = np.array(data["R_vec"], dtype=np.float64).reshape(3, 1)
    t_vec = np.array(data["t_vec"], dtype=np.float64).reshape(3, 1)
    obj = np.array(_obj_pts, dtype=np.float64)
    img = np.array(_img_pts, dtype=np.float64)
    cam_mat = _cam_matrix(intrinsics)
    res_px = _residuals_px(obj, img, r_vec, t_vec, cam_mat)
    res_mm = _residuals_mm(obj, res_px, r_vec, t_vec, intrinsics)
    print("\n" + "-" * 62)
    print(f"ВАЛИДАЦИЯ сохранённой калибровки ({out_path}) на {n} контрольных точках:")
    print(f"  ошибка, px: mean {res_px.mean():.1f}  median {np.median(res_px):.1f}  "
          f"max {res_px.max():.1f}")
    print(f"  ошибка, мм: mean {res_mm.mean():.1f}  median {np.median(res_mm):.1f}  "
          f"p95 {np.percentile(res_mm, 95):.1f}  max {res_mm.max():.1f}")
    for lbl, rpx, rmm in zip(_labels, res_px, res_mm):
        print(f"    {lbl:>4}: {rpx:6.1f} px  {rmm:7.1f} мм")
    print("-" * 62)


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
    # ── контроль качества (Задача 0) ──
    ap.add_argument("--target-points", type=int, default=30,
                    help="Рекомендуемое число точек (подсказка при solve)")
    ap.add_argument("--rms-warn-px", type=float, default=5.0,
                    help="Порог RMS (px): выше — предупреждение и запрос подтверждения")
    ap.add_argument("--min-inliers", type=int, default=15,
                    help="Минимум инлайеров: меньше — предупреждение")
    ap.add_argument("--outlier-px", type=float, default=8.0,
                    help="Порог остатка (px) для итеративной отбраковки выбросов")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="Доля точек для hold-out валидации")
    ap.add_argument("--log", default="captures/calibration_log.json",
                    help="Файл истории калибровок")
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
    print("► Разнеси точки и ПО ПОЛЮ КАДРА (левый/правый край, верх/низ)")
    print(f"► Набери >={args.target_points} точек → c = solve & save (с отчётом качества)")
    print("► v = валидация сохранённого extrinsic.json на текущих точках (без подгонки)\n")

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
            _solve_and_save(out_path, intrinsics, args)
        elif key == ord("v"):
            _validate_against_saved(out_path, intrinsics)
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
