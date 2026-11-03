"""
Указать место сборки кликом в кадре камеры.

Клик по пикселю → глубина в этой точке → 3D в системе камеры → через T_cr в
систему координат робота. Получается точка на поверхности, куда ставить первый
кубик. Результат сохраняется в task/build_point.json.

Зачем именно так. Точность пересчёта камера→робот измерена: hold-out медиана
6.9 мм, максимум 12.8 мм. Для установки кубика 40 мм на свободную поверхность
этого достаточно. А уровни башни считаются ПРИБАВКОЙ по вертикали от этой же
точки, поэтому от точности клика зависит только положение первого кубика —
стопка собирается с повторяемостью самого робота.

Кликаем ОДИН раз при настройке. Дальше точка живёт как обычная сохранённая
координата, и зрение в контуре управления не участвует: иначе промах хвата
стало бы не отличить от ошибки детекции, и эксперимент потерял бы смысл.

    python tools/pick_build_point.py
    python tools/pick_build_point.py --cube-mm 40

Клавиши: ЛКМ = указать точку, s = сохранить, q = выход.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from realsense_io import AppConfig, RealSenseCamera  # noqa: E402
from robot.services.fk_service import load_extrinsic  # noqa: E402

WINDOW = "Mesto sborki - klikni tochku"
_click = None


def _on_mouse(event, x, y, flags, param):
    global _click
    if event == cv2.EVENT_LBUTTONDOWN:
        _click = (x, y)


def pixel_to_camera(u: int, v: int, depth, intr, scale: float, win: int = 4):
    """Пиксель + глубина → точка в системе камеры (метры). None, если нет глубины.

    Берём медиану по окну: единичный пиксель на краю предмета даёт выброс.
    """
    h, w = depth.shape[:2]
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float64) * scale
    vals = patch[(patch > 0.15) & (patch < 1.8)]
    if vals.size < 5:
        return None
    z = float(np.median(vals))
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z], dtype=np.float64)


def camera_to_robot(p_cam: np.ndarray, T_cr: np.ndarray) -> np.ndarray:
    """Точка камеры → система координат робота (мм)."""
    T_rc = np.linalg.inv(T_cr)
    p = T_rc @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return p[:3] * 1000.0


def draw_hud(img, roi, p_rob, p_cam, saved: bool):
    x0, y0, x1, y1 = roi
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 220, 0), 2)
    cv2.putText(img, "WORK ZONE", (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 220, 0), 2)
    lines = ["LKM = ukazat tochku,  s = sohranit,  q = vyhod"]
    if p_rob is not None:
        lines.append(f"robot: X={p_rob[0]:+.1f} Y={p_rob[1]:+.1f} Z={p_rob[2]:+.1f} mm")
        lines.append(f"camera: z={p_cam[2]*1000:.0f} mm")
    if saved:
        lines.append("SOHRANENO")
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)


def save_point(path: Path, p_rob, p_cam, uv, cube_mm: float) -> None:
    """Сохранить точку. Z поднимается на половину кубика: клик даёт поверхность,
    а хват держит кубик за середину."""
    data = {
        "frame": "robot_base",
        "surface_xyz_mm": [round(float(v), 2) for v in p_rob],
        "place_xyz_mm": [round(float(p_rob[0]), 2), round(float(p_rob[1]), 2),
                         round(float(p_rob[2] + cube_mm / 2.0), 2)],
        "cube_mm": cube_mm,
        "pixel": [int(uv[0]), int(uv[1])],
        "camera_xyz_m": [round(float(v), 4) for v in p_cam],
        "note": ("place_xyz_mm — поза установки ПЕРВОГО кубика: поверхность плюс "
                 "половина ребра, потому что губки держат кубик за середину. "
                 "Уровень N считается прибавкой (N-1)*cube_mm по Z."),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Сохранено → {path}")
    print(f"     поверхность: X={p_rob[0]:+.1f} Y={p_rob[1]:+.1f} Z={p_rob[2]:+.1f}")
    print(f"     установка 1: Z={p_rob[2] + cube_mm/2:+.1f} (поверхность + половина кубика)")


def handle_click(uv_click, depth, intr, scale, T_cr):
    """Клик → (пиксель, точка камеры, точка робота) или None, если нет глубины."""
    u, v = uv_click
    pc = pixel_to_camera(u, v, depth, intr, scale)
    if pc is None:
        print(f"[!] В точке ({u},{v}) нет глубины — ткни рядом")
        return None
    p_rob = camera_to_robot(pc, T_cr)
    print(f"[точка] пиксель=({u},{v})  робот: X={p_rob[0]:+.1f} "
          f"Y={p_rob[1]:+.1f} Z={p_rob[2]:+.1f} мм")
    return (u, v), pc, p_rob


def run_loop(camera, roi, T_cr, out_path: Path, cube_mm: float) -> None:
    """Показ кадра, обработка кликов и сохранение точки."""
    global _click
    p_rob = p_cam = uv = None
    saved = False
    while True:
        frame = camera.get_aligned_frames()
        if frame is None:
            continue
        color, depth, intr, scale = frame
        img = color.copy()
        if _click is not None:
            got = handle_click(_click, depth, intr, scale, T_cr)
            _click = None
            if got is not None:
                uv, p_cam, p_rob = got
                saved = False
        if uv is not None:
            cv2.drawMarker(img, uv, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(img, uv, 10, (0, 0, 255), 2)
        draw_hud(img, roi, p_rob, p_cam, saved)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and p_rob is not None:
            save_point(out_path, p_rob, p_cam, uv, cube_mm)
            saved = True


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Указать место сборки кликом")
    ap.add_argument("--cube-mm", type=float, default=40.0)
    ap.add_argument("--out", default="task/build_point.json")
    args = ap.parse_args()

    cfg = AppConfig(enable_robot_control=False, show_debug_masks=False,
                    perf_monitor=False, show_o3d_window=False)
    T_cr = load_extrinsic(cfg.extrinsic_path)
    if T_cr is None:
        print("[ОШИБКА] Нет extrinsic.json — пересчёт в координаты робота невозможен")
        return 1
    roi = tuple(int(v) for v in cfg.collision_obj_roi_px)

    camera = RealSenseCamera(cfg)
    camera.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, _on_mouse)
    print("[OK] Кликни в точку, куда ставить первый кубик. s = сохранить.")
    try:
        run_loop(camera, roi, T_cr, Path(args.out), args.cube_mm)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
