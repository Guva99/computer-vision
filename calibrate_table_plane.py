"""
Калибровка опорной плоскости стола в системе координат КАМЕРЫ (одноразовая).

Зачем: детекция объектов должна опираться не на цвет, а на геометрию —
«объект = всё, что торчит над столом на height_min..height_max и не рука».
Для этого нужна плоскость стола. Она НЕ ищется каждый кадр (RANSAC хватает не ту
плоскость → мигание/objs=0), а калибруется ОДИН раз и замораживается: стол и
камера неподвижны, значит плоскость — константа.

Важно: плоскость меряется из облака глубины камеры и НЕ зависит от калибровки
робота (T_cr, extrinsic.json) — тот же принцип, по которому уже считается
дистанция до руки. Слабый extrinsic (6 инлайеров, RMS 27px) здесь ни при чём.

Плоскость: (a,b,c,d), |n|=1, ориентирована так, что точки ВЫШЕ стола (ближе к
камере) имеют height(p)=a·x+b·y+c·z+d > 0, сам стол ≈ 0.

Порядок работы:
  1. Убрать кубики со стола (нужна чистая поверхность) ИЛИ отметить мышью
     прямоугольник чистого участка стола (перетаскивание ЛКМ).
  2. Нажать SPACE — по точкам внутри ROI (накопленным за N кадров) строится
     RANSAC-плоскость. Появляется live-heatmap высоты над ней.
  3. Вернуть кубики: убедиться, что они «загораются» зелёным (проходят гейт
     height_min..height_max), а стол остаётся тёмным (высота ≈ 0).
  4. Нажать S — плоскость сохраняется в table_plane.json.

Клавиши: ЛКМ-перетаскивание=ROI, SPACE=подогнать плоскость, H=heatmap on/off,
         [ / ] = height_min ∓, - / = = height_max ∓, R=сброс ROI, S=сохранить,
         Q/ESC=выход.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pointcloud_pipeline import build_cloud_arrays
from realsense_io import AppConfig, RealSenseCamera

OUT_PATH = "table_plane.json"
ACCUM_FRAMES = 30          # сколько кадров ROI-точек копить для устойчивой подгонки
RANSAC_DIST_M = 0.006      # порог инлайера RANSAC (м)
RANSAC_ITERS = 400         # много итераций — плоскость важна, считаем один раз


def fit_plane_camera(points: np.ndarray) -> np.ndarray | None:
    """RANSAC-плоскость по точкам (кадр камеры) → (a,b,c,d), |n|=1 или None.

    Нормаль ориентируется так, чтобы объекты (меньший z, ближе к камере) имели
    ПОЛОЖИТЕЛЬНУЮ высоту: height=a·x+b·y+c·z+d>0. Стол получает height≈0.
    Санити: стол смотрит примерно на камеру (|c| велико); вертикаль (стена,
    профиль ограждения) с |c|~0 отвергается.
    """
    if len(points) < 100:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    try:
        model, inliers = cloud.segment_plane(
            distance_threshold=RANSAC_DIST_M, ransac_n=3, num_iterations=RANSAC_ITERS
        )
    except Exception:
        return None
    if len(inliers) < 50:
        return None
    a, b, c, d = (float(x) for x in model)
    n = (a * a + b * b + c * c) ** 0.5
    if n < 1e-9:
        return None
    a, b, c, d = a / n, b / n, c / n, d / n
    # Объекты ближе к камере (меньший z) должны давать height>0 → нужно c<0.
    if c > 0:
        a, b, c, d = -a, -b, -c, -d
    if abs(c) < 0.5:
        print(f"[WARN] |c|={abs(c):.2f} < 0.5 — похоже на стену/вертикаль, а не стол")
    return np.array([a, b, c, d], dtype=np.float64)


def height_image(points, valid_flat, plane, h, w) -> np.ndarray:
    """Карта высоты над плоскостью по пикселям (NaN вне valid). height=n·p+d."""
    a, b, c, d = plane
    heights = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
    img = np.full(h * w, np.nan, dtype=np.float32)
    img[valid_flat] = heights
    return img.reshape(h, w)


def render_heatmap(overlay, height_img, gate_min, gate_max) -> np.ndarray:
    """Зелёным подсветить пиксели в гейте (объекты), синим — «стол» (высота ≈0)."""
    out = overlay.copy()
    finite = np.isfinite(height_img)
    in_gate = finite & (height_img >= gate_min) & (height_img <= gate_max)
    near_table = finite & (np.abs(height_img) < gate_min)
    # Стол — приглушённый синий (виден, что распознан как поверхность).
    out[near_table] = (0.5 * out[near_table] + 0.5 * np.array([120, 40, 0])).astype(np.uint8)
    # Объекты над столом — яркий зелёный.
    out[in_gate] = (0.35 * out[in_gate] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    return out


class RoiSelector:
    def __init__(self):
        self.p0 = None
        self.p1 = None
        self.dragging = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p0 = (x, y)
            self.p1 = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.p1 = (x, y)
            self.dragging = False

    def rect(self):
        if self.p0 is None or self.p1 is None:
            return None
        x0, x1 = sorted((self.p0[0], self.p1[0]))
        y0, y1 = sorted((self.p0[1], self.p1[1]))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        return x0, y0, x1, y1

    def reset(self):
        self.p0 = self.p1 = None
        self.dragging = False


def main():
    cfg = AppConfig()
    camera = RealSenseCamera(cfg)
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] Камера недоступна: {e}")
        return
    print("[OK] Камера запущена")
    print("[INFO] ЛКМ-перетаскивание = отметить чистый участок стола, SPACE = подогнать")

    win = "calibrate_table_plane"
    cv2.namedWindow(win)
    roi = RoiSelector()
    cv2.setMouseCallback(win, roi.on_mouse)

    accum: list = []        # кольцевой буфер ROI-точек (кадр камеры)
    plane = None
    show_heatmap = True
    gate_min = 0.015
    gate_max = 0.30

    while True:
        frame = camera.get_aligned_frames()
        if frame is None:
            continue
        color_bgr, depth, intrinsics, depth_scale = frame
        h, w = color_bgr.shape[:2]
        points, colors, valid_flat = build_cloud_arrays(
            color_bgr, depth, intrinsics, depth_scale, cfg.depth_min_m, cfg.depth_max_m
        )

        # Накопление точек внутри ROI (для устойчивой подгонки по нескольким кадрам).
        rect = roi.rect()
        if rect is not None and len(points) > 0:
            x0, y0, x1, y1 = rect
            roi_mask = np.zeros((h, w), dtype=bool)
            roi_mask[y0:y1, x0:x1] = True
            sel = roi_mask.reshape(-1)[valid_flat]
            if np.any(sel):
                accum.append(points[sel])
                if len(accum) > ACCUM_FRAMES:
                    accum.pop(0)

        # Отрисовка.
        overlay = color_bgr.copy()
        if plane is not None and show_heatmap and len(points) > 0:
            hi = height_image(points, valid_flat, plane, h, w)
            overlay = render_heatmap(overlay, hi, gate_min, gate_max)

        if rect is not None:
            x0, y0, x1, y1 = rect
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 2)

        status = "plane: OK" if plane is not None else "plane: --- (SPACE)"
        cv2.putText(overlay, f"{status}   gate: {gate_min*100:.1f}..{gate_max*100:.0f} cm",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "LMB=ROI SPACE=fit H=heatmap [ ] -/= gate R=reset S=save Q=quit",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if plane is not None:
            a, b, c, d = plane
            cv2.putText(overlay, f"n=({a:+.2f},{b:+.2f},{c:+.2f}) d={d:+.3f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if len(accum) == 0:
                print("[WARN] Сначала отметь ROI на чистом столе (перетаскивание ЛКМ)")
            else:
                pts = np.vstack(accum)
                plane = fit_plane_camera(pts)
                if plane is None:
                    print("[WARN] Плоскость не найдена — увеличь ROI/убери объекты")
                else:
                    a, b, c, d = plane
                    print(f"[OK] Плоскость: a={a:+.4f} b={b:+.4f} c={c:+.4f} d={d:+.4f} "
                          f"(по {len(pts)} точкам)")
        elif key == ord("h"):
            show_heatmap = not show_heatmap
        elif key == ord("r"):
            roi.reset()
            accum.clear()
        elif key == ord("["):
            gate_min = max(0.0, gate_min - 0.005)
        elif key == ord("]"):
            gate_min += 0.005
        elif key == ord("-"):
            gate_max = max(gate_min + 0.01, gate_max - 0.02)
        elif key == ord("="):
            gate_max += 0.02
        elif key == ord("s"):
            if plane is None:
                print("[WARN] Нет плоскости — сначала SPACE")
            else:
                data = {
                    "frame": "camera",
                    "plane": [float(x) for x in plane],
                    "gate_min_m": float(gate_min),
                    "gate_max_m": float(gate_max),
                    "ransac_dist_m": RANSAC_DIST_M,
                    "created": datetime.now().isoformat(timespec="seconds"),
                }
                Path(OUT_PATH).write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(f"[SAVED] {OUT_PATH}: {data}")

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
