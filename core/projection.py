"""
Проекция из кадра камеры в пиксели (модель камеры-обскура).

Чистая математика без cv2: нужна и для рисования рамок, и для тестов «попал ли
центроид под маску руки», и для растеризации точек объекта в маску. Раньше
формула `u = fx·X/Z + cx` была вписана в четырёх местах — здесь она одна.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def project_points(points: np.ndarray, intrinsics: Dict[str, float]) -> np.ndarray:
    """Точки (N,3) в кадре камеры → пиксели (N,2), float.

    Точки с Z <= 0 отбрасываются: за плоскостью камеры проекция не определена.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    z = pts[:, 2]
    ok = z > 1e-4
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[ok, 0] = fx * pts[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * pts[ok, 1] / z[ok] + cy
    return out


def project_point(point: Sequence[float], intrinsics: Dict[str, float]) -> Optional[Tuple[int, int]]:
    """Одна точка → (u, v) целыми пикселями или None, если она за камерой."""
    uv = project_points(np.asarray(point, dtype=np.float64).reshape(1, 3), intrinsics)[0]
    if not np.isfinite(uv).all():
        return None
    return int(round(uv[0])), int(round(uv[1]))


def project_aabb_to_image(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Габарит объекта (3D AABB) → охватывающая рамка (x, y, w, h) или None.

    Рамка строится по восьми углам, поэтому она ЗАВЕДОМО ШИРЕ самого объекта в
    кадре (проекция параллелепипеда). Для аккуратного прямоугольника у
    SceneObject есть `bbox_2d` — тугой бокс самой компоненты.
    """
    h, w = image_shape[:2]
    lo = np.asarray(aabb_min, dtype=np.float64)
    hi = np.asarray(aabb_max, dtype=np.float64)
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                        [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
                        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]]], dtype=np.float64)
    uv = project_points(corners, intrinsics)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0:
        return None
    x0 = int(max(0, np.floor(uv[:, 0].min())))
    y0 = int(max(0, np.floor(uv[:, 1].min())))
    x1 = int(min(w - 1, np.ceil(uv[:, 0].max())))
    y1 = int(min(h - 1, np.ceil(uv[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0
