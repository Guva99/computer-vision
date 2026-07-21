"""
Геометрические утилиты.
"""
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает 4 точки многоугольника: [tl, tr, br, bl].
    
    Args:
        pts: Массив из 4 точек
        
    Returns:
        Упорядоченный массив точек
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def project_points_to_pixels(points_xyz: np.ndarray, intrinsics) -> np.ndarray:
    """
    Проецирует 3D точки на плоскость изображения.
    
    Args:
        points_xyz: Массив 3D точек (N, 3)
        intrinsics: Внутренние параметры камеры
        
    Returns:
        Массив 2D точек в пикселях (N, 2)
    """
    if points_xyz.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    x = points_xyz[:, 0]
    y = -points_xyz[:, 1]
    z = -points_xyz[:, 2]

    valid = z > 1e-6
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = (fx * (x / z) + ppx).astype(np.int32)
    v = (fy * (y / z) + ppy).astype(np.int32)
    return np.stack([u, v], axis=1)

