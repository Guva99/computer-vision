"""
Геометрия восприятия: прореживание, высота над плоскостью, дистанции, боксы.

Единственное место, где живут эти расчёты. До выделения ядра они были
размножены: высота над плоскостью `p·n + d` считалась в четырёх местах
(collision.py ×2, vision/collision_service.py ×2), прореживание существовало в
двух вариантах, пересечение боксов — в двух файлах. Любая правка требовала
помнить про все копии.

Всё на numpy: ни cv2, ни Open3D. Прореживание «по центроиду» раньше шло через
`open3d.voxel_down_sample`; здесь оно воспроизведено на numpy и проверено на
совпадение с Open3D бит-в-бит (tests/test_core_functions.py). Побочная выгода —
не создаётся облако Open3D на каждый вызов в горячем цикле.
"""
from typing import List, Sequence, Tuple

import numpy as np

from core.types import LinkSphere

# Режимы прореживания. Разница НЕ косметическая, поэтому вызывающий код
# обязан выбирать режим осознанно:
#   CENTROID      — усреднённая точка вокселя (как Open3D). Применяется к облаку
#                   ТЕЛА руки: сглаживает шум измеренной глубины.
#   REPRESENTATIVE— реальная точка из входа. Применяется к облаку ОБЪЕКТА:
#                   дистанция меряется до действительно наблюдённой точки, а не
#                   до середины вокселя.
CENTROID = "centroid"
REPRESENTATIVE = "representative"


def voxel_down(points: np.ndarray, voxel_m: float, mode: str = CENTROID) -> np.ndarray:
    """Проредить облако вокселями размера `voxel_m` (метры, кадр камеры)."""
    pts = np.asarray(points)
    if len(pts) == 0 or voxel_m <= 0.0:
        return pts
    if mode == REPRESENTATIVE:
        keys = np.floor(pts / voxel_m).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[idx]
    if mode != CENTROID:
        raise ValueError(f"неизвестный режим прореживания: {mode!r}")
    pts = pts.astype(np.float64, copy=False)
    # Конвенция Open3D: начало отсчёта вокселей — min_bound минус половина
    # вокселя. Повторена ровно, иначе дистанции сдвинулись бы на доли вокселя.
    origin = pts.min(axis=0) - voxel_m * 0.5
    keys = np.floor((pts - origin) / voxel_m).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(uniq)).astype(np.float64)
    out = np.empty((len(uniq), 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.bincount(inverse, weights=pts[:, d], minlength=len(uniq)) / counts
    return out


def plane_height(points: np.ndarray, plane: Sequence[float]) -> np.ndarray:
    """Высота точек над плоскостью (a·x + b·y + c·z + d), метры.

    Плоскость нормирована при калибровке (`calibrate_table_plane.py`), поэтому
    значение — это сразу расстояние со знаком, а не масштабированная невязка.
    """
    a, b, c, d = (float(v) for v in np.asarray(plane).ravel()[:4])
    pts = np.asarray(points)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    return pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d


def box_overlap(a: Tuple[int, int, int, int],
                b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """(IoU, containment = inter / min_area) двух 2D-боксов (x, y, w, h).

    Второе число нужно для дедупа: вложенная рамка даёт containment ≈ 1 при
    маленьком IoU, и именно так опознаются две записи об одном объекте.
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0
    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    cont = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, cont


def min_dist_point_to_cloud(obj_points: np.ndarray, body_points: np.ndarray,
                            chunk: int = 512, percentile: float = 0.0) -> float:
    """Дистанция между облаком объекта и облаком тела робота (метры).

    Оба облака — в кадре камеры, поэтому результат НЕ зависит от внешней
    калибровки `T_cr`. Считается по частям, чтобы ограничить память.

    `percentile` > 0 возвращает низкий перцентиль расстояний вместо минимума:
    защита от единичных краевых пикселей, которые иначе дают ложный DANGER.
    """
    if len(obj_points) == 0 or len(body_points) == 0:
        return float("inf")
    body = body_points[None, :, :]
    nearest = np.empty(len(obj_points), dtype=np.float64)
    for i in range(0, len(obj_points), chunk):
        ch = obj_points[i:i + chunk][:, None, :]
        d = np.linalg.norm(ch - body, axis=2)
        nearest[i:i + chunk] = d.min(axis=1)
    if percentile <= 0.0:
        return float(nearest.min())
    return float(np.percentile(nearest, percentile))


def min_distance_to_spheres(spheres: List[LinkSphere],
                            obj_points: np.ndarray) -> Tuple[float, str, np.ndarray]:
    """Дистанция от облака объекта до объединения сфер (формула 10 статьи).

    Возвращает (дистанция, имя ближайшей части робота, центр её сферы).
    Отрицательные значения обрезаются нулём: точка внутри сферы — это контакт,
    а не «минус пять сантиметров».
    """
    if len(spheres) == 0 or len(obj_points) == 0:
        return float("inf"), "none", np.zeros(3, dtype=np.float64)
    best_dist = float("inf")
    best_part = "none"
    best_center = np.zeros(3, dtype=np.float64)
    for sp in spheres:
        diff = obj_points - sp.center.reshape(1, 3)
        dists = np.linalg.norm(diff, axis=1) - sp.radius
        d_min = float(np.min(dists))
        if d_min < best_dist:
            best_dist = d_min
            best_part = sp.part
            best_center = sp.center.copy()
    return max(0.0, best_dist), best_part, best_center
