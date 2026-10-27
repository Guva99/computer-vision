"""
Растеризация объектов сцены в бинарную маску.

Нужна в двух местах: для дампа масок (метрика IoU в статье) и для debug-мозаики.
Операция дорогая — питоновский проход по точкам, — поэтому вызывается только на
кадрах дампа и при включённой отладке, а не каждый кадр.

Живёт в `vision/`, а не в ядре: использует морфологию cv2.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.projection import project_points
from core.types import SceneObject


def scene_objects_image_mask(
    objects: List[SceneObject],
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Точки объектов → бинарная маска (0/255) в кадре изображения."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in objects:
        uv = project_points(obj.points, intrinsics)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) == 0:
            continue
        us = np.rint(uv[:, 0]).astype(np.int64)
        vs = np.rint(uv[:, 1]).astype(np.int64)
        inside = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        mask[vs[inside], us[inside]] = 255
    if np.count_nonzero(mask) > 0:
        # Точки облака разрежены (дырки глубины) — замыкаем их в связную область,
        # иначе IoU считался бы по «сетке» пикселей, а не по силуэту объекта.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask
