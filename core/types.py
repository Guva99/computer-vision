"""
Типы домена: сфера звена, объект сцены, результат оценки.

Чистые структуры данных на numpy — без cv2, Open3D и привязки к источнику
кадров. Благодаря этому их можно создавать в тестах руками и прогонять через
геометрию и политики без железа.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LinkSphere:
    """Сфера на оси звена манипулятора (кадр камеры, метры).

    Манипулятор представляется объединением таких сфер: дистанция до робота —
    это минимум по всем сферам (формула 10 статьи). `part` отвечает на вопрос
    «какая часть робота ближе»: gripper / wrist / linkN.
    """
    center: np.ndarray  # (3,) кадр камеры, метры
    radius: float
    part: str


@dataclass
class SceneObject:
    """Посторонний объект сцены: облако точек, центроид и габарит."""
    obj_id: int
    points: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    # Признаки протечки маски руки (заполняет детектор): блоб прилип к маске
    # руки и совпал с ней по глубине — скорее всего это пиксели самой руки.
    touches_arm: bool = False
    arm_depth_delta_m: float = float("inf")
    # Медианная высота над калиброванной плоскостью стола (м), NaN если
    # плоскость не задана.
    height_above_table_m: float = float("nan")
    # Тугой 2D-bbox самой компоненты в пикселях (x, y, w, h) — для аккуратного
    # бокса на RGB без инфляции от проекции 3D-AABB и для дедупа. None у 3D-ветки.
    bbox_2d: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CollisionResult:
    """Итог оценки одного объекта: дистанция, уровень и ближайшая часть робота."""
    obj_id: int
    min_dist_m: float
    level: str
    part: str
