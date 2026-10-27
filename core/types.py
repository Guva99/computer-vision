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
class DetectionReport:
    """Результат детекции за кадр: объекты + всё, что о них стоит знать снаружи.

    До этого детектор отдавал объекты возвратом, а записи об отбракованных
    кандидатах — через модульную переменную `collision._last_candidates`, которую
    вызывающий забирал отдельным вызовом. Скрытый канал мешал дважды: состояние
    жило между кадрами, и два конвейера в одном процессе (две камеры, сравнение
    2D с 3D) затирали данные друг друга.

    Диагностику детектор тоже больше не печатает — он возвращает `rejects`, а
    печатает вызывающий, у которого есть номер кадра для троттлинга.
    """
    objects: list                      # прошедшие гейты SceneObject
    mask: "np.ndarray"                 # маска кандидатов (для debug-мозаики)
    candidates: list = None            # строки objects.csv по отбракованным
    rejects: dict = None               # агрегат «причина → сколько» для консоли
    n_components: int = 0              # сколько связных компонент пришло в гейты

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []
        if self.rejects is None:
            self.rejects = {}


@dataclass
class CollisionResult:
    """Итог оценки одного объекта: дистанция, уровень и ближайшая часть робота."""
    obj_id: int
    min_dist_m: float
    level: str
    part: str
