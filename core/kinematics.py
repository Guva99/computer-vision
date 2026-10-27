"""
Модель манипулятора для коллизий: сферы на звеньях в кадре камеры.

Раньше жило в `collision.py`, из-за чего слой робота (`robot/services/
fk_service.py`) импортировал слой зрения — зависимость шла вверх. Здесь это
ядро: FK-поза плюс внешняя матрица дают набор сфер, и никакой обработки
изображений в расчёте нет.
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from core.types import LinkSphere
from kuka_fk import fk_joints


@dataclass(frozen=True)
class SphereModel:
    """Параметры сферической модели манипулятора (метры).

    Отдельный объект, а не четыре аргумента: это одна связная сущность — «чем
    мы приближаем робота», — и она целиком приходит из конфига. Заодно
    описывает, насколько грубо приближение: радиусы звеньев 6…16 см, то есть
    модель заведомо консервативна (сферы толще реального корпуса).
    """
    link_radii_m: Tuple[float, ...]
    spheres_per_link: int = 8
    gripper_length_m: float = 0.0
    gripper_radius_m: float = 0.05

    @classmethod
    def from_config(cls, cfg) -> "SphereModel":
        return cls(
            link_radii_m=tuple(cfg.collision_link_radii_m),
            spheres_per_link=int(cfg.collision_spheres_per_link),
            gripper_length_m=float(cfg.fk_gripper_length_m),
            gripper_radius_m=float(cfg.collision_gripper_radius_m),
        )


def link_part_name(link_index: int) -> str:
    """Имя части робота по номеру звена: им помечается ближайшая к объекту сфера."""
    if link_index == 5:
        return "gripper"
    if link_index == 4:
        return "wrist"
    return f"link{link_index + 1}"


def build_link_spheres(
    joint_angles: Sequence[float],
    T_cr: np.ndarray,
    model: SphereModel,
) -> List[LinkSphere]:
    """Сферы вдоль звеньев DH-модели, пересчитанные в кадр камеры.

    При gripper_length_m > 0 добавляется сегмент J6 (фланец) → TCP (кончик
    хвата), помеченный 'gripper': иначе дистанция мерилась бы до фланца, а
    касается препятствия первыми пальцы.

    Внимание: положение этих сфер зависит от внешней калибровки `T_cr` и от
    заявленной длины инструмента. На стенде `fk_gripper_length_m = 0.137`, а
    подбор по кликам дал ~0.088 — завышение сдвигает сферы хвата вперёд
    реального (см. docs/STAND_NOTES.md, запись от 30.09).
    """
    positions_robot = fk_joints(tuple(joint_angles[:6]),
                                gripper_length_m=model.gripper_length_m)
    spheres: List[LinkSphere] = []
    n_per = max(2, int(model.spheres_per_link))
    radii = model.link_radii_m

    segments = []  # (i0, i1, radius_m, part)
    for link_i in range(6):
        r = float(radii[link_i]) if link_i < len(radii) else 0.08
        segments.append((link_i, link_i + 1, r, link_part_name(link_i)))
    if model.gripper_length_m > 0.0 and len(positions_robot) > 7:
        segments.append((6, 7, float(model.gripper_radius_m), "gripper"))

    for i0, i1, radius, part in segments:
        p0 = np.asarray(positions_robot[i0], dtype=np.float64)
        p1 = np.asarray(positions_robot[i1], dtype=np.float64)
        h0 = np.array([p0[0], p0[1], p0[2], 1.0], dtype=np.float64)
        h1 = np.array([p1[0], p1[1], p1[2], 1.0], dtype=np.float64)
        c0 = (T_cr @ h0)[:3]
        c1 = (T_cr @ h1)[:3]
        # Звено целиком за камерой — сфер не ставим (проекция была бы бессмысленной).
        if c0[2] <= 1e-4 and c1[2] <= 1e-4:
            continue
        for j in range(n_per):
            t = j / (n_per - 1) if n_per > 1 else 0.5
            center = c0 + t * (c1 - c0)
            if center[2] <= 1e-4:
                continue
            spheres.append(
                LinkSphere(center=center.astype(np.float64), radius=radius, part=part)
            )
    return spheres
