"""
Ожидаемое состояние сцены: что в рабочей зоне стоять ДОЛЖНО.

Измерено 2026-11-09 на записи `T2_baseline_tower`: 1460 кадров из 5279 (27.7%)
система провела в DANGER, и все срабатывания ложные — человека в зоне не было
ни секунды. Все 2266 принятых объектов оказались собственной башней робота на
высотах 0.085..0.235 м. Геометрия при этом работала верно: предмет есть, рука
рядом. Не хватало знания о том, что рука там и должна быть.

Почему ожидаемые объекты описаны ВЕРТИКАЛЬНЫМИ КОЛОННАМИ по XY, а не коробками.
Губки в этом техпроцессе смотрят вниз, поэтому TCP всегда строго под фланцем, и
XY башни равен XY заученной позы `build_base` ТОЧНО. А вот Z потребовал бы длины
хвата, которая на стенде до сих пор спорная: 137 мм в конфиге, ~88 мм по
подбору, 191.7 мм по двум заученным инструментам пульта. Колонна по XY от этого
параметра не зависит вовсе, и устойчива к той части ошибки калибровки, которая
приходится на высоту.

Правило сопоставления намеренно несимметрично: совпадение по положению И ПО
РАЗМЕРУ. Пятно заметно шире ожидаемого — значит это не только башня, и тревога
обязана остаться. Без этого ограничения рука, поднесённая к стопке, пряталась бы
за ней, и правило из полезного стало бы опасным.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

CUBE_MM = 40.0
# Половина диагонали кубика: наибольший горизонтальный полуразмер при любом
# развороте вокруг вертикали.
CUBE_HALF_DIAG_MM = CUBE_MM * 0.5 * 1.4143


@dataclass(frozen=True)
class Column:
    """Вертикальная область, занятая ожидаемым предметом."""

    x_mm: float
    y_mm: float
    r_mm: float          # радиус допуска по XY
    z_top_mm: float      # верх ожидаемого (для диагностики, в сравнение не входит)
    label: str


def camera_to_robot_mm(p_cam_m: Sequence[float], T_cr: np.ndarray) -> np.ndarray:
    """Точка камеры (м) → координаты робота (мм). T_cr переводит робота в камеру."""
    p = np.linalg.inv(T_cr) @ np.array([p_cam_m[0], p_cam_m[1], p_cam_m[2], 1.0])
    return p[:3] * 1000.0


def expected_columns(state, build_xy, flange_xy,
                     cube_mm: float = CUBE_MM,
                     pad_mm: float = 60.0) -> List[Column]:
    """Ожидаемые предметы для текущей фазы.

    `pad_mm` — допуск по XY: медиана ошибки внешней калибровки 20.5 мм плюс
    половина диагонали кубика плюс запас на шум глубины.

    Магазин в список не входит: он вне рабочей ROI детекции, и его кубики в
    кандидаты не попадают.
    """
    if state is None:
        return []
    out: List[Column] = []
    n = int(getattr(state, "cubes_placed", 0) or 0)
    if n > 0 and build_xy is not None:
        out.append(Column(float(build_xy[0]), float(build_xy[1]),
                          CUBE_HALF_DIAG_MM + pad_mm, n * cube_mm, "tower"))
    if bool(getattr(state, "cube_in_gripper", False)) and flange_xy is not None:
        out.append(Column(float(flange_xy[0]), float(flange_xy[1]),
                          CUBE_HALF_DIAG_MM + pad_mm, float("nan"), "carried"))
    return out


def object_radius_mm(points_cam_m, centroid_cam_m, T_cr) -> float:
    """Горизонтальный полуразмер наблюдаемого пятна в координатах робота.

    Берётся 95-й процентиль, а не максимум: одна шумная точка глубины не должна
    раздувать объект и ломать сопоставление.
    """
    if points_cam_m is None or len(points_cam_m) < 4:
        return 0.0
    pts = np.asarray(points_cam_m, dtype=np.float64)
    R = np.linalg.inv(T_cr)[:3, :3]
    c = np.asarray(centroid_cam_m, dtype=np.float64)
    rel = (R @ (pts - c).T).T * 1000.0
    return float(np.percentile(np.hypot(rel[:, 0], rel[:, 1]), 95))


def match(p_rob_mm, obj_radius_mm_val: float, columns: Sequence[Column],
          max_width_mm: float) -> Optional[str]:
    """Совпал ли объект с ожидаемым. Возвращает метку или None.

    Два условия, оба обязательны:
      • центр внутри колонны по XY;
      • пятно не шире ожидаемого — иначе рядом с предметом что-то ещё.
    """
    if obj_radius_mm_val > max_width_mm:
        return None
    for col in columns:
        d = float(np.hypot(p_rob_mm[0] - col.x_mm, p_rob_mm[1] - col.y_mm))
        if d <= col.r_mm:
            return col.label
    return None
