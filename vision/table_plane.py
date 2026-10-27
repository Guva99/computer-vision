"""
Калибровка плоскости стола: чтение, объект-значение, действующие пороги.

Плоскость мерится один раз (`calibrate_table_plane.py`) в кадре камеры и от
внешней калибровки камера↔робот (`T_cr`) НЕ зависит — поэтому живёт отдельно.
Она задаёт высотный гейт детекции, то есть прямо определяет, предмет какой
высоты система вообще способна увидеть.

Почему отдельным модулем. Раньше загрузчик писал пороги прямо в AppConfig
(`self.cfg.collision_obj_plane_height_min_m = ...`), и действующее значение
(0.04 из калибровки) расходилось с тем, что лежит в конфиге (0.025) и попадает
в мету прогона. Здесь чтение отделено от применения: функция возвращает
значения, а вызывающий код применяет их явно, создавая КОПИЮ конфига.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TablePlane:
    """Калиброванная плоскость стола и её гейты (кадр камеры)."""
    plane: np.ndarray                      # (a, b, c, d): a·x + b·y + c·z + d = высота над столом
    gate_min_m: Optional[float] = None     # ниже — шум стола, не объект
    gate_max_m: Optional[float] = None     # выше — не препятствие на столе
    roi_px: Optional[tuple] = None         # рабочая зона (x0, y0, x1, y1)
    source: str = ""                       # откуда прочитано (для меты прогона)


def load_table_plane(cfg) -> Optional[TablePlane]:
    """Прочитать table_plane.json. None → высотный гейт недоступен, фолбэк на цвет."""
    if not bool(cfg.collision_obj_plane_gate):
        return None
    path = Path(cfg.table_plane_path)
    if not path.exists():
        print(f"[COLLISION] table_plane.json не найден ({path}) — цвет-гейт (фолбэк). "
              f"Запусти calibrate_table_plane.py.")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plane = np.asarray(data["plane"], dtype=np.float64).ravel()[:4]
        roi = data.get("roi_px") or None
        return TablePlane(
            plane=plane,
            gate_min_m=float(data["gate_min_m"]) if "gate_min_m" in data else None,
            gate_max_m=float(data["gate_max_m"]) if "gate_max_m" in data else None,
            roi_px=tuple(int(v) for v in roi[:4]) if roi else None,
            source=str(path),
        )
    except Exception as e:
        print(f"[COLLISION] Ошибка чтения table_plane.json: {e} — цвет-гейт (фолбэк)")
        return None


def apply_to_config(cfg, cal: Optional[TablePlane]) -> Tuple[object, Optional[tuple]]:
    """Вернуть (эффективный конфиг, ROI из калибровки).

    Значения калибровки имеют приоритет над конфигом, но применяются к КОПИИ:
    объект, переданный извне (его видят раннер, рекордер, мета прогона),
    остаётся нетронутым.
    """
    if cal is None:
        return cfg, None
    eff = {}
    if cal.gate_min_m is not None:
        eff["collision_obj_plane_height_min_m"] = cal.gate_min_m
    if cal.gate_max_m is not None:
        eff["collision_obj_plane_height_max_m"] = cal.gate_max_m
    cfg_eff = replace(cfg, **eff) if eff else cfg
    p = cal.plane
    print(f"[COLLISION] Плоскость стола загружена: n=({p[0]:+.2f},{p[1]:+.2f},"
          f"{p[2]:+.2f}) d={p[3]:+.3f}, гейт "
          f"{cfg_eff.collision_obj_plane_height_min_m * 100:.1f}.."
          f"{cfg_eff.collision_obj_plane_height_max_m * 100:.0f} см (из {cal.source})")
    return cfg_eff, cal.roi_px
