"""
Восстановление состояния техпроцесса из записи — для офлайн-прогонов.

Зачем. Базовая линия `T2_baseline_tower` снята до того, как появился файл
состояния: цикл тогда ничего о себе не писал. Пересниматься ради этого нельзя —
сравнение «до и после» обязано идти на ОДНИХ И ТЕХ ЖЕ кадрах, иначе разница
окажется разницей между прогонами, а не между правилами.

Выручает то, что углы записаны покадрово (5278 из 5279). FK от них даёт позу
фланца, а позы `build_base` и координаты магазина известны точно.

Фазу при этом знать не нужно вовсе — достаточно чередования визитов:

    визит к месту сборки с кубиком      → поставил:  стопка +1, руки пусты
    визит к месту сборки без кубика     → снял:      стопка −1, кубик в руке
    визит к магазину без кубика         → взял:      кубик в руке
    визит к магазину с кубиком          → вернул:    руки пусты

Сборка и разборка получаются сами, без отдельного признака. Начальное
состояние задаётся вызывающим: для `T2_baseline_tower` это пустой стол и
полный магазин.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from kuka_fk import fk_tcp
from task.state import ProcessState

CUBE_MM = 40.0
# Допуски распознавания визита. По XY — жёстко: место сборки и магазин разнесены
# на сотни миллиметров, спутать их нельзя. По Z — мягче: уровень опознаётся с
# запасом на то, что рука проходит его насквозь между кадрами при 5.9 FPS.
BUILD_XY_TOL_MM = 45.0
LEVEL_Z_TOL_MM = 25.0
MAG_X_MAX_MM = -150.0
MAG_Y_MIN_MM = 300.0


def flange_xyz_mm(angles) -> np.ndarray:
    return fk_tcp(tuple(angles))[0] * 1000.0


def read_joints(path: str) -> List[Dict]:
    """joints.csv → [{frame, angles|None}] по порядку кадров."""
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                a = [float(r[f"A{i}"]) for i in range(1, 7)]
            except (KeyError, ValueError, TypeError):
                a = None
            out.append({"frame": int(r["frame"]), "angles": a})
    return out


def _where(p, build_xy) -> str:
    """Где рука: build, magazine или free. Уровень здесь НЕ определяется."""
    if p[0] < MAG_X_MAX_MM and p[1] > MAG_Y_MIN_MM:
        return "magazine"
    near = (abs(p[0] - build_xy[0]) <= BUILD_XY_TOL_MM
            and abs(p[1] - build_xy[1]) <= BUILD_XY_TOL_MM)
    return "build" if near else "free"


def _level_of(z_mm: float, level_zs) -> int:
    """Уровень по САМОЙ НИЗКОЙ точке визита: там кубик ставится или берётся.

    По последнему замеру перед выходом считать нельзя — рука уходит вверх на
    зависание 100 мм, и подъём попадает в допуск уровня на два выше.
    """
    d = [abs(z_mm - z) for z in level_zs]
    n = int(min(range(len(d)), key=lambda k: d[k]))
    return n + 1 if d[n] <= LEVEL_Z_TOL_MM else 0


def _apply(st: ProcessState, place: str, level: int) -> ProcessState:
    """Применить завершённый визит к состоянию."""
    if place == "magazine":
        if st.cube_in_gripper:
            return replace(st, cube_in_gripper=False, phase="unstack")
        return replace(st, cube_in_gripper=True, phase="assemble")
    if level <= 0:
        return st
    if st.cube_in_gripper:
        return replace(st, cube_in_gripper=False, phase="assemble",
                       level=level, cubes_placed=level)
    return replace(st, cube_in_gripper=True, phase="unstack",
                   level=level, cubes_placed=max(0, level - 1))


def reconstruct(joints_csv: str, build_xy, build_z_mm: float,
                levels: int = 6, start: Optional[ProcessState] = None):
    """Состояние на каждый кадр записи.

    Визит засчитывается по ВЫХОДУ из зоны: пока рука там, кубик ещё не
    отпущен. Уровень берётся по самой низкой точке за визит.
    """
    level_zs = [build_z_mm + (n - 1) * CUBE_MM for n in range(1, levels + 1)]
    st = start or ProcessState()
    out = {}
    where_prev, z_min = "free", float("inf")
    for rec in read_joints(joints_csv):
        if rec["angles"] is not None:
            p = flange_xyz_mm(rec["angles"])
            where = _where(p, build_xy)
            if where != "free":
                z_min = min(z_min, float(p[2]))
            elif where_prev != "free":
                st = _apply(st, where_prev, _level_of(z_min, level_zs))
                z_min = float("inf")
            where_prev = where
        out[rec["frame"]] = st
    return out


def load_build_pose(poses_yaml: str):
    """(XY, Z) заученной позы build_base в координатах робота, мм."""
    import yaml
    data = yaml.safe_load(Path(poses_yaml).read_text(encoding="utf-8")) or {}
    c = (data.get("poses") or {}).get("build_base", {}).get("cartesian")
    if not c:
        raise SystemExit(f"[task] в {poses_yaml} нет build_base с координатами")
    return (float(c["X"]), float(c["Y"])), float(c["Z"])
