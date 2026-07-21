"""
Покадровый лог решений и производительности 2D-контура (валидация, Задача 1).

Пишет два CSV (формат совместим с 3D-контуром для tools/compute_metrics.py):
  * perf-лог: t, frame, fps, cpu_proc_pct, ram_proc_mb, threads,
    collision_level, n_objects, min_dist_m, latency_ms, robot_paused, scenario_id
  * objects-лог: frame, obj_id, part, dist_m, level, reject_reason, scenario_id

Включается флагом ENABLE_DECISION_LOG в src/constants/config.py
(по умолчанию False — файлы не создаются, поведение не меняется).
"""
import csv
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return ""
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return ""


def gripper_object_distance_m(gripper_info: Optional[dict], obj: dict,
                              intrinsics) -> Optional[float]:
    """Метрическая дистанция хват↔объект для лога (детекция не меняется!).

    Восстанавливается из уже вычисленных данных: 3D-позиция хвата
    (position_3d) и депроекция центра объекта по его глубине. Если 3D-позиции
    хвата нет — деградация до |Δглубины| (модуль разницы высот).
    """
    if gripper_info is None:
        return None
    g3d = gripper_info.get("position_3d")
    g_depth = gripper_info.get("depth_m")
    d = obj.get("depth_m")
    if d is None:
        return None
    if g3d is not None and intrinsics is not None:
        cx, cy = obj.get("center", (None, None))
        if cx is not None:
            ox = (cx - intrinsics.ppx) * d / intrinsics.fx
            oy = (cy - intrinsics.ppy) * d / intrinsics.fy
            return math.sqrt((ox - g3d[0]) ** 2 + (oy - g3d[1]) ** 2
                             + (d - g3d[2]) ** 2)
    if g_depth is not None:
        return abs(d - g_depth)
    return None


def min_gripper_object_distance_m(gripper_info: Optional[dict],
                                  objects: List[dict],
                                  intrinsics) -> Optional[float]:
    """Минимум gripper_object_distance_m по всем объектам кадра."""
    best: Optional[float] = None
    for obj in objects or []:
        dist = gripper_object_distance_m(gripper_info, obj, intrinsics)
        if dist is not None and (best is None or dist < best):
            best = dist
    return best


class PerfLogger2D:
    PERF_FIELDS = ["t", "frame", "fps", "cpu_proc_pct", "ram_proc_mb",
                   "threads", "collision_level", "n_objects", "min_dist_m",
                   "latency_ms", "robot_paused", "scenario_id"]
    OBJ_FIELDS = ["frame", "obj_id", "part", "dist_m", "level",
                  "reject_reason", "scenario_id"]

    def __init__(self, perf_path: str, objects_path: str, scenario_id: str = ""):
        self.perf_path = Path(perf_path)
        self.objects_path = Path(objects_path)
        self.scenario_id = scenario_id
        self._perf_file = None
        self._perf_writer = None
        self._obj_file = None
        self._obj_writer = None

    def _open(self, path: Path, fields):
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        f = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        return f, w

    def sample(self, frame: int, fps: float, cpu: float, ram_mb: float,
               threads: int, collision_level: str, n_objects: int,
               min_dist_m: Optional[float], latency_ms: float,
               robot_paused: bool) -> None:
        if self._perf_writer is None:
            self._perf_file, self._perf_writer = self._open(
                self.perf_path, self.PERF_FIELDS
            )
        self._perf_writer.writerow({
            "t": round(time.time(), 3),
            "frame": frame,
            "fps": round(fps, 1),
            "cpu_proc_pct": round(cpu, 1),
            "ram_proc_mb": round(ram_mb, 1),
            "threads": threads,
            "collision_level": collision_level,
            "n_objects": n_objects,
            "min_dist_m": _fmt(min_dist_m),
            "latency_ms": round(latency_ms, 1),
            "robot_paused": int(bool(robot_paused)),
            "scenario_id": self.scenario_id,
        })

    def log_objects(self, frame: int, objects: Iterable[dict],
                    rejects: Iterable[dict] = ()) -> None:
        rows = []
        for o in objects:
            rows.append({
                "frame": frame,
                "obj_id": o.get("obj_id", -1),
                "part": o.get("part", "gripper"),
                "dist_m": _fmt(o.get("dist_m")),
                "level": o.get("level", ""),
                "reject_reason": "",
                "scenario_id": self.scenario_id,
            })
        for r in rejects:
            rows.append({
                "frame": frame,
                "obj_id": r.get("obj_id", -1),
                "part": r.get("part", ""),
                "dist_m": _fmt(r.get("dist_m")),
                "level": r.get("level", ""),
                "reject_reason": r.get("reject_reason", "unknown"),
                "scenario_id": self.scenario_id,
            })
        if not rows:
            return
        if self._obj_writer is None:
            self._obj_file, self._obj_writer = self._open(
                self.objects_path, self.OBJ_FIELDS
            )
        self._obj_writer.writerows(rows)

    def close(self) -> None:
        for f in (self._perf_file, self._obj_file):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._perf_file = self._perf_writer = None
        self._obj_file = self._obj_writer = None
