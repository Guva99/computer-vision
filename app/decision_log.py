"""
Покадровый лог решений по объектам (валидация, Задачи 1 и 8).

Пишет captures/objects.csv: по строке на каждый объект сцены в кадре —
frame, obj_id, part, dist_m, level, reject_reason, scenario_id.
Отбракованные кандидаты (reject_reason != "") пишутся с obj_id=-1: по ним
считается сводка «на какую причину отбраковки пришлись пропуски» (Задача 8).

Включается флагом AppConfig.enable_decision_log (по умолчанию False —
никаких файлов не создаётся, поведение системы не меняется).
"""
import csv
from pathlib import Path
from typing import Iterable, Optional


class ObjectsCsvLogger:
    FIELDS = ["frame", "obj_id", "part", "dist_m", "level",
              "reject_reason", "scenario_id"]

    def __init__(self, path: str, scenario_id: str = ""):
        self.path = Path(path)
        self.scenario_id = scenario_id
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def _ensure_open(self) -> None:
        if self._writer is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if new_file:
            self._writer.writeheader()

    def log_frame(self, frame: int, objects: Iterable[dict],
                  rejects: Iterable[dict] = ()) -> None:
        """objects: [{obj_id, part, dist_m, level}], rejects: [{reject_reason, ...}]."""
        rows = []
        for o in objects:
            rows.append({
                "frame": frame,
                "obj_id": o.get("obj_id", -1),
                "part": o.get("part", ""),
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
        self._ensure_open()
        self._writer.writerows(rows)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        import math
        if math.isnan(float(v)) or math.isinf(float(v)):
            return ""
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""
