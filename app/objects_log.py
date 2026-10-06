"""
Лог объектов (A4): строка на каждого рассмотренного кандидата.

Пишет captures/objects.csv в схеме, которую читает tools/compute_metrics.py:

    scenario_id, frame, obj_id, area_px, height_m, dist_m, level, reject_reason

Зачем. По decisions.csv видно ТОЛЬКО итог кадра (SAFE/WARN/DANGER). Когда
система пропускает опасную ситуацию (FN), из него не понять, что произошло:
объект вообще не выделился, выделился и был отбракован гейтом, или выделился,
но дистанция посчиталась завышенной. objects.csv отвечает на это напрямую —
compute_metrics сводит FN-кадры по `reject_reason` и показывает, какой гейт
стоит за пропусками.

Причина пустая = объект дошёл до оценки расстояния (тогда заполнены dist_m и
level). Непустая — какой этап его снял:
    area, shape, border, pts3d, keepz, span   — 2D-гейты детекции (collision.py)
    arm_reject, not_confirmed                 — трекер (vision/collision_service.py)

Включается флагом cfg.enable_objects_log (по умолчанию ВЫКЛ): при выключенном
CollisionService не собирает записи, а объект логгера не создаётся.
"""
import csv
import math
from pathlib import Path
from typing import Optional

_FIELDS = [
    "scenario_id",
    "frame",
    "obj_id",
    "area_px",
    "height_m",
    "dist_m",
    "level",
    "reject_reason",
]


def _num(v, fmt: str = "{:.4f}") -> str:
    """Число в CSV; нечисло и бесконечность → пусто (так их трактует метрика)."""
    if v is None or v == "":
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return fmt.format(x) if math.isfinite(x) else ""


class ObjectsLogger:
    """Буферизованная запись строк по объектам (flush пачками)."""

    def __init__(self, csv_path: str, scenario_id: str = "", flush_every: int = 200):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_id = str(scenario_id or "")
        self.flush_every = max(1, int(flush_every))
        self._rows: list = []
        self.n_written = 0
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writeheader()

    def update(self, frame: int, records: Optional[list]) -> None:
        if not records:
            return
        for r in records:
            self._rows.append({
                "scenario_id": self.scenario_id,
                "frame": int(frame),
                "obj_id": r.get("obj_id", ""),
                "area_px": r.get("area_px", ""),
                "height_m": _num(r.get("height_m")),
                "dist_m": _num(r.get("dist_m")),
                "level": r.get("level", "") or "",
                "reject_reason": r.get("reject_reason", "") or "",
            })
        if len(self._rows) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerows(self._rows)
        self.n_written += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self._flush()
        print(f"[OBJLOG] Записей по объектам: {self.n_written} -> {self.path}")
