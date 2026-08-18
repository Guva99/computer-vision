"""
Пер-кадровый лог решений системы — вход для tools/compute_metrics.py.

Пишет CSV в схеме, которую ждёт compute_metrics: колонки `frame`,
`collision_level`, `min_dist_m`, `scenario_id`. По этому логу + разметке GT
(tools/annotate.py) считаются проценты качества: precision/recall/F1 по DANGER,
доля ложных остановов (FP) и пропущенных опасностей (FN), MAE/RMSE дистанции.

Зачем: оценивать изменения детекции числами на ОДНОЙ записи, а не на глаз по
скриншотам — тогда видно, реально ли правка помогла и что она не сломала.

Ничего не меняет в детекции: только читает готовый результат кадра. Включается
флагом cfg.enable_decision_log (по умолчанию ВЫКЛ) — при выключенном флаге
объект не создаётся и накладных расходов нет.
"""
import csv
import time
from pathlib import Path
from typing import Optional

_FIELDS = [
    "frame",          # номер кадра — ключ сшивки с разметкой GT
    "t_wall",         # unix-время кадра (для задержек/синхронизации)
    "scenario_id",    # идентификатор сценария из scenarios.yaml
    "collision_level",  # SAFE/WARN/DANGER — предсказание системы
    "min_dist_m",     # минимальная дистанция рука↔объект, м
    "focus_part",     # какая часть робота ближе: gripper/arm/wrist
    "n_objects",      # сколько объектов система видит в кадре
    "fps",            # мгновенный FPS (контекст: просадки влияют на задержку)
]


class DecisionLogger:
    """Буферизованная запись решений по кадрам в CSV (flush пачками)."""

    def __init__(self, csv_path: str, scenario_id: str = "", flush_every: int = 60):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_id = str(scenario_id or "")
        self.flush_every = max(1, int(flush_every))
        self._rows: list = []
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        if not self._header_written:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
            self._header_written = True

    def update(self, frame: int, level: str, min_dist_m: float = float("inf"),
               focus_part: str = "", n_objects: int = 0,
               fps: Optional[float] = None) -> None:
        d = float(min_dist_m)
        self._rows.append({
            "frame": int(frame),
            "t_wall": f"{time.time():.6f}",
            "scenario_id": self.scenario_id,
            "collision_level": str(level),
            # inf → пусто: compute_metrics трактует нечисло как «нет значения»
            "min_dist_m": f"{d:.4f}" if d == d and d != float("inf") else "",
            "focus_part": str(focus_part or ""),
            "n_objects": int(n_objects),
            "fps": f"{fps:.2f}" if fps is not None else "",
        })
        if len(self._rows) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerows(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self._flush()
