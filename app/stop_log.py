"""
Лог событий останова робота (валидация, Задача 2).

Измеряет цепочку «кадр → решение DANGER → робот физически остановился»:
  * t_capture  — захват кадра, на котором подтверждён DANGER;
  * t_danger   — момент решения (вызов stop_movement);
  * t_stop     — момент, когда угловая скорость ВСЕХ суставов по $AXIS_ACT
                 держится ниже stop_speed_eps_deg_s в течение
                 stop_confirm_frames_stopped кадров подряд.

Пишет captures/stop_events.csv:
  event_id, contour(2D/3D), scenario_id, frame_danger, t_capture_s, t_danger_s,
  t_stop_s, detect_latency_ms, stop_time_ms, total_reaction_ms,
  min_dist_m_at_danger, link_speed_m_s, resumed(0/1)

detect_latency_ms = t_danger - t_capture; total_reaction_ms = t_stop - t_capture.
link_speed_m_s — модуль линейной скорости ближайшего звена на момент DANGER
(по разнице FK-положений между кадрами); вместе с min_dist_m_at_danger даёт
пост-фактум оценку минимальной безопасной дистанции в tools/compute_metrics.py:
  d_safe ≈ min_dist_m_at_danger − link_speed_m_s · total_reaction_ms.

Включается флагом enable_stop_log (по умолчанию False — файл не создаётся).
"""
import csv
import math
from pathlib import Path
from typing import List, Optional, Sequence


def estimate_link_speed_m_s(prev_positions, positions, dt_s: float,
                            part: Optional[str] = None) -> Optional[float]:
    """Линейная скорость звена по разнице FK-положений между кадрами.

    positions — список точек fk_joints() [base, J1..J6, (TCP)].
    part: 'gripper' → TCP/фланец, 'wrist' → J5, иначе — максимум по
    подвижным точкам (консервативная оценка сверху).
    """
    if prev_positions is None or positions is None or dt_s <= 0:
        return None
    n = min(len(prev_positions), len(positions))
    if n < 2:
        return None

    def _speed(i: int) -> float:
        d = positions[i] - prev_positions[i]
        return float(math.sqrt(float(d[0]) ** 2 + float(d[1]) ** 2
                               + float(d[2]) ** 2)) / dt_s

    if part == "gripper":
        return _speed(n - 1)
    if part == "wrist" and n > 5:
        return _speed(5)
    return max(_speed(i) for i in range(1, n))


class StopEventTracker:
    """Отслеживает события останова и накапливает строки stop_events.csv.

    Использование (оба контура):
      * каждый кадр:      tracker.update(frame, t_now, joint_angles)
                          (joint_angles можно передавать только пока
                          tracker.waiting_stop — чтобы не читать робота зря);
      * при stop_movement: tracker.mark_danger(frame, t_capture, t_danger,
                          min_dist_m, link_speed_m_s);
      * при resume:        tracker.mark_resumed();
      * в конце прогона:   tracker.close() — записывает CSV.
    """

    def __init__(self, csv_path: str, contour: str, scenario_id: str = "",
                 speed_eps_deg_s: float = 0.5, confirm_frames: int = 3):
        self.csv_path = Path(csv_path)
        self.contour = contour
        self.scenario_id = scenario_id
        self.speed_eps = float(speed_eps_deg_s)
        self.confirm_frames = max(1, int(confirm_frames))

        self._events: List[dict] = []
        self._pending: Optional[dict] = None   # событие в ожидании t_stop
        self._slow_streak = 0
        self._prev_angles: Optional[Sequence[float]] = None
        self._prev_t: Optional[float] = None

    @property
    def waiting_stop(self) -> bool:
        """True, пока ждём физической остановки — в это время нужны углы."""
        return self._pending is not None

    def mark_danger(self, frame: int, t_capture: float, t_danger: float,
                    min_dist_m: Optional[float],
                    link_speed_m_s: Optional[float] = None) -> None:
        if self._pending is not None:
            # Предыдущее событие так и не дошло до t_stop — фиксируем как есть.
            self._finalize_pending(t_stop=None)
        self._pending = {
            "event_id": len(self._events) + 1,
            "contour": self.contour,
            "scenario_id": self.scenario_id,
            "frame_danger": frame,
            "t_capture_s": t_capture,
            "t_danger_s": t_danger,
            "min_dist_m_at_danger": min_dist_m,
            "link_speed_m_s": link_speed_m_s,
            "resumed": 0,
        }
        self._slow_streak = 0
        self._prev_angles = None
        self._prev_t = None

    def update(self, frame: int, t_now: float,
               joint_angles: Optional[Sequence[float]]) -> None:
        """Покадровое обновление: детект физической остановки по $AXIS_ACT."""
        if self._pending is None or joint_angles is None:
            return
        if self._prev_angles is not None and self._prev_t is not None:
            dt = t_now - self._prev_t
            if dt > 1e-6:
                max_speed = max(
                    abs(float(a) - float(b)) / dt
                    for a, b in zip(joint_angles, self._prev_angles)
                )
                if max_speed < self.speed_eps:
                    self._slow_streak += 1
                    if self._slow_streak >= self.confirm_frames:
                        self._finalize_pending(t_stop=t_now)
                else:
                    self._slow_streak = 0
        self._prev_angles = tuple(joint_angles)
        self._prev_t = t_now

    def mark_resumed(self) -> None:
        """Робот возобновил движение после события останова."""
        if self._pending is not None:
            # Resume до подтверждения остановки — закрываем событие как есть.
            self._finalize_pending(t_stop=None, resumed=1)
        elif self._events:
            self._events[-1]["resumed"] = 1

    def _finalize_pending(self, t_stop: Optional[float],
                          resumed: int = 0) -> None:
        ev = self._pending
        self._pending = None
        if ev is None:
            return
        ev["t_stop_s"] = t_stop
        ev["resumed"] = resumed
        self._events.append(ev)

    def close(self) -> None:
        if self._pending is not None:
            self._finalize_pending(t_stop=None)
        if not self._events:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["event_id", "contour", "scenario_id", "frame_danger",
                  "t_capture_s", "t_danger_s", "t_stop_s",
                  "detect_latency_ms", "stop_time_ms", "total_reaction_ms",
                  "min_dist_m_at_danger", "link_speed_m_s", "resumed"]
        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for ev in self._events:
                t_cap = ev["t_capture_s"]
                t_dng = ev["t_danger_s"]
                t_stp = ev.get("t_stop_s")
                row = {
                    "event_id": ev["event_id"],
                    "contour": ev["contour"],
                    "scenario_id": ev["scenario_id"],
                    "frame_danger": ev["frame_danger"],
                    "t_capture_s": _r(t_cap, 6),
                    "t_danger_s": _r(t_dng, 6),
                    "t_stop_s": _r(t_stp, 6),
                    "detect_latency_ms": _r((t_dng - t_cap) * 1000.0, 1),
                    "stop_time_ms": _r((t_stp - t_dng) * 1000.0, 1)
                    if t_stp is not None else "",
                    "total_reaction_ms": _r((t_stp - t_cap) * 1000.0, 1)
                    if t_stp is not None else "",
                    "min_dist_m_at_danger": _r(ev.get("min_dist_m_at_danger"), 4),
                    "link_speed_m_s": _r(ev.get("link_speed_m_s"), 4),
                    "resumed": ev.get("resumed", 0),
                }
                w.writerow(row)
        print(f"[STOP-LOG] {len(self._events)} событий → {self.csv_path}")
        self._events = []


def _r(v, nd: int):
    if v is None:
        return ""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return ""
        return round(f, nd)
    except (TypeError, ValueError):
        return ""
