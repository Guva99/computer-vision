"""
Лог остановов (A3): сколько времени проходит от кадра до фактической остановки.

Пишет captures/stop_events.csv — вход для tools/compute_metrics.py (задержки и
минимальная безопасная дистанция d_safe). Колонки:

    scenario_id, frame_danger, t_frame, t_decision, t_command, t_stopped,
    detect_latency_ms, stop_time_ms, total_reaction_ms, min_dist_m_at_danger,
    link_speed_m_s, resumed, contact
    + t_stopped_precision_ms, stop_detect_source   (см. ниже)

Точки на оси времени (часы монотонные, time.perf_counter):
    t_frame     — захват кадра камерой;
    t_decision  — решение перешло в DANGER (после дебаунса confirm_frames);
    t_command   — запись $OV_PRO=0 вернула управление, команда ушла;
    t_stopped   — контроллер сообщает, что движения больше нет.

ЧЕСТНОСТЬ t_stopped. Прямого «скорость TCP = 0» контроллер не отдаёт, поэтому
момент остановки ДЕТЕКТИРУЕТСЯ опросом: отдельный поток после срабатывания
DANGER читает переменную состояния и ловит момент, когда движение прекратилось.
Источник выбирается автоматически при первом запуске:

    $VEL_ACT    — если читается, берём её (скорость напрямую);
    $PRO_STATE  — состояние программы (#P_STOP / #P_FREE);
    $AXIS_ACT   — фолбэк: углы перестали меняться (max|dA| < eps несколько
                  опросов подряд).

Разрешение измерения ограничено периодом опроса (контроллер отвечает ~60 мс),
поэтому в CSV пишутся ДВЕ служебные колонки: `t_stopped_precision_ms` —
фактический средний период опроса, и `stop_detect_source` — какая переменная
использовалась. Обе нужны, чтобы указать погрешность в статье, а не выдавать
оценку за прямое измерение.

Опрос идёт по ОТДЕЛЬНОМУ соединению с контроллером: JointAngleReader и
RobotService держат свои, смешивать запросы в одном сокете нельзя.

Включается флагом cfg.enable_stop_log (по умолчанию ВЫКЛ).
"""
import csv
import threading
import time
from pathlib import Path
from typing import Optional

from robot.services.joint_reader import parse_joint_angles

try:
    from robot.drivers.openshowvar import OpenShowVar
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

_FIELDS = [
    "scenario_id",
    "frame_danger",            # кадр, на котором решение стало DANGER
    "t_frame",                 # perf_counter захвата кадра
    "t_decision",              # perf_counter перехода решения в DANGER
    "t_command",               # perf_counter подтверждения записи $OV_PRO=0
    "t_stopped",               # perf_counter обнаруженной остановки
    "detect_latency_ms",       # t_decision - t_frame
    "stop_time_ms",            # t_command - t_decision
    "total_reaction_ms",       # t_stopped - t_frame
    "min_dist_m_at_danger",    # min_dist_m на кадре срабатывания
    "link_speed_m_s",          # скорость ближайшей точки FK-скелета
    "resumed",                 # робот поехал дальше после этого стопа (0/1)
    "contact",                 # был ли физический контакт — заполняет оператор
    "t_stopped_precision_ms",  # фактический период опроса = погрешность t_stopped
    "stop_detect_source",      # $VEL_ACT | $PRO_STATE | $AXIS_ACT | none
]

# Кандидаты на прямое измерение остановки, в порядке предпочтения.
_SOURCES = ("$VEL_ACT", "$PRO_STATE", "$AXIS_ACT")


class _StopEvent:
    """Одно срабатывание DANGER: копится в памяти, дописывается по ходу дела."""

    def __init__(self, scenario_id, frame, t_frame, t_decision, t_command,
                 min_dist_m, link_speed_m_s):
        self.scenario_id = scenario_id
        self.frame = int(frame)
        self.t_frame = float(t_frame)
        self.t_decision = float(t_decision)
        self.t_command = float(t_command)
        self.min_dist_m = min_dist_m
        self.link_speed_m_s = link_speed_m_s
        self.t_stopped: Optional[float] = None
        self.precision_ms: Optional[float] = None
        self.source = "none"
        self.resumed = 0
        self.contact = ""

    def row(self) -> dict:
        def ms(a, b):
            return f"{(a - b) * 1000.0:.2f}" if (a is not None and b is not None) else ""

        def num(v, fmt="{:.4f}"):
            try:
                x = float(v)
                return fmt.format(x) if x == x and abs(x) != float("inf") else ""
            except (TypeError, ValueError):
                return ""

        return {
            "scenario_id": self.scenario_id,
            "frame_danger": self.frame,
            "t_frame": f"{self.t_frame:.6f}",
            "t_decision": f"{self.t_decision:.6f}",
            "t_command": f"{self.t_command:.6f}",
            "t_stopped": f"{self.t_stopped:.6f}" if self.t_stopped else "",
            "detect_latency_ms": ms(self.t_decision, self.t_frame),
            "stop_time_ms": ms(self.t_command, self.t_decision),
            "total_reaction_ms": ms(self.t_stopped, self.t_frame),
            "min_dist_m_at_danger": num(self.min_dist_m),
            "link_speed_m_s": num(self.link_speed_m_s),
            "resumed": int(self.resumed),
            "contact": self.contact,
            "t_stopped_precision_ms": num(self.precision_ms, "{:.1f}"),
            "stop_detect_source": self.source,
        }


class StopWatcher:
    """Поток опроса контроллера: ловит момент, когда движение прекратилось."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.interval = float(getattr(cfg, "stop_poll_interval_s", 0.02))
        self.timeout = float(getattr(cfg, "stop_poll_timeout_s", 5.0))
        self.eps_deg = float(getattr(cfg, "stop_still_eps_deg", 0.05))
        self.need_still = max(2, int(getattr(cfg, "stop_still_samples", 3)))
        self._robot = None
        self.source = "none"
        self._connect()

    def _connect(self) -> None:
        if not ROBOT_AVAILABLE:
            print("[STOP] Драйвер робота недоступен — t_stopped писаться не будет")
            return
        cfg = self.cfg
        try:
            robot = OpenShowVar(ip=cfg.robot_ip, port=cfg.robot_port,
                                read_timeout=cfg.robot_read_timeout_s)
            if not robot.can_connect:
                print("[STOP] Контроллер недоступен — t_stopped писаться не будет")
                return
            self._robot = robot
        except Exception as e:
            print(f"[STOP] Не удалось открыть соединение для опроса: {e}")
            return
        # Выбрать лучший читаемый источник один раз при старте.
        for var in _SOURCES:
            try:
                raw = self._robot.read(var, debug=False)
            except Exception:
                raw = None
            if raw:
                self.source = var
                break
        if self.source == "none":
            print("[STOP] Ни одна переменная состояния не читается — t_stopped пуст")
        else:
            note = ("прямое измерение" if self.source != "$AXIS_ACT"
                    else "по остановке изменения углов")
            print(f"[STOP] Источник момента остановки: {self.source} ({note})")

    # ── разбор ответа контроллера ────────────────────────────────────────────
    def _sample(self):
        """Текущее показание источника или None, если прочитать не вышло."""
        if self._robot is None or self.source == "none":
            return None
        try:
            raw = self._robot.read(self.source, debug=False)
        except Exception:
            return None
        if not raw:
            return None
        text = raw.decode(errors="ignore")
        if self.source == "$AXIS_ACT":
            return parse_joint_angles(text)
        if self.source == "$PRO_STATE":
            return text.strip().upper()
        try:
            return (float(text.strip().replace(",", ".")),)
        except ValueError:
            return None

    def _is_stopped(self, cur, prev) -> bool:
        """Признак «движения больше нет» для выбранного источника."""
        if cur is None:
            return False
        if self.source == "$PRO_STATE":
            return ("P_STOP" in cur) or ("P_FREE" in cur) or ("P_END" in cur)
        if self.source == "$VEL_ACT":
            return abs(float(cur[0])) < 1e-3
        if prev is None or len(cur) != len(prev):
            return False
        return max(abs(a - b) for a, b in zip(cur, prev)) < self.eps_deg

    def watch(self, event: _StopEvent) -> None:
        """Запустить наблюдение за одним срабатыванием (не блокирует цикл)."""
        if self._robot is None or self.source == "none":
            return
        threading.Thread(target=self._watch_loop, args=(event,), daemon=True).start()

    def _watch_loop(self, event: _StopEvent) -> None:
        t0 = time.perf_counter()
        prev = None
        still = 0
        first_still_t = None
        stamps = []
        while (time.perf_counter() - t0) < self.timeout:
            t_poll = time.perf_counter()
            stamps.append(t_poll)
            cur = self._sample()
            if self._is_stopped(cur, prev):
                still += 1
                if first_still_t is None:
                    first_still_t = t_poll
                if still >= self.need_still:
                    # Момент остановки — ПЕРВЫЙ «тихий» опрос: движение прекратилось
                    # между ним и предыдущим, и ширина этого окна = период опроса.
                    event.t_stopped = first_still_t
                    break
            else:
                still = 0
                first_still_t = None
            prev = cur
            lag = self.interval - (time.perf_counter() - t_poll)
            if lag > 0:
                time.sleep(lag)
        if len(stamps) >= 2:
            event.precision_ms = (stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000.0
        event.source = self.source
        if event.t_stopped is None:
            print(f"[STOP] Кадр {event.frame}: остановка не обнаружена за "
                  f"{self.timeout:.1f} с — total_reaction_ms останется пустым")

    def close(self) -> None:
        if self._robot is not None:
            try:
                self._robot.sock.close()
            except Exception:
                pass


class StopEventLogger:
    """Накапливает события DANGER и пишет stop_events.csv при закрытии.

    Строки копятся в памяти, потому что `resumed` и `t_stopped` становятся
    известны ПОЗЖЕ самого события. Прогон длится 30 секунд, событий единицы —
    память не проблема, зато каждая строка уходит на диск уже полной.
    """

    def __init__(self, cfg, csv_path: Optional[str] = None, scenario_id: str = ""):
        self.cfg = cfg
        self.path = Path(csv_path or getattr(cfg, "stop_log_path",
                                             "captures/stop_events.csv"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_id = str(scenario_id or getattr(cfg, "scenario_id", "") or "")
        self.events: list = []
        self.watcher = StopWatcher(cfg)

    def on_danger(self, frame, t_frame, t_decision, t_command,
                  min_dist_m, link_speed_m_s) -> None:
        ev = _StopEvent(self.scenario_id, frame, t_frame, t_decision, t_command,
                        min_dist_m, link_speed_m_s)
        self.events.append(ev)
        self.watcher.watch(ev)

    def on_resume(self) -> None:
        """Робот поехал дальше: отметить последнее событие как возобновлённое."""
        if self.events:
            self.events[-1].resumed = 1

    def close(self, wait_s: float = 2.0) -> None:
        # Дать незавершённым наблюдателям дописать t_stopped.
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < wait_s and any(
            e.t_stopped is None and e.precision_ms is None for e in self.events
        ):
            time.sleep(0.05)
        self.watcher.close()
        exists = self.path.exists() and self.path.stat().st_size > 0
        with open(self.path, "a" if exists else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(e.row() for e in self.events)
        print(f"[STOP] Событий DANGER записано: {len(self.events)} -> {self.path}")
