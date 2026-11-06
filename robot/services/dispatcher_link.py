"""
Связь с диспетчером dist3 на контроллере KUKA — в отдельном потоке.

Зачем поток. Чтение `$POS_ACT` во время движения стоит 70–90 мс (замерено
2026-11-03, см. `krl/README.md`), а на стоящем роботе — 4 мс. Если держать
связь в потоке кадров, картинка встанет ровно тогда, когда на руку надо
смотреть. Поэтому команды и опрос позы живут здесь, а окно только читает
снимок состояния.

Команды диспетчера, которые использует этот модуль:
    CASE 2  — `$TOOL = COM_FRAME`   (смена TCP, движения нет)
    CASE 4  — `LIN` по массиву `COM_FRAME_ARRAY`

Про нулевую строку массива: KRL читает индексы с единицы, поэтому шлём две
строки, а `COM_LENGTH = 1`. Так же устроен штатный драйвер.
"""
from __future__ import annotations

import queue
import threading
import time

from robot.drivers.openshowvar import OpenShowVar

CASE_SET_TOOL = 2
CASE_LIN_ARRAY = 4
ZERO_FRAME = "{FRAME: X 0.0, Y 0.0, Z 0.0, A 0.0, B 0.0, C 0.0}"


def frame_str(xyz, abc) -> str:
    return (f"{{FRAME: X {xyz[0]:.3f}, Y {xyz[1]:.3f}, Z {xyz[2]:.3f}, "
            f"A {abc[0]:.3f}, B {abc[1]:.3f}, C {abc[2]:.3f}}}")


class RobotLink:
    """Связь с контроллером в отдельном потоке: окно не замирает на движении.

    Чтение `$POS_ACT` во время движения стоит 70–90 мс, поэтому в потоке кадров
    его делать нельзя — картинка встанет ровно тогда, когда на неё надо смотреть.
    """

    def __init__(self, ip: str, port: int):
        self.osv = OpenShowVar(ip=ip, port=port, read_timeout=3.0)
        self.cmds: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.status = "ожидание"
        self.pose = None
        self.busy = False
        self.stop_flag = False
        self.tool_frame = ZERO_FRAME
        self.hold_wait_s = 120.0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # --- служебное ---
    def read(self, name: str):
        raw = self.osv.read(name, False)
        return raw.decode(errors="ignore").strip() if raw is not None else None

    def _set(self, status: str, busy: bool | None = None):
        with self.lock:
            self.status = status
            if busy is not None:
                self.busy = busy
        print(f"[робот] {status}")

    def snapshot(self):
        with self.lock:
            return self.status, self.pose, self.busy

    def _wait_idle(self, timeout_s: float) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if (self.read("COM_CASEVAR") or "").strip() in ("0", "0.0"):
                return True
        return False

    def set_tool(self, frame: str) -> bool:
        """Поставить `$TOOL` через CASE 2. Движения нет, но все последующие
        команды начинают вести TCP, а не фланец. Вызывать до старта потока."""
        if self.read("$PRO_STATE") != "#P_ACTIVE":
            print("[ОШИБКА] dist3 не крутится — $TOOL поставить нечем.")
            return False
        self.osv.write("COM_FRAME", frame, False)
        self.osv.write("COM_CASEVAR", str(CASE_SET_TOOL), False)
        if not self._wait_idle(10.0):
            print("[ОШИБКА] CASE 2 не отработал.")
            return False
        self.tool_frame = self.read("$TOOL")
        print(f"[робот] $TOOL = {self.tool_frame}")
        return True

    # --- поток ---
    def start(self):
        self.thread.start()

    def _loop(self):
        last_poll = 0.0
        while not self.stop_flag:
            try:
                target, abc, timeout_s = self.cmds.get(timeout=0.2)
            except queue.Empty:
                if time.perf_counter() - last_poll > 1.0:
                    last_poll = time.perf_counter()
                    with self.lock:
                        self.pose = self.read("$POS_ACT")
                continue
            self._execute(target, abc, timeout_s)

    def _preflight(self) -> str | None:
        """Проверки перед отправкой. Строка — причина отказа, None — можно."""
        mode = self.read("$MODE_OP")
        if mode != "#T1":
            return f"режим {mode}, ожидался #T1"
        if (self.read("COM_CASEVAR") or "").strip() not in ("0", "0.0"):
            return "предыдущая команда не завершена"
        return None

    def _execute(self, target, abc, timeout_s: float):
        if not self.wait_active(self.hold_wait_s):
            self._set("ОТКАЗ: удержания так и не было", busy=False)
            self._drop_rest()
            return
        why = self._preflight()
        if why is not None:
            self._set(f"ОТКАЗ: {why}", busy=False)
            self._drop_rest()
            return
        f = frame_str(target, abc)
        self._set(f"едет -> X{target[0]:+.0f} Y{target[1]:+.0f} Z{target[2]:+.0f}", busy=True)
        self.osv.write("COM_FRAME_ARRAY[0]", f, False)
        self.osv.write("COM_FRAME_ARRAY[1]", f, False)
        self.osv.write("COM_LENGTH", "1", False)
        self.osv.write("COM_CASEVAR", str(CASE_LIN_ARRAY), False)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s and not self.stop_flag:
            if (self.read("COM_CASEVAR") or "").strip() in ("0", "0.0"):
                with self.lock:
                    self.pose = self.read("$POS_ACT")
                self._set(f"пришёл за {time.perf_counter() - t0:.1f} с", busy=False)
                return
        self._set("ТАЙМАУТ — смотри сообщение на пульте", busy=False)
        self._drop_rest()

    def _drop_rest(self) -> None:
        """Отрезок не прошёл — остальные отменяются. Иначе рука поедет дальше
        по маршруту, причину которого оператор ещё не понял."""
        n = 0
        while True:
            try:
                self.cmds.get_nowait()
                n += 1
            except queue.Empty:
                break
        if n:
            print(f"[робот] отменено отрезков маршрута: {n}")

    def wait_active(self, timeout_s: float = 300.0) -> bool:
        """Дождаться, пока dist3 запустят на пульте.

        Любое ручное перемещение пультом останавливает программу, поэтому
        между подводкой руки и запуском инструмента она почти всегда стоит.
        Падать на этом незачем — проще подождать и сказать, чего ждём.
        """
        if self.read("$PRO_STATE") == "#P_ACTIVE":
            return True
        print("")
        self._set("жду удержания на пульте — dist3 стоит")
        print(">>> ВОЗЬМИ УДЕРЖАНИЕ / ЗАПУСТИ dist3 — программа стоит. Жду...")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if self.read("$PRO_STATE") == "#P_ACTIVE":
                print(f">>> dist3 пошла, ждал {time.perf_counter() - t0:.0f} с")
                return True
        print(">>> Не дождался. Выхожу.")
        return False

    def shutdown(self):
        """Снять зависшую команду, если программа не крутится, и закрыть сокет."""
        self.stop_flag = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        try:
            if self.tool_frame != ZERO_FRAME:
                if self.set_tool(ZERO_FRAME):
                    print("[робот] $TOOL возвращён в ноль — магазин снова корректен")
                else:
                    print("[!!!] $TOOL ОСТАЛСЯ НЕНУЛЕВЫМ. Команды магазина "
                          "CASE 20/21 уедут мимо. Выбери TOOL 1 на пульте.")
            pending = (self.read("COM_CASEVAR") or "").strip()
            if pending not in ("0", "0.0") and self.read("$PRO_STATE") != "#P_ACTIVE":
                self.osv.write("COM_CASEVAR", "0", False)
                print(f"[робот] снял зависшую команду COM_CASEVAR={pending} "
                      f"(программа не крутится — иначе она исполнилась бы при старте)")
        except Exception:
            pass
        self.osv.close()


def parse_pose(pose: str | None):
    """XYZ из строки `$POS_ACT`. None, если строка не разобралась."""
    if not pose or "X " not in pose:
        return None
    try:
        return [float(pose.split(f"{ax} ")[1].split(",")[0]) for ax in ("X", "Y", "Z")]
    except (IndexError, ValueError):
        return None

