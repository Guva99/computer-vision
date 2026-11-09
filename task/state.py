"""
Состояние техпроцесса: что робот делает прямо сейчас.

Зачем файл, а не вызов. Циклом сборки управляет один процесс
(`tools/build_tower.py`), а зрение живёт в другом (`tools/run_live.py`).
Связывать их в один процесс ради обмена состоянием — значит поставить
управление роботом в зависимость от частоты кадров. Поэтому цикл пишет
состояние в маленький JSON, а зрение его читает: дёшево, без сети и без
общей памяти.

Состояние намеренно УСТАРЕВАЕТ. Если цикл упал или его остановили, зрение
не должно продолжать считать, что башня растёт: через `max_age_s` запись
считается негодной и ожидаемая сцена становится пустой. Пустая ожидаемая
сцена = прежнее поведение, то есть любой предмет снова тревога. Отказ в
безопасную сторону.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_PATH = "task/state.json"


@dataclass
class ProcessState:
    """Фаза процесса и то, что из неё следует для сцены.

    `cubes_placed` — сколько кубиков СТОИТ на столе в стопке прямо сейчас.
    Это и есть высота башни в кубиках; из неё считается ожидаемая колонна.

    `cube_in_gripper` — несёт ли рука кубик. Он движется вместе с TCP и для
    зрения выглядит отдельным предметом вплотную к руке.
    """

    phase: str = "idle"          # assemble | unstack | idle
    level: int = 0               # уровень, с которым идёт работа
    slot: int = 0                # слот магазина в этой операции
    cubes_placed: int = 0        # сколько кубиков в стопке
    cube_in_gripper: bool = False
    t_wall: float = 0.0          # время записи, для проверки свежести


def write_state(state: ProcessState, path: str = DEFAULT_PATH) -> None:
    """Записать состояние атомарно.

    Через временный файл и `os.replace`: зрение читает файл в своём темпе, и
    поймать его наполовину записанным нельзя. Ошибка записи не должна ронять
    цикл — робот важнее телеметрии, поэтому исключение гасится.
    """
    state.t_wall = time.time()
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        print(f"[task] состояние не записано ({e}) — цикл продолжает работу")


def read_state(path: str = DEFAULT_PATH,
               max_age_s: float = 10.0) -> Optional[ProcessState]:
    """Прочитать состояние. None — нет файла, он битый или протух.

    None означает «про процесс ничего не известно», и вызывающий обязан
    трактовать это как пустую ожидаемую сцену.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        st = ProcessState(**data)
    except TypeError:
        return None
    if max_age_s > 0 and (time.time() - float(st.t_wall)) > max_age_s:
        return None
    return st


def clear_state(path: str = DEFAULT_PATH) -> None:
    """Убрать состояние: процесс закончился, ожидать нечего."""
    try:
        Path(path).unlink()
    except OSError:
        pass


# ── Подмена состояния для офлайн-прогонов ────────────────────────────────────
# Реплей гоняет записанные кадры, и файла состояния для них нет: запись сделана
# до того, как цикл научился себя описывать. Восстановленное из углов состояние
# кладётся сюда, и `current()` отдаёт его вместо файла. Живой прогон этой ветки
# не касается — словарь пуст.
_frame_states: dict = {}


def set_frame_states(mapping: dict) -> None:
    """Задать состояние по номерам кадров (офлайн). Пустой словарь — сброс."""
    global _frame_states
    _frame_states = dict(mapping or {})


def current(frame: int = -1, path: str = DEFAULT_PATH,
            max_age_s: float = 10.0) -> Optional[ProcessState]:
    """Состояние для кадра: из подмены, иначе из файла, иначе None."""
    if _frame_states:
        return _frame_states.get(int(frame))
    return read_state(path, max_age_s)
