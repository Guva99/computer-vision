"""
Воспроизведение записи (A2): кадры и углы с диска вместо камеры и робота.

Нужно, чтобы пересчитывать метрики и делать абляцию (менять пороги, гейты,
фильтры глубины) БЕЗ повторной съёмки на стенде: один и тот же вход прогоняется
сколько угодно раз, и разница в числах — это разница в алгоритме, а не в сцене.

Два класса, оба повторяют интерфейс своих «живых» аналогов:
  • PlaybackCamera     ← RealSenseCamera (start / get_aligned_frames / stop)
  • PlaybackJointReader ← JointAngleReader (connect / read / close)

Номер кадра берётся из ИМЕНИ ФАЙЛА, а не из счётчика цикла: в записи бывают
пропуски (writer не успел за диском), и только имя файла гарантирует, что
логи playback-прогона сошьются с той же разметкой GT, что и логи съёмки.

Управление роботом при source_mode='playback' принудительно выключается в
AppRunner: движения нет, «останавливать» нечего.
"""
import csv
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class PlaybackError(RuntimeError):
    pass


def _load_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise PlaybackError(f"Нет meta.json в записи: {root}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


class PlaybackCamera:
    """Отдаёт кадры записи последовательно, как камера.

    Совместим с AppRunner: start() / get_aligned_frames() / stop().
    Когда кадры кончились, выставляет `finished = True` — раннер завершает цикл
    (иначе `frame is None` в главном цикле крутился бы вечно).
    """

    def __init__(self, config):
        self.config = config
        self.root = Path(getattr(config, "playback_path", "") or "")
        if not self.root.exists():
            raise PlaybackError(f"Каталог записи не найден: {self.root}")
        self.meta = _load_meta(self.root)
        intr = dict(self.meta.get("intrinsics", {}))
        self.intrinsics = {
            "fx": float(intr["fx"]), "fy": float(intr["fy"]),
            "cx": float(intr["cx"]), "cy": float(intr["cy"]),
        }
        self.depth_scale = float(self.meta.get("depth_scale", 0.001))
        self.files = sorted((self.root / "frames").glob("frame_*.npz"))
        if not self.files:
            raise PlaybackError(f"Нет кадров в {self.root / 'frames'}")
        self._i = 0
        self.finished = False
        self.last_frame_no = 0
        self._fps = float(getattr(config, "playback_fps", 0.0) or 0.0)
        self._t_prev = None

    def start(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[PLAYBACK] {len(self.files)} кадров из {self.root}")
        print(f"[PLAYBACK] scenario_id={self.meta.get('scenario_id', '')!r} "
              f"commit={str(self.meta.get('git_commit', ''))[:8]} "
              f"записано {self.meta.get('created', '?')}")

    def get_aligned_frames(self):
        if self._i >= len(self.files):
            self.finished = True
            return None
        path = self.files[self._i]
        self._i += 1
        try:
            self.last_frame_no = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            self.last_frame_no = self._i
        # Троттлинг до заданного FPS: нужен, только если смотришь глазами;
        # по умолчанию (0) гоним максимально быстро.
        if self._fps > 0.0:
            dt = 1.0 / self._fps
            if self._t_prev is not None:
                lag = dt - (time.perf_counter() - self._t_prev)
                if lag > 0:
                    time.sleep(lag)
            self._t_prev = time.perf_counter()
        try:
            data = np.load(path)
            color = np.ascontiguousarray(data["color"])
            depth = np.ascontiguousarray(data["depth"])
        except Exception as e:
            print(f"[PLAYBACK] Кадр {path.name} не прочитан: {e} — пропуск")
            return None
        return color, depth, self.intrinsics, self.depth_scale

    def stop(self) -> None:
        pass


class PlaybackJointReader:
    """Углы A1..A6 из joints.csv записи по НОМЕРУ КАДРА.

    Интерфейс как у JointAngleReader, поэтому PerceptionPipeline подменяет одно
    другим без правок конвейера. Если для кадра строки нет (запись велась без
    робота или строка потерялась) — отдаём последние известные углы, ровно как
    живой reader делает при таймауте контроллера.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(getattr(cfg, "playback_path", "") or "")
        self._by_frame: dict = {}
        self._last: Optional[Tuple[float, ...]] = None

    def connect(self) -> "PlaybackJointReader":
        path = self.root / "joints.csv"
        if not path.exists():
            print(f"[PLAYBACK] Нет {path} — FK будет недоступен")
            return self
        n = 0
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    angles = tuple(float(row[f"A{i + 1}"]) for i in range(6))
                except (KeyError, TypeError, ValueError):
                    continue  # строка без углов (робот не был подключён)
                try:
                    self._by_frame[int(row["frame"])] = angles
                except (KeyError, ValueError):
                    continue
                n += 1
        print(f"[PLAYBACK] Углы суставов: {n} кадров из {path.name}")
        return self

    @property
    def last_angles(self) -> Optional[Tuple[float, ...]]:
        return self._last

    def read(self, frame_count: int) -> Optional[Tuple[float, ...]]:
        angles = self._by_frame.get(int(frame_count))
        if angles is not None:
            self._last = angles
        return self._last

    def close(self) -> None:
        pass
