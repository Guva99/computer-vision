"""
Playback-сервис камеры 2D-контура (валидация, Задача 3).

Повторяет интерфейс CameraService (start/stop/get_frames/get_intrinsics/
get_depth_scale), но кадры берёт из записи рекордера (формат app/recorder.py:
<dir>/frames/frame_%06d.npz + meta.json). Кадры оборачиваются в лёгкие
обёртки с .get_data(), чтобы остальной код работал без изменений.

По концу записи get_frames() возвращает (None, None, None) и выставляет
finished=True — контроллер завершает цикл.
"""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class _ArrayFrame:
    """Обёртка numpy-массива под интерфейс rs.frame (.get_data())."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


class _Intrinsics:
    """Минимальный аналог rs.intrinsics (fx, fy, ppx, ppy)."""

    def __init__(self, fx: float, fy: float, ppx: float, ppy: float):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class PlaybackCameraService:
    """Источник кадров из записи; интерфейс совместим с CameraService."""

    def __init__(self, playback_path: str, **_ignored):
        self.path = Path(playback_path)
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(
                f"meta.json не найден в '{self.path}' — это не запись рекордера"
            )
        meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        intr = meta["intrinsics"]
        self._intrinsics = _Intrinsics(
            float(intr["fx"]), float(intr["fy"]),
            float(intr.get("cx", intr.get("ppx", 0.0))),
            float(intr.get("cy", intr.get("ppy", 0.0))),
        )
        self.depth_scale = float(meta["depth_scale"])
        self._files = sorted((self.path / "frames").glob("frame_*.npz"))
        self._idx = 0
        self.finished = False
        self.is_running = False
        if not self._files:
            raise FileNotFoundError(f"В {self.path / 'frames'} нет кадров")

    def start(self) -> bool:
        self._idx = 0
        self.finished = False
        self.is_running = True
        print(f"▶ Playback: {len(self._files)} кадров из {self.path}")
        return True

    def stop(self):
        self.is_running = False

    def get_frames(self) -> Tuple[Optional[object], Optional[object], Optional[object]]:
        if self._idx >= len(self._files):
            self.finished = True
            return None, None, None
        data = np.load(self._files[self._idx])
        self._idx += 1
        color = _ArrayFrame(np.ascontiguousarray(data["color"]))
        depth = _ArrayFrame(np.ascontiguousarray(data["depth"]))
        return depth, color, None

    def get_intrinsics(self, color_frame=None):
        return self._intrinsics

    def get_depth_scale(self) -> float:
        return self.depth_scale
