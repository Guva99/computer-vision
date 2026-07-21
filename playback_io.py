"""
Оффлайн-воспроизведение записей рекордера (валидация, Задача 3).

PlaybackCamera повторяет интерфейс RealSenseCamera:
  start() / stop() / get_aligned_frames() -> (color_bgr, depth_uint16,
  intrinsics, depth_scale); None по концу записи.

Формат записи — app/recorder.py: <dir>/frames/frame_%06d.npz + joints.csv
+ meta.json. Углы суставов для реплея читает load_recorded_joints()
(используется JointAngleReader в режиме playback).

Дренажа очереди кадров здесь нет по построению — кадры отдаются строго
последовательно, что и требуется для воспроизводимого оффлайн-сравнения.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_recorded_joints(recording_path: str) -> Dict[int, Tuple[float, ...]]:
    """joints.csv записи → {frame_idx: (A1..A6)}. Кадры без углов пропущены."""
    path = Path(recording_path) / "joints.csv"
    joints: Dict[int, Tuple[float, ...]] = {}
    if not path.exists():
        print(f"[PLAYBACK] joints.csv не найден в {recording_path} — "
              "углы недоступны (FK/маска руки работать не будут).")
        return joints
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals = [row[f"A{i}"] for i in range(1, 7)]
                if any(v == "" for v in vals):
                    continue
                joints[int(row["frame"])] = tuple(float(v) for v in vals)
            except (KeyError, ValueError):
                continue
    return joints


class PlaybackCamera:
    """Камера-эмулятор поверх записи рекордера (интерфейс RealSenseCamera)."""

    def __init__(self, cfg):
        self.path = Path(getattr(cfg, "playback_path", "") or "")
        if not self.path.exists():
            raise FileNotFoundError(
                f"playback_path '{self.path}' не существует — укажи каталог записи"
            )
        meta_path = self.path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} не найден — это не запись рекордера")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.intrinsics = {k: float(v) for k, v in meta["intrinsics"].items()
                           if k in ("fx", "fy", "cx", "cy")}
        self.depth_scale = float(meta["depth_scale"])
        self.meta = meta
        self._files: List[Path] = sorted((self.path / "frames").glob("frame_*.npz"))
        self._idx = 0
        self.finished = False
        if not self._files:
            raise FileNotFoundError(f"В {self.path / 'frames'} нет кадров")
        print(f"[PLAYBACK] {len(self._files)} кадров из {self.path} "
              f"(контур записи: {meta.get('contour', '?')})")

    def start(self) -> None:
        self._idx = 0
        self.finished = False

    def stop(self) -> None:
        pass

    @property
    def n_frames(self) -> int:
        return len(self._files)

    def get_aligned_frames(self) -> Optional[tuple]:
        if self._idx >= len(self._files):
            self.finished = True
            return None
        data = np.load(self._files[self._idx])
        self._idx += 1
        color = np.ascontiguousarray(data["color"])
        depth = np.ascontiguousarray(data["depth"])
        return color, depth, dict(self.intrinsics), self.depth_scale
