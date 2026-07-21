"""
Рекордер RGB-D + углов суставов (валидация, Задача 3) — копия app/recorder.py
из 3D-контура (2D-контур не видит родительский пакет app). Формат записи
общий: реплей возможен обоими контурами на одних и тех же данных.

В live-режиме синхронно сохраняет по кадрам:
  * <dir>/frames/frame_%06d.npz — color (H,W,3 uint8), depth (H,W uint16),
    t (perf_counter захвата), wall_t (time.time);
  * <dir>/joints.csv — frame, t, wall_t, A1..A6 (пусто, если углов нет);
  * <dir>/meta.json — intrinsics, depth_scale, размеры, контур, время старта.

Углы обязательны для 3D-реплея (FK и маска манипулятора), поэтому пишутся
на каждый кадр (переиспользуется последнее валидное значение читателя углов).
Воспроизведение — playback_io.PlaybackCamera с тем же интерфейсом камеры.

Включается флагом enable_recording (по умолчанию False).
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class FrameRecorder:
    def __init__(self, recording_path: str, contour: str = "3D",
                 compress: bool = False, scenario_id: str = ""):
        base = Path(recording_path) if recording_path else Path(
            "captures/recordings"
        ) / f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Не смешивать две записи в одном каталоге
        if (base / "frames").exists() and any((base / "frames").iterdir()):
            base = base.parent / f"{base.name}_{datetime.now().strftime('%H%M%S')}"
        self.dir = base
        self.frames_dir = base / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.contour = contour
        self.compress = bool(compress)
        self.scenario_id = scenario_id
        self._meta_written = False
        self._n = 0
        self._joints_file = open(base / "joints.csv", "w", newline="",
                                 encoding="utf-8")
        self._joints_writer = csv.writer(self._joints_file)
        self._joints_writer.writerow(
            ["frame", "t", "wall_t", "A1", "A2", "A3", "A4", "A5", "A6"]
        )
        print(f"[REC] Запись в {self.dir}")

    def record(self, frame_idx: int, color_bgr: np.ndarray,
               depth_u16: np.ndarray, intrinsics: dict, depth_scale: float,
               joint_angles: Optional[Sequence[float]],
               t_capture: float) -> None:
        if not self._meta_written:
            self._write_meta(color_bgr, intrinsics, depth_scale)
        wall_t = time.time()
        path = self.frames_dir / f"frame_{frame_idx:06d}.npz"
        save = np.savez_compressed if self.compress else np.savez
        save(path, color=color_bgr, depth=depth_u16,
             t=np.float64(t_capture), wall_t=np.float64(wall_t))
        row = [frame_idx, f"{t_capture:.6f}", f"{wall_t:.3f}"]
        if joint_angles is not None and len(joint_angles) >= 6:
            row += [f"{float(a):.4f}" for a in joint_angles[:6]]
        else:
            row += [""] * 6
        self._joints_writer.writerow(row)
        self._n += 1

    def _write_meta(self, color_bgr, intrinsics: dict,
                    depth_scale: float) -> None:
        meta = {
            "contour": self.contour,
            "scenario_id": self.scenario_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "width": int(color_bgr.shape[1]),
            "height": int(color_bgr.shape[0]),
            "intrinsics": {k: float(v) for k, v in intrinsics.items()},
            "depth_scale": float(depth_scale),
            "format": "npz-per-frame v1 (color u8 BGR, depth u16) + joints.csv",
        }
        (self.dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        self._meta_written = True

    def close(self) -> None:
        try:
            self._joints_file.close()
        except Exception:
            pass
        # дописать число кадров в meta
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["n_frames"] = self._n
                meta_path.write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
            except Exception:
                pass
        print(f"[REC] Записано {self._n} кадров → {self.dir}")
