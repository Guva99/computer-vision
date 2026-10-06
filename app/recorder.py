"""
Рекордер кадров (A1): сырые кадры + углы суставов + метаданные записи.

Пишет то, что читают tools/annotate.py (разметка GT) и app/playback.py
(воспроизведение без камеры):

    <path>/frames/frame_%06d.npz   color (BGR uint8), depth (uint16, ДО фильтров)
    <path>/joints.csv              frame, t_wall, A1..A6
    <path>/meta.json               intrinsics, depth_scale, git_commit, branch,
                                   scenario_id, created

Номер в имени файла совпадает с `frame` в joints.csv, decisions.csv и
objects.csv — по нему сшивается вся аналитика.

ПОЧЕМУ ФОНОВЫЙ ПОТОК. 640x480 даёт ~1.5 МБ на кадр, при 30 FPS это ~46 МБ/с.
Синхронная запись сажает главный цикл, а он принимает решение об остановке
робота. Поэтому кадры уходят в очередь, а writer пишет их отдельным потоком.
Если очередь переполнена (диск не успевает), кадр ТЕРЯЕТСЯ, а не тормозит
контур безопасности: пропуск в записи аналитика переживёт (annotate.py идёт по
существующим файлам), просадка реакции робота — нет. Потери считаются и
печатаются при закрытии, их число попадает в meta.json.

Включается флагом cfg.enable_recording (по умолчанию ВЫКЛ) — при выключенном
объект не создаётся, накладных расходов нет.
"""
import csv
import json
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_JOINT_FIELDS = ["frame", "t_wall", "A1", "A2", "A3", "A4", "A5", "A6"]


def git_info(root: Path) -> tuple:
    """(commit, branch) рабочего дерева; пустые строки, если git недоступен."""
    def _run(args):
        try:
            out = subprocess.run(
                args, cwd=str(root), capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""
    return (_run(["git", "rev-parse", "HEAD"]),
            _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]))


class FrameRecorder:
    """Запись кадров и углов в фоновом потоке.

    Использование:
        rec = FrameRecorder(cfg, path, scenario_id)
        rec.begin(intrinsics, depth_scale)      # один раз, когда известна камера
        rec.add(frame_no, color, depth, angles, t_wall)
        rec.close()
    """

    def __init__(self, cfg, path: Optional[str] = None, scenario_id: str = ""):
        self.cfg = cfg
        self.root = Path(path or getattr(cfg, "recording_path", "captures/recording"))
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_id = str(scenario_id or getattr(cfg, "scenario_id", "") or "")
        self.compress = bool(getattr(cfg, "recording_compress", False))
        # Режим «только углы»: кадры на диск не пишем, joints.csv ведём как обычно.
        self.joints_only = bool(getattr(cfg, "recording_joints_only", False))
        self._put_timeout = float(getattr(cfg, "recording_put_timeout_s", 0.5))
        self._q: queue.Queue = queue.Queue(
            maxsize=max(4, int(getattr(cfg, "recording_queue_max", 120)))
        )
        self._stop = threading.Event()
        self._meta_written = False
        self.n_written = 0
        self.n_dropped = 0
        self._joints_path = self.root / "joints.csv"
        with open(self._joints_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_JOINT_FIELDS).writeheader()
        self._joints_rows: list = []
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    # ── метаданные ───────────────────────────────────────────────────────────
    def begin(self, intrinsics: dict, depth_scale: float) -> None:
        """Записать meta.json (один раз, когда известны параметры камеры)."""
        if self._meta_written:
            return
        commit, branch = git_info(Path(__file__).resolve().parent.parent)
        intr = dict(intrinsics or {})
        # annotate.py берёт width/height из meta — кладём их всегда.
        intr.setdefault("width", int(getattr(self.cfg, "width", 640)))
        intr.setdefault("height", int(getattr(self.cfg, "height", 480)))
        meta = {
            "intrinsics": {
                k: (int(v) if k in ("width", "height") else float(v))
                for k, v in intr.items()
            },
            "depth_scale": float(depth_scale),
            "git_commit": commit,
            "branch": branch,
            "scenario_id": self.scenario_id,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        (self.root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._meta_written = True

    # ── приём кадра из главного цикла ────────────────────────────────────────
    def add(self, frame_no: int, color: np.ndarray, depth: np.ndarray,
            joint_angles: Optional[Sequence[float]] = None,
            t_wall: Optional[float] = None) -> bool:
        """Поставить кадр в очередь. False = кадр потерян (диск не успевает).

        Копии обязательны: буферы RealSense переиспользуются, и writer читал бы
        уже следующий кадр.
        """
        item = (
            int(frame_no),
            np.array(color, copy=True),
            np.array(depth, copy=True),
            tuple(joint_angles) if joint_angles is not None else None,
            float(t_wall if t_wall is not None else time.time()),
        )
        try:
            self._q.put(item, timeout=self._put_timeout)
            return True
        except queue.Full:
            self.n_dropped += 1
            if self.n_dropped in (1, 10, 100) or self.n_dropped % 500 == 0:
                print(f"[REC] Очередь переполнена, кадров потеряно: {self.n_dropped}")
            return False

    # ── фоновый writer ───────────────────────────────────────────────────────
    def _writer_loop(self) -> None:
        save = np.savez_compressed if self.compress else np.savez
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            frame_no, color, depth, angles, t_wall = item
            try:
                if not self.joints_only:
                    save(str(self.frames_dir / f"frame_{frame_no:06d}.npz"),
                         color=color, depth=depth)
                row = {"frame": frame_no, "t_wall": f"{t_wall:.6f}"}
                for i in range(6):
                    ok = angles is not None and i < len(angles)
                    row[f"A{i + 1}"] = f"{float(angles[i]):.4f}" if ok else ""
                self._joints_rows.append(row)
                if len(self._joints_rows) >= 60:
                    self._flush_joints()
                self.n_written += 1
            except Exception as e:
                print(f"[REC] Ошибка записи кадра {frame_no}: {e}")
            finally:
                self._q.task_done()

    def _flush_joints(self) -> None:
        if not self._joints_rows:
            return
        with open(self._joints_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_JOINT_FIELDS).writerows(self._joints_rows)
        self._joints_rows.clear()

    # ── завершение ───────────────────────────────────────────────────────────
    def close(self, timeout_s: float = 30.0) -> None:
        """Дописать хвост очереди и остановить writer."""
        t0 = time.time()
        while not self._q.empty() and (time.time() - t0) < timeout_s:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._flush_joints()
        # Дописать статистику записи в meta.json (сколько реально легло на диск).
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["n_frames"] = int(self.n_written)
                meta["n_dropped"] = int(self.n_dropped)
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                print(f"[REC] Не удалось дописать meta.json: {e}")
        print(f"[REC] Записано кадров: {self.n_written}, потеряно: {self.n_dropped} "
              f"-> {self.root}")


class MaskDumper:
    """Дамп масок для расчёта IoU (A5).

    Кладёт `frame_%06d_obstacle.png` и `frame_%06d_manip.png` в <path>/masks/.
    Имена обязаны совпадать с тем, что tools/annotate.py пишет в gt_masks —
    compute_metrics сопоставляет пары файлов по имени, и расхождение в один
    символ обнулит IoU молча.

    Шаг задаётся cfg.dump_masks_every_n (0 = выключено): каждый кадр не нужен,
    соседние кадры почти одинаковы, а разметка руками стоит дорого.
    """

    def __init__(self, cfg, path: Optional[str] = None):
        self.every_n = max(0, int(getattr(cfg, "dump_masks_every_n", 0)))
        base = path or getattr(cfg, "masks_dump_path", "") or (
            str(Path(getattr(cfg, "recording_path", "captures/recording")) / "masks")
        )
        self.root = Path(base)
        self.n_written = 0
        if self.every_n > 0:
            self.root.mkdir(parents=True, exist_ok=True)

    def due(self, frame_no: int) -> bool:
        return self.every_n > 0 and int(frame_no) % self.every_n == 0

    def dump(self, frame_no: int, obstacle_mask, manip_mask) -> None:
        import cv2  # локально: модуль нужен только на кадрах дампа

        for kind, mask in (("obstacle", obstacle_mask), ("manip", manip_mask)):
            if mask is None:
                continue
            binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
            cv2.imwrite(str(self.root / f"frame_{int(frame_no):06d}_{kind}.png"), binary)
        self.n_written += 1

    def close(self) -> None:
        if self.every_n > 0:
            print(f"[MASKS] Дампов масок: {self.n_written} -> {self.root}")
