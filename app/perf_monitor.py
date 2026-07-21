"""
Мониторинг ресурсов ПК во время работы системы технического зрения.

Снимает CPU (система/процесс), RAM (процесс/система) через psutil и загрузку
GPU (Intel Iris Xe) через счётчики Windows (win32pdh). Пишет CSV-лог для
оффлайн-анализа, рисует значения на кадре и печатает сводку (min/avg/max) в конце.

Всё опционально и деградирует мягко: если psutil/win32pdh нет — пропускает.
"""
import csv
import os
import platform
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import win32pdh
    _PDH = True
except ImportError:
    _PDH = False


class _GpuSampler:
    """Загрузка GPU через perf-счётчики Windows: сумма по 3D+Compute движкам."""

    def __init__(self):
        self.ok = False
        self._q = None
        self._handles: list = []
        if not _PDH:
            return
        try:
            paths = win32pdh.ExpandCounterPath(r"\GPU Engine(*)\Utilization Percentage")
            paths = [p for p in paths if ("engtype_3D" in p or "engtype_Compute" in p)]
            if not paths:
                return
            self._q = win32pdh.OpenQuery()
            self._handles = [win32pdh.AddCounter(self._q, p) for p in paths]
            win32pdh.CollectQueryData(self._q)  # первичный замер для дельты
            self.ok = True
        except Exception:
            self.ok = False

    def read(self):
        if not self.ok:
            return None
        try:
            win32pdh.CollectQueryData(self._q)
            total = 0.0
            for h in self._handles:
                try:
                    _, v = win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_DOUBLE)
                    total += v
                except Exception:
                    pass
            return min(total, 100.0)
        except Exception:
            return None

    def close(self):
        if self._q is not None:
            try:
                win32pdh.CloseQuery(self._q)
            except Exception:
                pass


class PerfWindow:
    """Отдельное окно с графиками CPU/RAM/FPS/GPU (как в 2D-программе)."""

    def __init__(self, title: str = "Process Monitor (3D)", history_size: int = 200):
        self.title = title
        self.history_size = history_size
        self._open = False
        self.cpu_history = deque(maxlen=history_size)
        self.ram_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.width = 420
        self.height = 475
        self.bg = (30, 30, 30)
        self.grid = (60, 60, 60)
        self.txt = (200, 200, 200)
        self.cpu_c = (0, 255, 0)
        self.ram_c = (0, 165, 255)
        self.fps_c = (0, 255, 255)
        self.gpu_c = (255, 0, 255)

    def update(self, cpu, ram, fps, threads=0, gpu=None):
        self.cpu_history.append(cpu)
        self.ram_history.append(ram)
        self.fps_history.append(fps)
        self.gpu_history.append(gpu if gpu is not None else 0.0)

        img = np.full((self.height, self.width, 3), self.bg, dtype=np.uint8)
        cv2.putText(img, "PROCESS MONITOR", (self.width // 2 - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.txt, 2, cv2.LINE_AA)
        cv2.putText(img, "(3D: cloud + collisions)", (self.width // 2 - 80, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        gh, gw, ml, mt, sp = 70, self.width - 80, 60, 55, 95
        y = mt
        max_cpu = max(100.0, max(self.cpu_history) if self.cpu_history else 100.0)
        self._graph(img, self.cpu_history, ml, y, gw, gh, f"CPU: {cpu:.1f}%", self.cpu_c, max_cpu)
        y += sp
        max_ram = max((max(self.ram_history) if self.ram_history else 100.0) * 1.2, 100.0)
        self._graph(img, self.ram_history, ml, y, gw, gh, f"RAM: {ram:.0f} MB", self.ram_c, max_ram)
        y += sp
        max_fps = max((max(self.fps_history) if self.fps_history else 30.0) * 1.1, 30.0)
        self._graph(img, self.fps_history, ml, y, gw, gh, f"FPS: {fps:.1f}", self.fps_c, max_fps)
        y += sp
        gpu_lbl = f"GPU: {gpu:.1f}%" if gpu is not None else "GPU: n/a"
        self._graph(img, self.gpu_history, ml, y, gw, gh, gpu_lbl, self.gpu_c, 100.0)

        y += sp + 10
        cv2.line(img, (10, y - 5), (self.width - 10, y - 5), self.grid, 1)
        cv2.putText(img, f"Threads: {threads}", (20, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.txt, 1, cv2.LINE_AA)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_ram = sum(self.ram_history) / len(self.ram_history) if self.ram_history else 0
        cv2.putText(img, f"Avg CPU: {avg_cpu:.1f}%", (140, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cpu_c, 1, cv2.LINE_AA)
        cv2.putText(img, f"Avg RAM: {avg_ram:.0f}MB", (270, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ram_c, 1, cv2.LINE_AA)

        if not self._open:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)
            self._open = True
        cv2.imshow(self.title, img)

    def _graph(self, img, data, x, y, w, h, label, color, max_val):
        data = list(data)
        cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.grid, 1)
        for i in range(1, 4):
            ly = y + int(h * i / 4)
            cv2.line(img, (x, ly), (x + w, ly), self.grid, 1)
        cv2.putText(img, f"{max_val:.0f}", (x - 40, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.txt, 1, cv2.LINE_AA)
        cv2.putText(img, "0", (x - 15, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.txt, 1, cv2.LINE_AA)
        if len(data) >= 2:
            pts = []
            for i, val in enumerate(data):
                px = x + int(i * w / self.history_size)
                py = y + h - int((min(val, max_val) / max_val) * h)
                pts.append((px, max(y, min(y + h - 1, py))))
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 4, color, -1)
        cv2.putText(img, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def close(self):
        if self._open:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            self._open = False


class PerfMonitor:
    def __init__(self, cfg):
        self.enabled = bool(getattr(cfg, "perf_monitor", True))
        self.overlay = bool(getattr(cfg, "perf_overlay", True))
        self.gpu_every = max(1, int(getattr(cfg, "perf_gpu_every_n", 15)))
        self.log_path = Path(getattr(cfg, "perf_log_path", "captures/perf_log.csv"))

        self._rows: list = []
        self._stats = defaultdict(list)
        self._last: dict = {}
        self._gpu_val = None

        self.proc = None
        self.n_cores = 1
        self.cpu_name = "CPU"
        if self.enabled and _PSUTIL:
            self.proc = psutil.Process(os.getpid())
            self.proc.cpu_percent(None)   # прайм (первый вызов = 0)
            psutil.cpu_percent(None)       # прайм системного CPU
            self.n_cores = psutil.cpu_count(logical=True) or 1
            self.cpu_name = platform.processor() or "CPU"
        self.gpu = _GpuSampler() if self.enabled else None
        # Отдельное окно с графиками (как в 2D-программе)
        self.window = None
        if self.enabled and bool(getattr(cfg, "perf_window", True)):
            self.window = PerfWindow()

        if self.enabled and not _PSUTIL:
            print("[PERF] psutil не установлен — мониторинг CPU/RAM отключён "
                  "(pip install psutil).")
        if self.enabled:
            gpu_state = "GPU: win32pdh OK" if (self.gpu and self.gpu.ok) else "GPU: n/a"
            print(f"[PERF] Мониторинг включён. CPU cores={self.n_cores}  {gpu_state}  "
                  f"log -> {self.log_path}")

    def sample(self, frame_count: int, fps: float, stage_times: dict = None):
        if not (self.enabled and _PSUTIL):
            return
        cpu_sys = psutil.cpu_percent(None)
        # CPU процесса «на ядро» (>100% возможно) — та же система, что в 2D-мониторе
        cpu_proc = self.proc.cpu_percent(None)
        threads = self.proc.num_threads()
        vm = psutil.virtual_memory()
        ram_proc_mb = self.proc.memory_info().rss / (1024 * 1024)

        if self.gpu and self.gpu.ok and frame_count % self.gpu_every == 0:
            g = self.gpu.read()
            if g is not None:
                self._gpu_val = g

        row = {
            "t": round(time.time(), 3),
            "frame": frame_count,
            "fps": round(fps, 1),
            "cpu_sys_pct": round(cpu_sys, 1),
            "cpu_proc_pct": round(cpu_proc, 1),   # на ядро (как в 2D)
            "ram_proc_mb": round(ram_proc_mb, 1),
            "ram_sys_pct": round(vm.percent, 1),
            "ram_sys_used_mb": round(vm.used / (1024 * 1024), 1),
            "gpu_pct": round(self._gpu_val, 1) if self._gpu_val is not None else "",
            "threads": threads,
        }
        if stage_times:
            for k, v in stage_times.items():
                row[f"t_{k}_ms"] = round(v * 1000.0, 1)

        self._rows.append(row)
        self._last = row
        for k, v in row.items():
            if k not in ("t", "frame") and isinstance(v, (int, float)):
                self._stats[k].append(v)

        # Обновляем окно графиков (CPU «на ядро», RAM МБ, FPS, GPU, потоки)
        if self.window is not None:
            self.window.update(
                cpu_proc, ram_proc_mb, fps, threads,
                gpu=self._gpu_val,
            )

    def overlay_lines(self) -> List[str]:
        if not (self.enabled and self.overlay and self._last):
            return []
        l = self._last
        g = l.get("gpu_pct", "")
        return [
            f"CPU proc {l.get('cpu_proc_pct','?')}% (/core)  sys {l.get('cpu_sys_pct','?')}%",
            f"RAM {l.get('ram_proc_mb','?')}MB  sys {l.get('ram_sys_pct','?')}%",
            (f"GPU {g}%" if g != "" else "GPU n/a") + f"  thr {l.get('threads','?')}",
        ]

    def close(self):
        if self.gpu:
            self.gpu.close()
        if self.window is not None:
            self.window.close()
        if not self._rows:
            return
        # CSV (объединение всех ключей)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        all_keys: list = []
        for r in self._rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        try:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                w.writerows(self._rows)
        except Exception as e:
            print(f"[PERF] Не удалось записать CSV: {e}")

        def agg(key):
            xs = self._stats.get(key, [])
            return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None

        print("\n" + "=" * 56)
        print("PERF SUMMARY (за весь прогон) — 3D метод (облако + коллизии)")
        print(f"  CPU: {self.cpu_name}  logical cores={self.n_cores}")
        for key, label, unit in [
            ("cpu_proc_pct", "CPU процесс", "% (на ядро)"),
            ("cpu_sys_pct", "CPU система", "% (машина)"),
            ("ram_proc_mb", "RAM процесс", "MB"),
            ("ram_sys_pct", "RAM система", "%"),
            ("gpu_pct", "GPU", "%"),
            ("fps", "FPS", ""),
            ("threads", "Потоки", ""),
        ]:
            a = agg(key)
            if a:
                print(f"  {label:14s}: min {a[0]:6.1f}  avg {a[1]:6.1f}  max {a[2]:6.1f} {unit}")
        print(f"  Кадров записано: {len(self._rows)}")
        print(f"  CSV: {self.log_path.resolve()}")
        print("=" * 56)
