"""
Утилиты для мониторинга ПРОЦЕССА программы.

Измеряет потребление ресурсов ТОЛЬКО текущим процессом,
а не всей системы - для точного анализа влияния программы на ПК.
"""
import csv
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import psutil

try:
    import win32pdh
    _PDH = True
except ImportError:
    _PDH = False


# Глобальный объект процесса (кэшируем для производительности)
_process = None


class GpuSampler:
    """Загрузка GPU через perf-счётчики Windows: сумма по 3D+Compute движкам.

    Деградирует мягко: если win32pdh нет (не Windows / нет pywin32) — ok=False,
    read() возвращает None.
    """

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
            win32pdh.CollectQueryData(self._q)
            self.ok = True
        except Exception:
            self.ok = False

    def read(self) -> Optional[float]:
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


class PerfLogger:
    """Пишет метрики по кадрам в CSV и печатает сводку (min/avg/max) при закрытии."""

    def __init__(self, log_path: str = "captures/perf_log_2d.csv"):
        self.log_path = Path(log_path)
        self._rows: list = []
        self._stats = defaultdict(list)
        self.cpu_name = platform.processor() or "CPU"
        self.n_cores = psutil.cpu_count(logical=True) or 1

    def log(self, **fields):
        self._rows.append(fields)
        for k, v in fields.items():
            if isinstance(v, (int, float)):
                self._stats[k].append(v)

    def close(self):
        if not self._rows:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        keys: list = []
        for r in self._rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        try:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._rows)
        except Exception as e:
            print(f"[PERF] Не удалось записать CSV: {e}")

        def agg(key):
            xs = self._stats.get(key, [])
            return (min(xs), sum(xs) / len(xs), max(xs)) if xs else None

        print("\n" + "=" * 56)
        print("PERF SUMMARY (за весь прогон) — 2D метод, камера сверху")
        print(f"  CPU: {self.cpu_name}  logical cores={self.n_cores}")
        for key, label, unit in [
            ("cpu_pct", "CPU процесс", "% (на ядро)"),
            ("ram_mb", "RAM процесс", "MB"),
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


def get_process() -> psutil.Process:
    """Получает объект текущего процесса."""
    global _process
    if _process is None:
        _process = psutil.Process(os.getpid())
    return _process


def sample_process_stats() -> Tuple[float, float, int]:
    """
    Собирает статистику ТОЛЬКО ТЕКУЩЕГО ПРОЦЕССА.
    
    Returns:
        Кортеж (cpu_percent, memory_mb, thread_count)
        
    Note:
        - cpu_percent: загрузка CPU только этим процессом (%)
        - memory_mb: RAM только этого процесса (в MB!)
        - thread_count: количество потоков процесса
    """
    process = get_process()
    
    # CPU только этого процесса (может быть >100% на многоядерных системах)
    cpu_percent = process.cpu_percent(interval=None)
    
    # RAM только этого процесса (в MB, не в %)
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Потоки только этого процесса
    thread_count = process.num_threads()
    
    return cpu_percent, memory_mb, thread_count


def sample_system_stats() -> Tuple[float, float, int]:
    """
    Обратная совместимость - теперь измеряет ПРОЦЕСС.
    
    Returns:
        Кортеж (cpu_percent, memory_mb, thread_count)
    """
    return sample_process_stats()


def get_detailed_stats() -> dict:
    """
    Получает детальную статистику процесса.
    
    Returns:
        Словарь с метриками:
        - cpu_percent: загрузка CPU
        - memory_rss_mb: физическая память (MB)
        - memory_vms_mb: виртуальная память (MB)
        - threads: количество потоков
        - pid: ID процесса
    """
    process = get_process()
    mem_info = process.memory_info()
    
    return {
        'cpu_percent': process.cpu_percent(interval=None),
        'memory_rss_mb': mem_info.rss / (1024 * 1024),
        'memory_vms_mb': mem_info.vms / (1024 * 1024),
        'threads': process.num_threads(),
        'pid': process.pid,
    }


class FPSCounter:
    """Класс для подсчёта FPS."""
    
    def __init__(self, update_interval: int = 30):
        """
        Инициализация счётчика FPS.
        
        Args:
            update_interval: Интервал обновления FPS (в кадрах)
        """
        self.update_interval = update_interval
        self.counter = 0
        self.start_time = None
        self.current_fps = 0.0
    
    def update(self) -> float:
        """
        Обновляет счётчик FPS.
        
        Returns:
            Текущий FPS или 0 если ещё не готов
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.counter += 1
        
        if self.counter % self.update_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.current_fps = self.counter / elapsed if elapsed > 0 else 0
            self.counter = 0
            self.start_time = current_time
            return self.current_fps
        
        return 0.0
    
    def get_fps(self) -> float:
        """
        Получает текущий FPS.
        
        Returns:
            Текущий FPS
        """
        return self.current_fps
