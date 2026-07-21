"""
Утилиты для мониторинга ПРОЦЕССА программы.

Измеряет потребление ресурсов ТОЛЬКО текущим процессом,
а не всей системы - для точного анализа влияния программы на ПК.
"""
import time
import psutil
import os
from typing import Tuple


# Глобальный объект процесса (кэшируем для производительности)
_process = None


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
