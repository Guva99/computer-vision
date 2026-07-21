"""
Отдельное окно для графиков мониторинга ПРОЦЕССА (CPU, RAM, FPS).

Показывает потребление ресурсов ТОЛЬКО программой Cloud Point.
"""
import cv2
import numpy as np
from typing import List, Tuple
from collections import deque


class MonitorWindow:
    """Окно с графиками мониторинга процесса."""
    
    def __init__(self, title: str = 'Process Monitor', history_size: int = 200):
        """
        Инициализация окна мониторинга.
        
        Args:
            title: Заголовок окна
            history_size: Размер истории для графиков
        """
        self.title = title
        self.history_size = history_size
        self._window_open = False
        
        # История данных
        self.cpu_history = deque(maxlen=history_size)
        self.ram_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        
        # Размеры окна
        self.width = 420
        self.height = 380
        
        # Цвета
        self.bg_color = (30, 30, 30)
        self.grid_color = (60, 60, 60)
        self.text_color = (200, 200, 200)
        self.cpu_color = (0, 255, 0)      # Зелёный
        self.ram_color = (0, 165, 255)    # Оранжевый
        self.fps_color = (0, 255, 255)    # Жёлтый
    
    def update(self, cpu: float, ram: float, fps: float, threads: int = 0):
        """
        Обновляет данные и перерисовывает окно.
        
        Args:
            cpu: Загрузка CPU процессом в %
            ram: Использование RAM процессом в MB (!)
            fps: Кадры в секунду
            threads: Количество потоков процесса
        """
        # Добавляем данные в историю
        self.cpu_history.append(cpu)
        self.ram_history.append(ram)
        self.fps_history.append(fps)
        
        # Создаём изображение
        img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        # Заголовок
        cv2.putText(img, "PROCESS MONITOR", (self.width // 2 - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(img, "(Cloud Point)", (self.width // 2 - 55, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Параметры графиков
        graph_height = 70
        graph_width = self.width - 80
        margin_left = 60
        margin_top = 55
        graph_spacing = 95
        
        # === ГРАФИК CPU ===
        y_pos = margin_top
        # CPU может быть > 100% на многоядерных системах
        max_cpu = max(100, max(list(self.cpu_history)) if self.cpu_history else 100)
        self._draw_graph(
            img, list(self.cpu_history),
            margin_left, y_pos, graph_width, graph_height,
            f"CPU: {cpu:.1f}%", self.cpu_color, max_cpu
        )
        
        # === ГРАФИК RAM (в MB!) ===
        y_pos += graph_spacing
        # Динамическая шкала для RAM
        max_ram = max(list(self.ram_history)) if self.ram_history else 500
        max_ram = max(max_ram * 1.2, 100)  # +20% запас, минимум 100 MB
        self._draw_graph(
            img, list(self.ram_history),
            margin_left, y_pos, graph_width, graph_height,
            f"RAM: {ram:.0f} MB", self.ram_color, max_ram
        )
        
        # === ГРАФИК FPS ===
        y_pos += graph_spacing
        max_fps = max(list(self.fps_history)) if self.fps_history else 30
        max_fps = max(max_fps * 1.1, 30)
        self._draw_graph(
            img, list(self.fps_history),
            margin_left, y_pos, graph_width, graph_height,
            f"FPS: {fps:.1f}", self.fps_color, max_fps
        )
        
        # === ИНФОРМАЦИОННАЯ ПАНЕЛЬ ===
        y_pos += graph_spacing + 10
        cv2.line(img, (10, y_pos - 5), (self.width - 10, y_pos - 5), self.grid_color, 1)
        
        # Потоки
        cv2.putText(img, f"Threads: {threads}", (20, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
        
        # Средние значения
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_ram = sum(self.ram_history) / len(self.ram_history) if self.ram_history else 0
        
        cv2.putText(img, f"Avg CPU: {avg_cpu:.1f}%", (130, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cpu_color, 1, cv2.LINE_AA)
        
        cv2.putText(img, f"Avg RAM: {avg_ram:.0f}MB", (250, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ram_color, 1, cv2.LINE_AA)
        
        # Вторая строка
        cv2.putText(img, f"Avg FPS: {avg_fps:.1f}", (20, y_pos + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.fps_color, 1, cv2.LINE_AA)
        
        # MIN/MAX FPS
        if self.fps_history:
            min_fps = min(self.fps_history)
            max_fps_val = max(self.fps_history)
            cv2.putText(img, f"FPS: min={min_fps:.0f} max={max_fps_val:.0f}", 
                        (130, y_pos + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.fps_color, 1, cv2.LINE_AA)
        
        # Показываем окно
        if not self._window_open:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, self.width, self.height)
            self._window_open = True
        
        cv2.imshow(self.title, img)
    
    def _draw_graph(
        self,
        img: np.ndarray,
        data: List[float],
        x: int, y: int,
        width: int, height: int,
        label: str,
        color: Tuple[int, int, int],
        max_val: float
    ):
        """Рисует один график с сеткой и подписями."""
        # Фон графика
        cv2.rectangle(img, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), self.grid_color, 1)
        
        # Горизонтальные линии сетки
        for i in range(1, 4):
            line_y = y + int(height * i / 4)
            cv2.line(img, (x, line_y), (x + width, line_y), self.grid_color, 1)
        
        # Подписи значений слева
        cv2.putText(img, f"{max_val:.0f}", (x - 40, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(img, f"{max_val/2:.0f}", (x - 35, y + height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(img, "0", (x - 15, y + height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)
        
        # Рисуем линию графика
        if len(data) >= 2:
            points = []
            for i, val in enumerate(data):
                px = x + int(i * width / self.history_size)
                py = y + height - int((min(val, max_val) / max_val) * height)
                py = max(y, min(y + height - 1, py))
                points.append((px, py))
            
            # Заливка под графиком (полупрозрачная)
            if len(points) >= 2:
                fill_points = points.copy()
                fill_points.append((points[-1][0], y + height))
                fill_points.append((points[0][0], y + height))
                fill_pts = np.array(fill_points, dtype=np.int32)
                
                overlay = img.copy()
                cv2.fillPoly(overlay, [fill_pts], color)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            
            # Линия графика
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 2, cv2.LINE_AA)
            
            # Текущее значение (точка)
            if points:
                cv2.circle(img, points[-1], 4, color, -1)
        
        # Подпись графика
        cv2.putText(img, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    
    def close(self):
        """Закрывает окно."""
        if self._window_open:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            self._window_open = False
