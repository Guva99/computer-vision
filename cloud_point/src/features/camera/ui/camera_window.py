"""
Окно отображения видеопотока камеры.
"""
import os
from typing import Dict, List, Tuple, Optional, Callable

import cv2
import numpy as np

from src.utils.geometry import project_points_to_pixels

try:
    import pytesseract
    tess_path = os.environ.get("TESSERACT_PATH")
    if tess_path and os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
except Exception:
    pytesseract = None


def recognize_green_digit(roi_bgr: np.ndarray) -> str:
    """Распознаёт зелёную цифру в области интереса."""
    if roi_bgr is None or roi_bgr.size == 0:
        return ""

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, None, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return ""

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h < 80:
        return ""

    digit_roi = roi_bgr[y:y + h, x:x + w]
    if digit_roi.size == 0:
        return ""

    gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw

    if pytesseract is None:
        return ""

    try:
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(bw, config=config)
        text = text.strip()
        for ch in text:
            if ch.isdigit():
                return ch
    except Exception:
        return ""
    return ""


class CameraWindow:
    """Окно для отображения видеопотока камеры с наложением информации."""

    def __init__(self, title: str = 'Camera', on_click_callback: Optional[Callable] = None):
        """
        Инициализация окна камеры.
        
        Args:
            title: Заголовок окна
            on_click_callback: Callback функция для обработки кликов
        """
        self.title = title
        self._window_open = False
        self.mouse_point: Optional[Tuple[int, int]] = None
        self.on_click_callback = on_click_callback
        self.last_depth_frame = None
        self.last_depth_scale = 1.0
        self.last_intrinsics = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback для отслеживания мыши."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_point = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)
    
    def _handle_click(self, x: int, y: int):
        """Обрабатывает клик мыши."""
        if self.last_depth_frame is None or self.last_intrinsics is None:
            return
        
        depth_image = np.asanyarray(self.last_depth_frame.get_data())
        
        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
            return
        
        depth_value_mm = depth_image[y, x]
        distance_m = depth_value_mm * self.last_depth_scale
        
        if distance_m <= 0:
            return
        
        # Вычисляем 3D координаты
        z = distance_m
        x_3d = (x - self.last_intrinsics.ppx) * z / self.last_intrinsics.fx
        y_3d = (y - self.last_intrinsics.ppy) * z / self.last_intrinsics.fy
        
        # Формула для Z робота
        distance_mm = distance_m * 1000
        robot_z = -407 - (880 - distance_mm - 120)
        
        print(f"\n=== Click at ({x}, {y}) ===")
        print(f"Camera (m): X={x_3d:.3f}, Y={y_3d:.3f}, Z={z:.3f}")
        print(f"Distance: {distance_m:.3f} m ({distance_mm:.1f} mm)")
        print(f"Robot Z: {robot_z:.1f} mm")
        
        if self.on_click_callback:
            self.on_click_callback(x_3d, y_3d, z, distance_m, robot_z)

    def display(
        self,
        frame: np.ndarray,
        *,
        depth_frame=None,
        depth_scale: float = 1.0,
        stable_count: int = 0,
        current_count: int = 0,
        stable_frames: int = 0,
        stable_threshold: int = 0,
        time_to_change: float = 0,
        cpu_percent: float = 0,
        memory_percent: float = 0,
        thread_count: int = 0,
        fps: float = 0,
        tracks: Optional[Dict] = None,
        detections: Optional[List] = None,
        intrinsics=None,
        track_persist: int = 0,
    ) -> bool:
        """
        Отображает кадр с наложенной информацией.
        
        Returns:
            True для продолжения, False для выхода
        """
        if tracks is None:
            tracks = {}
        if detections is None:
            detections = []
        
        self.last_depth_frame = depth_frame
        self.last_depth_scale = depth_scale
        self.last_intrinsics = intrinsics
        
        overlay = frame.copy()
        
        # FPS
        if fps > 0:
            cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Подсказка о клике
        if self.on_click_callback is not None:
            cv2.putText(overlay, "Click to send to robot",
                        (20, overlay.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2, cv2.LINE_AA)
        
        # Информация при наведении мыши
        self._draw_mouse_info(overlay, depth_frame, depth_scale, intrinsics)
        
        # Устанавливаем callback мыши при первом показе
        if not self._window_open:
            cv2.namedWindow(self.title)
            cv2.setMouseCallback(self.title, self._mouse_callback)
        
        cv2.imshow(self.title, overlay)
        self._window_open = True
        
        key = cv2.waitKey(1)
        return not (key & 0xFF == ord('q'))
    
    def _draw_mouse_info(self, overlay: np.ndarray, depth_frame, depth_scale: float, intrinsics):
        """Отрисовывает информацию при наведении мыши."""
        if depth_frame is None or self.mouse_point is None or intrinsics is None:
            return
        
        depth_image = np.asanyarray(depth_frame.get_data())
        x, y = self.mouse_point
        
        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
            return
        
        depth_value_mm = depth_image[y, x]
        distance_m = depth_value_mm * depth_scale
        
        if distance_m <= 0:
            return
        
        z = distance_m
        x_3d = (x - intrinsics.ppx) * z / intrinsics.fx
        y_3d = (y - intrinsics.ppy) * z / intrinsics.fy
        
        cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)
        
        cv2.putText(overlay, f"Dist: {distance_m:.3f} m", (x + 10, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"X: {x_3d:.3f} m", (x + 10, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Y: {y_3d:.3f} m", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Z: {z:.3f} m", (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    def close(self):
        """Закрывает окно."""
        if self._window_open:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            self._window_open = False

