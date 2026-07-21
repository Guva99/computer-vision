"""
Сервис для работы с камерой Intel RealSense.
"""
import pyrealsense2 as rs
import numpy as np
from typing import Optional, Tuple


class CameraService:
    """Сервис управления камерой RealSense."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Инициализация сервиса камеры.
        
        Args:
            width: Ширина кадра
            height: Высота кадра
            fps: Частота кадров
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.001
        self.is_running: bool = False
    
    def start(self) -> bool:
        """
        Запускает камеру.
        
        Returns:
            True если запуск успешен, False иначе
        """
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Настройка потоков
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            
            # Запуск pipeline
            profile = self.pipeline.start(self.config)
            
            # Получение масштаба глубины
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Настройка выравнивания
            self.align = rs.align(rs.stream.color)
            
            self.is_running = True
            print(f"✅ Камера запущена: {self.width}x{self.height}@{self.fps}fps")
            print(f"   Depth scale: {self.depth_scale}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка запуска камеры: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Останавливает камеру."""
        if self.pipeline and self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("📷 Камера остановлена")
    
    def get_frames(self) -> Tuple[Optional[object], Optional[object], Optional[object]]:
        """
        Получает выровненные кадры с камеры.
        
        Returns:
            Кортеж (depth_frame, color_frame, frameset) или (None, None, None) при ошибке
        """
        if not self.is_running or not self.pipeline:
            return None, None, None
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            return depth_frame, color_frame, frames
            
        except Exception as e:
            print(f"❌ Ошибка получения кадров: {e}")
            return None, None, None
    
    def get_intrinsics(self, color_frame):
        """
        Получает внутренние параметры камеры.
        
        Args:
            color_frame: Цветной кадр
            
        Returns:
            Внутренние параметры камеры
        """
        return color_frame.profile.as_video_stream_profile().intrinsics
    
    def get_depth_scale(self) -> float:
        """
        Возвращает масштаб глубины.
        
        Returns:
            Масштаб глубины
        """
        return self.depth_scale

