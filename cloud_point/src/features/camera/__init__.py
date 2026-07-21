"""
Camera Feature - Модуль работы с камерой RealSense.

Содержит:
- services/ - Сервисы для работы с камерой
- processors/ - Обработчики изображений и облака точек
- ui/ - Компоненты пользовательского интерфейса
"""
from src.features.camera.services.camera_service import CameraService
from src.features.camera.processors.depth_processor import DepthProcessor
from src.features.camera.processors.gripper_detector import GripperDetector
from src.features.camera.processors.object_tracker import ObjectTracker

__all__ = [
    'CameraService',
    'DepthProcessor', 
    'GripperDetector',
    'ObjectTracker'
]

