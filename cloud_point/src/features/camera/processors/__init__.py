"""
Camera Processors - Обработчики изображений и облака точек.
"""
from src.features.camera.processors.depth_processor import DepthProcessor
from src.features.camera.processors.gripper_detector import GripperDetector
from src.features.camera.processors.object_tracker import ObjectTracker
from src.features.camera.processors.obstacle_detector import ObstacleDetector
from src.features.camera.processors.pointcloud_detector import PointCloudDetector

__all__ = [
    'DepthProcessor', 
    'GripperDetector', 
    'ObjectTracker', 
    'ObstacleDetector',
    'PointCloudDetector'
]

