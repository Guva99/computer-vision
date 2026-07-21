"""
Realtime сервис обработки облака точек.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np

from src.features.camera.processors.pointcloud_detector import PointCloudDetector


class RealTimePointCloudService:
    """
    Сервис для realtime обработки point cloud кадров.
    """

    def __init__(self, detector: Optional[PointCloudDetector] = None):
        self.detector = detector if detector is not None else PointCloudDetector()
        self.processed_frames = 0

    def process_frame(
        self,
        color_frame,
        depth_frame,
        depth_scale: float,
        intrinsics
    ) -> Tuple[bool, list, Optional[Dict[str, Any]], np.ndarray]:
        """
        Обрабатывает один realtime кадр через PointCloudDetector.

        Returns:
            collision, detected_objects, gripper_info, rendered_image
        """
        color_image = np.asanyarray(color_frame.get_data())

        gripper_info, detected_objects = self.detector.detect(
            color_image=color_image,
            depth_frame=depth_frame,
            depth_scale=depth_scale,
            intrinsics=intrinsics
        )

        collision, _ = self.detector.check_collision(
            gripper_info, detected_objects
        )

        # В 3D режиме не рисуем 2D оверлеи.
        rendered_image = color_image

        self.processed_frames += 1
        return collision, detected_objects, gripper_info, rendered_image
