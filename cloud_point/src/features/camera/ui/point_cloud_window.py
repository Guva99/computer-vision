"""
Окно визуализации облака точек.
"""
import os
import sys

os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "true"

import numpy as np
import cv2
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class PointCloudWindow:
    """Окно визуализации облака точек с Open3D."""

    def __init__(
        self,
        title: str = 'Point Cloud',
        width: int = 960,
        height: int = 940,
        depth_trunc: float = 3.0,
        show_depth_stream: bool = False
    ):
        """
        Инициализация окна облака точек.
        
        Args:
            title: Заголовок окна
            width: Ширина окна
            height: Высота окна
            depth_trunc: Максимальная глубина (метры)
            show_depth_stream: Показывать ли окно потока глубины
        """
        self.width = width
        self.height = height
        self.depth_trunc = depth_trunc
        self.show_depth_stream = show_depth_stream
        
        # Open3D визуализатор
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=width, height=height)
        
        # Инициализация пустого облака точек
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        self._first_render = True
        
        # Матрица трансформации для корректировки ориентации
        self.transform_matrix = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

    def render(self, geometries):
        """
        Рендерит список геометрий.
        
        Args:
            geometries: Список Open3D геометрий для отображения
        """
        self.vis.clear_geometries()
        
        for geom in geometries:
            self.vis.add_geometry(geom, reset_bounding_box=self._first_render)
        
        if self._first_render and geometries:
            self._first_render = False
            self.vis.reset_view_point(True)
        
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_point_cloud(self, pcd: o3d.geometry.PointCloud):
        """
        Обновляет облако точек.
        
        Args:
            pcd: Облако точек для отображения
        """
        self.pcd.points = pcd.points
        self.pcd.colors = pcd.colors if pcd.has_colors() else o3d.utility.Vector3dVector()
        
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Закрывает окно."""
        cv2.destroyAllWindows()
        self.vis.destroy_window()

