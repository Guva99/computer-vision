"""
Обёртка 3D-окна Open3D для сырого облака точек.

Окно отключаемо (cfg.show_o3d_window=False → vis=None, экономит ~60 мс/кадр).
Для скриншотов в статью поставь show_o3d_window=True в AppConfig.
Тяжёлый рендер обновляется раз в N кадров (1-й кадр всегда).
"""
import open3d as o3d

from realsense_io import AppConfig


class CloudVisualizer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.pcd = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.vis = None
        if cfg.show_o3d_window:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Raw point cloud (RGB colors, no filters)",
                width=1200, height=800,
            )
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.coordinate_frame)

    def render(self, points, colors, frame_count: int) -> None:
        cfg = self.cfg
        if self.vis is not None and (frame_count == 1 or (frame_count % cfg.o3d_render_every_n == 0)):
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)
            if frame_count == 1:
                self.vis.reset_view_point(True)
                print("[INFO] Open3D view reset")
            self.vis.poll_events()
            self.vis.update_renderer()

    def destroy(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
