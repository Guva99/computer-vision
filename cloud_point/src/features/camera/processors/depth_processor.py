"""
Модуль для обработки глубины и детекции объектов в облаке точек.
"""
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from typing import List, Tuple, Optional

from src.utils.geometry import project_points_to_pixels
from src.constants.config import (
    HEIGHT_MIN, HEIGHT_MAX, MIN_CLUSTER_POINTS, MIN_CLUSTER_SIDE,
    MIN_CLUSTER_VOLUME, MERGE_DISTANCE, VOXEL_SIZE, OUTLIER_NB_NEIGHBORS,
    OUTLIER_STD_RATIO, RADIUS_NB_POINTS, RADIUS_VALUE, WHITE_HSV_LOWER,
    WHITE_HSV_UPPER, MIN_SHEET_AREA, MIN_SHEET_RATIO, MAX_SHEET_RATIO,
    MIN_POINTS_ON_SHEET, DEPTH_TRUNC
)


class DepthProcessor:
    """Класс для обработки глубины и детекции объектов."""
    
    def __init__(self, depth_scale: float):
        """
        Инициализация процессора глубины.
        
        Args:
            depth_scale: Масштаб глубины камеры RealSense
        """
        self.depth_scale = depth_scale
        self.last_sheet_mask: Optional[np.ndarray] = None
        
        # Инициализация фильтров RealSense
        self._init_depth_filters()
    
    def _init_depth_filters(self):
        """Инициализирует фильтры постобработки глубины RealSense."""
        self.dec_filter = rs.decimation_filter()
        self.dec_filter.set_option(rs.option.filter_magnitude, 1)
        
        self.spat_filter = rs.spatial_filter()
        self.spat_filter.set_option(rs.option.holes_fill, 2)
        
        self.temp_filter = rs.temporal_filter()
        self.temp_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temp_filter.set_option(rs.option.filter_smooth_delta, 50)
        self.temp_filter.set_option(rs.option.holes_fill, 3)
        
        self.hole_filter = rs.hole_filling_filter()
    
    def apply_depth_filters(self, depth_frame) -> rs.depth_frame:
        """
        Применяет фильтры постобработки к кадру глубины.
        
        Args:
            depth_frame: Исходный кадр глубины
            
        Returns:
            Обработанный кадр глубины
        """
        try:
            depth_processed = self.dec_filter.process(depth_frame)
            depth_processed = self.spat_filter.process(depth_processed)
            depth_processed = self.temp_filter.process(depth_processed)
            depth_processed = self.hole_filter.process(depth_processed)
            return depth_processed.as_depth_frame()
        except Exception:
            return depth_frame
    
    def detect_white_sheet(self, color_image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Обнаруживает белый лист на изображении.
        
        Args:
            color_image: Цветное изображение BGR
            
        Returns:
            Кортеж (маска листа, флаг применения маски)
        """
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array(WHITE_HSV_LOWER, dtype=np.uint8),
            np.array(WHITE_HSV_UPPER, dtype=np.uint8)
        )
        
        # Морфологическая очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Находим самый большой контур
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sheet_mask = np.zeros_like(white_mask)
        apply_mask = False
        
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > MIN_SHEET_AREA:
                cv2.drawContours(sheet_mask, [c], -1, 255, thickness=cv2.FILLED)
                area_ratio = float(cv2.countNonZero(sheet_mask)) / float(sheet_mask.size)
                apply_mask = MIN_SHEET_RATIO <= area_ratio <= MAX_SHEET_RATIO
                
                if apply_mask:
                    return (sheet_mask > 0).astype(np.uint8), True
        
        return None, False
    
    def get_effective_mask(self, color_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Получает эффективную маску рабочей области.
        
        Args:
            color_image: Цветное изображение
            
        Returns:
            Маска рабочей области или None
        """
        sheet_mask, apply_mask = self.detect_white_sheet(color_image)
        
        if apply_mask and sheet_mask is not None:
            self.last_sheet_mask = sheet_mask
            return sheet_mask
        elif self.last_sheet_mask is not None:
            return self.last_sheet_mask
        
        return None
    
    def create_point_cloud(
        self,
        depth_image: np.ndarray,
        color_image: np.ndarray,
        intrinsics,
        sheet_mask: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """
        Создает облако точек из изображений глубины и цвета.
        
        Args:
            depth_image: Изображение глубины
            color_image: Цветное изображение
            intrinsics: Внутренние параметры камеры
            sheet_mask: Маска рабочей области
            
        Returns:
            Облако точек Open3D
        """
        # Фильтрация глубины
        depth_filtered = cv2.medianBlur(depth_image, 3)
        
        # Применяем маску
        if sheet_mask is not None:
            if sheet_mask.shape != depth_filtered.shape:
                sheet_mask_resized = cv2.resize(
                    sheet_mask,
                    (depth_filtered.shape[1], depth_filtered.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                sheet_mask_resized = sheet_mask
            depth_filtered[sheet_mask_resized == 0] = 0
        
        # Создание RGBD изображения
        depth_o3d = o3d.geometry.Image(depth_filtered)
        color_o3d = o3d.geometry.Image(color_image)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1 / self.depth_scale,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )
        
        # Создание камеры
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy
        )
        
        # Создание облака точек
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic
        )
        
        # Трансформация системы координат
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
        
        # Окраска по глубине
        pcd = self._colorize_by_depth(pcd)
        
        return pcd
    
    def _colorize_by_depth(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Окрашивает облако точек по глубине."""
        if len(pcd.points) == 0:
            return pcd
        
        points = np.asarray(pcd.points)
        z_values = points[:, 2]
        
        z_min, z_max = z_values.min(), z_values.max()
        if z_max - z_min > 1e-6:
            z_norm = (z_values - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_values)
        
        colors = np.zeros((len(points), 3))
        colors[:, 0] = 1.0 - z_norm
        colors[:, 2] = z_norm
        colors[:, 1] = 0.2
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Применяет фильтры к облаку точек."""
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=RADIUS_NB_POINTS,
            radius=RADIUS_VALUE
        )
        return pcd
    
    def filter_by_sheet_mask(
        self,
        pcd: o3d.geometry.PointCloud,
        sheet_mask: np.ndarray,
        intrinsics
    ) -> Optional[o3d.geometry.PointCloud]:
        """Фильтрует облако точек по маске рабочей области."""
        if len(pcd.points) == 0:
            return None
        
        pts = np.asarray(pcd.points)
        px = project_points_to_pixels(pts, intrinsics)
        
        if px.size == 0:
            return None
        
        h, w = sheet_mask.shape[:2]
        u = np.clip(px[:, 0], 0, w - 1)
        v = np.clip(px[:, 1], 0, h - 1)
        on_sheet_mask = sheet_mask[v, u] > 0
        
        if not np.any(on_sheet_mask):
            return None
        
        valid_indices = np.where(on_sheet_mask)[0].tolist()
        return pcd.select_by_index(valid_indices)
    
    def detect_objects(
        self,
        pcd: o3d.geometry.PointCloud,
        intrinsics,
        sheet_mask: Optional[np.ndarray] = None
    ) -> List[Tuple]:
        """Обнаруживает объекты в облаке точек."""
        detections = []
        
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.006,
                ransac_n=3,
                num_iterations=800
            )
        except Exception:
            plane_model, inliers = None, []
        
        objects_pcd = pcd.select_by_index(inliers, invert=True) if len(inliers) else pcd
        
        if plane_model is not None and len(objects_pcd.points):
            objects_pcd = self._filter_by_height(objects_pcd, plane_model)
        
        labels, max_label = self._cluster_objects(objects_pcd)
        
        if max_label >= 0:
            for lbl in range(int(max_label) + 1):
                detection = self._process_cluster(
                    objects_pcd, labels, lbl, intrinsics, sheet_mask
                )
                if detection is not None:
                    detections.append(detection)
        
        if len(detections) > 1:
            detections = self._merge_close_detections(detections)
        
        return detections
    
    def _filter_by_height(
        self,
        pcd: o3d.geometry.PointCloud,
        plane_model: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """Фильтрует точки по высоте над плоскостью."""
        a, b, c, d = plane_model
        pts = np.asarray(pcd.points)
        
        if pts.size == 0:
            return pcd
        
        denom = max(np.sqrt(a * a + b * b + c * c), 1e-6)
        dist = np.abs(pts @ np.array([a, b, c]) + d) / denom
        mask_h = (dist >= HEIGHT_MIN) & (dist <= HEIGHT_MAX)
        
        if np.any(mask_h):
            return pcd.select_by_index(np.where(mask_h)[0].tolist())
        
        return pcd
    
    def _cluster_objects(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, int]:
        """Выполняет кластеризацию объектов."""
        labels = np.array([])
        max_label = -1
        
        if len(pcd.points) == 0:
            return labels, max_label
        
        pts = np.asarray(pcd.points)
        
        try:
            kdt = o3d.geometry.KDTreeFlann(pcd)
            sample_idx = np.linspace(0, len(pts) - 1, num=min(1500, len(pts)), dtype=int)
            d10 = []
            
            for i in sample_idx:
                _, _, d2 = kdt.search_knn_vector_3d(pts[i], 11)
                if len(d2) >= 11:
                    d10.append(np.sqrt(d2[-1]))
            
            base = float(np.median(d10)) if len(d10) > 0 else 0.015
        except Exception:
            base = 0.015
        
        voxel_size = 0.003
        eps = max(float(np.clip(base * 2.0, 0.008, 0.05)), voxel_size * 5.0)
        min_pts = 8
        
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts))
        max_label = labels.max() if labels.size else -1
        
        return labels, max_label
    
    def _process_cluster(
        self,
        objects_pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
        lbl: int,
        intrinsics,
        sheet_mask: Optional[np.ndarray]
    ) -> Optional[Tuple]:
        """Обрабатывает один кластер."""
        idx_local = np.where(labels == lbl)[0]
        
        if idx_local.size == 0:
            return None
        
        cluster = objects_pcd.select_by_index(idx_local.tolist())
        
        if len(cluster.points) < MIN_CLUSTER_POINTS:
            return None
        
        try:
            obb = cluster.get_oriented_bounding_box()
            center = obb.center
            extent = obb.extent
        except RuntimeError:
            aabb = cluster.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()
            obb = aabb
        
        e_sorted = np.sort(np.asarray(extent))
        min_side = float(e_sorted[0])
        volume = float(np.prod(e_sorted))
        
        if min_side < MIN_CLUSTER_SIDE or volume < MIN_CLUSTER_VOLUME:
            return None
        
        if sheet_mask is not None and intrinsics is not None:
            if not self._check_cluster_on_sheet(cluster, center, sheet_mask, intrinsics):
                return None
        elif sheet_mask is None:
            return None
        
        return (center, extent, obb, cluster)
    
    def _check_cluster_on_sheet(
        self,
        cluster: o3d.geometry.PointCloud,
        center: np.ndarray,
        sheet_mask: np.ndarray,
        intrinsics
    ) -> bool:
        """Проверяет, находится ли кластер на листе."""
        center_px = project_points_to_pixels(np.array([center]), intrinsics)
        if center_px.size > 0:
            h, w = sheet_mask.shape[:2]
            u_center = int(np.clip(center_px[0, 0], 0, w - 1))
            v_center = int(np.clip(center_px[0, 1], 0, h - 1))
            if sheet_mask[v_center, u_center] == 0:
                return False
        
        pts = np.asarray(cluster.points)
        px = project_points_to_pixels(pts, intrinsics)
        
        if px.size == 0:
            return False
        
        h, w = sheet_mask.shape[:2]
        u = np.clip(px[:, 0], 0, w - 1)
        v = np.clip(px[:, 1], 0, h - 1)
        on_sheet = sheet_mask[v, u] > 0
        
        if not np.any(on_sheet):
            return False
        
        ratio = np.count_nonzero(on_sheet) / len(on_sheet)
        return ratio >= MIN_POINTS_ON_SHEET
    
    def _merge_close_detections(self, detections: List[Tuple]) -> List[Tuple]:
        """Сливает близкие детекции."""
        used = [False] * len(detections)
        merged = []
        
        for i in range(len(detections)):
            if used[i]:
                continue
            
            group_idx = [i]
            ci = np.array(detections[i][0])
            
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                cj = np.array(detections[j][0])
                if np.linalg.norm(ci - cj) < MERGE_DISTANCE:
                    group_idx.append(j)
                    used[j] = True
            
            used[i] = True
            best = max(group_idx, key=lambda k: len(np.asarray(detections[k][3].points)))
            merged.append(detections[best])
        
        return merged

