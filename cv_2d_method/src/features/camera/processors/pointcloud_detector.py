"""
Модуль для обнаружения объектов и хвата на основе облака точек XYZRGB.

ОПТИМИЗИРОВАННАЯ ВЕРСИЯ:
- Быстрое создание облака точек (без лишних операций)
- Улучшенная фильтрация по позиции (исключаем края кадра)
- Точная классификация хвата по позиции и цвету
- Корректное определение препятствий по высоте
"""
import cv2
import numpy as np
import open3d as o3d
import logging
from typing import Optional, List, Dict, Tuple
from src.utils.text_renderer import put_russian_text, get_text_size
from src.constants import config as app_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PointCloudDetector:
    """
    Оптимизированный детектор объектов на основе XYZRGB облака точек.
    """
    
    def __init__(self):
        """Инициализация детектора."""
        # === ПАРАМЕТРЫ ГЛУБИНЫ ===
        self.z_min = 0.4    # Минимальная глубина (м) - отсекаем близкие объекты
        self.z_max = 0.95   # Максимальная глубина (м) - отсекаем фон
        
        # === ОПТИМИЗАЦИЯ СКОРОСТИ ===
        self.voxel_size = 0.005       # 5мм - крупнее = быстрее
        self.skip_pixels = 2          # Пропускаем каждый N-й пиксель для скорости
        
        # === RANSAC (быстрые параметры) ===
        self.ransac_distance = 0.01   # 1см толщина плоскости
        self.ransac_iterations = 300  # Меньше итераций = быстрее
        
        # === DBSCAN ===
        self.dbscan_eps = 0.025       # 2.5см - расстояние между точками
        self.dbscan_min_points = 20   # Минимум точек в кластере
        
        # === ФИЛЬТРЫ ОБЪЕКТОВ ===
        self.min_object_points = 50   # Минимум точек
        self.min_object_size = 0.02   # 2см минимальный размер
        self.max_object_size = 0.12   # 12см максимальный размер
        
        # === ЗОНА ИСКЛЮЧЕНИЯ (края кадра, крепления) ===
        self.exclude_left_ratio = 0.12   # Исключаем 12% слева (крепления)
        self.exclude_right_ratio = 0.05  # Исключаем 5% справа
        self.exclude_top_ratio = 0.05    # Исключаем 5% сверху
        self.exclude_bottom_ratio = 0.15 # Исключаем 15% снизу (корпус робота)
        
        # === КЛАССИФИКАЦИЯ ХВАТА ===
        # Хват - тёмный объект в центральной зоне
        self.gripper_brightness_max = 0.35  # Максимальная яркость
        self.gripper_center_zone = 0.4      # Хват должен быть в центральных 40% по X
        
        # === ПРЕПЯТСТВИЯ ===
        self.height_tolerance = 0.05  # 5см допуск по высоте

        # === SELF FILTER (манипулятор) ===
        self.self_filter_enabled = getattr(app_config, 'POINTCLOUD_SELF_FILTER_ENABLED', True)
        self.self_filter_boxes = getattr(
            app_config,
            'POINTCLOUD_SELF_FILTER_BOXES',
            [(-0.18, 0.18, -0.05, 0.25, 0.30, 0.65)]
        )
        
        # === КЭШИРОВАНИЕ ===
        self.last_gripper_info: Optional[Dict] = None
        self.last_objects: List[Dict] = []
        self._intrinsics = None
        self._frame_count = 0
        
        # Сглаживание
        self.smooth_alpha = 0.5
    
    def detect(
        self,
        color_image: np.ndarray,
        depth_frame,
        depth_scale: float,
        intrinsics
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Основной метод детекции - ОПТИМИЗИРОВАННЫЙ.
        """
        self._intrinsics = intrinsics
        self._frame_count += 1
        
        if depth_frame is None or color_image is None:
            return self.last_gripper_info, self.last_objects
        
        height, width = color_image.shape[:2]
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # === 1. БЫСТРОЕ СОЗДАНИЕ ОБЛАКА ТОЧЕК ===
        pcd, pixel_coords = self._create_fast_pointcloud(
            color_image, depth_image, depth_scale, intrinsics
        )
        
        if pcd is None or len(pcd.points) < 100:
            return self.last_gripper_info, self.last_objects
        
        # === 2. СЕГМЕНТАЦИЯ ПЛОСКОСТИ (RANSAC) ===
        pcd_objects = self._segment_plane_fast(pcd)
        
        if pcd_objects is None or len(pcd_objects.points) < 30:
            return self.last_gripper_info, self.last_objects
        
        # === 3. КЛАСТЕРИЗАЦИЯ (DBSCAN) ===
        labels = np.array(pcd_objects.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points
        ))
        
        max_label = labels.max() if len(labels) > 0 else -1
        
        if max_label < 0:
            return self.last_gripper_info, self.last_objects
        
        # === 4. АНАЛИЗ КЛАСТЕРОВ ===
        points = np.asarray(pcd_objects.points)
        colors = np.asarray(pcd_objects.colors)
        
        gripper_info = None
        objects = []
        gripper_candidates = []
        
        for label_id in range(max_label + 1):
            mask = labels == label_id
            cluster_points = points[mask]
            cluster_colors = colors[mask]
            
            if len(cluster_points) < self.min_object_points:
                continue
            
            # Анализируем кластер
            info = self._analyze_cluster_fast(cluster_points, cluster_colors, width, height)
            
            if info is None:
                continue
            
            # Проверяем размер
            extent = info['extent']
            min_dim = np.min(extent[:2])  # X, Y размеры
            max_dim = np.max(extent[:2])
            
            if min_dim < self.min_object_size or max_dim > self.max_object_size:
                continue
            
            # Классификация
            is_gripper_candidate = self._is_gripper_candidate(info, width)
            
            if is_gripper_candidate:
                gripper_candidates.append(info)
            else:
                # Проверяем - это препятствие или безопасный объект
                info['is_obstacle'] = self._check_is_obstacle(info)
                objects.append(info)
        
        # Выбираем лучший хват
        if gripper_candidates:
            gripper_info = self._select_best_gripper(gripper_candidates)
            
            # Обновляем статус препятствий относительно хвата
            if gripper_info:
                gripper_depth = gripper_info['depth_m']
                for obj in objects:
                    obj['is_obstacle'] = self._check_is_obstacle(obj, gripper_depth)
        
        # Сглаживание хвата
        if gripper_info and self.last_gripper_info:
            gripper_info = self._smooth_gripper(gripper_info)
        
        # Обновляем кэш
        if gripper_info:
            self.last_gripper_info = gripper_info
        if objects:
            self.last_objects = objects
        
        return gripper_info, objects
    
    def _create_fast_pointcloud(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_scale: float,
        intrinsics
    ) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[np.ndarray]]:
        """Быстрое создание облака точек с пропуском пикселей."""
        height, width = depth_image.shape
        
        # Пропускаем пиксели для скорости
        skip = self.skip_pixels
        v_indices = np.arange(0, height, skip)
        u_indices = np.arange(0, width, skip)
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)
        
        # Глубина
        z = depth_image[v_grid, u_grid] * depth_scale
        
        # Зона исключения по позиции
        exclude_left = int(width * self.exclude_left_ratio)
        exclude_right = int(width * (1 - self.exclude_right_ratio))
        exclude_top = int(height * self.exclude_top_ratio)
        exclude_bottom = int(height * (1 - self.exclude_bottom_ratio))
        
        # Маска валидных точек
        valid_mask = (
            (z > self.z_min) & (z < self.z_max) &
            (u_grid >= exclude_left) & (u_grid <= exclude_right) &
            (v_grid >= exclude_top) & (v_grid <= exclude_bottom)
        )
        
        if np.sum(valid_mask) < 100:
            return None, None
        
        # 3D координаты
        z_valid = z[valid_mask]
        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        
        x = (u_valid - intrinsics.ppx) * z_valid / intrinsics.fx
        y = (v_valid - intrinsics.ppy) * z_valid / intrinsics.fy
        
        points = np.stack([x, y, z_valid], axis=-1)
        
        if self.self_filter_enabled and len(points) > 0:
            keep_mask = np.ones(len(points), dtype=bool)
            for box in self.self_filter_boxes:
                x_min, x_max, y_min, y_max, z_min, z_max = box
                in_box = (
                    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                )
                keep_mask &= ~in_box
            
            if np.sum(keep_mask) < 100:
                return None, None
            
            points = points[keep_mask]
            u_valid = u_valid[keep_mask]
            v_valid = v_valid[keep_mask]
        
        # Цвета (BGR -> RGB, нормализация)
        colors = color_image[v_valid, u_valid].astype(np.float64) / 255.0
        colors = colors[:, ::-1]  # BGR -> RGB
        
        # Создаём облако точек
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Быстрый даунсемплинг
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        return pcd, np.stack([u_valid, v_valid], axis=-1)
    
    def _segment_plane_fast(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Optional[o3d.geometry.PointCloud]:
        """Быстрая сегментация плоскости."""
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.ransac_distance,
                ransac_n=3,
                num_iterations=self.ransac_iterations
            )
            return pcd.select_by_index(inliers, invert=True)
        except:
            return pcd
    
    def _analyze_cluster_fast(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        img_width: int,
        img_height: int
    ) -> Optional[Dict]:
        """Быстрый анализ кластера."""
        if len(points) == 0:
            return None
        
        # Центр и размеры
        center = np.mean(points, axis=0)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        extent = max_bound - min_bound
        
        # Средняя глубина
        depth_m = center[2]
        
        # Средний цвет и яркость
        mean_color = np.mean(colors, axis=0)
        brightness = np.mean(mean_color)
        
        # Проверка на синий (провода)
        is_blue = self._check_blue_fast(colors)
        
        # Проекция на 2D
        bbox = self._project_bbox_fast(min_bound, max_bound)
        
        return {
            'center_3d': center,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'extent': extent,
            'depth_m': depth_m,
            'brightness': brightness,
            'mean_color': mean_color,
            'is_blue': is_blue,
            'num_points': len(points),
            'bbox': bbox
        }
    
    def _check_blue_fast(self, colors: np.ndarray) -> bool:
        """Быстрая проверка на синий цвет."""
        # Простая проверка: B > R и B > G
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        blue_dominant = (b > r + 0.1) & (b > g + 0.1) & (b > 0.3)
        return np.mean(blue_dominant) > 0.4
    
    def _project_bbox_fast(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Быстрая проекция bbox на 2D."""
        if self._intrinsics is None:
            return (0, 0, 50, 50)
        
        fx, fy = self._intrinsics.fx, self._intrinsics.fy
        ppx, ppy = self._intrinsics.ppx, self._intrinsics.ppy
        
        # Проецируем только 2 угла (достаточно для bbox)
        z_avg = (min_bound[2] + max_bound[2]) / 2
        
        u_min = int(min_bound[0] * fx / z_avg + ppx)
        u_max = int(max_bound[0] * fx / z_avg + ppx)
        v_min = int(min_bound[1] * fy / z_avg + ppy)
        v_max = int(max_bound[1] * fy / z_avg + ppy)
        
        x = max(0, min(u_min, u_max))
        y = max(0, min(v_min, v_max))
        w = max(20, abs(u_max - u_min))
        h = max(20, abs(v_max - v_min))
        
        return (x, y, w, h)
    
    def _is_gripper_candidate(self, info: Dict, img_width: int) -> bool:
        """Проверяет, является ли кластер кандидатом на хват."""
        # Исключаем синие объекты
        if info['is_blue']:
            return False
        
        # Хват должен быть тёмным
        if info['brightness'] > self.gripper_brightness_max:
            return False
        
        # Хват должен быть в центральной зоне по X
        center_x_3d = info['center_3d'][0]
        # Примерный диапазон X для центра (в метрах)
        # Зависит от настройки камеры, но обычно центр около 0
        if abs(center_x_3d) > 0.15:  # Не дальше 15см от центра
            return False
        
        # Хват не должен быть слишком большим
        extent = info['extent']
        if np.max(extent) > 0.08:  # Не больше 8см
            return False
        
        return True
    
    def _check_is_obstacle(
        self,
        info: Dict,
        gripper_depth: Optional[float] = None
    ) -> bool:
        """Проверяет, является ли объект препятствием."""
        obj_depth = info['depth_m']
        
        # Синие объекты (провода) - не препятствия
        if info['is_blue']:
            return False
        
        if gripper_depth is not None:
            # Препятствие если объект выше хвата (меньше depth)
            # или на той же высоте (в пределах допуска)
            height_diff = gripper_depth - obj_depth
            return height_diff > -self.height_tolerance
        else:
            # Без хвата - используем фиксированный порог
            return obj_depth < 0.65
    
    def _select_best_gripper(self, candidates: List[Dict]) -> Optional[Dict]:
        """Выбирает лучший кандидат на хват."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Оценка: темнее + больше точек + ближе к центру
        best = None
        best_score = -1
        
        for c in candidates:
            dark_score = 1.0 - c['brightness']
            size_score = min(c['num_points'] / 500.0, 1.0)
            center_score = 1.0 - min(abs(c['center_3d'][0]) / 0.15, 1.0)
            
            score = dark_score * 0.4 + size_score * 0.3 + center_score * 0.3
            
            if score > best_score:
                best_score = score
                best = c
        
        return best
    
    def _smooth_gripper(self, current: Dict) -> Dict:
        """Сглаживает информацию о хвате."""
        if self.last_gripper_info is None:
            return current
        
        alpha = self.smooth_alpha
        prev = self.last_gripper_info
        
        # Сглаживаем центр
        current['center_3d'] = alpha * current['center_3d'] + (1 - alpha) * prev['center_3d']
        current['depth_m'] = alpha * current['depth_m'] + (1 - alpha) * prev['depth_m']
        
        # Сглаживаем bbox
        cx, cy, cw, ch = current['bbox']
        px, py, pw, ph = prev['bbox']
        current['bbox'] = (
            int(alpha * cx + (1 - alpha) * px),
            int(alpha * cy + (1 - alpha) * py),
            int(alpha * cw + (1 - alpha) * pw),
            int(alpha * ch + (1 - alpha) * ph)
        )
        
        return current
    
    def check_collision(
        self,
        gripper_info: Optional[Dict],
        objects: List[Dict],
        margin: float = 0.02
    ) -> Tuple[bool, List[Dict]]:
        """Проверяет столкновение хвата с препятствиями."""
        if gripper_info is None or not objects:
            return False, []
        
        gripper_center = gripper_info['center_3d']
        gripper_extent = gripper_info['extent']
        gripper_depth = gripper_info['depth_m']
        
        colliding = []
        
        for obj in objects:
            if not obj.get('is_obstacle', False):
                continue
            
            obj_center = obj['center_3d']
            obj_extent = obj['extent']
            
            # Проверяем пересечение в 3D
            dist = np.abs(gripper_center - obj_center)
            combined = (gripper_extent + obj_extent) / 2 + margin
            
            if np.all(dist < combined):
                colliding.append(obj)
        
        return len(colliding) > 0, colliding
    
    def draw_results(
        self,
        image: np.ndarray,
        gripper_info: Optional[Dict],
        objects: List[Dict],
        collision: bool = False
    ) -> np.ndarray:
        """Отрисовывает результаты."""
        overlay = image.copy()
        img_h, img_w = image.shape[:2]
        
        # Рисуем объекты
        for obj in objects:
            x, y, w, h = obj['bbox']
            depth_m = obj['depth_m']
            is_obstacle = obj.get('is_obstacle', False)
            
            # Пропускаем синие (провода)
            if obj.get('is_blue', False):
                continue
            
            if is_obstacle:
                color = (0, 0, 255)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Полупрозрачная заливка
                overlay_fill = overlay.copy()
                cv2.rectangle(overlay_fill, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(overlay_fill, 0.2, overlay, 0.8, 0, overlay)
                
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                
                label = f"ОПАСНО={depth_m:.2f}"
                overlay = put_russian_text(
                    overlay, label, (x, y - 15),
                    font_size=11, color=(255, 255, 255),
                    bg_color=color, padding=2
                )
            else:
                # Безопасный - только точка и высота
                color = (255, 150, 0)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(overlay, (cx, cy), 4, color, -1)
                cv2.putText(
                    overlay, f"{depth_m:.2f}", (cx + 8, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
                )
        
        # Рисуем хват
        if gripper_info is not None:
            x, y, w, h = gripper_info['bbox']
            depth_m = gripper_info['depth_m']
            
            color = (0, 0, 255) if collision else (0, 255, 0)
            fill_alpha = 0.3 if collision else 0.15
            
            # Заливка
            overlay_fill = overlay.copy()
            cv2.rectangle(overlay_fill, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay_fill, fill_alpha, overlay, 1 - fill_alpha, 0, overlay)
            
            # Рамка
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
            
            # Угловые маркеры
            corner_len = 15
            corner_color = (0, 255, 255)
            ct = 3
            cv2.line(overlay, (x, y), (x + corner_len, y), corner_color, ct)
            cv2.line(overlay, (x, y), (x, y + corner_len), corner_color, ct)
            cv2.line(overlay, (x + w, y), (x + w - corner_len, y), corner_color, ct)
            cv2.line(overlay, (x + w, y), (x + w, y + corner_len), corner_color, ct)
            cv2.line(overlay, (x, y + h), (x + corner_len, y + h), corner_color, ct)
            cv2.line(overlay, (x, y + h), (x, y + h - corner_len), corner_color, ct)
            cv2.line(overlay, (x + w, y + h), (x + w - corner_len, y + h), corner_color, ct)
            cv2.line(overlay, (x + w, y + h), (x + w, y + h - corner_len), corner_color, ct)
            
            # Центр
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 2)
            cv2.line(overlay, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 2)
            cv2.line(overlay, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 2)
            
            # Заголовок
            label = "СТОЛКНОВЕНИЕ!" if collision else "ЗОНА ХВАТА"
            label += f"={depth_m:.2f}"
            
            label_y = y - 5 if y > 25 else y + h + 5
            overlay = put_russian_text(
                overlay, label, (x, label_y - 15),
                font_size=12, color=(255, 255, 255),
                bg_color=color, padding=3
            )
        
        # Предупреждение
        if collision:
            warning_text = "ВНИМАНИЕ: СТОЛКНОВЕНИЕ!"
            text_w, text_h = get_text_size(warning_text, 18)
            wx = (img_w - text_w) // 2
            overlay = put_russian_text(
                overlay, warning_text, (wx, 10),
                font_size=18, color=(255, 255, 255),
                bg_color=(0, 0, 200), padding=6
            )
        
        # Счётчик препятствий
        num_obstacles = sum(1 for o in objects if o.get('is_obstacle', False))
        if num_obstacles > 0:
            overlay = put_russian_text(
                overlay, f"Опасно: {num_obstacles}",
                (img_w - 100, 15), font_size=12, color=(0, 0, 255)
            )
        
        return overlay
    
    def log_detection(self, gripper_info: Optional[Dict], objects: List[Dict]):
        """Логирует результаты."""
        if gripper_info:
            logger.info(
                f"GRIPPER: depth={gripper_info['depth_m']:.3f}m, "
                f"points={gripper_info['num_points']}"
            )
