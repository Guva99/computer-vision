"""
Модуль для обнаружения и отслеживания хвата (gripper) манипулятора.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, List
from src.utils.text_renderer import put_russian_text, get_text_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GripperDetector:
    """Класс для обнаружения хвата манипулятора в кадре камеры."""
    
    def __init__(self):
        """Инициализация детектора хвата."""
        # Параметры HSV для обнаружения чёрного/тёмного хвата
        self.hsv_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.hsv_upper = np.array([180, 255, 80], dtype=np.uint8)  # Увеличил порог яркости
        
        # Параметры фильтрации для отдельных частей хвата
        self.min_finger_area = 1000     # Минимальная площадь одного "пальца"
        self.max_finger_area = 25000    # Максимальная площадь одного "пальца"
        
        # Параметры для объединённой зоны хвата
        self.min_gripper_area = 1500    # Минимальная площадь всей зоны хвата
        self.max_gripper_area = 80000   # Максимальная площадь всей зоны хвата
        
        self.search_region_y_ratio = 0.95
        
        # Сглаживание
        self.last_gripper_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_gripper_position: Optional[Tuple[int, int]] = None
        self.last_gripper_height: Optional[float] = None
        self.smooth_alpha = 0.4
        
        # Счётчик пропусков
        self.miss_count = 0
        self.max_misses = 10
        
        # Параметры зоны хвата
        self.gripper_zone_expand = 20  # Расширение зоны хвата в пикселях
        
        # Расширение для поиска всего хвата (оба пальца)
        self.gripper_search_expand_x = 50  # Расширение по X для поиска второго пальца
        self.gripper_search_expand_y = 30  # Расширение по Y
    
    def detect(
        self,
        color_image: np.ndarray,
        depth_frame=None,
        depth_scale: float = 1.0,
        intrinsics=None
    ) -> Optional[Dict]:
        """
        Обнаруживает хват манипулятора в кадре.
        Находит все тёмные части хвата и объединяет их в единую зону.
        """
        height, width = color_image.shape[:2]
        search_height = int(height * self.search_region_y_ratio)
        search_region = color_image[:search_height, :]
        
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Находим белый лист (рабочая область)
        white_lower = np.array([0, 0, 150], dtype=np.uint8)
        white_upper = np.array([180, 60, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Маска тёмных объектов (хват)
        dark_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Расширяем белую маску для поиска хвата рядом с листом
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        white_expanded = cv2.dilate(white_mask, kernel_expand, iterations=2)
        
        # Хват на белом листе или рядом
        gripper_search_mask = cv2.bitwise_and(dark_mask, white_expanded)
        
        # Морфологические операции для очистки
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        gripper_search_mask = cv2.morphologyEx(gripper_search_mask, cv2.MORPH_OPEN, kernel_small)
        gripper_search_mask = cv2.morphologyEx(gripper_search_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Дополнительное расширение для объединения близких частей хвата
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        gripper_connected = cv2.dilate(gripper_search_mask, kernel_connect, iterations=1)
        gripper_connected = cv2.erode(gripper_connected, kernel_connect, iterations=1)
        
        # Находим все контуры (части хвата)
        contours, _ = cv2.findContours(gripper_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self._handle_miss()
            return None
        
        # Фильтруем контуры и находим части хвата
        gripper_parts = self._find_gripper_parts(contours, width, height)
        
        if not gripper_parts:
            self._handle_miss()
            return None
        
        # Объединяем близкие части хвата в единую зону
        gripper_zone = self._merge_gripper_parts(gripper_parts, width, height)
        
        if gripper_zone is None:
            self._handle_miss()
            return None
        
        # Сглаживание bbox
        x, y, w, h = gripper_zone['bbox']
        
        if self.last_gripper_bbox is not None:
            lx, ly, lw, lh = self.last_gripper_bbox
            x = int(self.smooth_alpha * x + (1 - self.smooth_alpha) * lx)
            y = int(self.smooth_alpha * y + (1 - self.smooth_alpha) * ly)
            w = int(self.smooth_alpha * w + (1 - self.smooth_alpha) * lw)
            h = int(self.smooth_alpha * h + (1 - self.smooth_alpha) * lh)
        
        self.last_gripper_bbox = (x, y, w, h)
        
        # Центр зоны хвата
        cx = x + w // 2
        cy = y + h // 2
        
        self.last_gripper_position = (cx, cy)
        self.miss_count = 0
        
        result = {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'gripper_zone': gripper_zone,
            'parts': gripper_parts,
            'area': w * h,
            'depth_m': None,
            'height_m': None,
            'position_3d': None
        }
        
        # Вычисляем глубину по центру зоны хвата
        if depth_frame is not None:
            self._calculate_depth(result, depth_frame, depth_scale, intrinsics, cx, cy)
        
        return result
    
    def _find_gripper_parts(
        self,
        contours: List,
        width: int,
        height: int
    ) -> List[Dict]:
        """Находит части хвата (пальцы) среди контуров."""
        gripper_parts = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_finger_area or area > self.max_finger_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтр по позиции - исключаем левый край (крепления)
            if x < width * 0.08:
                continue
            
            # Центр контура
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Фильтр - хват должен быть в центральной области
            center_x = width // 2
            dist_from_center_x = abs(cx - center_x)
            max_dist_from_center = width * 0.45
            
            if dist_from_center_x > max_dist_from_center:
                continue
            
            gripper_parts.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            })
        
        return gripper_parts
    
    def _merge_gripper_parts(
        self,
        parts: List[Dict],
        width: int,
        height: int
    ) -> Optional[Dict]:
        """Объединяет части хвата в единую зону."""
        if not parts:
            return None
        
        # Группируем близкие части
        groups = self._group_nearby_parts(parts, max_distance=100)
        
        if not groups:
            return None
        
        # Выбираем лучшую группу (ближе к центру и больше по площади)
        center_x = width // 2
        best_group = None
        best_score = -1
        
        for group in groups:
            # Вычисляем общий bbox группы
            all_x = []
            all_y = []
            all_x2 = []
            all_y2 = []
            total_area = 0
            
            for part in group:
                x, y, w, h = part['bbox']
                all_x.append(x)
                all_y.append(y)
                all_x2.append(x + w)
                all_y2.append(y + h)
                total_area += part['area']
            
            min_x = min(all_x) - self.gripper_zone_expand
            min_y = min(all_y) - self.gripper_zone_expand
            max_x = max(all_x2) + self.gripper_zone_expand
            max_y = max(all_y2) + self.gripper_zone_expand
            
            # Ограничиваем границами изображения
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(width, max_x)
            max_y = min(height, max_y)
            
            group_cx = (min_x + max_x) // 2
            dist_from_center = abs(group_cx - center_x)
            
            # Оценка: больше площадь + ближе к центру + больше частей
            centrality = 1.0 - (dist_from_center / (width / 2))
            score = total_area * centrality * len(group)
            
            if score > best_score:
                best_score = score
                best_group = {
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'parts': group,
                    'total_area': total_area,
                    'num_parts': len(group)
                }
        
        # Проверяем минимальную площадь зоны хвата
        if best_group:
            x, y, w, h = best_group['bbox']
            if w * h < self.min_gripper_area:
                return None
            if w * h > self.max_gripper_area:
                return None
        
        return best_group
    
    def _group_nearby_parts(
        self,
        parts: List[Dict],
        max_distance: int = 100
    ) -> List[List[Dict]]:
        """Группирует близкие части хвата."""
        if not parts:
            return []
        
        used = [False] * len(parts)
        groups = []
        
        for i, part in enumerate(parts):
            if used[i]:
                continue
            
            group = [part]
            used[i] = True
            
            # Находим все близкие части
            for j, other in enumerate(parts):
                if used[j]:
                    continue
                
                # Расстояние между центрами
                cx1, cy1 = part['center']
                cx2, cy2 = other['center']
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                
                if dist < max_distance:
                    group.append(other)
                    used[j] = True
            
            groups.append(group)
        
        return groups
    
    def _calculate_depth(
        self,
        result: Dict,
        depth_frame,
        depth_scale: float,
        intrinsics,
        cx: int,
        cy: int
    ):
        """Вычисляет глубину хвата."""
        depth_image = np.asanyarray(depth_frame.get_data())
        
        if not (0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]):
            return
        
        # Берём область вокруг центра
        x, y, w, h = result['bbox']
        x1 = max(0, x + w // 4)
        x2 = min(depth_image.shape[1], x + 3 * w // 4)
        y1 = max(0, y + h // 4)
        y2 = min(depth_image.shape[0], y + 3 * h // 4)
        
        roi_depth = depth_image[y1:y2, x1:x2]
        valid_depths = roi_depth[roi_depth > 0]
        
        if len(valid_depths) == 0:
            return
        
        depth_mm = np.median(valid_depths)
        depth_m = depth_mm * depth_scale
        result['depth_m'] = depth_m
        
        if self.last_gripper_height is not None:
            depth_m = self.smooth_alpha * depth_m + (1 - self.smooth_alpha) * self.last_gripper_height
        self.last_gripper_height = depth_m
        result['height_m'] = depth_m
        
        if intrinsics is not None:
            z = depth_m
            x_3d = (cx - intrinsics.ppx) * z / intrinsics.fx
            y_3d = (cy - intrinsics.ppy) * z / intrinsics.fy
            result['position_3d'] = (x_3d, y_3d, z)
    
    def _handle_miss(self):
        """Обрабатывает пропуск детекции."""
        self.miss_count += 1
        if self.miss_count > self.max_misses:
            self.last_gripper_bbox = None
            self.last_gripper_position = None
            self.last_gripper_height = None
    
    def check_collision(
        self,
        gripper_info: Optional[Dict],
        obstacles: List[Dict],
        proximity_margin: int = 5
    ) -> Tuple[bool, List[Dict]]:
        """
        Проверяет столкновение хвата с препятствиями.
        
        Args:
            gripper_info: Информация о хвате
            obstacles: Список объектов (проверяем is_obstacle=True)
            proximity_margin: Небольшой отступ для погрешности (пиксели)
            
        Returns:
            (есть_столкновение, список_препятствий_в_зоне)
        """
        if gripper_info is None or not obstacles:
            return False, []
        
        gx, gy, gw, gh = gripper_info['bbox']
        gripper_depth = gripper_info.get('depth_m')
        
        # Небольшое расширение зоны хвата для учёта погрешности
        gx_ext = gx - proximity_margin
        gy_ext = gy - proximity_margin
        gw_ext = gw + 2 * proximity_margin
        gh_ext = gh + 2 * proximity_margin
        
        colliding = []
        
        for obs in obstacles:
            # Проверяем только препятствия
            if not obs.get('is_obstacle', False):
                continue
            
            obx, oby, obw, obh = obs['bbox']
            obs_depth = obs.get('depth_m')
            
            # Проверяем РЕАЛЬНОЕ пересечение bbox
            # Столкновение только если bbox действительно перекрываются
            bbox_overlap = (gx_ext < obx + obw and gx_ext + gw_ext > obx and
                           gy_ext < oby + obh and gy_ext + gh_ext > oby)
            
            if not bbox_overlap:
                continue
            
            # Проверяем по глубине - столкновение если объект на опасной высоте
            if gripper_depth is not None and obs_depth is not None:
                # Объект выше хвата (ближе к камере) - столкновение
                if obs_depth <= gripper_depth:
                    colliding.append(obs)
                # Объект примерно на той же высоте (±8см) - столкновение
                elif abs(obs_depth - gripper_depth) <= 0.08:
                    colliding.append(obs)
            else:
                # Если нет данных о глубине - считаем столкновением для безопасности
                colliding.append(obs)
        
        return len(colliding) > 0, colliding
    
    def draw_gripper(
        self,
        image: np.ndarray,
        gripper_info: Optional[Dict],
        collision: bool = False,
        thickness: int = 3
    ) -> np.ndarray:
        """Отрисовывает зону хвата на изображении."""
        if gripper_info is None:
            return image
        
        overlay = image.copy()
        
        x, y, w, h = gripper_info['bbox']
        img_h, img_w = image.shape[:2]
        
        # Цвет зависит от столкновения
        if collision:
            color = (0, 0, 255)  # КРАСНЫЙ - столкновение!
            corner_color = (0, 0, 255)
            fill_alpha = 0.3
        else:
            color = (0, 255, 0)  # Зелёный - безопасно
            corner_color = (0, 255, 255)  # Жёлтый
            fill_alpha = 0.15
        
        # Заливка зоны хвата с прозрачностью
        overlay_fill = image.copy()
        cv2.rectangle(overlay_fill, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay_fill, fill_alpha, overlay, 1 - fill_alpha, 0, overlay)
        
        # Рисуем основную рамку зоны хвата
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        
        # Угловые маркеры для выделения зоны
        corner_len = 20
        ct = thickness + 1
        # Верхний левый
        cv2.line(overlay, (x, y), (x + corner_len, y), corner_color, ct)
        cv2.line(overlay, (x, y), (x, y + corner_len), corner_color, ct)
        # Верхний правый
        cv2.line(overlay, (x + w, y), (x + w - corner_len, y), corner_color, ct)
        cv2.line(overlay, (x + w, y), (x + w, y + corner_len), corner_color, ct)
        # Нижний левый
        cv2.line(overlay, (x, y + h), (x + corner_len, y + h), corner_color, ct)
        cv2.line(overlay, (x, y + h), (x, y + h - corner_len), corner_color, ct)
        # Нижний правый
        cv2.line(overlay, (x + w, y + h), (x + w - corner_len, y + h), corner_color, ct)
        cv2.line(overlay, (x + w, y + h), (x + w, y + h - corner_len), corner_color, ct)
        
        # Рисуем контуры отдельных частей хвата (тёмные части)
        if 'parts' in gripper_info:
            for part in gripper_info['parts']:
                cv2.drawContours(overlay, [part['contour']], -1, (255, 0, 255), 2)
        
        # Центр зоны хвата
        cx, cy = gripper_info['center']
        center_color = (0, 0, 255) if collision else (0, 0, 255)
        cv2.circle(overlay, (cx, cy), 8, center_color, -1)
        cv2.circle(overlay, (cx, cy), 10, (255, 255, 255), 2)
        
        # Перекрестие в центре
        cv2.line(overlay, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 2)
        cv2.line(overlay, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 2)
        
        # Заголовок на русском
        if collision:
            label = "СТОЛКНОВЕНИЕ!"
        else:
            label = "ЗОНА ХВАТА"
        
        # Добавляем высоту к заголовку
        if gripper_info['depth_m'] is not None:
            label += f"={gripper_info['depth_m']:.2f}"
        
        # Позиция заголовка
        label_x = x
        label_y = y - 5 if y > 25 else y + h + 5
        
        # Рисуем текст на русском
        overlay = put_russian_text(overlay, label, (label_x, label_y - 15), 
                                  font_size=14, color=(255, 255, 255), bg_color=color, padding=3)
        
        # 3D координаты
        if gripper_info['position_3d'] is not None:
            x_3d, y_3d, z_3d = gripper_info['position_3d']
            coord_text = f"X:{x_3d:.2f} Y:{y_3d:.2f}"
            overlay = put_russian_text(overlay, coord_text, (label_x, label_y + 5), 
                                      font_size=10, color=(200, 200, 200))
        
        # === ПРЕДУПРЕЖДЕНИЕ О СТОЛКНОВЕНИИ ===
        if collision:
            # Предупреждение на русском в центре экрана
            warning_text = "ВНИМАНИЕ: СТОЛКНОВЕНИЕ!"
            text_w, text_h = get_text_size(warning_text, 20)
            
            # Позиция по центру сверху
            wx = (img_w - text_w) // 2
            wy = 10
            
            # Рисуем предупреждение
            overlay = put_russian_text(overlay, warning_text, (wx, wy), 
                                      font_size=20, color=(255, 255, 255), 
                                      bg_color=(0, 0, 200), padding=8)
        
        return overlay
    
    def log_gripper_info(self, gripper_info: Optional[Dict]):
        """Выводит информацию о хвате в лог."""
        if gripper_info is None:
            return
        
        cx, cy = gripper_info['center']
        x, y, w, h = gripper_info['bbox']
        depth_m = gripper_info.get('depth_m')
        position_3d = gripper_info.get('position_3d')
        num_parts = gripper_info.get('gripper_zone', {}).get('num_parts', 1)
        
        log_msg = f"GRIPPER ZONE: center=({cx}, {cy}), size={w}x{h}, parts={num_parts}"
        
        if depth_m is not None:
            log_msg += f", height={depth_m:.3f}m"
        
        if position_3d is not None:
            x_3d, y_3d, z_3d = position_3d
            log_msg += f", 3D=({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f})"
        
        logger.info(log_msg)
