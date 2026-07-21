"""
Модуль для обнаружения препятствий и объектов на рабочей области.
Определяет высоту объектов и классифицирует их как препятствия или безопасные.
"""
import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from src.utils.text_renderer import put_russian_text, get_text_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObstacleDetector:
    """Детектор препятствий и объектов на рабочей области."""
    
    def __init__(self):
        """Инициализация детектора препятствий."""
        # Порог высоты для препятствия (в метрах)
        # Объекты ближе к камере (меньше depth) = выше = препятствие
        self.obstacle_threshold = 0.60  # 0.6м - критическая высота
        
        # Допуск по высоте - объекты в этом диапазоне от хвата считаются опасными
        # Если объект в пределах ±tolerance от высоты хвата - это препятствие
        self.height_tolerance = 0.08  # 8см допуск для безопасности
        
        # Минимальная площадь объекта (пиксели)
        self.min_object_area = 800
        self.max_object_area = 15000  # Уменьшили чтобы не ловить хват
        
        # Фильтр формы - исключаем тонкие длинные объекты (провода)
        self.min_aspect_ratio = 0.2   # Минимальное соотношение сторон (ширина/высота)
        self.max_aspect_ratio = 5.0   # Максимальное соотношение сторон
        self.min_solidity = 0.3       # Минимальная "плотность" объекта (площадь/выпуклая оболочка)
        
        # Параметры поиска объектов
        self.search_region_ratio = 0.95
        
        # Сглаживание
        self.smooth_alpha = 0.3
        self.last_objects: List[Dict] = []
    
    def detect_objects(
        self,
        color_image: np.ndarray,
        depth_frame,
        depth_scale: float,
        gripper_height: Optional[float] = None,
        sheet_mask: Optional[np.ndarray] = None,
        gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
        reject_log: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Обнаруживает объекты на рабочей области и классифицирует их.

        Args:
            color_image: Цветное изображение BGR
            depth_frame: Кадр глубины
            depth_scale: Масштаб глубины
            gripper_height: Высота хвата (для сравнения)
            sheet_mask: Маска рабочей области (белый лист)
            gripper_bbox: Bounding box хвата (x, y, w, h) для исключения из поиска
            reject_log: (Задача 8) список для записей об отбракованных
                кандидатах с причиной; на логику детекции не влияет

        Returns:
            Список объектов с информацией о них
        """
        if depth_frame is None:
            return []

        height, width = color_image.shape[:2]
        depth_image = np.asanyarray(depth_frame.get_data())

        # Определяем рабочую область
        if sheet_mask is None:
            sheet_mask = self._detect_work_area(color_image)

        if sheet_mask is None:
            if reject_log is not None:
                reject_log.append({"reject_reason": "no_work_area"})
            return []

        # Находим объекты на рабочей области
        objects = self._find_objects_on_sheet(
            color_image, depth_image, depth_scale, sheet_mask, gripper_height,
            gripper_bbox, reject_log
        )

        return objects
    
    def _detect_work_area(self, color_image: np.ndarray) -> Optional[np.ndarray]:
        """Определяет рабочую область (белый лист)."""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Белый цвет
        white_lower = np.array([0, 0, 150], dtype=np.uint8)
        white_upper = np.array([180, 60, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Находим самый большой контур
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10000:
            return None
        
        mask = np.zeros_like(white_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
    
    def _find_objects_on_sheet(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_scale: float,
        sheet_mask: np.ndarray,
        gripper_height: Optional[float],
        gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
        reject_log: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Находит объекты на рабочей области."""
        height, width = color_image.shape[:2]
        objects = []

        def _reject(reason: str, area_px) -> None:
            # Сбор причин отбраковки (Задача 8); включается только когда
            # передан reject_log — иначе ноль накладных расходов.
            if reject_log is not None:
                reject_log.append({"reject_reason": reason,
                                   "area_px": int(area_px)})
        
        # Конвертируем в HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Находим НЕ белые объекты на белом листе
        # Объекты - это всё что не белое и находится на листе
        white_lower = np.array([0, 0, 160], dtype=np.uint8)
        white_upper = np.array([180, 50, 255], dtype=np.uint8)
        white_pixels = cv2.inRange(hsv, white_lower, white_upper)
        
        # Инвертируем - получаем не-белые пиксели
        non_white = cv2.bitwise_not(white_pixels)
        
        # Применяем маску листа - объекты только на листе
        objects_mask = cv2.bitwise_and(non_white, sheet_mask)
        
        # НЕ исключаем зону хвата из маски - проверим центр объекта позже
        # Это позволит детектировать объекты рядом с хватом
        
        # === ИСКЛЮЧАЕМ СИНИЕ ОБЪЕКТЫ (провода манипулятора) ===
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Синий цвет в HSV: H=100-130, S>50
        blue_mask = ((h_channel >= 100) & (h_channel <= 130) & (s_channel > 50))
        not_blue = ~blue_mask
        objects_mask = cv2.bitwise_and(objects_mask, not_blue.astype(np.uint8) * 255)
        
        # Убираем слишком тёмные области (тени, хват манипулятора)
        # Хват манипулятора - очень тёмный (V < 80)
        bright_enough = v_channel > 80  # Исключаем тёмный хват
        objects_mask = cv2.bitwise_and(objects_mask, bright_enough.astype(np.uint8) * 255)
        
        # Убираем слишком светлые области (блики)
        not_too_bright = v_channel < 250
        objects_mask = cv2.bitwise_and(objects_mask, not_too_bright.astype(np.uint8) * 255)
        
        # Объекты должны иметь некоторую насыщенность (цветные кубики)
        has_color = s_channel > 30
        objects_mask = cv2.bitwise_and(objects_mask, has_color.astype(np.uint8) * 255)
        
        # Морфология для очистки
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel_small)
        objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel)
        
        # Находим контуры объектов
        contours, _ = cv2.findContours(objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_object_area or area > self.max_object_area:
                _reject("area_min" if area < self.min_object_area else "area_max",
                        area)
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # === ФИЛЬТР ФОРМЫ - исключаем провода и тонкие объекты ===
            # Соотношение сторон
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                _reject("shape_aspect", area)
                continue  # Слишком тонкий/длинный объект (провод)

            # Плотность объекта (solidity) - отношение площади к выпуклой оболочке
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                _reject("shape_solidity", area)
                continue  # Объект слишком "рваный" (провод изгибается)

            # Компактность - провода имеют большой периметр относительно площади
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.1:
                _reject("shape_circularity", area)
                continue  # Слишком вытянутый объект
            
            # Центр объекта
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Проверяем что центр объекта НЕ глубоко внутри зоны хвата
            # Это исключает детекцию самого хвата как объекта, но не объекты рядом
            if gripper_bbox is not None:
                gx, gy, gw, gh = gripper_bbox
                # Уменьшаем зону проверки - только центральная часть хвата
                margin = 30  # Отступ от краёв
                inner_x1 = gx + margin
                inner_y1 = gy + margin
                inner_x2 = gx + gw - margin
                inner_y2 = gy + gh - margin
                
                # Если центр объекта глубоко внутри хвата - пропускаем
                if inner_x1 < cx < inner_x2 and inner_y1 < cy < inner_y2:
                    _reject("inside_gripper", area)
                    continue

            # Вычисляем глубину (высоту) объекта
            obj_depth = self._calculate_object_depth(
                depth_image, depth_scale, contour, x, y, w, h
            )

            if obj_depth is None:
                _reject("no_depth", area)
                continue
            
            # Находим самую высокую точку (минимальная глубина)
            highest_point = self._find_highest_point(
                depth_image, depth_scale, contour, x, y, w, h
            )
            
            # Классификация: препятствие или безопасный объект
            # Препятствие если:
            # 1. Объект ближе к камере (меньше depth) чем хват
            # 2. ИЛИ объект на той же высоте что и хват (в пределах tolerance)
            is_obstacle = False
            if gripper_height is not None:
                # Объект ближе к камере = выше = препятствие
                # ИЛИ объект примерно на той же высоте (опасная зона)
                height_diff = abs(obj_depth - gripper_height)
                is_obstacle = (obj_depth <= gripper_height) or (height_diff <= self.height_tolerance)
            else:
                # Используем фиксированный порог
                is_obstacle = obj_depth <= self.obstacle_threshold
            
            objects.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area,
                'depth_m': obj_depth,
                'highest_point': highest_point,
                'is_obstacle': is_obstacle
            })
        
        # Сортируем по глубине (ближайшие первые = самые высокие)
        objects.sort(key=lambda o: o['depth_m'])
        
        return objects
    
    def _calculate_object_depth(
        self,
        depth_image: np.ndarray,
        depth_scale: float,
        contour: np.ndarray,
        x: int, y: int, w: int, h: int
    ) -> Optional[float]:
        """Вычисляет среднюю глубину объекта."""
        # Создаём маску для объекта
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Берём глубину только внутри контура
        object_depths = depth_image[mask == 255]
        valid_depths = object_depths[object_depths > 0]
        
        if len(valid_depths) < 10:
            return None
        
        # Медиана для устойчивости к выбросам
        depth_mm = np.median(valid_depths)
        depth_m = depth_mm * depth_scale
        
        return depth_m
    
    def _find_highest_point(
        self,
        depth_image: np.ndarray,
        depth_scale: float,
        contour: np.ndarray,
        x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, int, float]]:
        """Находит самую высокую точку объекта (минимальная глубина)."""
        # Создаём маску для объекта
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Применяем маску
        masked_depth = depth_image.copy()
        masked_depth[mask == 0] = 0
        
        # Находим минимальную глубину (самая высокая точка)
        valid_mask = (masked_depth > 0) & (mask == 255)
        
        if not np.any(valid_mask):
            return None
        
        # Находим координаты минимальной глубины
        valid_depths = masked_depth[valid_mask]
        min_depth = np.min(valid_depths)
        
        # Находим пиксель с минимальной глубиной
        min_positions = np.where((masked_depth == min_depth) & valid_mask)
        
        if len(min_positions[0]) == 0:
            return None
        
        # Берём первую найденную точку
        py = int(min_positions[0][0])
        px = int(min_positions[1][0])
        height_m = min_depth * depth_scale
        
        return (px, py, height_m)
    
    def draw_objects(
        self,
        image: np.ndarray,
        objects: List[Dict],
        gripper_height: Optional[float] = None
    ) -> np.ndarray:
        """
        Отрисовывает объекты на изображении.
        
        Препятствия (выше хвата) - КРАСНЫЙ (полное выделение)
        Безопасные объекты - только точка центра и высота
        """
        overlay = image.copy()
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            cx, cy = obj['center']
            depth_m = obj['depth_m']
            is_obstacle = obj['is_obstacle']
            highest_point = obj.get('highest_point')
            
            if is_obstacle:
                # === ПРЕПЯТСТВИЕ - полное выделение ===
                color = (0, 0, 255)  # Красный
                
                # Рисуем контур объекта
                cv2.drawContours(overlay, [obj['contour']], -1, color, 2)
                
                # Рисуем bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Заливка с прозрачностью
                overlay_fill = overlay.copy()
                cv2.drawContours(overlay_fill, [obj['contour']], -1, color, -1)
                cv2.addWeighted(overlay_fill, 0.2, overlay, 0.8, 0, overlay)
                
                # Центр объекта
                cv2.circle(overlay, (cx, cy), 6, color, -1)
                cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 2)
                
                # Самая высокая точка
                if highest_point is not None:
                    hx, hy, h_depth = highest_point
                    cv2.drawMarker(overlay, (hx, hy), (0, 255, 255), 
                                   cv2.MARKER_TRIANGLE_UP, 15, 3)
                    cv2.circle(overlay, (hx, hy), 5, (0, 255, 255), -1)
                
                # Текст на русском: "ОПАСНО={высота}"
                info_y = y - 5 if y > 25 else y + h + 18
                label_text = f"ОПАСНО={depth_m:.2f}"
                overlay = put_russian_text(overlay, label_text, (x, info_y - 15), 
                                          font_size=11, color=(255, 255, 255), bg_color=color, padding=2)
            else:
                # === БЕЗОПАСНЫЙ ОБЪЕКТ - только точка и высота ===
                color = (255, 150, 0)  # Синий
                
                # Только точка центра
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), 1)
                
                # Только высота цифрами
                depth_text = f"{depth_m:.2f}"
                cv2.putText(overlay, depth_text, (cx + 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Статистика на русском (только если есть препятствия)
        num_obstacles = sum(1 for o in objects if o['is_obstacle'])
        
        if num_obstacles > 0:
            overlay = put_russian_text(overlay, f"Опасно: {num_obstacles}", 
                                      (image.shape[1] - 100, 15), font_size=14, 
                                      color=(0, 0, 255))
        
        return overlay
    
    def get_obstacles(self, objects: List[Dict]) -> List[Dict]:
        """Возвращает только препятствия."""
        return [o for o in objects if o['is_obstacle']]
    
    def get_safe_objects(self, objects: List[Dict]) -> List[Dict]:
        """Возвращает только безопасные объекты."""
        return [o for o in objects if not o['is_obstacle']]
    
    def get_highest_obstacle(self, objects: List[Dict]) -> Optional[Dict]:
        """Возвращает самое высокое препятствие."""
        obstacles = self.get_obstacles(objects)
        if not obstacles:
            return None
        # Минимальная глубина = самое высокое
        return min(obstacles, key=lambda o: o['depth_m'])
    
    def log_objects(self, objects: List[Dict], gripper_height: Optional[float] = None):
        """Логирует информацию об объектах."""
        if not objects:
            return
        
        obstacles = self.get_obstacles(objects)
        safe = self.get_safe_objects(objects)
        
        log_msg = f"OBJECTS: total={len(objects)}, obstacles={len(obstacles)}, safe={len(safe)}"
        
        if gripper_height:
            log_msg += f", gripper_h={gripper_height:.3f}m"
        
        logger.info(log_msg)
        
        # Логируем препятствия
        for i, obs in enumerate(obstacles):
            cx, cy = obs['center']
            depth = obs['depth_m']
            logger.warning(f"  OBSTACLE {i+1}: pos=({cx},{cy}), height={depth:.3f}m")

