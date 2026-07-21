"""
SystemController - Главный контроллер системы.

Связывает компоненты камеры и робота, управляет потоком данных
и координирует работу всей системы.

ГИБРИДНЫЙ ПОДХОД:
- 2D детекция (OpenCV) для быстрого поиска хвата и объектов
- Облако точек (Open3D) для 3D визуализации
- Комбинированная проверка столкновений (2D bbox + глубина)
"""
import time

import numpy as np
from typing import Optional, Callable

from src.features.camera.services.camera_service import CameraService
from src.features.camera.processors.depth_processor import DepthProcessor
from src.features.camera.processors.gripper_detector import GripperDetector
from src.features.camera.processors.object_tracker import ObjectTracker
from src.features.camera.processors.obstacle_detector import ObstacleDetector
from src.features.camera.ui.camera_window import CameraWindow
from src.features.camera.ui.point_cloud_window import PointCloudWindow
from src.features.camera.ui.monitor_window import MonitorWindow
from src.features.robot.services.robot_service import RobotService
from src.utils.system_monitor import sample_system_stats, FPSCounter
from src.utils.perf_logger import (
    PerfLogger2D, gripper_object_distance_m, min_gripper_object_distance_m
)
from src.utils.stop_event_logger import StopEventTracker
from src.constants.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    STABLE_THRESHOLD, TRACK_PERSIST, GRIPPER_LOG_INTERVAL,
    ENABLE_DECISION_LOG, DECISION_PERF_LOG_PATH, OBJECTS_CSV_PATH, SCENARIO_ID,
    ENABLE_STOP_LOG, STOP_EVENTS_CSV_PATH, STOP_SPEED_EPS_DEG_S,
    STOP_CONFIRM_FRAMES_STOPPED,
    ENABLE_RECORDING, RECORDING_PATH, RECORDING_JOINTS_EVERY_N,
    SOURCE_MODE, PLAYBACK_PATH, MAX_RUN_SECONDS,
    DUMP_MASKS_EVERY_N, MASKS_DUMP_DIR,
)


class SystemController:
    """
    Главный контроллер системы.
    
    Координирует работу камеры, обработки изображений и робота.
    Обеспечивает связь между компонентами и управляет жизненным циклом.

    Детекция хвата, препятствий и коллизий — по RGB и карте глубины RealSense.
    """
    
    def __init__(
        self,
        enable_robot: bool = True,
        enable_point_cloud: bool = False,
        robot_ip: Optional[str] = None,
        robot_port: Optional[int] = None
    ):
        """
        Инициализация контроллера.
        
        Args:
            enable_robot: Включить подключение к роботу
            enable_point_cloud: Окно Open3D (только визуализация, не детекция)
            robot_ip: IP-адрес робота (если отличается от конфига)
            robot_port: Порт робота (если отличается от конфига)
        """
        # Playback (Задача 3): робот принудительно отключён (безопасность)
        self._playback = SOURCE_MODE == 'playback'
        if self._playback and enable_robot:
            print("[PLAYBACK] enable_robot игнорируется — в режиме "
                  "воспроизведения робот не управляется")
            enable_robot = False
        self.enable_robot = enable_robot
        self.enable_point_cloud = enable_point_cloud

        # Рекордер (Задача 3); None при выключенном флаге или в playback
        self.recorder = None
        self._rec_last_joints = None
        if ENABLE_RECORDING and not self._playback:
            from src.utils.frame_recorder import FrameRecorder
            self.recorder = FrameRecorder(
                RECORDING_PATH, contour="2D", scenario_id=SCENARIO_ID
            )
        
        # Компоненты камеры
        self.camera_service: Optional[CameraService] = None
        self.depth_processor: Optional[DepthProcessor] = None
        self.gripper_detector: Optional[GripperDetector] = None
        self.obstacle_detector: Optional[ObstacleDetector] = None
        self.object_tracker: Optional[ObjectTracker] = None
        self.camera_window: Optional[CameraWindow] = None
        self.point_cloud_window: Optional[PointCloudWindow] = None
        self.monitor_window: Optional[MonitorWindow] = None
        
        # Компоненты робота
        self.robot_service: Optional[RobotService] = None
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        
        # Мониторинг
        self.fps_counter = FPSCounter()
        
        # Состояние
        self._running = False
        self._gripper_log_counter = 0
        self._last_collision_state = False  # Для отслеживания изменения состояния
        self._frame_idx = 0  # сквозной номер кадра (для логов валидации)

        # Покадровый лог решений (Задача 1); None при выключенном флаге
        self.perf_logger: Optional[PerfLogger2D] = None
        if ENABLE_DECISION_LOG:
            self.perf_logger = PerfLogger2D(
                DECISION_PERF_LOG_PATH, OBJECTS_CSV_PATH, scenario_id=SCENARIO_ID
            )

        # Лог событий останова (Задача 2); None при выключенном флаге
        self.stop_tracker: Optional[StopEventTracker] = None
        if ENABLE_STOP_LOG:
            self.stop_tracker = StopEventTracker(
                STOP_EVENTS_CSV_PATH, contour="2D", scenario_id=SCENARIO_ID,
                speed_eps_deg_s=STOP_SPEED_EPS_DEG_S,
                confirm_frames=STOP_CONFIRM_FRAMES_STOPPED,
            )
        # История 3D-позиции хвата — для оценки скорости звена в момент DANGER
        self._prev_gripper_3d = None
        self._prev_gripper_t = None
        
        # Callbacks
        self._on_gripper_detected: Optional[Callable] = None
        self._on_object_detected: Optional[Callable] = None
        self._on_cycle_position: Optional[Callable] = None
    
    def initialize(self) -> bool:
        """
        Инициализирует все компоненты системы.
        
        Returns:
            True если инициализация успешна
        """
        print("\n" + "=" * 60)
        print("  CLOUD POINT SYSTEM - Initializing...")
        print("=" * 60)
        
        # Инициализация камеры (live) или источника воспроизведения (Задача 3)
        if self._playback:
            from src.features.camera.services.playback_camera_service import (
                PlaybackCameraService
            )
            self.camera_service = PlaybackCameraService(PLAYBACK_PATH)
        else:
            self.camera_service = CameraService(
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS
            )

        if not self.camera_service.start():
            print("Failed to start camera!")
            return False
        
        # Инициализация процессоров
        self.depth_processor = DepthProcessor(
            depth_scale=self.camera_service.get_depth_scale()
        )
        self.gripper_detector = GripperDetector()
        self.obstacle_detector = ObstacleDetector()
        self.object_tracker = ObjectTracker()
        
        print("  [+] Detection: gripper / obstacles / collision (color + depth)")
        if self.enable_point_cloud:
            print("  [+] 3D point cloud window enabled")
        
        # Инициализация UI
        self.camera_window = CameraWindow(
            title='Camera View',
            on_click_callback=self._on_click if self.enable_robot else None
        )
        
        # Окно мониторинга системы (CPU, RAM, FPS)
        self.monitor_window = MonitorWindow(title='System Monitor')
        
        if self.enable_point_cloud:
            self.point_cloud_window = PointCloudWindow(title='Point Cloud')
        
        # Инициализация робота
        if self.enable_robot:
            self.robot_service = RobotService(
                ip=self.robot_ip if self.robot_ip else None,
                port=self.robot_port if self.robot_port else None,
                auto_connect=True
            )
            
            # Перемещаем робота в рабочую позицию при запуске
            if self.robot_service.is_connected:
                print("\n" + "-" * 40)
                print("  Перемещение в рабочую позицию...")
                print("-" * 40)
                if self.robot_service.move_to_home():
                    print("  Робот в рабочей позиции!")
                    print("-" * 40 + "\n")
                    
                    # Запускаем цикл движения влево-вправо
                    self.robot_service.start_cycle_movement(
                        on_position_reached=self._on_cycle_position_reached
                    )
                else:
                    print("  Не удалось переместить робота")
                    print("-" * 40 + "\n")
        
        print("\n" + "=" * 60)
        print("  System initialized successfully!")
        print("  Press 'q' to exit")
        print("=" * 60 + "\n")
        
        return True
    
    def run(self):
        """Запускает основной цикл обработки."""
        if not self.camera_service or not self.camera_service.is_running:
            print("Camera not initialized!")
            return
        
        self._running = True
        _run_start = time.perf_counter()

        try:
            while self._running:
                # Автозавершение по таймеру (сценарный раннер, Задача 4)
                if (MAX_RUN_SECONDS > 0
                        and time.perf_counter() - _run_start >= MAX_RUN_SECONDS):
                    print(f"[INFO] MAX_RUN_SECONDS={MAX_RUN_SECONDS:.0f}s — стоп.")
                    break
                if not self._process_frame():
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.shutdown()
    
    def _process_frame(self) -> bool:
        """
        Обрабатывает один кадр.
        
        Чистая 2D детекция (OpenCV) - быстро и стабильно.
        
        Returns:
            True для продолжения, False для выхода
        """
        # Получаем кадры (t_capture — точка отсчёта latency, Задача 1)
        t_capture = time.perf_counter()
        depth_frame, color_frame, _ = self.camera_service.get_frames()

        if depth_frame is None or color_frame is None:
            # Конец записи в playback → завершаем цикл
            if getattr(self.camera_service, 'finished', False):
                print("[PLAYBACK] Конец записи.")
                return False
            return True

        self._frame_idx += 1

        # Получаем данные
        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = self.camera_service.get_intrinsics(color_frame)
        depth_scale = self.camera_service.get_depth_scale()

        # ── запись RGB-D + углов (Задача 3), сырые кадры до фильтров ──
        if self.recorder is not None:
            if (self.robot_service and self.robot_service.is_connected
                    and self._frame_idx % RECORDING_JOINTS_EVERY_N == 1):
                angles = self.robot_service.get_joint_angles()
                if angles is not None:
                    self._rec_last_joints = angles
            self.recorder.record(
                self._frame_idx, color_image,
                np.asanyarray(depth_frame.get_data()),
                {"fx": intrinsics.fx, "fy": intrinsics.fy,
                 "cx": intrinsics.ppx, "cy": intrinsics.ppy},
                depth_scale, self._rec_last_joints, t_capture,
            )

        # Применяем фильтры глубины (только live: rs-фильтры требуют rs.frame)
        if not self._playback:
            depth_frame = self.depth_processor.apply_depth_filters(depth_frame)

        (collision, detected_objects, gripper_info, color_image,
         colliding, t_decision) = self._process_frame_2d(
            color_image=color_image,
            depth_frame=depth_frame,
            depth_scale=depth_scale,
            intrinsics=intrinsics
        )
        
        # === УПРАВЛЕНИЕ РОБОТОМ ПРИ СТОЛКНОВЕНИИ ===
        if self.robot_service and self.robot_service.is_connected:
            if collision and not self._last_collision_state:
                self.robot_service.stop_movement()
                # ── событие останова (Задача 2): t_danger + скорость хвата ──
                if self.stop_tracker is not None:
                    t_danger = time.perf_counter()
                    link_speed = self._estimate_gripper_speed(
                        gripper_info, t_danger
                    )
                    self.stop_tracker.mark_danger(
                        self._frame_idx, t_capture, t_danger,
                        min_gripper_object_distance_m(
                            gripper_info, colliding, intrinsics
                        ),
                        link_speed,
                    )
            elif not collision and self._last_collision_state:
                self.robot_service.resume_movement()
                if self.stop_tracker is not None:
                    self.stop_tracker.mark_resumed()
            self._last_collision_state = collision

        # ── детект физической остановки по $AXIS_ACT (Задача 2) ──
        # Углы читаются ТОЛЬКО пока ждём остановку — чтение блокирующее,
        # в обычном цикле оно не выполняется и FPS не страдает.
        if self.stop_tracker is not None:
            angles = None
            if (self.stop_tracker.waiting_stop and self.robot_service
                    and self.robot_service.is_connected):
                angles = self.robot_service.get_joint_angles()
            self.stop_tracker.update(
                self._frame_idx, time.perf_counter(), angles
            )
        # История позиции хвата для оценки скорости в момент DANGER
        if gripper_info is not None and gripper_info.get('position_3d') is not None:
            self._prev_gripper_3d = gripper_info['position_3d']
            self._prev_gripper_t = time.perf_counter()
        
        # Счётчики
        current_count = len(detected_objects)
        stable_count = current_count
        
        # Обновляем FPS
        fps = self.fps_counter.update() or self.fps_counter.get_fps()
        
        # Получаем системную статистику
        cpu, memory, threads = sample_system_stats()

        # Обновляем окно мониторинга
        if self.monitor_window:
            self.monitor_window.update(cpu, memory, fps, threads)

        # ── дамп масок для IoU (Задача 6) ──
        if DUMP_MASKS_EVERY_N > 0 and self._frame_idx % DUMP_MASKS_EVERY_N == 0:
            self._dump_masks(color_image.shape[:2], detected_objects, gripper_info)

        # ── Покадровый лог решений (Задача 1) ──
        if self.perf_logger is not None:
            robot_paused = bool(
                self.robot_service.is_paused
                if (self.robot_service and self.robot_service.is_connected)
                else False
            )
            min_dist = min_gripper_object_distance_m(
                gripper_info, detected_objects, intrinsics
            )
            self.perf_logger.sample(
                frame=self._frame_idx,
                fps=fps,
                cpu=cpu,
                ram_mb=memory,
                threads=threads,
                collision_level="DANGER" if collision else "SAFE",
                n_objects=len(detected_objects),
                min_dist_m=min_dist,
                latency_ms=(t_decision - t_capture) * 1000.0,
                robot_paused=robot_paused,
            )
            colliding_ids = {id(o) for o in colliding}
            obj_rows = []
            for i, obj in enumerate(detected_objects):
                if id(obj) in colliding_ids:
                    level = "DANGER"
                elif obj.get('is_obstacle', False):
                    level = "WARN"
                else:
                    level = "SAFE"
                obj_rows.append({
                    "obj_id": i,
                    "part": "gripper",
                    "dist_m": gripper_object_distance_m(
                        gripper_info, obj, intrinsics
                    ),
                    "level": level,
                })
            self.perf_logger.log_objects(self._frame_idx, obj_rows)
        
        # Опциональное окно облака точек (только визуализация, не детекция)
        if self.enable_point_cloud and self.point_cloud_window and self.depth_processor:
            try:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image_raw = np.asanyarray(color_frame.get_data())
                pcd = self.depth_processor.create_point_cloud(
                    depth_image=depth_image,
                    color_image=color_image_raw,
                    intrinsics=intrinsics
                )
                self.point_cloud_window.update_point_cloud(pcd)
            except Exception:
                pass
        
        # Отображаем в окне камеры
        continues = self.camera_window.display(
            color_image,
            depth_frame=depth_frame,
            depth_scale=depth_scale,
            stable_count=stable_count,
            current_count=current_count,
            stable_frames=self.object_tracker.stable_frames,
            stable_threshold=STABLE_THRESHOLD,
            time_to_change=self.object_tracker.get_time_to_change(),
            cpu_percent=cpu,
            memory_percent=memory,
            thread_count=threads,
            fps=fps,
            tracks=self.object_tracker.tracks,
            detections=[],
            intrinsics=intrinsics,
            track_persist=TRACK_PERSIST
        )
        
        return continues

    def _process_frame_2d(self, color_image, depth_frame, depth_scale, intrinsics):
        """Детекция хвата, препятствий, коллизий и отрисовка (RGB + depth)."""
        # Обработка рабочей области (белый лист)
        sheet_mask = self.depth_processor.get_effective_mask(color_image)
        
        # === 2D ДЕТЕКЦИЯ ХВАТА ===
        gripper_info = self.gripper_detector.detect(
            color_image,
            depth_frame=depth_frame,
            depth_scale=depth_scale,
            intrinsics=intrinsics
        )
        
        # Логируем хват
        self._gripper_log_counter += 1
        if self._gripper_log_counter >= GRIPPER_LOG_INTERVAL:
            if gripper_info is not None:
                self.gripper_detector.log_gripper_info(gripper_info)
                if self._on_gripper_detected:
                    self._on_gripper_detected(gripper_info)
            self._gripper_log_counter = 0
        
        # Получаем высоту и bbox хвата
        gripper_height = None
        gripper_bbox = None
        if gripper_info is not None:
            gripper_height = gripper_info.get('depth_m')
            gripper_bbox = gripper_info.get('bbox')
        
        # === 2D ДЕТЕКЦИЯ ОБЪЕКТОВ ===
        detected_objects = self.obstacle_detector.detect_objects(
            color_image,
            depth_frame=depth_frame,
            depth_scale=depth_scale,
            gripper_height=gripper_height,
            sheet_mask=sheet_mask,
            gripper_bbox=gripper_bbox
        )
        
        # === ПРОВЕРКА СТОЛКНОВЕНИЙ ===
        collision, colliding = self.gripper_detector.check_collision(
            gripper_info, detected_objects
        )
        # Момент принятия решения о коллизии — для latency_ms (Задача 1)
        t_decision = time.perf_counter()

        # === ОТРИСОВКА ===
        color_image = self.obstacle_detector.draw_objects(
            color_image, detected_objects, gripper_height
        )
        color_image = self.gripper_detector.draw_gripper(
            color_image, gripper_info, collision=collision
        )

        return collision, detected_objects, gripper_info, color_image, colliding, t_decision
    
    def _dump_masks(self, shape, detected_objects, gripper_info):
        """Бинарные маски препятствий и хвата для IoU (Задача 6).

        Имена совместимы с gt_masks инструмента разметки:
        <MASKS_DUMP_DIR>/2d/frame_%06d_{obstacle,manip}.png.
        """
        import cv2
        from pathlib import Path
        mask_dir = Path(MASKS_DUMP_DIR) / "2d"
        mask_dir.mkdir(parents=True, exist_ok=True)
        obstacle = np.zeros(shape, dtype=np.uint8)
        for obj in detected_objects:
            if obj.get('contour') is not None:
                cv2.drawContours(obstacle, [obj['contour']], -1, 255, -1)
        manip = np.zeros(shape, dtype=np.uint8)
        if gripper_info is not None:
            for part in gripper_info.get('parts', []):
                if part.get('contour') is not None:
                    cv2.drawContours(manip, [part['contour']], -1, 255, -1)
            if not gripper_info.get('parts') and gripper_info.get('bbox'):
                x, y, w, h = gripper_info['bbox']
                cv2.rectangle(manip, (x, y), (x + w, y + h), 255, -1)
        cv2.imwrite(str(mask_dir / f"frame_{self._frame_idx:06d}_obstacle.png"),
                    obstacle)
        cv2.imwrite(str(mask_dir / f"frame_{self._frame_idx:06d}_manip.png"),
                    manip)

    def _estimate_gripper_speed(self, gripper_info, t_now) -> Optional[float]:
        """Модуль линейной скорости хвата (м/с) по разнице 3D-позиций между
        кадрами — для оценки минимальной безопасной дистанции (Задача 2)."""
        if (gripper_info is None or gripper_info.get('position_3d') is None
                or self._prev_gripper_3d is None or self._prev_gripper_t is None):
            return None
        dt = t_now - self._prev_gripper_t
        if dt <= 1e-6:
            return None
        cur = gripper_info['position_3d']
        prev = self._prev_gripper_3d
        d = ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2
             + (cur[2] - prev[2]) ** 2) ** 0.5
        return d / dt

    def _on_click(self, x_3d: float, y_3d: float, z_3d: float, distance_m: float, robot_z: float):
        """
        Обработчик клика - отправляет команду роботу.
        
        Args:
            x_3d: X координата в метрах
            y_3d: Y координата в метрах
            z_3d: Z координата в метрах
            distance_m: Расстояние в метрах
            robot_z: Z координата для робота в мм
        """
        if self.robot_service and self.robot_service.is_connected:
            # Конвертируем в мм
            x_mm = x_3d * 1000
            y_mm = y_3d * 1000
            distance_mm = distance_m * 1000
            
            self.robot_service.move_to_target_async(x_mm, y_mm, distance_mm)
    
    def set_gripper_callback(self, callback: Callable):
        """Устанавливает callback для обнаружения хвата."""
        self._on_gripper_detected = callback
    
    def set_object_callback(self, callback: Callable):
        """Устанавливает callback для обнаружения объектов."""
        self._on_object_detected = callback
    
    def _on_cycle_position_reached(self, direction: str, position: list):
        """
        Callback при достижении позиции в цикле движения.
        
        Args:
            direction: 'left' или 'right'
            position: Координаты позиции [X, Y, Z, A, B, C]
        """
        # Здесь можно добавить логику при достижении позиции
        # Например: проверка препятствий, захват объекта и т.д.
        if self._on_cycle_position:
            self._on_cycle_position(direction, position)
    
    def set_cycle_position_callback(self, callback: Callable):
        """Устанавливает callback для достижения позиции в цикле."""
        self._on_cycle_position = callback
    
    def get_gripper_position(self) -> Optional[dict]:
        """
        Получает текущую позицию хвата.
        
        Returns:
            Информация о хвате или None
        """
        if self.gripper_detector:
            return {
                'position': self.gripper_detector.last_gripper_position,
                'height': self.gripper_detector.last_gripper_height
            }
        return None
    
    def get_robot_position(self) -> Optional[tuple]:
        """
        Получает текущую позицию робота.
        
        Returns:
            Кортеж (X, Y, Z, A, B, C) или None
        """
        if self.robot_service and self.robot_service.is_connected:
            return self.robot_service.get_current_position()
        return None
    
    def move_robot_home(self) -> bool:
        """Отправляет робота в домашнюю позицию."""
        if self.robot_service and self.robot_service.is_connected:
            return self.robot_service.move_to_home()
        return False
    
    def shutdown(self):
        """Завершает работу системы."""
        print("\n" + "=" * 60)
        print("  Shutting down...")
        print("=" * 60)
        
        self._running = False
        
        # Останавливаем цикл движения робота
        if self.robot_service and self.robot_service.is_cycle_running:
            self.robot_service.stop_cycle_movement()
        
        if self.camera_window:
            self.camera_window.close()
        
        if self.monitor_window:
            self.monitor_window.close()
        
        if self.point_cloud_window:
            self.point_cloud_window.close()
        
        if self.camera_service:
            self.camera_service.stop()
        
        if self.robot_service:
            self.robot_service.disconnect()

        if self.perf_logger is not None:
            self.perf_logger.close()

        if self.stop_tracker is not None:
            self.stop_tracker.close()

        if self.recorder is not None:
            self.recorder.close()

        print("  System stopped.")
        print("=" * 60 + "\n")

