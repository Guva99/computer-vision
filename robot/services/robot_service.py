"""
Сервис для управления роботом KUKA.
"""
import time
import numpy as np
from threading import Thread, Lock, Event
from typing import Optional, Tuple, Callable

from robot.drivers.kuka_driver import KukaDriver
from robot.drivers.openshowvar import OpenShowVar

# Конфигурация по умолчанию
ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 7000
ROBOT_BASE = 1
ROBOT_TOOL = 1
ROBOT_SPEED = 30
ROBOT_HOME_POSITION = [450, 0, 600, 180, 0, 180]
ROBOT_WORK_ANGLES = [180, 0, 180]
ROBOT_CYCLE_LEFT = [350, 200, 600, 180, 0, 180]
ROBOT_CYCLE_RIGHT = [350, -200, 600, 180, 0, 180]


class RobotService:
    """Сервис для управления роботом KUKA."""
    
    def __init__(
        self,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        auto_connect: bool = False
    ):
        """
        Инициализация сервиса робота.
        
        Args:
            ip: IP-адрес робота (по умолчанию из конфига)
            port: Порт робота (по умолчанию из конфига)
            auto_connect: Автоматически подключаться при создании
        """
        self.ip = ip if ip else ROBOT_IP
        self.port = port if port else ROBOT_PORT
        
        self._connection: Optional[OpenShowVar] = None
        self._driver: Optional[KukaDriver] = None
        self._connected: bool = False
        self._busy: bool = False
        self._busy_lock: Lock = Lock()
        
        # Цикл движения
        self._cycle_running: bool = False
        self._cycle_thread: Optional[Thread] = None
        self._cycle_stop_event: Event = Event()
        self._paused: bool = False
        self._pause_lock: Lock = Lock()
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """
        Подключается к роботу.
        
        Returns:
            True если подключение успешно
        """
        print(f"\n{'=' * 60}")
        print(f"Connecting to KUKA robot at {self.ip}:{self.port}...")
        print(f"{'=' * 60}")
        
        try:
            self._connection = OpenShowVar(ip=self.ip, port=self.port)
            
            if not self._connection.can_connect:
                raise ConnectionError("Cannot connect to robot")
            
            self._driver = KukaDriver(self._connection)
            self._driver.set_base(ROBOT_BASE)
            self._driver.set_tool(ROBOT_TOOL)
            self._driver.set_speed(ROBOT_SPEED)
            
            print(f"Robot connected: {self._driver.name}")
            print(f"Base: {ROBOT_BASE}, Tool: {ROBOT_TOOL}, Speed: {ROBOT_SPEED}%")
            print(f"{'=' * 60}\n")
            
            self._connected = True
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            print(f"Continuing without robot")
            print(f"{'=' * 60}\n")
            self._connected = False
            return False
    
    def disconnect(self):
        """Отключается от робота."""
        if self._connected:
            print("\nDisconnecting from robot...")
            self._connected = False
            self._driver = None
            if self._connection:
                self._connection.close()
            self._connection = None
    
    @property
    def is_connected(self) -> bool:
        """Проверяет подключение к роботу."""
        return self._connected
    
    @property
    def is_busy(self) -> bool:
        """Проверяет, занят ли робот."""
        with self._busy_lock:
            return self._busy
    
    def get_current_position(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Получает текущую позицию робота.
        
        Returns:
            Кортеж (X, Y, Z, A, B, C) или None при ошибке
        """
        if not self._connected or not self._driver:
            return None
        
        try:
            self._driver.read_cartesian()
            return (
                float(self._driver.x_cartesian),
                float(self._driver.y_cartesian),
                float(self._driver.z_cartesian),
                float(self._driver.A_cartesian),
                float(self._driver.B_cartesian),
                float(self._driver.C_cartesian)
            )
        except Exception as e:
            print(f"Error reading position: {e}")
            return None
    
    def get_joint_angles(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Получает текущие углы суставов робота.
        
        Returns:
            Кортеж (A1, A2, A3, A4, A5, A6) в градусах или None при ошибке
        """
        if not self._connected or not self._connection:
            return None
        
        try:
            axis_string = self._connection.read("$AXIS_ACT", False).decode()
            axis_string = axis_string.replace(',', '').replace('{', '').replace('}', '')
            parts = axis_string.split()
            
            angles = []
            for i in range(1, 13, 2):
                if i < len(parts):
                    angles.append(float(parts[i]))
            
            if len(angles) >= 6:
                return tuple(angles[:6])
            return None
            
        except Exception as e:
            print(f"Error reading joint angles: {e}")
            return None
    
    def move_to_home(self) -> bool:
        """
        Отправляет робота в домашнюю позицию.
        
        Returns:
            True если движение успешно
        """
        if not self._connected or not self._driver:
            print("Robot not connected!")
            return False
        
        try:
            print("\nMoving to home position...")
            
            current_pos = self.get_current_position()
            if current_pos is None:
                return False
            
            trajectory = np.array([
                np.array(list(current_pos)),
                np.array(ROBOT_HOME_POSITION)
            ])
            
            self._driver.ptp_continuous(trajectory)
            print("Robot at home position!")
            return True
            
        except Exception as e:
            print(f"Error moving to home: {e}")
            return False
    
    def move_to_target_async(
        self,
        x_cam_mm: float,
        y_cam_mm: float,
        distance_mm: float,
        on_complete: Optional[Callable] = None
    ) -> bool:
        """
        Асинхронно перемещает робота к цели.
        
        Args:
            x_cam_mm: X координата камеры (мм)
            y_cam_mm: Y координата камеры (мм)
            distance_mm: Расстояние до объекта (мм)
            on_complete: Callback при завершении
            
        Returns:
            True если команда принята
        """
        if not self._connected or not self._driver:
            print("Robot not connected!")
            return False
        
        with self._busy_lock:
            if self._busy:
                print("Robot is busy!")
                return False
            self._busy = True
        
        def execute():
            try:
                current_pos = self.get_current_position()
                if current_pos is None:
                    return
                
                z_target = self._calculate_z(distance_mm)
                
                target = [
                    float(current_pos[0] + x_cam_mm),
                    float(current_pos[1] + y_cam_mm),
                    float(z_target),
                    float(ROBOT_WORK_ANGLES[0]),
                    float(ROBOT_WORK_ANGLES[1]),
                    float(ROBOT_WORK_ANGLES[2])
                ]
                
                print(f"\nMoving to target: X={target[0]:.1f}, Y={target[1]:.1f}, Z={target[2]:.1f}")
                self._driver.ptp_continuous(np.array(target))
                print("Movement complete!")
                
                if on_complete:
                    on_complete()
                    
            except Exception as e:
                print(f"Movement error: {e}")
            finally:
                with self._busy_lock:
                    self._busy = False
        
        thread = Thread(target=execute, daemon=True)
        thread.start()
        return True
    
    def _calculate_z(self, distance_mm: float) -> float:
        """Вычисляет Z координату для робота."""
        return -407 - (880 - distance_mm - 120)
    
    def open_gripper(self):
        """Открывает хват."""
        if self._driver:
            self._driver.open_grip()
    
    def close_gripper(self):
        """Закрывает хват."""
        if self._driver:
            self._driver.close_grip()
    
    # =========================================================================
    # ЦИКЛ ДВИЖЕНИЯ ВЛЕВО-ВПРАВО
    # =========================================================================
    
    def start_cycle_movement(
        self,
        on_position_reached: Optional[Callable[[str, list], None]] = None
    ) -> bool:
        """
        Запускает цикл движения влево-вправо.
        
        Args:
            on_position_reached: Callback при достижении позиции (direction, position)
            
        Returns:
            True если цикл запущен
        """
        if not self._connected or not self._driver:
            print("Robot not connected!")
            return False
        
        if self._cycle_running:
            print("Cycle already running!")
            return False
        
        self._cycle_running = True
        self._cycle_stop_event.clear()
        self._paused = False
        
        def cycle_loop():
            print("\n[CYCLE] Starting cycle movement...")
            direction = 'right'  # Начинаем с движения вправо
            
            while not self._cycle_stop_event.is_set():
                # Проверяем паузу
                with self._pause_lock:
                    if self._paused:
                        time.sleep(0.1)
                        continue
                
                try:
                    # Выбираем целевую позицию
                    if direction == 'right':
                        target = ROBOT_CYCLE_RIGHT
                    else:
                        target = ROBOT_CYCLE_LEFT
                    
                    print(f"[CYCLE] Moving {direction}: X={target[0]:.1f}, Y={target[1]:.1f}")
                    
                    # Устанавливаем скорость перед движением
                    self._driver.set_speed(ROBOT_SPEED)
                    
                    # Выполняем движение
                    current_pos = self.get_current_position()
                    if current_pos is None:
                        time.sleep(0.5)
                        continue
                    
                    trajectory = np.array([
                        np.array(list(current_pos)),
                        np.array(target)
                    ])
                    
                    self._driver.ptp_continuous(trajectory)
                    
                    # Callback при достижении позиции
                    if on_position_reached:
                        on_position_reached(direction, target)
                    
                    # Меняем направление
                    direction = 'left' if direction == 'right' else 'right'
                    
                    # Пауза между движениями
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"[CYCLE] Error: {e}")
                    time.sleep(1.0)
            
            print("[CYCLE] Cycle stopped.")
        
        self._cycle_thread = Thread(target=cycle_loop, daemon=True)
        self._cycle_thread.start()
        return True
    
    def stop_cycle_movement(self):
        """Останавливает цикл движения."""
        if self._cycle_running:
            print("[CYCLE] Stopping cycle...")
            self._cycle_stop_event.set()
            self._cycle_running = False
            
            if self._cycle_thread:
                self._cycle_thread.join(timeout=2.0)
                self._cycle_thread = None
    
    @property
    def is_cycle_running(self) -> bool:
        """Проверяет, запущен ли цикл движения."""
        return self._cycle_running
    
    def stop_movement(self):
        """Останавливает движение робота (скорость = 0)."""
        if self._driver and self._connection:
            try:
                self._connection.write("$OV_PRO", "0")
                with self._pause_lock:
                    self._paused = True
                print("[ROBOT] Movement STOPPED (speed=0)")
            except Exception as e:
                print(f"[ROBOT] Error stopping: {e}")
    
    def resume_movement(self):
        """Возобновляет движение робота."""
        if self._driver and self._connection:
            try:
                self._connection.write("$OV_PRO", str(ROBOT_SPEED))
                with self._pause_lock:
                    self._paused = False
                print(f"[ROBOT] Movement RESUMED (speed={ROBOT_SPEED})")
            except Exception as e:
                print(f"[ROBOT] Error resuming: {e}")
    
    @property
    def is_paused(self) -> bool:
        """Проверяет, на паузе ли робот."""
        with self._pause_lock:
            return self._paused

