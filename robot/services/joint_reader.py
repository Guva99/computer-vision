"""
Сервис чтения углов суставов KUKA для real-time цикла зрения.

Отделён от RobotService (управление роботом): здесь только неблокирующее
чтение $AXIS_ACT с коротким таймаутом раз в N кадров и переиспользование
последних валидных углов при пропуске/таймауте.
"""
from typing import Optional, Tuple

from realsense_io import AppConfig

try:
    from robot.drivers.openshowvar import OpenShowVar

    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("[WARNING] Robot modules not found")


def parse_joint_angles(axis_string):
    try:
        cleaned = axis_string.replace(",", " ").replace("{", "").replace("}", "").replace("E6AXIS:", "")
        parts = cleaned.split()
        angles = []
        for i, part in enumerate(parts):
            if part.startswith("A") and len(part) == 2 and part[1].isdigit():
                if i + 1 < len(parts):
                    try:
                        angles.append(float(parts[i + 1]))
                    except ValueError:
                        pass
        return tuple(angles[:6]) if len(angles) >= 6 else None
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}, data: {axis_string[:100]}")
        return None


class JointAngleReader:
    """Неблокирующее чтение углов A1..A6 у контроллера KUKA."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._robot = None
        self._last_joint_angles: Optional[Tuple[float, ...]] = None
        self._playback_joints: Optional[dict] = None  # {frame: (A1..A6)} в playback

    def connect(self) -> "JointAngleReader":
        cfg = self.cfg
        # Режим воспроизведения (Задача 3): углы берутся из записи по номеру
        # кадра, соединение с роботом не открывается.
        if getattr(cfg, "source_mode", "live") == "playback":
            from playback_io import load_recorded_joints
            self._playback_joints = load_recorded_joints(
                getattr(cfg, "playback_path", "")
            )
            print(f"[PLAYBACK] JointAngleReader: {len(self._playback_joints)} "
                  "кадров с углами из записи")
            return self
        if ROBOT_AVAILABLE and cfg.use_robot_kinematics:
            print("\n" + "=" * 60)
            print("RAW POINT CLOUD + gripper outline (vision) + robot angles")
            print("=" * 60)
            try:
                robot = OpenShowVar(ip=cfg.robot_ip, port=cfg.robot_port,
                                    read_timeout=cfg.robot_read_timeout_s)
                if robot.can_connect:
                    print(f"[OK] Robot read: {cfg.robot_ip}:{cfg.robot_port}")
                    self._robot = robot
                else:
                    print("[WARN] Robot not reachable")
            except Exception as e:
                print(f"[ERROR] {e}")
                self._robot = None
        else:
            print("\n[INFO] Camera only (robot angles off)")
        return self

    def read(self, frame_count: int) -> Optional[Tuple[float, ...]]:
        """Читать углы WITHOUT блокировки цикла камеры.

         • короткий socket-таймаут (cfg.robot_read_timeout_s) — recv не висит
         • только раз в N кадров — углы меняются плавно
         • при таймауте/None — переиспользуем последние известные углы
        """
        cfg = self.cfg
        # Playback: углы записи по номеру кадра (пропуски → последние валидные)
        if self._playback_joints is not None:
            angles = self._playback_joints.get(frame_count)
            if angles is not None:
                self._last_joint_angles = angles
            return self._last_joint_angles
        if self._robot and (frame_count % cfg.robot_read_every_n == 0):
            try:
                raw = self._robot.read("$AXIS_ACT", debug=False)
                if raw is not None:
                    parsed = parse_joint_angles(raw.decode())
                    if parsed is not None:
                        self._last_joint_angles = parsed
            except Exception:
                pass  # timeout / busy controller → keep last_joint_angles
        return self._last_joint_angles

    def close(self) -> None:
        if self._robot:
            try:
                self._robot.sock.close()
            except Exception:
                pass
