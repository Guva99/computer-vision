"""
Замер границ рабочей зоны в базе робота — для workspace-фильтра коллизий.

Использует ТОТ ЖЕ FK-код (kuka_fk.fk_joints), что и проверка зоны в рантайме,
поэтому систематическая ошибка DH/калибровки сокращается: зона, замеренная
этим скриптом, согласована с координатами, которые увидит фильтр.

Окно камеры показывает текущую FK-точку TCP (перекрестие) и уже записанные
точки — видно, попадает ли точка в поле зрения камеры.

Порядок работы:
  1. Робот в T1, джойстиком подвести КОНЧИК ХВАТА (TCP) к точке границы зоны
     (углы стола, поверхность стола, верх зоны).
  2. В окне камеры нажать SPACE — точка записана.
  3. Минимум точек: 4 угла стола (или 2 диагональных), касание поверхности
     стола (нижний Z); верх зоны можно потом поднять руками в конфиге.
  4. Клавиши: SPACE=записать, u=удалить последнюю, q/ESC=закончить.
  5. Скрипт напечатает min/max по XYZ с запасом и готовые строки для AppConfig.

Если камера или extrinsic недоступны — консольный режим (Enter/q) без окна.
"""
import numpy as np

from kuka_fk import fk_joints, project_to_image
from realsense_io import AppConfig, RealSenseCamera
from robot.drivers.openshowvar import OpenShowVar
from robot.services.fk_service import load_extrinsic
from robot.services.joint_reader import parse_joint_angles

MARGIN_M = 0.03          # запас наружу от замеренных крайних точек
READ_EVERY_N_FRAMES = 3  # частота опроса $AXIS_ACT (как в основном цикле)


def read_tcp(robot: OpenShowVar, gripper_length_m: float):
    """Текущая позиция TCP в базе робота (м) + углы, или (None, None)."""
    try:
        raw = robot.read("$AXIS_ACT", debug=False)
    except Exception:
        return None, None
    if raw is None:
        return None, None
    angles = parse_joint_angles(raw.decode())
    if angles is None:
        return None, None
    positions = fk_joints(angles, gripper_length_m=gripper_length_m)
    return np.asarray(positions[-1], dtype=np.float64), angles


def print_point(idx: int, tcp: np.ndarray, angles) -> None:
    print(f"  [{idx}] TCP (база): X={tcp[0]:+.3f}  Y={tcp[1]:+.3f}  Z={tcp[2]:+.3f} м"
          f"   (A1..A6: {', '.join(f'{a:.1f}' for a in angles)})")


def print_result(points: list) -> None:
    if len(points) < 2:
        print("[ERROR] Нужно минимум 2 точки — границы не посчитаны")
        return
    pts = np.array(points)
    lo = pts.min(axis=0) - MARGIN_M
    hi = pts.max(axis=0) + MARGIN_M
    print("\n" + "=" * 60)
    print(f"Замерено точек: {len(points)}, запас: ±{MARGIN_M * 1000:.0f} мм")
    print(f"X: [{lo[0]:+.3f}, {hi[0]:+.3f}] м")
    print(f"Y: [{lo[1]:+.3f}, {hi[1]:+.3f}] м")
    print(f"Z: [{lo[2]:+.3f}, {hi[2]:+.3f}] м")
    print("\nСтроки для AppConfig (realsense_io.py):")
    print(f"    collision_workspace_x_m: tuple = ({lo[0]:.3f}, {hi[0]:.3f})")
    print(f"    collision_workspace_y_m: tuple = ({lo[1]:.3f}, {hi[1]:.3f})")
    print(f"    collision_workspace_z_m: tuple = ({lo[2]:.3f}, {hi[2]:.3f})")
    print("\nПодсказка: если верх зоны не замерялся TCP-ом, подними Z-максимум")
    print("вручную (до высоты, куда реально может попасть объект).")


def run_console(robot: OpenShowVar, cfg: AppConfig) -> list:
    """Fallback без камеры: Enter=записать, q=закончить."""
    points = []
    while True:
        cmd = input(f"[{len(points)} точек] Enter=записать, q=закончить: ").strip().lower()
        if cmd == "q":
            break
        tcp, angles = read_tcp(robot, cfg.fk_gripper_length_m)
        if tcp is None:
            print("  [WARN] не удалось прочитать углы, повтори")
            continue
        points.append(tcp)
        print_point(len(points), tcp, angles)
    return points


def run_camera(robot: OpenShowVar, cfg: AppConfig, camera: RealSenseCamera,
               T_cr: np.ndarray) -> list:
    import cv2

    points = []           # записанные TCP-позиции в базе робота (м)
    tcp = None            # текущая позиция TCP (база)
    angles = None
    frame_i = 0
    print("[INFO] Окно камеры: SPACE=записать точку, u=удалить последнюю, q/ESC=выход")

    while True:
        frame = camera.get_aligned_frames()
        if frame is None:
            continue
        color_bgr, _depth, intrinsics, _scale = frame
        frame_i += 1

        if frame_i % READ_EVERY_N_FRAMES == 1 or tcp is None:
            new_tcp, new_angles = read_tcp(robot, cfg.fk_gripper_length_m)
            if new_tcp is not None:
                tcp, angles = new_tcp, new_angles

        h, w = color_bgr.shape[:2]
        overlay = color_bgr.copy()

        # Записанные точки: проекция сохранённых base-координат в текущий кадр.
        for i, p in enumerate(points):
            uv = project_to_image(p, T_cr, intrinsics)
            if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(overlay, uv, 6, (255, 200, 0), 2)
                cv2.putText(overlay, str(i + 1), (uv[0] + 8, uv[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        # Текущая FK-точка TCP: перекрестие. Зелёное = в кадре, красный
        # текст = вне кадра/за камерой (камера её не видит).
        if tcp is not None:
            uv = project_to_image(tcp, T_cr, intrinsics)
            in_frame = uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h
            if in_frame:
                cv2.drawMarker(overlay, uv, (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(overlay, uv, 10, (0, 255, 0), 2)
            else:
                cv2.putText(overlay, "TCP OUT OF CAMERA VIEW", (10, h - 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"TCP base: X={tcp[0]:+.3f} Y={tcp[1]:+.3f} Z={tcp[2]:+.3f} m",
                (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
        else:
            cv2.putText(overlay, "NO ROBOT ANGLES", (10, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(overlay, f"points: {len(points)}   SPACE=save  u=undo  q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("measure_workspace", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if tcp is None:
                print("  [WARN] нет углов робота — точка не записана")
            else:
                points.append(tcp.copy())
                print_point(len(points), tcp, angles)
        elif key == ord("u") and points:
            points.pop()
            print(f"  [UNDO] осталось точек: {len(points)}")

    cv2.destroyAllWindows()
    return points


def main():
    cfg = AppConfig()
    robot = OpenShowVar(ip=cfg.robot_ip, port=cfg.robot_port,
                        read_timeout=cfg.robot_read_timeout_s)
    if not robot.can_connect:
        print(f"[ERROR] Контроллер {cfg.robot_ip}:{cfg.robot_port} недоступен")
        return
    print(f"[OK] Подключено: {cfg.robot_ip}:{cfg.robot_port}")
    print(f"[INFO] Длина хвата (fk_gripper_length_m): {cfg.fk_gripper_length_m:.3f} м")

    T_cr = load_extrinsic(cfg.extrinsic_path)
    camera = None
    if T_cr is not None:
        try:
            camera = RealSenseCamera(cfg)
            camera.start()
            print("[OK] Камера запущена")
        except Exception as e:
            print(f"[WARN] Камера недоступна ({e}) — консольный режим")
            camera = None
    else:
        print("[WARN] Нет extrinsic — консольный режим (без окна камеры)")

    try:
        if camera is not None:
            points = run_camera(robot, cfg, camera, T_cr)
        else:
            points = run_console(robot, cfg)
    finally:
        if camera is not None:
            camera.stop()
        try:
            robot.sock.close()
        except Exception:
            pass

    print_result(points)


if __name__ == "__main__":
    main()
