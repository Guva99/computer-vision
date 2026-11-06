"""
Клик в кадре → робот едет в эту точку.

Про TCP. На стенде выбран нулевой `TOOL_DATA[1]`, поэтому `LIN` ведёт ФЛАНЕЦ, и
кубик в губках оказывается в стороне на длину хвата. Ключ `--tool N` ставит
`$TOOL = TOOL_DATA[N]` через `CASE 2`, и тогда в кликнутую точку приходит TCP.
Рабочие инструменты и подвох с магазином — в `krl/README.md`.

Клик робота не двигает: промах мышью не должен приводить к движению. Движение —
по отдельной клавише, и первым делом на высоту зависания.

    python tools/click_move.py --tool 4            # TCP хвата, губки вниз

ЛКМ = указать точку, g = на высоту зависания, s = вниз, w = вверх, q = выход.
Требуется запущенная dist3 (`$PRO_STATE = #P_ACTIVE`) и режим T1 с удержанием.
Робот движется — оператор держит кнопку останова.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from realsense_io import AppConfig, RealSenseCamera  # noqa: E402
from robot.services.dispatcher_link import (  # noqa: E402
    ZERO_FRAME, RobotLink, parse_pose)
from robot.services.fk_service import load_extrinsic  # noqa: E402
from tools.pick_build_point import camera_to_robot, pixel_to_camera  # noqa: E402

WINDOW = "Click to move"
MAX_STEP_MM = 400.0

_click = None


def _on_mouse(event, x, y, flags, param):
    global _click
    if event == cv2.EVENT_LBUTTONDOWN:
        _click = (x, y)


def load_plane(path: str):
    """Плоскость стола (a,b,c,d) в кадре камеры: a·x+b·y+c·z+d = высота, м."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return np.asarray(data["plane"], dtype=np.float64).ravel()[:4]
    except Exception as e:
        print(f"[ВНИМАНИЕ] Плоскость стола не прочитана ({e}) — "
              f"проверка просвета отключена, спуск на свой страх.")
        return None


def height_above_table_mm(p_rob_mm, T_cr, plane) -> float | None:
    """Высота точки робота над столом, мм. None — плоскость недоступна."""
    if plane is None:
        return None
    p_cam = (T_cr @ np.array([*np.asarray(p_rob_mm) / 1000.0, 1.0]))[:3]
    return float(plane[:3] @ p_cam + plane[3]) * 1000.0



def split_leg(a, b, max_step: float):
    """Отрезок a→b, разбитый на куски не длиннее max_step. Возвращает точки без a.

    Нужен, чтобы длинный ход не отвергался предохранителем и при этом не шёл
    одним броском: `$ADVANCE = 0`, поэтому каждый кусок отрабатывается до полной
    остановки — между ними можно отпустить удержание и посмотреть на руку.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(b - a))
    n = max(1, int(np.ceil(d / max_step))) if d > 0 else 1
    return [list(a + (b - a) * (i / n)) for i in range(1, n + 1)]


def plan_path(p_click, z_to: float, cur, max_step: float):
    """Маршрут от текущей точки к цели: сперва горизонтально на своей высоте,
    потом вертикально вниз. Диагональ через всю зону непредсказуема на глаз,
    а эти два плеча видно заранее."""
    x, y = float(p_click[0]), float(p_click[1])
    over = [x, y, cur[2]]
    return split_leg(cur, over, max_step) + split_leg(over, [x, y, z_to], max_step)


def check_target(target, link: RobotLink, geom, args) -> str | None:
    """Геометрические проверки цели. Строка — причина отказа."""
    T_cr, plane = geom
    h = height_above_table_mm(target, T_cr, plane)
    if h is not None and h < args.min_clear_mm:
        return f"цель на {h:.0f} мм над столом, предел {args.min_clear_mm:.0f}"
    cur = parse_pose(link.snapshot()[1])
    if cur is not None:
        d = float(np.linalg.norm(np.asarray(target) - np.asarray(cur)))
        if d > args.max_step_mm:
            return (f"бросок {d:.0f} мм больше предела {args.max_step_mm:.0f} "
                    f"— подведи руку пультом или подними --max-step-mm")
    return None


def draw_hud(img, state, link: RobotLink, args):
    """Текст поверх кадра. Только латиница — cv2 не рисует кириллицу."""
    status, pose, busy = link.snapshot()
    led = "TCP" if link.tool_frame != ZERO_FRAME else "FLANEC"
    lines = ["LKM = tochka,  g = zavisnut,  s = vniz,  w = vverh,  q = vyhod",
             f"vedem: {led}   orientaciya A{args.abc[0]:.0f} B{args.abc[1]:.0f} "
             f"C{args.abc[2]:.0f}" + ("  (guby vniz)" if abs(args.abc[1] - 180) < 1 else "")]
    if state["p_rob"] is not None:
        p = state["p_rob"]
        lines.append(f"klik: X={p[0]:+.1f} Y={p[1]:+.1f} Z={p[2]:+.1f} mm")
        h = state["h"]
        lines.append(f"cel: Z={state['z_cmd']:+.1f}  nad stolom "
                     + (f"{h:+.0f} mm" if h is not None else "? (stol neizvesten)"))
    lines.append(f"robot: {'EDET' if busy else 'stoit'}  |  {status}")
    if pose:
        lines.append(pose.replace("{E6POS: ", "")[:72])
    lines.append(f"predel spuska: {args.min_clear_mm:.0f} mm nad stolom")
    for i, t in enumerate(lines):
        color = (0, 0, 255) if busy else (0, 255, 0)
        cv2.putText(img, t, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 2, cv2.LINE_AA)


def handle_key(key: int, state, link: RobotLink, args, geom) -> bool:
    """Обработка клавиши. False — выходим."""
    T_cr, plane = geom
    if key == ord("q"):
        return False
    if state["p_rob"] is None or key not in (ord("g"), ord("s"), ord("w")):
        return True
    if link.snapshot()[2]:
        print("[!] Робот ещё едет — дождись остановки.")
        return True
    if key == ord("g"):
        state["z_cmd"] = float(state["p_rob"][2]) + args.hover_mm
        send_goto(state, link, args, geom)
        return True
    state["z_cmd"] += -args.step_mm if key == ord("s") else args.step_mm
    target = [float(state["p_rob"][0]), float(state["p_rob"][1]), state["z_cmd"]]
    state["h"] = height_above_table_mm(target, T_cr, plane)
    why = check_target(target, link, geom, args)
    if why is not None:
        print(f"[СТОП] {why}")
        return True
    announce(target, state["h"])
    link.cmds.put((target, args.abc, args.timeout))
    return True


def announce(target, h) -> None:
    where = f"{h:+.0f} мм над столом" if h is not None else "высота над столом неизвестна"
    print(f"  ПОЕДЕТ: X={target[0]:+.1f} Y={target[1]:+.1f} "
          f"Z={target[2]:+.1f}  ({where})")


def send_goto(state, link: RobotLink, args, geom) -> None:
    """Подвести к точке: горизонтальное плечо на своей высоте, потом спуск."""
    T_cr, plane = geom
    cur = parse_pose(link.snapshot()[1])
    if cur is None:
        print("[СТОП] текущая поза не прочитана — не знаю, откуда ехать.")
        return
    target = [float(state["p_rob"][0]), float(state["p_rob"][1]), state["z_cmd"]]
    state["h"] = height_above_table_mm(target, T_cr, plane)
    if state["h"] is not None and state["h"] < args.min_clear_mm:
        print(f"[СТОП] цель на {state['h']:.0f} мм над столом, "
              f"предел {args.min_clear_mm:.0f}")
        return
    path = plan_path(state["p_rob"], state["z_cmd"], cur, args.max_step_mm)
    print(f"\nРОБОТ ПОЕДЕТ, {len(path)} отрезков (каждый до полной остановки):")
    for wp in path:
        announce(wp, height_above_table_mm(wp, T_cr, plane))
        link.cmds.put((wp, args.abc, args.timeout))


def run_loop(camera, link: RobotLink, args, geom) -> None:
    global _click
    T_cr, plane = geom
    state = {"p_rob": None, "uv": None, "z_cmd": 0.0, "h": None}
    while True:
        frame = camera.get_aligned_frames()
        if frame is None:
            continue
        color, depth, intr, scale = frame
        img = color.copy()
        if _click is not None:
            pc = pixel_to_camera(_click, depth, intr, scale)
            if pc is None:
                print(f"[!] В точке {_click} нет глубины — ткни рядом")
            else:
                state["uv"] = _click
                state["p_rob"] = camera_to_robot(pc, T_cr)
                state["z_cmd"] = float(state["p_rob"][2]) + args.hover_mm
                state["h"] = height_above_table_mm(
                    [*state["p_rob"][:2], state["z_cmd"]], T_cr, plane)
                p = state["p_rob"]
                print(f"[точка] пиксель={_click}  робот: X={p[0]:+.1f} "
                      f"Y={p[1]:+.1f} Z={p[2]:+.1f} мм")
            _click = None
        if state["uv"] is not None:
            cv2.drawMarker(img, state["uv"], (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(img, state["uv"], 10, (0, 0, 255), 2)
        draw_hud(img, state, link, args)
        cv2.imshow(WINDOW, img)
        if not handle_key(cv2.waitKey(1) & 0xFF, state, link, args, geom):
            break


def apply_tool(link: RobotLink, n: int) -> bool:
    """Прочитать TOOL_DATA[n] с пульта и поставить его как `$TOOL`.

    Значение берётся с контроллера, а не из константы в коде: инструмент заучен
    на пульте, и единственный достоверный источник — сам контроллер.
    """
    frame = link.read(f"TOOL_DATA[{n}]")
    if not frame or "FRAME" not in frame:
        print(f"[ОШИБКА] TOOL_DATA[{n}] не прочитан: {frame}")
        return False
    if frame.replace(" ", "") == ZERO_FRAME.replace(" ", ""):
        print(f"[ОШИБКА] TOOL_DATA[{n}] нулевой — это не TCP хвата. "
              f"Рабочие: 4 (Z 191.74) и 6 (Z 191.46).")
        return False
    print(f"[робот] TOOL_DATA[{n}] = {frame}")
    pose_flange = link.read("$POS_ACT")
    if not link.set_tool(frame):
        return False
    print(f"позиция (TCP)   : {link.read('$POS_ACT')}")
    print(f"       (фланец) : {pose_flange}")
    print("Разница двух строк и есть вынос TCP — сверь её с картинкой.")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Клик в кадре → движение робота")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--hover-mm", type=float, default=150.0,
                    help="высота зависания над кликнутой точкой, мм")
    ap.add_argument("--step-mm", type=float, default=50.0, help="шаг вверх/вниз, мм")
    ap.add_argument("--min-clear-mm", type=float, default=15.0,
                    help="ниже этого просвета над столом не опускаться")
    ap.add_argument("--max-step-mm", type=float, default=MAX_STEP_MM,
                    help="предел одного броска, мм")
    ap.add_argument("--tool", type=int, default=0, metavar="N",
                    help="поставить $TOOL = TOOL_DATA[N] (4 или 6 — TCP хвата); "
                         "0 = не трогать, вести фланец")
    ap.add_argument("--abc", nargs=3, type=float, default=[0.0, 180.0, 0.0],
                    metavar=("A", "B", "C"),
                    help="ориентация. По умолчанию A0 B180 C0 — ось инструмента "
                         "смотрит вниз (губки вниз). A0 B90 C180 — горизонтально, "
                         "как у магазина")
    ap.add_argument("--timeout", type=float, default=90.0)
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    cfg = AppConfig(enable_robot_control=False, show_debug_masks=False,
                    perf_monitor=False, perf_window=False, show_o3d_window=False)
    T_cr = load_extrinsic(cfg.extrinsic_path)
    if T_cr is None:
        print("[ОШИБКА] Нет extrinsic.json — пересчёт в координаты робота невозможен")
        return 1
    plane = load_plane(cfg.table_plane_path)

    link = RobotLink(args.ip, args.port)
    if not link.osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1
    print(f"режим={link.read('$MODE_OP')}  скорость={link.read('$OV_PRO')}%  "
          f"программа={link.read('$PRO_STATE')}")
    print(f"позиция (фланец): {link.read('$POS_ACT')}")
    if not link.wait_active():
        link.osv.close()
        return 1
    if args.tool and not apply_tool(link, args.tool):
        link.osv.close()
        return 1
    link.start()

    camera = RealSenseCamera(cfg)
    camera.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, _on_mouse)
    print("\nКлик = точка. g = зависнуть над ней. s/w = шаг вниз/вверх. q = выход.")
    print("РОБОТ ДВИЖЕТСЯ по g/s/w. T1, удержание, рука на кнопке останова.\n")
    try:
        run_loop(camera, link, args, (T_cr, plane))
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        link.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
