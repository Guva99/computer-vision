"""
Башня из кубиков: сборка, разборка и непрерывный цикл.

Техпроцесс для опытов со вторжением человека: роботу нужна работа, которую он
делает долго и одинаково, иначе реакцию системы обнаружения не на чем мерить.

Маршрут уровня N при СБОРКЕ:

    transport            PTP по углам — уход в свободное пространство
    магазин, слот S      CASE 20: подход, вход, смыкание, подъём, отход
    transport            PTP по углам — возврат наверх
    над стопкой          LIN в build_base + (N-1)*40 + запас
    уровень N            LIN вниз в build_base + (N-1)*40
    разжать              CASE 7, манипулятор стоит
    над стопкой          LIN вверх

При РАЗБОРКЕ всё зеркально: вместо «разжать» — «сомкнуть», вместо CASE 20 в
начале — CASE 21 в конце. Разбирать можно только с макушки: сняв нижний кубик,
уронишь верхние.

Почему для разборки не нужно ничего заучивать. Высота снятия уровня N — та же,
на которой кубик туда ставили. В момент разжатия губки стояли ровно в этой
точке вокруг этого кубика и поднялись, не задев соседей, — значит опускание
разомкнутых губок на ту же Z безопасно по построению.

Высота уровня считается от ЗАУЧЁНОЙ позы `build_base`, в которой кубик стоял на
столе, плюс ребро кубика на каждый уровень. Камера, внешняя калибровка и длина
инструмента в расчёт не входят: `$TOOL` нулевой, губки смотрят вниз, поэтому
сдвиг фланца по Z равен подъёму кубика.

    python tools/build_tower.py --plan --level 2        # маршрут, не ехать
    python tools/build_tower.py --level 1 --yes         # слот 1 -> уровень 1
    python tools/build_tower.py --unstack --level 6 --yes
    python tools/build_tower.py --cycle 1 --yes         # собрать и разобрать
    python tools/build_tower.py --cycle 0 --yes         # крутить до остановки

Робот движется. T1, удержание, рука на кнопке останова.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402
from tools.goto_pose import (  # noqa: E402
    load_poses, pose_xyz, preflight, read_var, send_axis, send_grip, send_lin)
from tools.magazine import CASE_PUT, CASE_TAKE, grip_is_closed  # noqa: E402
from tools.magazine import run_case as magazine_case  # noqa: E402
from task.state import ProcessState, clear_state, write_state  # noqa: E402

CUBE_MM = 40.0


def level_z(build_base: dict, level: int, cube_mm: float) -> float:
    """Высота фланца для уровня. Уровень 1 — сама заученная поза."""
    return float(build_base["cartesian"]["Z"]) + (level - 1) * cube_mm


def geom(poses: dict, level: int, args):
    """(точка над уровнем, точка на уровне, ориентация)."""
    bb = poses["build_base"]["cartesian"]
    z = level_z(poses["build_base"], level, args.cube_mm)
    return ((bb["X"], bb["Y"], z + args.hover_mm), (bb["X"], bb["Y"], z),
            (bb["A"], bb["B"], bb["C"]))


def plan_lines(poses: dict, level: int, slot: int, args) -> list:
    over, at, _ = geom(poses, level, args)
    bb = poses["build_base"]["cartesian"]
    base = f"(build_base {bb['Z']:.1f} + {(level - 1) * args.cube_mm:.0f})"
    if args.unstack:
        return [f"РАЗБОРКА: уровень {level} -> слот {slot}",
                "  1. transport        PTP по углам",
                f"  2. над уровнем      LIN Z={over[2]:.1f}",
                f"  3. уровень {level}        LIN Z={at[2]:.1f}   {base}",
                "  4. сомкнуть         CASE 7",
                f"  5. подъём           LIN Z={over[2]:.1f}",
                "  6. transport        PTP по углам",
                f"  7. слот {slot}           CASE 21, вернуть"]
    return [f"СБОРКА: слот {slot} -> уровень {level}",
            "  1. transport        PTP по углам",
            f"  2. магазин слот {slot}   CASE 20, взять",
            "  3. transport        PTP по углам",
            f"  4. над стопкой      LIN Z={over[2]:.1f}",
            f"  5. уровень {level}        LIN Z={at[2]:.1f}   {base}",
            "  6. разжать          CASE 7",
            f"  7. над стопкой      LIN Z={over[2]:.1f}"]


def approach(osv, poses: dict, level: int, args, close: bool) -> bool:
    """Подвести сверху, опуститься, сработать хватом, подняться.

    Общая часть сборки и разборки: отличается только тем, смыкаются губки или
    размыкаются. Высота в обоих случаях одна — см. шапку модуля.
    """
    over, at, abc = geom(poses, level, args)
    print(f"  над уровнем {level}, Z={over[2]:.1f}")
    if not send_lin(osv, over, abc, args.timeout):
        return False
    print(f"  вниз на уровень {level}, Z={at[2]:.1f}")
    if not send_lin(osv, at, abc, args.timeout):
        return False
    print("  сомкнуть губки" if close else "  разжать губки")
    if not send_grip(osv, close=close):
        return False
    if grip_is_closed(osv) is not close:
        print("[СТОП] хват не в ожидаемом состоянии после срабатывания")
        return False
    print(f"  подъём, Z={over[2]:.1f}")
    return send_lin(osv, over, abc, args.timeout)


def run_level(osv, poses: dict, level: int, slot: int, args) -> bool:
    """Сборка: взять из слота и поставить на уровень.

    Состояние пишется после КАЖДОГО изменения сцены, а не в начале операции:
    зрение должно узнать о кубике в губках ровно тогда, когда он там
    появился, иначе ожидаемая сцена отстаёт от наблюдаемой на целый ход.
    """
    tr = poses["transport"]["axis"]
    print(f"\n--- СБОРКА: слот {slot} -> уровень {level} ---")
    st = ProcessState(phase="assemble", level=level, slot=slot,
                      cubes_placed=level - 1, cube_in_gripper=False)
    write_state(st, args.state_path)
    if not send_axis(osv, tr, args.timeout):
        return False
    if not magazine_case(osv, CASE_TAKE, slot, args.timeout):
        return False
    st.cube_in_gripper = True
    write_state(st, args.state_path)
    if not send_axis(osv, tr, args.timeout):
        return False
    if not approach(osv, poses, level, args, close=False):
        return False
    st.cube_in_gripper, st.cubes_placed = False, level
    write_state(st, args.state_path)
    return True


def run_unstack(osv, poses: dict, level: int, slot: int, args) -> bool:
    """Разборка: снять с уровня и вернуть в слот."""
    tr = poses["transport"]["axis"]
    print(f"\n--- РАЗБОРКА: уровень {level} -> слот {slot} ---")
    st = ProcessState(phase="unstack", level=level, slot=slot,
                      cubes_placed=level, cube_in_gripper=False)
    write_state(st, args.state_path)
    if not send_axis(osv, tr, args.timeout):
        return False
    if not approach(osv, poses, level, args, close=True):
        return False
    st.cube_in_gripper, st.cubes_placed = True, level - 1
    write_state(st, args.state_path)
    if not send_axis(osv, tr, args.timeout):
        return False
    if not magazine_case(osv, CASE_PUT, slot, args.timeout):
        return False
    st.cube_in_gripper = False
    write_state(st, args.state_path)
    return True


def run_cycle(osv, poses: dict, args) -> bool:
    """Собрать башню снизу вверх, затем разобрать сверху вниз. Повторить."""
    n, rep = args.levels, 0
    bar = "=" * 58
    while args.cycle == 0 or rep < args.cycle:
        rep += 1
        print(f"\n{bar}\n  ПРОХОД {rep}: СБОРКА, {n} уровней\n{bar}")
        for lvl in range(1, n + 1):
            if not run_level(osv, poses, lvl, lvl, args):
                return False
        print(f"\n{bar}\n  ПРОХОД {rep}: РАЗБОРКА с макушки\n{bar}")
        for lvl in range(n, 0, -1):
            if not run_unstack(osv, poses, lvl, lvl, args):
                return False
        print(f"\n[ПРОХОД {rep} ЗАВЕРШЁН] магазин снова полон")
        if args.cycle == 0 or rep < args.cycle:
            time.sleep(args.pause_s)
    return True


def check_ready(osv, poses: dict) -> str | None:
    """Всё ли на месте до первого движения."""
    for name in ("build_base", "transport"):
        if name not in poses:
            return f"нет позы '{name}' — сними tools/teach_poses.py --only {name} --now"
    if not poses["build_base"].get("cartesian"):
        return "у build_base нет декартовых координат — уровни считать не от чего"
    if grip_is_closed(osv) is not False:
        return ("губки должны быть РАЗОМКНУТЫ перед началом: сейчас в них либо "
                "кубик, либо они просто закрыты. Освободи "
                "(python tools/goto_pose.py --grip open)")
    return None


def parse_args():
    ap = argparse.ArgumentParser(description="Башня: сборка, разборка, цикл")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--poses", default="task/poses.yaml")
    ap.add_argument("--level", type=int, default=1, help="уровень башни, 1 = на столе")
    ap.add_argument("--slot", type=int, default=0,
                    help="слот магазина; по умолчанию равен номеру уровня")
    ap.add_argument("--unstack", action="store_true",
                    help="снять кубик с уровня и вернуть в слот")
    ap.add_argument("--cycle", type=int, default=-1, metavar="N",
                    help="полный цикл сборка+разборка N раз; 0 = без остановки")
    ap.add_argument("--levels", type=int, default=6, help="высота башни в кубиках")
    ap.add_argument("--cube-mm", type=float, default=CUBE_MM, help="ребро кубика")
    ap.add_argument("--hover-mm", type=float, default=100.0,
                    help="запас над уровнем при подводе и отходе")
    ap.add_argument("--pause-s", type=float, default=3.0, help="пауза между проходами")
    ap.add_argument("--plan", action="store_true", help="показать маршрут и выйти")
    ap.add_argument("--yes", action="store_true", help="подтвердить движение")
    ap.add_argument("--state-path", default="task/state.json",
                    help="куда писать состояние процесса для зрения")
    ap.add_argument("--timeout", type=float, default=240.0)
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    poses = load_poses(ROOT / args.poses)
    if not poses:
        return 1
    slot = args.slot or args.level
    if not (1 <= slot <= 6) or args.level < 1 or not (1 <= args.levels <= 6):
        print("[СТОП] уровень и слот считаются с единицы, слотов всего 6")
        return 1

    if args.plan:
        if "build_base" not in poses:
            print("[ОШИБКА] Нет позы build_base")
            return 1
        print("\n".join(plan_lines(poses, args.level, slot, args)))
        return 0

    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1
    why = check_ready(osv, poses) or preflight(osv)
    if why is not None:
        print(f"[СТОП] {why}")
        osv.close()
        return 1
    if not args.yes:
        print("[СТОП] Нужен --yes для движения.")
        osv.close()
        return 1

    print(f"скорость={read_var(osv, '$OV_PRO')}%  старт из "
          f"X{(pose_xyz(osv) or [0, 0, 0])[0]:.0f}")
    if args.cycle >= 0:
        ok = run_cycle(osv, poses, args)
    else:
        print("\n".join(plan_lines(poses, args.level, slot, args)))
        runner = run_unstack if args.unstack else run_level
        ok = runner(osv, poses, args.level, slot, args)
    print(f"\nпозиция: {read_var(osv, '$POS_ACT')}")
    print(f"хват: $OUT[1]={read_var(osv, '$OUT[1]')} $OUT[2]={read_var(osv, '$OUT[2]')}")
    clear_state(args.state_path)  # процесс кончился — ожидать нечего
    print("\n[ГОТОВО]" if ok else "\n[ПРЕРВАНО] смотри причину выше.")
    osv.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
