"""
Один уровень башни: взять кубик из слота и поставить на стопку.

Запускается по одному уровню за раз. Так после каждого кубика есть
естественная остановка — посмотреть, поправить, прервать. Прогон всех трёх
уровней подряд прячет момент, когда что-то пошло не так.

Маршрут уровня N:

    transport            PTP по углам — уход в свободное пространство
    магазин, слот S      CASE 20: подход, вход, смыкание, подъём, отход
    transport            PTP по углам — возврат наверх
    над стопкой          LIN в build_base + (N-1)*40 + запас
    уровень N            LIN вниз в build_base + (N-1)*40
    разжать              CASE 7, манипулятор стоит
    над стопкой          LIN вверх

Почему именно так. Высота уровня считается от ЗАУЧЁНОЙ позы `build_base`, в
которой кубик стоял на столе, плюс ребро кубика на каждый уровень. Камера,
внешняя калибровка и длина инструмента в расчёт не входят: `$TOOL` нулевой,
губки смотрят вниз, поэтому сдвиг фланца по Z равен подъёму кубика.

Заход на уровень идёт СВЕРХУ: опуститься в `build_base` и потом подняться
значило бы поставить кубик туда, где уже стоит предыдущий.

    python tools/build_tower.py --plan              # показать маршрут, не ехать
    python tools/build_tower.py --level 1 --yes     # слот 1 -> уровень 1
    python tools/build_tower.py --level 2 --yes     # слот 2 -> уровень 2
    python tools/build_tower.py --level 3 --slot 5 --yes

Робот движется. T1, удержание, рука на кнопке останова.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402
from tools.goto_pose import (  # noqa: E402
    load_poses, pose_xyz, preflight, read_var, send_axis, send_grip, send_lin)
from tools.magazine import CASE_TAKE, grip_is_closed  # noqa: E402
from tools.magazine import run_case as magazine_case  # noqa: E402

CUBE_MM = 40.0


def level_z(build_base: dict, level: int, cube_mm: float) -> float:
    """Высота фланца для уровня. Уровень 1 — сама заученная поза."""
    return float(build_base["cartesian"]["Z"]) + (level - 1) * cube_mm


def plan_lines(poses: dict, level: int, slot: int, args) -> list:
    bb = poses["build_base"]["cartesian"]
    z = level_z(poses["build_base"], level, args.cube_mm)
    return [
        f"уровень {level}, кубик из слота {slot}",
        f"  1. transport        PTP по углам",
        f"  2. магазин слот {slot}   CASE 20, взять",
        f"  3. transport        PTP по углам",
        f"  4. над стопкой      LIN Z={z + args.hover_mm:.1f}",
        f"  5. уровень {level}        LIN Z={z:.1f}   (build_base {bb['Z']:.1f} "
        f"+ {(level - 1) * args.cube_mm:.0f})",
        f"  6. разжать          CASE 7",
        f"  7. над стопкой      LIN Z={z + args.hover_mm:.1f}",
    ]


def place(osv, poses: dict, level: int, args) -> bool:
    """Шаги 4..7: подвести сверху, опустить, отпустить, подняться."""
    bb = poses["build_base"]["cartesian"]
    z = level_z(poses["build_base"], level, args.cube_mm)
    abc = (bb["A"], bb["B"], bb["C"])
    over = (bb["X"], bb["Y"], z + args.hover_mm)
    print(f"\n[4/7] над стопкой, Z={over[2]:.1f}")
    if not send_lin(osv, over, abc, args.timeout):
        return False
    print(f"\n[5/7] вниз на уровень {level}, Z={z:.1f}")
    if not send_lin(osv, (bb["X"], bb["Y"], z), abc, args.timeout):
        return False
    print("\n[6/7] разжать губки")
    if not send_grip(osv, close=False):
        return False
    print(f"\n[7/7] подъём, Z={over[2]:.1f}")
    return send_lin(osv, over, abc, args.timeout)


def run_level(osv, poses: dict, level: int, slot: int, args) -> bool:
    """Полный цикл одного уровня."""
    tr = poses["transport"]["axis"]
    print(f"\n[1/7] transport")
    if not send_axis(osv, tr, args.timeout):
        return False
    print(f"\n[2/7] магазин, слот {slot}")
    if not magazine_case(osv, CASE_TAKE, slot, args.timeout):
        return False
    if grip_is_closed(osv) is not True:
        print("[СТОП] после взятия губки не сомкнуты — кубика в них нет")
        return False
    print(f"\n[3/7] transport")
    if not send_axis(osv, tr, args.timeout):
        return False
    return place(osv, poses, level, args)


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
                "(python tools/test_gripper.py --open)")
    return None


def parse_args():
    ap = argparse.ArgumentParser(description="Один уровень башни: взять и поставить")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--poses", default="task/poses.yaml")
    ap.add_argument("--level", type=int, default=1, help="уровень башни, 1 = на столе")
    ap.add_argument("--slot", type=int, default=0,
                    help="слот магазина; по умолчанию равен номеру уровня")
    ap.add_argument("--cube-mm", type=float, default=CUBE_MM, help="ребро кубика")
    ap.add_argument("--hover-mm", type=float, default=100.0,
                    help="запас над уровнем при подводе и отходе")
    ap.add_argument("--plan", action="store_true", help="показать маршрут и выйти")
    ap.add_argument("--yes", action="store_true", help="подтвердить движение")
    ap.add_argument("--timeout", type=float, default=240.0)
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    poses = load_poses(ROOT / args.poses)
    if not poses:
        return 1
    slot = args.slot or args.level
    if not (1 <= slot <= 6):
        print(f"[СТОП] Слот {slot} вне диапазона 1..6")
        return 1
    if args.level < 1:
        print("[СТОП] Уровень считается с единицы")
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

    print("\n".join(plan_lines(poses, args.level, slot, args)))
    print(f"\nскорость={read_var(osv, '$OV_PRO')}%  старт из "
          f"X{(pose_xyz(osv) or [0, 0, 0])[0]:.0f}")
    ok = run_level(osv, poses, args.level, slot, args)
    print(f"\nпозиция: {read_var(osv, '$POS_ACT')}")
    print(f"хват: $OUT[1]={read_var(osv, '$OUT[1]')} $OUT[2]={read_var(osv, '$OUT[2]')}")
    print("\n[ГОТОВО] уровень поставлен." if ok else "\n[ПРЕРВАНО] смотри причину выше.")
    osv.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
