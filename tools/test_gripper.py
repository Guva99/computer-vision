"""
Проверка хвата: открыть, закрыть, замерить время срабатывания.

Зачем отдельный инструмент. В техпроцессе после команды «закрыть» нужно выждать,
пока губки физически сомкнутся, и только потом поднимать кубик — иначе он
останется в магазине. Это время нигде не записано, и его надо замерить один раз.

Робот при этом НЕ ДВИЖЕТСЯ: команда идёт в CASE 7 (выходы OUT5/OUT6), это только
пневматика хвата. Но программа dist2 на пульте должна быть запущена, а в режиме
T1 оператор держит удержание — иначе KRL команду не подхватит.

    python tools/test_gripper.py --probe              # прочитать состояние, ничего не делать
    python tools/test_gripper.py --open               # разжать
    python tools/test_gripper.py --close              # сжать
    python tools/test_gripper.py --cycle 3            # открыть-закрыть 3 раза, замерить время
    python tools/test_gripper.py --vacuum-on          # вакуум (CASE 8), если используется

Как понять, взялся ли кубик: команда сама этого не знает — датчика в хвате нет.
Проверяется либо глазами, либо (в техпроцессе) зрением: после подъёма слот
должен опустеть. Этот же принцип ложится в модель задачи — «ожидаемое состояние
сцены против наблюдаемого».
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

CASE_GRIPPER = 7
CASE_VACUUM = 8


def read_var(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="ignore").strip() if raw is not None else None


def wait_done(osv, timeout_s: float = 10.0):
    """Ждать обнуления COM_CASEVAR; вернуть время в мс или None при таймауте."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            return (time.perf_counter() - t0) * 1000.0
    return None


def send(osv, out5: str, out6: str, case: int = CASE_GRIPPER):
    """Выставить выходы и дёрнуть CASE. Возвращает время отклика KRL, мс."""
    osv.write("OUT5", out5, False)
    osv.write("OUT6", out6, False)
    osv.write("COM_CASEVAR", str(case), False)
    return wait_done(osv)


def show_state(osv) -> None:
    print(f"режим={read_var(osv, '$MODE_OP')}  COM_CASEVAR={read_var(osv, 'COM_CASEVAR')}  "
          f"$PRO_STATE={read_var(osv, '$PRO_STATE')}")
    outs = {n: read_var(osv, n) for n in ("OUT5", "OUT6", "OUT7")}
    print(f"выходы: {outs}")
    print(f"$OUT[1]={read_var(osv, '$OUT[1]')}  $OUT[2]={read_var(osv, '$OUT[2]')}  "
          f"$OUT[3]={read_var(osv, '$OUT[3]')}")


def run_cycle(osv, n: int, dwell: float) -> None:
    """Открыть-закрыть n раз, печатая время отклика KRL на каждую команду."""
    for i in range(1, n + 1):
        t_open = send(osv, "False", "True")
        print(f"  [{i}] разжать: отклик KRL {t_open:.0f} мс" if t_open
              else f"  [{i}] разжать: ТАЙМАУТ")
        time.sleep(dwell)
        t_close = send(osv, "True", "False")
        print(f"  [{i}] сжать  : отклик KRL {t_close:.0f} мс" if t_close
              else f"  [{i}] сжать  : ТАЙМАУТ")
        time.sleep(dwell)


def parse_args():
    ap = argparse.ArgumentParser(description="Проверка хвата (CASE 7 / CASE 8)")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--probe", action="store_true", help="только чтение состояния")
    ap.add_argument("--open", action="store_true", help="разжать губки")
    ap.add_argument("--close", action="store_true", help="сжать губки")
    ap.add_argument("--cycle", type=int, default=0, help="открыть-закрыть N раз")
    ap.add_argument("--dwell", type=float, default=1.0, help="пауза между командами, с")
    ap.add_argument("--vacuum-on", action="store_true")
    ap.add_argument("--vacuum-off", action="store_true")
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1

    show_state(osv)
    if args.probe or not (args.open or args.close or args.cycle
                          or args.vacuum_on or args.vacuum_off):
        print("\n[PROBE] Только чтение. Дальше: --open / --close / --cycle 3")
        return 0

    print("\nРобот не движется — это только пневматика хвата.")
    if args.open:
        t = send(osv, "False", "True")
        print(f"разжать: отклик KRL {t:.0f} мс" if t else "разжать: ТАЙМАУТ "
              "(программа dist2 не запущена или нет удержания)")
    if args.close:
        t = send(osv, "True", "False")
        print(f"сжать: отклик KRL {t:.0f} мс" if t else "сжать: ТАЙМАУТ "
              "(программа dist2 не запущена или нет удержания)")
    if args.cycle:
        print(f"цикл {args.cycle} раз, пауза {args.dwell} с:")
        run_cycle(osv, args.cycle, args.dwell)
    if args.vacuum_on or args.vacuum_off:
        osv.write("OUT7", "True" if args.vacuum_on else "False", False)
        osv.write("COM_CASEVAR", str(CASE_VACUUM), False)
        t = wait_done(osv)
        print(f"вакуум {'вкл' if args.vacuum_on else 'выкл'}: "
              f"{f'отклик {t:.0f} мс' if t else 'ТАЙМАУТ'}")

    print()
    show_state(osv)
    print("\nВзялся ли кубик — датчика нет, смотри глазами. В техпроцессе это "
          "проверяется зрением: после подъёма слот должен опустеть.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
