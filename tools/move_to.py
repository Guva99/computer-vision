"""
Движение в заданную декартову точку и относительные смещения.

Нужен, чтобы проверить досягаемость места сборки до того, как туда поедет
техпроцесс. Движение идёт через уже существующие команды диспетчера:

    CASE 4  — LIN по массиву COM_FRAME_ARRAY (прямая линия, предсказуемо)
    CASE 12 — LIN_REL, относительное смещение

Про массив: KRL читает индексы с единицы, а нулевая строка служит слотом под
текущую позицию — поэтому отправляются две строки, а COM_LENGTH = 1. Так же
устроен и штатный драйвер.

    python tools/move_to.py --probe
    python tools/move_to.py --xyz 251.6 147.3 63 --yes      # в точку
    python tools/move_to.py --dz -50 --yes                  # опуститься на 50 мм

Ориентация по умолчанию A=0 B=90 C=180 — та же, что у магазина, губки
горизонтально. Робот движется: режим T1, удержание, рука на кнопке останова.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

CASE_LIN_ARRAY = 4
CASE_LIN_REL = 12
MAX_STEP_MM = 300.0


def read_var(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="ignore").strip() if raw is not None else None


def wait_done(osv, timeout_s: float):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            return time.perf_counter() - t0
    return None


def frame_str(xyz, abc) -> str:
    """Строка FRAME для KRL: позиция и ориентация — две связные тройки."""
    return (f"{{FRAME: X {xyz[0]:.3f}, Y {xyz[1]:.3f}, Z {xyz[2]:.3f}, "
            f"A {abc[0]:.3f}, B {abc[1]:.3f}, C {abc[2]:.3f}}}")


def goto(osv, xyz, abc, timeout_s: float):
    """LIN в абсолютную точку через массив (две строки, исполняется вторая)."""
    f = frame_str(xyz, abc)
    print(f"\n[LIN] → {f}")
    osv.write("COM_FRAME_ARRAY[0]", f, False)
    osv.write("COM_FRAME_ARRAY[1]", f, False)
    osv.write("COM_LENGTH", "1", False)
    osv.write("COM_CASEVAR", str(CASE_LIN_ARRAY), False)
    return wait_done(osv, timeout_s)


def shift(osv, dxyz, timeout_s: float):
    """Относительное смещение в системе BASE."""
    f = frame_str(dxyz, (0.0, 0.0, 0.0))
    print(f"\n[LIN_REL] → {f}")
    osv.write("COM_FRAME", f, False)
    osv.write("COM_CASEVAR", str(CASE_LIN_REL), False)
    return wait_done(osv, timeout_s)


def parse_args():
    ap = argparse.ArgumentParser(description="Движение в точку / относительное смещение")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--abc", nargs=3, type=float, default=[0.0, 90.0, 180.0],
                    metavar=("A", "B", "C"))
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--yes", action="store_true", help="подтвердить движение")
    ap.add_argument("--timeout", type=float, default=90.0)
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1

    print(f"режим={read_var(osv, '$MODE_OP')}  скорость={read_var(osv, '$OV_PRO')}%  "
          f"программа={read_var(osv, '$PRO_STATE')}")
    print(f"позиция: {read_var(osv, '$POS_ACT')}")
    d = (args.dx, args.dy, args.dz)
    if args.probe or (args.xyz is None and max(abs(v) for v in d) == 0.0):
        print("\n[PROBE] Только чтение.")
        return 0
    if not args.yes:
        print("[СТОП] Нужен --yes для движения.")
        return 1
    if max(abs(v) for v in d) > MAX_STEP_MM:
        print(f"[СТОП] Смещение больше {MAX_STEP_MM:.0f} мм за раз.")
        return 1

    dt = goto(osv, args.xyz, args.abc, args.timeout) if args.xyz else shift(osv, d, args.timeout)
    if dt is None:
        print("    ТАЙМАУТ — движение не завершилось. Проверь пульт: возможно, "
              "контроллер отказал (недосягаемая точка или предел сустава).")
        return 1
    print(f"    выполнено за {dt:.1f} с")
    print(f"позиция: {read_var(osv, '$POS_ACT')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
