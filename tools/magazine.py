"""
Команды магазина: взять кубик из слота и вернуть обратно.

Работает с программой krl/dist3.src на контроллере — это прежний диспетчер
dist2 плюс две команды:

    CASE 20 — взять кубик из слота COM_VALUE1
    CASE 21 — вернуть кубик в слот COM_VALUE1

Траектории внутри CASE не выдуманы: они повторяют рабочие программы стенда
takeCube_1..6 (подход X=-205, вход X=-258, смыкание губок, подъём на 2 мм,
отход). Координаты слотов взяты оттуда же, шаг по высоте 61.5 мм.

    python tools/magazine.py --probe          # состояние, ничего не делать
    python tools/magazine.py --take 1         # взять кубик из нижнего слота
    python tools/magazine.py --put 1          # вернуть в нижний слот
    python tools/magazine.py --take 1 --put 1 # взять и сразу вернуть (проверка)

РОБОТ ПРИ ЭТОМ ДВИЖЕТСЯ. Нужен режим T1, удержание и запущенная dist3.
Движение длится секунды, поэтому ожидание тут долгое, в отличие от команд хвата.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

CASE_TAKE = 20
CASE_PUT = 21


def read_var(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="ignore").strip() if raw is not None else None


def show_state(osv) -> None:
    print(f"режим={read_var(osv, '$MODE_OP')}  скорость={read_var(osv, '$OV_PRO')}%  "
          f"программа={read_var(osv, '$PRO_STATE')}  COM_CASEVAR={read_var(osv, 'COM_CASEVAR')}")
    print(f"позиция: {read_var(osv, '$POS_ACT')}")
    print(f"хват: $OUT[1]={read_var(osv, '$OUT[1]')}  $OUT[2]={read_var(osv, '$OUT[2]')}")


def run_case(osv, case: int, slot: int, timeout_s: float) -> bool:
    """Выполнить команду магазина и дождаться её завершения."""
    what = "взять из" if case == CASE_TAKE else "вернуть в"
    print(f"\n[{case}] {what} слот {slot}. Робот поедет — держи кнопку останова.")
    osv.write("COM_VALUE1", str(int(slot)), False)
    osv.write("COM_CASEVAR", str(case), False)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            print(f"    выполнено за {time.perf_counter() - t0:.1f} с")
            return True
    print(f"    ТАЙМАУТ {timeout_s:.0f} с — команда не завершилась.")
    print("    Проверь: запущена ли dist3, держишь ли удержание, не ноль ли $OV_PRO.")
    return False


def parse_args():
    ap = argparse.ArgumentParser(description="Взять и вернуть кубик из магазина")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--probe", action="store_true", help="только состояние")
    ap.add_argument("--take", type=int, default=0, metavar="N", help="взять из слота N (1..6)")
    ap.add_argument("--put", type=int, default=0, metavar="N", help="вернуть в слот N (1..6)")
    ap.add_argument("--pause", type=float, default=2.0, help="пауза между take и put, с")
    ap.add_argument("--timeout", type=float, default=60.0, help="ожидание команды, с")
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1

    show_state(osv)
    if args.probe or not (args.take or args.put):
        print("\n[PROBE] Только чтение. Дальше: --take 1 (взять из нижнего слота).")
        return 0
    for n in (args.take, args.put):
        if n and not (1 <= n <= 6):
            print(f"[СТОП] Слот {n} вне диапазона 1..6.")
            return 1
    if read_var(osv, "$MODE_OP") != "#T1":
        print("[СТОП] Ожидался режим #T1.")
        return 1

    if args.take:
        if not run_case(osv, CASE_TAKE, args.take, args.timeout):
            return 1
    if args.take and args.put:
        time.sleep(args.pause)
    if args.put:
        if not run_case(osv, CASE_PUT, args.put, args.timeout):
            return 1

    print()
    show_state(osv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
