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


def grip_is_closed(osv):
    """TRUE/FALSE/None. $OUT[1] — сомкнуть, $OUT[2] — разомкнуть."""
    o1, o2 = read_var(osv, "$OUT[1]"), read_var(osv, "$OUT[2]")
    if o1 is None or o2 is None:
        return None
    if o1 == "TRUE" and o2 == "FALSE":
        return True
    if o1 == "FALSE" and o2 == "TRUE":
        return False
    return None


def check_grip(osv, case: int):
    """Состояние хвата должно соответствовать команде.

    Взятие сомкнутыми губками бессмысленно: либо в них уже кубик и его
    раздавит о магазин, либо они просто закрыты и захватить нечем. Возврат
    разомкнутыми — тоже: возвращать нечего, а рука поедет к слоту впустую.
    Проверка дешёвая, а ошибка стоит кубика и удара о стойку.
    """
    closed = grip_is_closed(osv)
    if closed is None:
        return ("состояние хвата не прочитано ($OUT[1]/$OUT[2]) — "
                "без него брать и возвращать нельзя")
    if case == CASE_TAKE and closed:
        return ("губки СОМКНУТЫ, а команда — взять. Если в них кубик, сперва "
                "верни его (--put N); если пусто, разожми "
                "(python tools/test_gripper.py --open)")
    if case == CASE_PUT and not closed:
        return "губки РАЗОМКНУТЫ, а команда — вернуть. Возвращать нечего"
    return None



def _send_once(osv, case: int, slot: int, timeout_s: float, pos_before):
    """Одна попытка. Возвращает ok / not_executed / fail."""
    osv.write("COM_VALUE1", str(int(slot)), False)
    for k in range(3):
        osv.write("COM_CASEVAR", str(case), False)
        t_ack = time.perf_counter()
        while time.perf_counter() - t_ack < 1.0:
            if (read_var(osv, "COM_CASEVAR") or "").strip() == str(case):
                break
        else:
            print(f"    [запись {k + 1}] контроллер не принял CASE {case}")
            continue
        break
    else:
        return "not_executed"
    t0 = time.perf_counter()
    # Часы стоят, пока стоит программа: в T1 оператор отпускает удержание
    # между движениями, и пауза не должна выглядеть как зависшая команда.
    deadline = t0 + timeout_s
    next_check = t0 + 5.0
    while time.perf_counter() < deadline:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            dt = time.perf_counter() - t0
            # Проверка ПО ФАКТУ: ноль не доказывает исполнения, dist3
            # обнуляет переменную при старте программы.
            moved = read_var(osv, "$POS_ACT") != pos_before
            if grip_is_closed(osv) is (case == CASE_TAKE) and moved:
                print(f"    выполнено за {dt:.1f} с")
                return "ok"
            return "not_executed"
        now = time.perf_counter()
        if now >= next_check:
            next_check = now + 5.0
            if read_var(osv, "$PRO_STATE") != "#P_ACTIVE":
                deadline = now + timeout_s
    print(f"    ТАЙМАУТ {timeout_s:.0f} с — команда не завершилась.")
    print("    Проверь: запущена ли dist3, держишь ли удержание, не ноль ли $OV_PRO.")
    return "fail"


def run_case(osv, case: int, slot: int, timeout_s: float) -> bool:
    """Команда магазина с повтором, если она не исполнилась.

    Перезапуск программы с пульта обнуляет COM_CASEVAR в начале dist3 и
    стирает уже отправленную команду. Проверка по факту это ловит, а
    повтор безопасен: CASE 20/21 ведут по АБСОЛЮТНОЙ траектории, второй
    запуск приводит туда же. Отказывать тут нечего — оператор переселектирует
    программу часто, и обрывать из-за этого цикл бессмысленно.
    """
    what = "взять из" if case == CASE_TAKE else "вернуть в"
    why = check_grip(osv, case)
    if why is not None:
        print(f"[СТОП] {why}")
        return False
    print("")
    print(f"[{case}] {what} слот {slot}. Робот поедет — держи кнопку останова.")
    for attempt in range(1, 4):
        res = _send_once(osv, case, slot, timeout_s, read_var(osv, "$POS_ACT"))
        if res == "ok":
            return True
        if res == "fail":
            return False
        print(f"    [попытка {attempt}] не исполнилась — похоже, программу "
              f"перезапустили. Отправляю снова.")
    print("    [ОШИБКА] три попытки подряд не исполнились.")
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
