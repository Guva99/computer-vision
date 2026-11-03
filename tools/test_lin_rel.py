"""
Проверка относительного линейного смещения (CASE 12) — поэтапно.

CASE 12 выполняет `LIN_REL COM_FRAME #BASE`: смещение на заданный вектор в
системе BASE. Нужен для подхода и отхода в техпроцессе — вынуть кубик из слота,
опустить на стопку, подняться.

ВАЖНО: COM_FRAME — полный кадр, у него есть углы. Скрипт всегда пишет
A = B = C = 0, иначе вместе со смещением произошёл бы поворот.

Этапы:
  1) --probe        ЧТЕНИЕ: поза, $BASE, $TOOL, режим. Движения нет.
  2) --dz 5 --yes   Реальное смещение на 5 мм (знак задаёт направление).
                    Только режим T1, только с явным подтверждением.

Запуск:
  python tools/test_lin_rel.py --probe
  python tools/test_lin_rel.py --dz 5 --yes      # вверх на 5 мм
  python tools/test_lin_rel.py --dz -5 --yes     # вниз на 5 мм
  python tools/test_lin_rel.py --dx 10 --yes     # вбок по X базы
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

CASE_LIN_REL = 12
MAX_MM = 50.0  # предохранитель: больше за одну пробу не двигаем


def read_var(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="ignore").strip() if raw is not None else None


def read_pos(osv):
    """$POS_ACT → dict X,Y,Z (мм) или None."""
    txt = read_var(osv, "$POS_ACT")
    if not txt:
        return None
    cleaned = txt.replace("{", " ").replace("}", " ").replace(",", " ").replace(":", " ")
    parts = cleaned.split()
    out = {}
    for key in ("X", "Y", "Z"):
        if key in parts:
            i = parts.index(key)
            if i + 1 < len(parts):
                try:
                    out[key] = float(parts[i + 1])
                except ValueError:
                    pass
    return out if len(out) == 3 else None


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Проверка CASE 12 (LIN_REL)")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--probe", action="store_true", help="только чтение")
    ap.add_argument("--dx", type=float, default=0.0, help="смещение по X базы, мм")
    ap.add_argument("--dy", type=float, default=0.0, help="смещение по Y базы, мм")
    ap.add_argument("--dz", type=float, default=0.0, help="смещение по Z базы, мм")
    ap.add_argument("--yes", action="store_true", help="подтвердить реальное движение")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    osv = OpenShowVar(ip=args.ip, port=args.port)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1

    mode = read_var(osv, "$MODE_OP")
    ov = read_var(osv, "$OV_PRO")
    case = read_var(osv, "COM_CASEVAR")
    pos0 = read_pos(osv)
    print(f"режим={mode}  скорость={ov}%  COM_CASEVAR={case}")
    print(f"$BASE = {read_var(osv, '$BASE')}")
    print(f"$TOOL = {read_var(osv, '$TOOL')}")
    print(f"поза X/Y/Z = {pos0}")

    if args.probe or (args.dx == args.dy == args.dz == 0.0):
        print("\n[PROBE] Только чтение. Дальше: --dz 5 --yes (смещение вверх на 5 мм).")
        return 0

    delta = (args.dx, args.dy, args.dz)
    if max(abs(v) for v in delta) > MAX_MM:
        print(f"[СТОП] Больше {MAX_MM:.0f} мм за пробу не двигаем.")
        return 1
    if not args.yes:
        print("[СТОП] Нужен --yes для реального движения.")
        return 1
    if mode != "#T1":
        print(f"[СТОП] Режим {mode}, ожидался #T1.")
        return 1

    frame = ("{{FRAME: X {:.3f}, Y {:.3f}, Z {:.3f}, "
             "A 0.0, B 0.0, C 0.0}}").format(*delta)
    print(f"\n[MOVE] LIN_REL {frame} в системе BASE. Держи кнопку останова под рукой.")
    osv.write("COM_FRAME", frame, False)
    osv.write("COM_CASEVAR", str(CASE_LIN_REL), False)

    t0 = time.time()
    accepted = False
    while time.time() - t0 < args.timeout:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            accepted = True
            break
    time.sleep(0.3)
    pos1 = read_pos(osv)
    print(f"после: поза X/Y/Z = {pos1}")
    if pos0 and pos1:
        moved = {k: round(pos1[k] - pos0[k], 2) for k in ("X", "Y", "Z")}
        print(f"[MOVE] принято={accepted}  фактическое смещение={moved}")
        want = {"X": args.dx, "Y": args.dy, "Z": args.dz}
        err = max(abs(moved[k] - want[k]) for k in want)
        if accepted and err <= 1.0:
            print(f"[OK] CASE 12 работает: расхождение с заданием {err:.2f} мм.")
        elif accepted:
            print(f"[!] Команда принята, но смещение не совпало (расхождение {err:.2f} мм).\n"
                  f"    Вероятно, $BASE повёрнут — тогда нужен вариант #TOOL в KRL.")
        else:
            print("[!] COM_CASEVAR не обнулился: обработчика CASE 12 нет или он не завершился.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
