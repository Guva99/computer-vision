"""
Проверка осевого (E6AXIS) канала движения — поэтапно, от безопасного к реальному.

Зачем: декартов PTP шлёт FRAME без битов Status/Turn, конфигурацию суставов
выбирает обратная кинематика контроллера, и при A4 ≈ 168.7° она уходила за
границу оборота ±180° → останов KSS01211. Осевая команда убирает пересчёт.

Этапы:
  1) --probe   ЧТЕНИЕ: углы, режим, COM_CASEVAR. Движения нет.
  2) --null    Послать цель = ТЕКУЩИЕ углы. Робот физически не двигается, но
               видно, принимает ли KRL команду (COM_CASEVAR обнуляется).
  3) --move N  Реальное движение: A1 сдвигается на N градусов (по модулю <= 10).
               Только режим T1, только с явным подтверждением.

Запуск:
  python tools/test_axis_motion.py --probe
  python tools/test_axis_motion.py --null --casevar 8
  python tools/test_axis_motion.py --move 5 --casevar 8 --yes
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.drivers.kuka_driver import KukaDriver
from robot.drivers.openshowvar import OpenShowVar


def read_angles(osv):
    """$AXIS_ACT → [A1..A6] градусы, или None."""
    raw = osv.read("$AXIS_ACT", False)
    if raw is None:
        return None
    txt = raw.decode(errors="ignore")
    cleaned = txt.replace("{", " ").replace("}", " ").replace(",", " ").replace(":", " ")
    parts = cleaned.split()
    out = []
    for ax in ("A1", "A2", "A3", "A4", "A5", "A6"):
        if ax in parts:
            i = parts.index(ax)
            if i + 1 < len(parts):
                try:
                    out.append(float(parts[i + 1]))
                except ValueError:
                    pass
    return out if len(out) == 6 else None


def read_str(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="replace").strip() if raw is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--casevar", type=int, default=8,
                    help="номер CASE в KRL, выполняющий PTP COM_E6AXIS")
    ap.add_argument("--probe", action="store_true", help="только чтение")
    ap.add_argument("--null", action="store_true",
                    help="цель = текущие углы (движения быть не должно)")
    ap.add_argument("--move", type=float, default=None,
                    help="сдвиг A1 в градусах (|N| <= 10)")
    ap.add_argument("--yes", action="store_true", help="подтвердить движение")
    args = ap.parse_args()

    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print("[ERROR] Контроллер недоступен.")
        return 1

    angles = read_angles(osv)
    mode = read_str(osv, "$MODE_OP")
    ov = read_str(osv, "$OV_PRO")
    case_now = read_str(osv, "COM_CASEVAR")
    print(f"режим={mode}  скорость={ov}%  COM_CASEVAR={case_now}")
    if angles is None:
        print("[ERROR] Не удалось прочитать $AXIS_ACT.")
        return 1
    print("углы A1..A6: " + "  ".join(f"{v:+8.2f}" for v in angles))
    print(f"COM_E6AXIS  = {read_str(osv, 'COM_E6AXIS')}")

    if args.probe or (not args.null and args.move is None):
        print("\n[PROBE] Только чтение. Дальше: --null (проверка канала), "
              "потом --move N --yes (реальное движение).")
        return 0

    # Безопасность: движение только в T1 (ручной режим, ограниченная скорость).
    if mode and "T1" not in str(mode):
        print(f"[STOP] Режим {mode}, а нужен T1. Переключи на пульте.")
        return 1

    drv = KukaDriver(osv)

    if args.null:
        print(f"\n[NULL] Цель = текущие углы, CASE={args.casevar}. "
              f"Робот двигаться НЕ должен.")
        ok = drv.ptp_axis(angles, casevar=args.casevar, timeout_s=15.0)
        after = read_angles(osv)
        moved = max(abs(a - b) for a, b in zip(angles, after)) if after else -1.0
        print(f"[NULL] принято={ok}  макс. смещение оси={moved:.3f}°")
        if ok and moved < 0.5:
            print("[OK] Канал работает: команда принята, движения нет. "
                  "Номер CASE верный.")
        elif not ok:
            print("[FAIL] COM_CASEVAR не обнулился — вероятно, неверный CASE "
                  "или в KRL нет обработчика PTP COM_E6AXIS.")
        return 0 if ok else 1

    if args.move is not None:
        if abs(args.move) > 10.0:
            print("[STOP] |--move| должно быть <= 10 градусов.")
            return 1
        if not args.yes:
            print("[STOP] Реальное движение требует флага --yes.")
            return 1
        target = list(angles)
        target[0] = angles[0] + float(args.move)
        print(f"\n[MOVE] A1: {angles[0]:+.2f} → {target[0]:+.2f}, "
              f"остальные оси без изменений. Держи кнопку останова под рукой.")
        time.sleep(1.0)
        ok = drv.ptp_axis(target, casevar=args.casevar, timeout_s=30.0)
        after = read_angles(osv)
        if after:
            print("после: " + "  ".join(f"{v:+8.2f}" for v in after))
            d_other = max(abs(after[i] - angles[i]) for i in range(1, 6))
            print(f"[MOVE] принято={ok}  A1 факт={after[0]:+.2f}  "
                  f"макс. уход осей A2..A6={d_other:.2f}°")
            if d_other < 1.0:
                print("[OK] Оси A2..A6 остались на месте — осевой режим работает, "
                      "A4 больше не пересчитывается кинематикой.")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
