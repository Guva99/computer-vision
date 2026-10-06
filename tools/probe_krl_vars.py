"""
Разведка переменных KRL на контроллере KUKA (ТОЛЬКО ЧТЕНИЕ, движения нет).

Нужна перед переходом на осевое (E6AXIS) управление: показывает, какие
переменные уже объявлены в программе контроллера, есть ли готовая E6AXIS-цель
и какие номера COM_CASEVAR заняты. По результату решаем, что дописывать в KRL.

Запуск:  python tools/probe_krl_vars.py [--ip 192.168.17.2] [--port 7000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.drivers.openshowvar import OpenShowVar

# Кандидаты: системные (должны быть всегда) + контрактные COM_* (наша KRL) +
# возможные готовые осевые цели под разными типовыми именами.
CANDIDATES = [
    # системные — проверка, что канал вообще живой
    "$AXIS_ACT", "$POS_ACT", "$OV_PRO", "$MODE_OP",
    # контракт текущей KRL (используются в kuka_driver.py)
    "COM_CASEVAR", "COM_FRAME", "COM_LENGTH", "COM_VALUE1",
    "COM_FRAME_ARRAY[0]",
    # возможные готовые ОСЕВЫЕ цели (главное, что ищем)
    "COM_E6AXIS", "COM_AXIS", "COM_A6AXIS", "COM_POS_AXIS",
    "COM_AXIS_ARRAY[0]", "COM_E6AXIS_ARRAY[0]",
    # прочее, что встречается в таких программах
    "COM_ACTION", "COM_ROBCOR", "COM_KEY",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    args = ap.parse_args()

    print(f"Подключение к {args.ip}:{args.port} ...")
    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=2.0)
    if not osv.can_connect:
        print("[ERROR] Контроллер недоступен. Проверь питание, сеть и KUKAVARPROXY.")
        return 1
    print("[OK] Соединение установлено. Только чтение — робот не двинется.\n")

    found, missing = [], []
    for name in CANDIDATES:
        try:
            raw = osv.read(name, False)
            val = raw.decode(errors="replace").strip() if raw is not None else None
        except Exception as e:
            val = None
            print(f"  {name:<24} [ошибка чтения: {e}]")
            continue
        if val:
            found.append((name, val))
            print(f"  {name:<24} = {val[:96]}")
        else:
            missing.append(name)

    print("\n" + "=" * 62)
    print("ЕСТЬ:", ", ".join(n for n, _ in found) or "(ничего)")
    print("НЕТ :", ", ".join(missing) or "(все найдены)")

    axis_vars = [n for n, _ in found
                 if "AXIS" in n.upper() and not n.startswith("$")]
    print("=" * 62)
    if axis_vars:
        print(f"[!] Осевая цель УЖЕ объявлена: {', '.join(axis_vars)}")
        print("    Возможно, в KRL есть и готовый CASE — проверь программу.")
    else:
        print("[!] Готовой осевой цели (E6AXIS) не найдено — её нужно добавить")
        print("    в KRL: объявление переменной + новый CASE с PTP по осям.")
    try:
        osv.sock.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
