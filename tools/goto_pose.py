"""
Переход в заучёную позу по УГЛАМ и команды хвата.

Почему по углам. Заученная поза хранит `A1..A6`, снятые с контроллера. Отправка
их обратно через `CASE 11` (`PTP COM_E6AXIS`) воспроизводит ту же конфигурацию
суставов, что и была. Декартов PTP этого не гарантирует: выбор конфигурации
остаётся за обратной кинематикой контроллера, и на стенде это уже приводило к
прокрутке A4 и отказу KSS01211 (см. `krl/README.md`).

Камера в цепочке не участвует вовсе: поза снята касанием, движение идёт по
углам. Ни внешняя калибровка, ни плоскость стола сюда не входят.

    python tools/goto_pose.py --list
    python tools/goto_pose.py --pose transport --yes
    python tools/goto_pose.py --pose build_base --yes
    python tools/goto_pose.py --grip open
    python tools/goto_pose.py --pose build_base --dz 40 --yes   # уровень 2

`--dz` поднимает цель по вертикали относительно заученной позы и потому идёт
не углами, а декартовым `LIN` от неё: уровни башни — это та же точка плюс
ребро кубика. Губки при этом смотрят вниз, поэтому смещение фланца по Z
переносится в высоту кубика один к одному.

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
from robot.services.dispatcher_link import frame_str  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

CASE_SET_TOOL = 2
CASE_GRIPPER = 7
CASE_LIN_ARRAY = 4
CASE_PTP_AXIS = 11
ZERO_FRAME = "{FRAME: X 0.0, Y 0.0, Z 0.0, A 0.0, B 0.0, C 0.0}"


def read_var(osv, name):
    raw = osv.read(name, False)
    return raw.decode(errors="ignore").strip() if raw is not None else None


def load_poses(path: Path) -> dict:
    import yaml
    if not path.exists():
        print(f"[ОШИБКА] Нет файла поз {path} — сними их tools/teach_poses.py")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("poses") or {}


def axis_str(a) -> str:
    """Список углов → E6AXIS для KRL."""
    names = ("A1", "A2", "A3", "A4", "A5", "A6")
    body = ", ".join(f"{n} {v:.4f}" for n, v in zip(names, a))
    return "{E6AXIS: " + body + ", E1 0.0, E2 0.0, E3 0.0, E4 0.0, E5 0.0, E6 0.0}"


def wait_done(osv, timeout_s: float):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = read_var(osv, "COM_CASEVAR")
        if r is not None and r.strip() in ("0", "0.0"):
            return time.perf_counter() - t0
    return None


def pose_xyz(osv):
    """XYZ из $POS_ACT или None."""
    txt = read_var(osv, "$POS_ACT")
    if not txt or "X " not in txt:
        return None
    try:
        return [float(txt.split(f"{ax} ")[1].split(",")[0]) for ax in ("X", "Y", "Z")]
    except (IndexError, ValueError):
        return None


def run_case(osv, case: int, timeout_s: float) -> bool:
    """Запустить команду диспетчера и дождаться её ЗАВЕРШЕНИЯ, а не нуля.

    Запись в KUKAVARPROXY может потеряться. Тогда COM_CASEVAR так и остаётся
    нулём, и наивное ожидание «пока не ноль» рапортует успех, хотя команда не
    начиналась — ровно так 2026-11-06 подъём на 100 мм был принят за
    выполненный, а рука не двинулась. Поэтому сперва ждём, пока переменная
    ПРИМЕТ наш номер, и при неудаче пишем ещё раз. Повтор безопасен: CASE 4 и
    CASE 11 ведут в АБСОЛЮТНУЮ точку, второй запуск туда же ничего не меняет.
    """
    for attempt in range(3):
        osv.write("COM_CASEVAR", str(case), False)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            if (read_var(osv, "COM_CASEVAR") or "").strip() == str(case):
                break
        else:
            if attempt < 2:
                print(f"    [повтор] контроллер не принял CASE {case}")
            continue
        dt = wait_done(osv, timeout_s)
        if dt is None:
            print("    ТАЙМАУТ — смотри сообщение на пульте")
            osv.write("COM_CASEVAR", "0", False)
            return False
        print(f"    отработано за {dt:.1f} с")
        return True
    print(f"    [ОШИБКА] контроллер так и не принял CASE {case} за 3 попытки")
    return False


def reached(osv, target, tol_mm: float = 1.0) -> bool:
    """Проверка по факту: рука действительно в целевой точке."""
    cur = pose_xyz(osv)
    if cur is None:
        print("    [ВНИМАНИЕ] позу не прочитать — результат не проверен")
        return True
    d = max(abs(c - t) for c, t in zip(cur, target))
    if d > tol_mm:
        print(f"    [ОШИБКА] НЕ ДОЕХАЛ: расхождение {d:.1f} мм "
              f"(сейчас X{cur[0]:.1f} Y{cur[1]:.1f} Z{cur[2]:.1f})")
        return False
    return True



def preflight(osv, hold_wait_s: float = 300.0) -> str | None:
    mode = read_var(osv, "$MODE_OP")
    if mode != "#T1":
        return f"режим {mode}, ожидался #T1"
    state = read_var(osv, "$PRO_STATE")
    if state != "#P_ACTIVE":
        # Удержание в T1 отпускают после каждого движения, и программа
        # встаёт. Отказывать на этом бессмысленно — ждём, когда возьмут.
        print("")
        print(">>> ВОЗЬМИ УДЕРЖАНИЕ / ЗАПУСТИ dist3 — программа стоит. Жду...")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < hold_wait_s:
            if read_var(osv, "$PRO_STATE") == "#P_ACTIVE":
                print(f">>> пошла, ждал {time.perf_counter() - t0:.0f} с")
                break
        else:
            return "удержания так и не было"
    if (read_var(osv, "COM_CASEVAR") or "").strip() not in ("0", "0.0"):
        return "предыдущая команда не завершена"
    tool = read_var(osv, "$TOOL")
    if tool and tool.replace(" ", "") != ZERO_FRAME.replace(" ", ""):
        return f"$TOOL не нулевой ({tool}) — позы сняты для фланца"
    return None


def send_axis(osv, angles, timeout_s: float) -> bool:
    """PTP в углы через CASE 11."""
    s = axis_str(angles)
    print(f"[PTP углы] -> {s}")
    osv.write("COM_E6AXIS", s, False)
    return run_case(osv, CASE_PTP_AXIS, timeout_s)


def send_lin(osv, xyz, abc, timeout_s: float) -> bool:
    """LIN в декартову точку через массив (две строки, исполняется вторая)."""
    f = frame_str(xyz, abc)
    print(f"[LIN] -> {f}")
    osv.write("COM_FRAME_ARRAY[0]", f, False)
    osv.write("COM_FRAME_ARRAY[1]", f, False)
    osv.write("COM_LENGTH", "1", False)
    if not run_case(osv, CASE_LIN_ARRAY, timeout_s):
        return False
    return reached(osv, xyz)


def send_grip(osv, close: bool) -> bool:
    """Хват через CASE 7. Проверка ПО ВЫХОДАМ, а не по эху команды.

    CASE 7 отрабатывает за единицы миллисекунд: контроллер успевает
    обнулить COM_CASEVAR раньше, чем мы её прочитаем, и команда выглядит
    непринятой, хотя исполнена. 2026-11-06 это дало ложный отказ и три
    лишних отправки. Единственный надёжный признак — сами $OUT.
    """
    want = ("TRUE", "FALSE") if close else ("FALSE", "TRUE")
    osv.write("OUT5", "TRUE" if close else "FALSE", False)
    osv.write("OUT6", "FALSE" if close else "TRUE", False)
    t0 = time.perf_counter()
    for _ in range(3):
        osv.write("COM_CASEVAR", str(CASE_GRIPPER), False)
        while time.perf_counter() - t0 < 5.0:
            got = (read_var(osv, "$OUT[1]"), read_var(osv, "$OUT[2]"))
            if got == want:
                word = "сжать" if close else "разжать"
                print(f"{word}: "
                      f"$OUT[1]={got[0]} $OUT[2]={got[1]}  "
                      f"за {(time.perf_counter() - t0) * 1000:.0f} мс")
                return True
            if time.perf_counter() - t0 > 1.5:
                break
    print(f"    [ОШИБКА] хват не перешёл в состояние "
          f"{'сжат' if close else 'разжат'} — выходы не изменились")
    return False


def do_pose(osv, poses: dict, args) -> int:
    """Переход в позу.

    Без `--dz` — PTP по углам: точное возвращение в снятую конфигурацию.
    С `--dz` — прямой LIN в смещённую по вертикали точку, БЕЗ захода в саму
    позу. Это важно для уровней башни: зайти сначала в `build_base`, а потом
    подняться, значило бы опустить кубик на высоту уровня 1 — туда, где уже
    стоит предыдущий кубик.
    """
    p = poses.get(args.pose)
    if p is None:
        print(f"[ОШИБКА] Позы {args.pose!r} нет. Есть: {chr(44).join(poses) or chr(45)}")
        return 1
    if not args.yes:
        print("[СТОП] Нужен --yes для движения.")
        return 1
    if abs(args.dz) < 1e-6:
        return 0 if send_axis(osv, p["axis"], args.timeout) else 1
    c = p.get("cartesian")
    if not c:
        print("[ОШИБКА] У позы нет декартовых координат — --dz невозможен")
        return 1
    xyz = (c["X"], c["Y"], c["Z"] + args.dz)
    abc = (c["A"], c["B"], c["C"])
    print(f"LIN в {args.pose} со сдвигом {args.dz:+.1f} мм по Z "
          f"(без захода в саму позу)")
    return 0 if send_lin(osv, xyz, abc, args.timeout) else 1


def parse_args():
    ap = argparse.ArgumentParser(description="Переход в заучёную позу по углам")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--poses", default="task/poses.yaml")
    ap.add_argument("--list", action="store_true", help="показать позы и выйти")
    ap.add_argument("--pose", default="", help="имя позы из файла")
    ap.add_argument("--dz", type=float, default=0.0,
                    help="сдвиг по Z от заученной позы, мм (уровни башни)")
    ap.add_argument("--grip", choices=("open", "close"), default="",
                    help="хват: без движения манипулятора")
    ap.add_argument("--yes", action="store_true", help="подтвердить движение")
    ap.add_argument("--timeout", type=float, default=120.0)
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    poses = load_poses(ROOT / args.poses)
    if args.list or not (args.pose or args.grip):
        for n, p in poses.items():
            c = p.get("cartesian") or {}
            print(f"{n:14} углы {p['axis']}")
            if c:
                print(f"{'':14} фланец X={c['X']} Y={c['Y']} Z={c['Z']} "
                      f"A={c['A']} B={c['B']} C={c['C']}")
        return 0

    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1
    why = preflight(osv)
    if why is not None:
        print(f"[СТОП] {why}")
        return 1
    print(f"режим={read_var(osv, '$MODE_OP')}  скорость={read_var(osv, '$OV_PRO')}%")
    print(f"позиция: {read_var(osv, '$POS_ACT')}")

    rc = 0
    if args.grip:
        rc = 0 if send_grip(osv, args.grip == "close") else 1
    if rc == 0 and args.pose:
        rc = do_pose(osv, poses, args)
    print(f"\nпозиция: {read_var(osv, '$POS_ACT')}")
    osv.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
