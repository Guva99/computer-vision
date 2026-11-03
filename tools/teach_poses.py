"""
Снятие опорных поз робота с пульта в файл.

Как работает: подводишь робота руками к очередной позе, жмёшь Enter — скрипт
читает `$AXIS_ACT` (углы) и `$POS_ACT` (декартову позу) и складывает их в YAML.
Ничего не двигает сам, только читает.

Зачем именно так. Точность техпроцесса держится на позах, заученных на пульте, а
не на вычислениях: DH-модель врёт (вылет хвата в конфиге 137 мм против ~88 мм по
подбору), внешняя калибровка даёт свои миллиметры. Записанные углы от этого не
зависят вовсе — робот возвращается туда же, откуда их сняли.

Промежуточные позиции НЕ заучиваются: слоты магазина идут с постоянным шагом,
уровни башни — на высоту кубика. Их считает автомат процесса и отрабатывает
относительным смещением (CASE 12), поэтому поз нужно всего пять.

    python tools/teach_poses.py                       # весь список
    python tools/teach_poses.py --only home build_base
    python tools/teach_poses.py --out task/poses.yaml

Робота двигать в режиме T1 с удержанием. Программа dist2 для снятия поз не
нужна — переменные читаются и при остановленной программе.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

# Список опорных поз. Описание печатается оператору перед снятием.
POSES = [
    ("home", "стартовая поза: рука отведена, зона сборки свободна"),
    ("transport", "безопасная высота переноса между магазином и зоной сборки"),
    ("magazine_slot1", "губки напротив ВЕРХНЕГО слота магазина, готовы сомкнуться"),
    ("magazine_slot6", "губки напротив НИЖНЕГО слота магазина"),
    ("build_base", "губки над местом сборки, кубик на уровне 1 (на поверхности)"),
]


def read_var(osv, name, tries: int = 3):
    for _ in range(tries):
        raw = osv.read(name, False)
        if raw is not None:
            return raw.decode(errors="ignore").strip()
    return None


def parse_axis(txt):
    """{E6AXIS: A1 ..., A6 ...} → [A1..A6] или None."""
    if not txt:
        return None
    out = []
    for ax in ("A1", "A2", "A3", "A4", "A5", "A6"):
        m = re.search(rf"\b{ax}\s+(-?[0-9.]+)", txt)
        if not m:
            return None
        out.append(round(float(m.group(1)), 4))
    return out


def parse_pos(txt):
    """{E6POS: X ..., C ..., S .., T ..} → dict или None."""
    if not txt:
        return None
    out = {}
    for key in ("X", "Y", "Z", "A", "B", "C"):
        m = re.search(rf"\b{key}\s+(-?[0-9.]+)", txt)
        if not m:
            return None
        out[key] = round(float(m.group(1)), 3)
    for key in ("S", "T"):
        m = re.search(rf"\b{key}\s+(-?[0-9]+)", txt)
        if m:
            out[key] = int(m.group(1))
    return out


def capture(osv, name: str, hint: str):
    """Снять одну позу. Возвращает dict или None, если оператор пропустил."""
    print(f"\n── {name} ──  {hint}")
    ans = input("   Подведи робота и нажми Enter (s = пропустить, q = закончить): ")
    low = ans.strip().lower()
    if low == "q":
        return "quit"
    if low == "s":
        print("   пропущено")
        return None
    axis = parse_axis(read_var(osv, "$AXIS_ACT"))
    pos = parse_pos(read_var(osv, "$POS_ACT"))
    if axis is None:
        print("   [ОШИБКА] углы не прочитаны — повтори")
        return None
    print(f"   углы   A1..A6 = {axis}")
    if pos:
        print(f"   декарт X={pos['X']:.1f} Y={pos['Y']:.1f} Z={pos['Z']:.1f} "
              f"A={pos['A']:.1f} B={pos['B']:.1f} C={pos['C']:.1f}"
              + (f"  S={pos.get('S')} T={pos.get('T')}" if "S" in pos else ""))
    return {"axis": axis, "cartesian": pos, "captured": datetime.now().isoformat(timespec="seconds")}


def dump_yaml(data: dict, path: Path) -> None:
    """Пишем YAML вручную: pyyaml не обязателен, а формат тут простой."""
    lines = ["# Опорные позы робота, снятые с пульта (tools/teach_poses.py).",
             "# axis — углы A1..A6 в градусах, по ним идёт движение (CASE 11).",
             "# cartesian — поза фланца для справки и проверки.",
             "# Промежуточные позиции не заучиваются: слоты магазина и уровни",
             "# башни считаются смещением (CASE 12).",
             ""]
    meta = data.get("meta", {})
    lines.append("meta:")
    for k, v in meta.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("poses:")
    for name, p in data["poses"].items():
        lines.append(f"  {name}:")
        lines.append(f"    axis: [{', '.join(f'{v:.4f}' for v in p['axis'])}]")
        if p.get("cartesian"):
            c = p["cartesian"]
            inner = ", ".join(f"{k}: {v}" for k, v in c.items())
            lines.append(f"    cartesian: {{{inner}}}")
        lines.append(f"    captured: {p['captured']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Снятие опорных поз робота в YAML")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--out", default="task/poses.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="снять только эти позы")
    ap.add_argument("--slot-pitch-mm", type=float, default=70.0,
                    help="шаг между слотами магазина (замерено линейкой)")
    ap.add_argument("--cube-mm", type=float, default=0.0,
                    help="ребро кубика, мм (0 = ещё не замерено)")
    return ap.parse_args()


def main() -> int:
    setup_console()
    args = parse_args()
    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=3.0)
    if not osv.can_connect:
        print(f"[ОШИБКА] Нет связи с {args.ip}:{args.port}")
        return 1
    print(f"режим={read_var(osv, '$MODE_OP')}  связь есть. Робот не двигается — только чтение.")

    todo = [(n, h) for n, h in POSES if not args.only or n in args.only]
    captured = {}
    for name, hint in todo:
        res = capture(osv, name, hint)
        if res == "quit":
            break
        if res:
            captured[name] = res

    if not captured:
        print("\nНичего не снято.")
        return 0

    out = Path(args.out)
    data = {
        "meta": {
            "slot_pitch_mm": args.slot_pitch_mm,
            "cube_edge_mm": args.cube_mm,
            "robot": "KUKA KR4 R600",
            "base": 0, "tool": 0,
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        "poses": captured,
    }
    if out.exists():
        import shutil
        shutil.copy(out, out.with_suffix(".yaml.bak"))
        print(f"\nпрежний файл сохранён как {out.with_suffix('.yaml.bak').name}")
    dump_yaml(data, out)
    print(f"\n[OK] Снято поз: {len(captured)} → {out}")
    for n in captured:
        print(f"   {n}")
    missing = [n for n, _ in POSES if n not in captured]
    if missing:
        print(f"   не снято: {', '.join(missing)} — допиши через --only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
