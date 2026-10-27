"""
Регрессия поведения детекции на записях (оракул рефакторинга).

Зачем. У проекта нет юнит-покрытия логики восприятия, зато есть 41 запись и
детерминированный playback: один и тот же вход даёт побитово один и тот же
decisions.csv (проверено 2026-10-20 — реплей совпал с прогоном на стенде
177/177 строк). Значит любой рефакторинг, не меняющий алгоритм, обязан оставить
решения теми же, и это проверяется автоматически.

Сравниваются ТОЛЬКО решения: frame, collision_level, min_dist_m, focus_part,
n_objects. Колонки `t_wall` и `fps` зависят от машины и загрузки, в сравнение не
входят.

    python tools/regress.py                      # весь golden-набор
    python tools/regress.py --fast               # быстрый набор (для pytest)
    python tools/regress.py --update             # перезаписать golden (только на зелёном HEAD!)
    python tools/regress.py --compare collision_object_permanence=False
                                                 # абляция: набор с другим параметром против golden

Набор записей объявлен в tests/golden/manifest.json. Каждый реплей идёт
ОТДЕЛЬНЫМ процессом: в collision.py есть модульное глобальное состояние
(счётчики диагностики, буфер кандидатов), и прогоны в одном процессе влияли бы
друг на друга. После шага S2.5 это ограничение уйдёт, но изоляция процессов
останется честнее.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.console import setup_console  # noqa: E402

MANIFEST = ROOT / "tests" / "golden" / "manifest.json"
KEY_COLUMNS = ("collision_level", "min_dist_m", "focus_part", "n_objects")


def load_manifest(fast_only: bool = False) -> list:
    if not MANIFEST.exists():
        raise SystemExit(f"Нет манифеста golden-набора: {MANIFEST}")
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["recordings"]
    if fast_only:
        items = [it for it in items if it.get("fast")]
    return items


def read_decisions(path: Path) -> dict:
    """decisions.csv → {номер кадра: (level, dist, part, n_objects)}."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                frame = int(r["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[frame] = tuple(str(r.get(c, "")) for c in KEY_COLUMNS)
    return rows


def summarize(rows: dict) -> dict:
    """Сводка прогона: распределение уровней и кадр первого DANGER."""
    levels = {}
    first_danger = None
    for frame in sorted(rows):
        lvl = rows[frame][0]
        levels[lvl] = levels.get(lvl, 0) + 1
        if lvl == "DANGER" and first_danger is None:
            first_danger = frame
    return {"frames": len(rows), "levels": levels, "first_danger": first_danger}


def compare(golden: dict, actual: dict) -> dict:
    """Сравнить два прогона по кадрам. Возвращает отчёт с первыми различиями."""
    missing = sorted(set(golden) - set(actual))
    extra = sorted(set(actual) - set(golden))
    diffs = []
    for frame in sorted(set(golden) & set(actual)):
        if golden[frame] != actual[frame]:
            diffs.append((frame, golden[frame], actual[frame]))
    return {
        "common": len(set(golden) & set(actual)),
        "missing": missing,
        "extra": extra,
        "diffs": diffs,
        "identical": not (missing or extra or diffs),
    }


def run_replay(recording: Path, out_csv: Path, scenario_id: str,
               overrides: list | None = None) -> float:
    """Прогнать одну запись отдельным процессом. Возвращает время прогона, с."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()
    cmd = [sys.executable, str(ROOT / "tools" / "replay.py"),
           "--recording", str(recording), "--out", str(out_csv),
           "--scenario-id", scenario_id]
    if overrides:
        cmd += ["--set", *overrides]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_csv.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-6:])
        err = "\n".join((proc.stderr or "").splitlines()[-6:])
        raise RuntimeError(f"Реплей упал ({recording.name}):\n{tail}\n{err}")
    return dt


def fmt_levels(levels: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(levels.items()))


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Регрессия решений на записях")
    ap.add_argument("--fast", action="store_true", help="только быстрый набор")
    ap.add_argument("--update", action="store_true", help="перезаписать golden-файлы")
    ap.add_argument("--compare", nargs="*", default=None, metavar="ИМЯ=ЗНАЧЕНИЕ",
                    help="прогнать с переопределениями и сравнить с golden (абляция)")
    ap.add_argument("--out-dir", default=None,
                    help="куда писать прогоны (по умолчанию каталог во временных файлах)")
    args = ap.parse_args()

    items = load_manifest(fast_only=args.fast)
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "captures" / "_regress")
    mode = "ОБНОВЛЕНИЕ golden" if args.update else (
        "АБЛЯЦИЯ: " + " ".join(args.compare) if args.compare else "РЕГРЕССИЯ")
    print(f"[REGRESS] {mode}; записей: {len(items)}")

    failures, total_time = 0, 0.0
    for it in items:
        rec = ROOT / it["recording"]
        golden_path = ROOT / "tests" / "golden" / it["golden"]
        sid = it["scenario_id"]
        if not (rec / "meta.json").exists():
            print(f"  ПРОПУСК {sid}: нет записи {rec}")
            continue

        target = golden_path if args.update else (out_dir / f"{golden_path.stem}.csv")
        dt = run_replay(rec, target, sid, args.compare)
        total_time += dt

        if args.update:
            print(f"  записан golden {golden_path.name}  ({dt:.1f} с)")
            continue

        if not golden_path.exists():
            print(f"  ПРОПУСК {sid}: нет golden {golden_path.name} (запусти --update)")
            continue

        rep = compare(read_decisions(golden_path), read_decisions(target))
        s_new = summarize(read_decisions(target))
        if rep["identical"]:
            print(f"  OK  {sid:28s} {s_new['frames']:4d} кадров  {fmt_levels(s_new['levels'])}  ({dt:.1f} с)")
            continue

        failures += 1
        verdict = "РАСХОЖДЕНИЕ" if not args.compare else "ОТЛИЧАЕТСЯ"
        s_old = summarize(read_decisions(golden_path))
        print(f"  {verdict} {sid}: кадров {len(rep['diffs'])} из {rep['common']}"
              f"{', нет в прогоне: %d' % len(rep['missing']) if rep['missing'] else ''}"
              f"{', лишних: %d' % len(rep['extra']) if rep['extra'] else ''}")
        print(f"      golden : {fmt_levels(s_old['levels'])}  первый DANGER: {s_old['first_danger']}")
        print(f"      прогон : {fmt_levels(s_new['levels'])}  первый DANGER: {s_new['first_danger']}")
        for frame, g, a in rep["diffs"][:5]:
            print(f"      кадр {frame}: golden {g} → прогон {a}")

    if args.update:
        print(f"[REGRESS] golden обновлён за {total_time:.0f} с")
        return 0
    if args.compare:
        print(f"[REGRESS] абляция: отличий в {failures} записях из {len(items)}; {total_time:.0f} с")
        return 0  # отличия тут — результат, а не провал
    if failures:
        print(f"[REGRESS] ПРОВАЛ: расхождения в {failures} записях из {len(items)}")
        return 1
    print(f"[REGRESS] ВСЁ СОВПАДАЕТ: {len(items)} записей, {total_time:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
