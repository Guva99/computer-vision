"""
Сценарный раннер матрицы испытаний (валидация, Задача 4).

Читает scenarios.yaml, для каждого сценария × повтора:
  * печатает оператору описание и ждёт подтверждения (препятствие ставится
    вручную);
  * запускает прогон выбранного контура с включёнными логами валидации
    (Задачи 1-2), записью RGB-D (Задача 3) и проставленным scenario_id;
  * складывает артефакты в captures/scenarios/<scenario_id>/run_<k>/;
  * дописывает сводную строку в captures/scenarios_index.csv.

Контуры:
  3d — in-process (AppConfig + AppRunner), робот выполняет цикл движения;
  2d — subprocess `python main.py` в cv_2d_method/ с CLI-переопределениями.

Запуск (из корня репозитория):
  python tools/run_scenarios.py --contour 3d
  python tools/run_scenarios.py --contour 2d --scenarios scenarios.yaml
  python tools/run_scenarios.py --contour 3d --only S04_cube_on_path --repeats 3

Метрики по собранным логам считает tools/compute_metrics.py (без железа).
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_FIELDS = ["run_id", "scenario_id", "repeat", "contour", "started_at",
                "duration_s", "robot_ctrl_speed", "obstacle",
                "start_distance_m", "lighting", "occlusion", "danger_expected",
                "n_frames", "n_danger_frames", "n_stop_events",
                "recording_path", "perf_log", "objects_log", "stop_events_log",
                "status"]


def load_scenarios(path: Path) -> list:
    """scenarios.yaml → список сценариев с применёнными defaults."""
    try:
        import yaml
    except ImportError:
        print("[ERROR] Нужен PyYAML: pip install pyyaml")
        sys.exit(1)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {}) or {}
    scenarios = []
    for sc in data.get("scenarios", []) or []:
        merged = dict(defaults)
        merged.update(sc or {})
        if "scenario_id" not in merged:
            print(f"[WARN] Сценарий без scenario_id пропущен: {sc}")
            continue
        scenarios.append(merged)
    return scenarios


def count_csv_stats(perf_log: Path, stop_events_log: Path) -> tuple:
    """(n_frames, n_danger_frames, n_stop_events) по записанным логам."""
    n_frames = n_danger = n_stops = 0
    if perf_log.exists():
        with open(perf_log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n_frames += 1
                if row.get("collision_level") == "DANGER":
                    n_danger += 1
    if stop_events_log.exists():
        with open(stop_events_log, newline="", encoding="utf-8") as f:
            n_stops = sum(1 for _ in csv.DictReader(f))
    return n_frames, n_danger, n_stops


def append_index(index_path: Path, row: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not index_path.exists() or index_path.stat().st_size == 0
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_3d(sc: dict, run_dir: Path, no_robot: bool) -> str:
    """Прогон 3D-контура in-process. Возвращает status."""
    from realsense_io import AppConfig
    from app.runner import AppRunner

    cfg = AppConfig(
        scenario_id=sc["scenario_id"],
        enable_decision_log=True,
        enable_stop_log=True,
        enable_recording=True,
        recording_path=str(run_dir / "recording"),
        perf_log_path=str(run_dir / "perf_log.csv"),
        objects_csv_path=str(run_dir / "objects.csv"),
        stop_events_csv_path=str(run_dir / "stop_events.csv"),
        max_run_seconds=float(sc.get("duration_s", 60)),
        robot_ctrl_speed=int(sc.get("robot_ctrl_speed", 30)),
        enable_robot_control=not no_robot,
    )
    try:
        AppRunner(cfg).run()
        return "ok"
    except KeyboardInterrupt:
        return "interrupted"
    except Exception as e:
        print(f"[ERROR] Прогон упал: {e}")
        return f"error: {e}"


def run_2d(sc: dict, run_dir: Path, no_robot: bool) -> str:
    """Прогон 2D-контура subprocess'ом (cv_2d_method/main.py)."""
    cmd = [
        sys.executable, "main.py",
        "--scenario-id", sc["scenario_id"],
        "--duration-s", str(sc.get("duration_s", 60)),
        "--enable-decision-log",
        "--enable-stop-log",
        "--enable-recording",
        "--recording-path", str(run_dir / "recording"),
        "--perf-log", str(run_dir / "perf_log.csv"),
        "--objects-log", str(run_dir / "objects.csv"),
        "--stop-events-log", str(run_dir / "stop_events.csv"),
    ]
    if no_robot:
        cmd.append("--no-robot")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT / "cv_2d_method"))
        return "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
    except KeyboardInterrupt:
        return "interrupted"


def main():
    ap = argparse.ArgumentParser(description="Матрица испытаний SAFE/WARN/DANGER")
    ap.add_argument("--scenarios", default=str(ROOT / "scenarios.yaml"))
    ap.add_argument("--contour", choices=["3d", "2d"], default="3d")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Запустить только перечисленные scenario_id")
    ap.add_argument("--repeats", type=int, default=None,
                    help="Переопределить число повторов для всех сценариев")
    ap.add_argument("--out-dir", default=str(ROOT / "captures" / "scenarios"))
    ap.add_argument("--index", default=str(ROOT / "captures" / "scenarios_index.csv"))
    ap.add_argument("--no-robot", action="store_true",
                    help="Без управления роботом (только зрение) — для отладки")
    ap.add_argument("--yes", action="store_true",
                    help="Не ждать подтверждения оператора (осторожно!)")
    args = ap.parse_args()

    scenarios = load_scenarios(Path(args.scenarios))
    if args.only:
        scenarios = [s for s in scenarios if s["scenario_id"] in set(args.only)]
    if not scenarios:
        print("[ERROR] Нет сценариев для запуска.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    index_path = Path(args.index)
    print(f"Сценариев: {len(scenarios)}  контур: {args.contour}  "
          f"индекс: {index_path}")

    for sc in scenarios:
        sid = sc["scenario_id"]
        repeats = args.repeats or int(sc.get("repeats", 1))
        for rep in range(1, repeats + 1):
            print("\n" + "=" * 70)
            print(f"СЦЕНАРИЙ {sid}  повтор {rep}/{repeats}  контур {args.contour}")
            print(f"  {sc.get('description', '')}")
            print(f"  препятствие: {sc.get('obstacle')}  "
                  f"дистанция: {sc.get('start_distance_m')}  "
                  f"скорость: {sc.get('robot_ctrl_speed')}%  "
                  f"свет: {sc.get('lighting')}  перекрытие: {sc.get('occlusion')}")
            print("=" * 70)
            if not args.yes:
                ans = input("Установи препятствие. Enter=старт, s=пропустить, "
                            "q=выход: ").strip().lower()
                if ans == "q":
                    print("Выход.")
                    return
                if ans == "s":
                    continue

            run_dir = out_dir / sid / f"run_{rep:02d}_{args.contour}"
            run_dir.mkdir(parents=True, exist_ok=True)
            started = datetime.now().isoformat(timespec="seconds")
            t0 = time.time()
            if args.contour == "3d":
                status = run_3d(sc, run_dir, args.no_robot)
            else:
                status = run_2d(sc, run_dir, args.no_robot)
            dur = time.time() - t0

            n_frames, n_danger, n_stops = count_csv_stats(
                run_dir / "perf_log.csv", run_dir / "stop_events.csv"
            )
            append_index(index_path, {
                "run_id": f"{sid}_r{rep:02d}_{args.contour}",
                "scenario_id": sid,
                "repeat": rep,
                "contour": args.contour.upper(),
                "started_at": started,
                "duration_s": round(dur, 1),
                "robot_ctrl_speed": sc.get("robot_ctrl_speed"),
                "obstacle": sc.get("obstacle"),
                "start_distance_m": sc.get("start_distance_m"),
                "lighting": sc.get("lighting"),
                "occlusion": sc.get("occlusion"),
                "danger_expected": sc.get("danger_expected"),
                "n_frames": n_frames,
                "n_danger_frames": n_danger,
                "n_stop_events": n_stops,
                "recording_path": str(run_dir / "recording"),
                "perf_log": str(run_dir / "perf_log.csv"),
                "objects_log": str(run_dir / "objects.csv"),
                "stop_events_log": str(run_dir / "stop_events.csv"),
                "status": status,
            })
            print(f"[INDEX] {sid} r{rep}: frames={n_frames} danger={n_danger} "
                  f"stops={n_stops} status={status}")
            if status == "interrupted":
                print("Прервано оператором — выход.")
                return

    print(f"\nГотово. Сводка: {index_path}")


if __name__ == "__main__":
    main()
