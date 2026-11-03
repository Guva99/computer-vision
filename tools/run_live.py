"""
Запуск на стенде: камера + детекция, с роботом или без.

Зачем отдельный лаунчер. `python main.py` берёт AppConfig как есть, а там
`enable_robot_control = True` — то есть запуск СРАЗУ подключается к контроллеру,
выезжает в home и стартует цикл движения. Для первой проверки камеры или новой
KRL-программы это не то, что нужно.

Здесь умолчание обратное и безопасное: **без управления роботом**. Движение
включается только явным `--robot`.

    python tools/run_live.py                      # камера + детекция, робот не трогается
    python tools/run_live.py --duration 20        # ограничить прогон по времени
    python tools/run_live.py --robot              # с движением и остановом по коллизии
    python tools/run_live.py --robot --record --scenario T1_assembly
    python tools/run_live.py --set show_debug_masks=False perf_window=False

Чтение углов суставов (`use_robot_kinematics`) остаётся включённым и без
`--robot`: оно нужно FK-самофильтру, но роботом не управляет.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.console import setup_console  # noqa: E402
from realsense_io import AppConfig  # noqa: E402
from tools.replay import config_types, parse_overrides  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Прогон на стенде (камера + детекция)")
    ap.add_argument("--robot", action="store_true",
                    help="ВКЛЮЧИТЬ управление роботом: выезд в home, цикл, останов по коллизии")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="длительность прогона, с (0 = до нажатия q)")
    ap.add_argument("--record", action="store_true", help="писать кадры и углы на диск")
    ap.add_argument("--scenario", default="", help="scenario_id во всех логах")
    ap.add_argument("--logs", action="store_true",
                    help="включить логи решений, объектов и остановов")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    metavar="ИМЯ=ЗНАЧЕНИЕ", help="переопределить поля AppConfig")
    args = ap.parse_args()

    kwargs = {
        "enable_robot_control": bool(args.robot),
        "run_duration_s": float(args.duration),
        "scenario_id": str(args.scenario),
    }
    if args.record:
        kwargs["enable_recording"] = True
    if args.logs or args.record:
        kwargs["enable_decision_log"] = True
        kwargs["enable_objects_log"] = True
        if args.robot:
            kwargs["enable_stop_log"] = True
    kwargs.update(parse_overrides(args.overrides, config_types()))

    cfg = AppConfig(**kwargs)

    print("\n" + "=" * 64)
    print(f"  Управление роботом : {'ВКЛЮЧЕНО — робот поедет' if cfg.enable_robot_control else 'выключено'}")
    print(f"  Запись кадров      : {'да' if cfg.enable_recording else 'нет'}")
    print(f"  Логи               : решения={cfg.enable_decision_log} объекты={cfg.enable_objects_log} стопы={cfg.enable_stop_log}")
    print(f"  Длительность       : {'до q' if cfg.run_duration_s <= 0 else f'{cfg.run_duration_s:.0f} с'}")
    print(f"  Сегментатор        : {cfg.segmenter}")
    print("=" * 64)
    if cfg.enable_robot_control:
        # Движение требует T1 с удержанием и $OV_PRO > 0 — это делается на
        # пульте руками, приложение режим не переключает.
        ans = input("\nРОБОТ ПОЕДЕТ. Оператор на удержании, зона свободна? [y/N] ")
        if ans.strip().lower() not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return 1

    from app.runner import AppRunner  # импорт здесь: тянет cv2/open3d/камеру
    AppRunner(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
