"""
Одиночный реплей записи: прогнать записанные кадры через конвейер и записать
решения в CSV.

Заменяет однострочник `python -c "from realsense_io import AppConfig; ..."`,
который до этого стоял в docs/VALIDATION.md: его нельзя было ни прочитать, ни
повторить без ошибок, а параметры приходилось править внутри строки.

Прогон идёт БЕЗ окон и без мониторинга ресурсов (`show_cv_windows=False`,
`perf_monitor=False`) — это и есть «канонический конфиг регрессии». Детекция от
этих флагов не зависит: окна лишь показывают уже посчитанный кадр, монитор лишь
пишет свой CSV. Управление роботом в playback выключается принудительно самим
AppRunner.

Примеры:

    python tools/replay.py --recording captures/scenarios/S04_cube_on_path/run_01_3d/recording \\
                           --out captures/replay/S04.csv

    # абляция: тот же вход, другой параметр
    python tools/replay.py --recording <dir> --out off.csv --set collision_object_permanence=False

Любое поле AppConfig переопределяется через --set имя=значение; тип берётся из
объявления поля, так что «False» станет bool, а «0.08» — float.
"""
from __future__ import annotations  # Python 3.9: иначе `dict | None` в аннотации падает

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from realsense_io import AppConfig  # noqa: E402

# Конфиг регрессии: единственный источник правды о том, в каких условиях
# снимаются golden-файлы. Менять его — значит обесценить все golden сразу,
# поэтому он лежит здесь, а не растаскивается по вызовам.
REGRESSION_OVERRIDES = {
    "source_mode": "playback",
    "show_cv_windows": False,
    "show_debug_masks": False,
    "show_o3d_window": False,
    "perf_monitor": False,
    "enable_robot_control": False,
    "enable_decision_log": True,
    "run_duration_s": 0.0,
}


def _coerce(name: str, raw: str, declared_type):
    """Привести строку из --set к типу поля AppConfig."""
    if declared_type is bool or isinstance(declared_type, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{name}: ожидался bool, получено {raw!r}")
    if declared_type is int or isinstance(declared_type, int):
        return int(raw)
    if declared_type is float or isinstance(declared_type, float):
        return float(raw)
    if declared_type is tuple or isinstance(declared_type, tuple):
        parts = [p for p in raw.replace("(", "").replace(")", "").split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                try:
                    out.append(float(p))
                except ValueError:
                    out.append(p.strip())
        return tuple(out)
    return raw


def parse_overrides(pairs, cfg_types) -> dict:
    """Разобрать список «имя=значение» в словарь переопределений конфига."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set ожидает имя=значение, получено: {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if name not in cfg_types:
            raise SystemExit(f"--set: в AppConfig нет поля {name!r}")
        out[name] = _coerce(name, raw, cfg_types[name])
    return out


def config_types() -> dict:
    """Имя поля AppConfig → его тип (по объявлению, иначе по значению)."""
    types = {}
    defaults = AppConfig()
    for f in fields(AppConfig):
        declared = f.type
        if isinstance(declared, str):  # from __future__ annotations / строковые аннотации
            declared = {"bool": bool, "int": int, "float": float,
                        "str": str, "tuple": tuple}.get(declared.strip(),
                                                        type(getattr(defaults, f.name)))
        types[f.name] = declared
    return types


def build_config(recording: str, out_csv: str, scenario_id: str,
                 extra: dict | None = None) -> AppConfig:
    """Конфиг реплея: канонические переопределения + путь записи + --set."""
    kwargs = dict(REGRESSION_OVERRIDES)
    kwargs["playback_path"] = str(recording)
    kwargs["decision_log_path"] = str(out_csv)
    kwargs["scenario_id"] = str(scenario_id or "")
    if extra:
        kwargs.update(extra)
    return AppConfig(**kwargs)


def scenario_from_path(recording: str) -> str:
    """Вытащить scenario_id из пути записи captures/scenarios/<sid>/run_XX_3d/recording."""
    parts = Path(recording).resolve().parts
    if "scenarios" in parts:
        i = parts.index("scenarios")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Реплей записи через конвейер восприятия")
    ap.add_argument("--recording", required=True,
                    help="каталог записи (внутри frames/, joints.csv, meta.json)")
    ap.add_argument("--out", required=True, help="куда писать decisions.csv")
    ap.add_argument("--scenario-id", default=None,
                    help="id сценария для логов (по умолчанию из пути записи)")
    ap.add_argument("--task-states", default=None, metavar="POSES_YAML",
                    help="восстановить фазу техпроцесса из углов записи и "
                         "подсунуть её модели задачи (для абляции ожидаемой сцены)")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    metavar="ИМЯ=ЗНАЧЕНИЕ", help="переопределить поля AppConfig")
    args = ap.parse_args()

    rec = Path(args.recording)
    if not (rec / "meta.json").exists():
        raise SystemExit(f"Не похоже на запись (нет meta.json): {rec}")
    out = Path(args.out)
    if out.exists():
        out.unlink()  # DecisionLogger дописывает в существующий файл

    extra = parse_overrides(args.overrides, config_types())
    sid = args.scenario_id if args.scenario_id is not None else scenario_from_path(args.recording)
    cfg = build_config(str(rec), str(out), sid, extra)

    if args.task_states:
        # Запись T2_baseline_tower снята до появления файла состояния, поэтому
        # фаза восстанавливается из записанных углов. Пересниматься нельзя:
        # сравнение правил обязано идти на ОДНИХ И ТЕХ ЖЕ кадрах.
        from task.reconstruct import load_build_pose, reconstruct
        from task.state import set_frame_states
        bxy, bz = load_build_pose(args.task_states)
        states = reconstruct(str(rec / "joints.csv"), bxy, bz)
        set_frame_states(states)
        n_tower = sum(1 for v in states.values() if v.cubes_placed > 0)
        print(f"[REPLAY] состояние восстановлено: {len(states)} кадров, "
              f"со стопкой на столе {n_tower}")

    from app.runner import AppRunner  # импорт здесь: тянет cv2/open3d
    AppRunner(cfg).run()
    print(f"[REPLAY] Решения → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
