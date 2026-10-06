"""
Мастер экспериментов: ведёт оператора по сериям от калибровки до отчёта.

    python tools/run_experiments.py --stage calib   # проверка калибровки
    python tools/run_experiments.py --stage e1      # статика (точность восприятия)
    python tools/run_experiments.py --stage e2      # динамика (сценарии)
    python tools/run_experiments.py --stage e3      # серия скорости (время реакции)
    python tools/run_experiments.py --stage report  # метрики и отчёт
    python tools/run_experiments.py --stage e2 --resume   # продолжить после перерыва

Перед каждой записью мастер печатает, что надо сделать руками, ждёт клавишу и
ведёт обратный отсчёт. Состояние пишется в captures/progress.json после каждого
шага, поэтому серию можно прервать (Ctrl+C) и продолжить с --resume.

Прогоны идут В ЭТОМ ЖЕ ПРОЦЕССЕ (AppRunner с cfg.run_duration_s). Это проще, чем
подпроцессы, но означает, что падение одного прогона не должно ронять сессию —
каждый прогон обёрнут в try/except, а прогресс сохраняется до и после.

Что меряется в какой серии:
  Э1  блок расстояний — ОДИН объект в кадре: тогда min_dist_m относится к нему
      и gt_<config_id>.csv даёт ошибку расстояния напрямую;
      остальные блоки — несколько объектов, и метрика там не расстояние, а доля
      обнаруженных (n_objects из decisions.csv против числа расставленных);
  Э2  обнаружение и остановы в динамике по сценариям S01-S08;
  Э3  время реакции на четырёх скоростях; ЗАПИСЬ КАДРОВ ВЫКЛЮЧЕНА, чтобы
      задержки не измерялись под дисковой нагрузкой.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from realsense_io import AppConfig  # noqa: E402

CAPTURES = ROOT / "captures"
PROGRESS = CAPTURES / "progress.json"
DANGER_M = float(AppConfig.collision_danger_dist_m)

# ── Э1: конфигурации статической серии ───────────────────────────────────────
# metric='distance' — один объект в кадре, эталон расстояния меряется линейкой;
# metric='detection' — несколько объектов, эталон = их ЧИСЛО (доля обнаруженных).
E1_CONFIGS = [
    # Блок расстояний: по одному объекту, восемь записей. Так min_dist_m
    # однозначно относится к этому объекту и ошибка расстояния считается честно.
    *[
        {
            "config_id": f"e1_dist_{d:03d}mm",
            "block": "Расстояние",
            "metric": "distance",
            "n_objects": 1,
            "nominal_mm": [d],
            "setup": f"Поставь ОДИН кубик на расстоянии {d} мм от траектории руки. "
                     f"В кадре больше ничего быть не должно.",
        }
        for d in (20, 30, 50, 80, 100, 120, 150, 200)
    ],
    # Блок высоты: минимальная различимая высота объекта. Нижняя граница задана
    # высотным гейтом (gate_min_m из table_plane.json) — этап calib его печатает.
    *[
        {
            "config_id": f"e1_height_{h:03d}mm",
            "block": "Высота объекта",
            "metric": "distance",
            "n_objects": 1,
            "nominal_mm": [80],
            "height_mm": h,
            "setup": f"Поставь ОДИН предмет высотой {h} мм на расстоянии 80 мм от "
                     f"траектории руки. Высоту померь, это ключевой параметр записи.\n"
                     f"    ВАЖНО: 80 мм — ровно порог DANGER, поэтому вердикт "
                     f"DANGER/SAFE тут на границе и сам по себе ничего не\n"
                     f"    доказывает. Смысл блока в другом: ВИДИТ ли система "
                     f"предмет такой высоты вообще (доля обнаруженных).",
        }
        for h in (25, 40, 60, 90, 130)
    ],
    # Блок освещения: три кубика, метрика — доля обнаруженных.
    *[
        {
            "config_id": f"e1_light_{key}",
            "block": "Освещение",
            "metric": "detection",
            "n_objects": 3,
            "nominal_mm": [30, 80, 150],
            "lighting": key,
            "setup": f"Три кубика на 30, 80 и 150 мм. Освещение: {ru}.",
        }
        for key, ru in (("normal", "штатное"), ("dim", "слабое"),
                        ("side", "боковое, с тенями"))
    ],
    # Блок типов объекта: тёмный вынесен отдельно — ложный DANGER на чёрном это
    # известная граница применимости, её надо показать, а не обойти.
    {
        "config_id": "e1_type_cube_box",
        "block": "Тип объекта",
        "metric": "detection",
        "n_objects": 3,
        "nominal_mm": [30, 80, 150],
        "object_type": "cube+box",
        "setup": "Кубик и коробка на 30, 80 и 150 мм (три предмета).",
    },
    {
        "config_id": "e1_type_dark",
        "block": "Тип объекта",
        "metric": "detection",
        "n_objects": 3,
        "nominal_mm": [30, 80, 150],
        "object_type": "dark",
        "setup": "ТЁМНЫЕ (чёрные/матовые) предметы на 30, 80 и 150 мм. "
                 "Ожидается ухудшение — это заявленная граница применимости.",
    },
    {
        "config_id": "e1_type_glossy",
        "block": "Тип объекта",
        "metric": "detection",
        "n_objects": 3,
        "nominal_mm": [30, 80, 150],
        "object_type": "glossy",
        "setup": "Предметы С БЛИКАМИ (глянец/металл) на 30, 80 и 150 мм.",
    },
    # Блок поз манипулятора: те же три кубика, разные позы руки.
    *[
        {
            "config_id": f"e1_pose_{i}",
            "block": "Поза манипулятора",
            "metric": "detection",
            "n_objects": 3,
            "nominal_mm": [30, 80, 150],
            "arm_pose": i,
            "setup": f"Три кубика на 30, 80 и 150 мм. Поза руки №{i} "
                     f"(точка {i} из цикла — выставь вручную и оставь).",
        }
        for i in (1, 2, 3)
    ],
]

E1_DURATION_S = 10
E1_MASK_EVERY_N = 30
E2_REPEATS = 3
E2_DURATION_S = 30
E2_SCENARIOS = ("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08")
E3_LEVELS = [
    {"speed": 30, "scenario_id": "S04_cube_on_path"},
    {"speed": 50, "scenario_id": "S05_cube_on_path_fast"},
    {"speed": 70, "scenario_id": "S09_speed_70"},
    {"speed": 100, "scenario_id": "S10_speed_100"},
]
E3_EPISODES_PER_LEVEL = 5
E3_DURATION_S = 30
E3_TOPUP_MARGIN_M = 0.010   # остаточная дистанция ближе 10 мм к нулю → добрать

SCENARIOS_INDEX_FIELDS = [
    "scenario_id", "run", "contour", "obstacle", "start_distance_m",
    "robot_ctrl_speed", "lighting", "occlusion", "danger_expected",
    "n_stop_events", "n_danger_frames", "contact", "path",
]
GT_FIELDS = ["frame", "obj_present", "danger_true", "dist_true_m"]
E1_INDEX_FIELDS = [
    "config_id", "block", "metric", "n_objects_true", "nominal_mm",
    "measured_mm", "height_mm", "lighting", "object_type", "arm_pose",
    "duration_s", "n_frames", "path",
]
E3_INDEX_FIELDS = [
    "scenario_id", "speed_pct", "run", "episode", "contact",
    "min_dist_m_at_danger", "total_reaction_ms", "path",
]


# ── интерфейс оператора ──────────────────────────────────────────────────────

def say(text: str = "") -> None:
    print(text, flush=True)


def head(title: str) -> None:
    say("\n" + "=" * 72)
    say(title)
    say("=" * 72)


def wait_enter(prompt: str = "Готово? Нажми Enter, чтобы продолжить") -> None:
    try:
        input(f"\n>>> {prompt} ")
    except EOFError:
        raise SystemExit("\nВвод недоступен — мастер рассчитан на интерактивный запуск.")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f">>> {prompt}{suffix}: ").strip()
    except EOFError:
        raise SystemExit("\nВвод недоступен — мастер рассчитан на интерактивный запуск.")
    return val or default


def ask_yes(prompt: str, default: bool = True) -> bool:
    d = "Д/н" if default else "д/Н"
    while True:
        val = ask(f"{prompt} ({d})").lower()
        if not val:
            return default
        if val in ("д", "da", "y", "yes", "да", "1"):
            return True
        if val in ("н", "n", "no", "нет", "0"):
            return False
        say("    Ответь «д» или «н».")


def ask_float(prompt: str, allow_empty: bool = False):
    while True:
        val = ask(prompt)
        if not val and allow_empty:
            return None
        try:
            return float(val.replace(",", "."))
        except ValueError:
            say("    Нужно число, например 82.5")


def ask_int(prompt: str, default=None):
    while True:
        val = ask(prompt, "" if default is None else str(default))
        try:
            return int(val)
        except ValueError:
            say("    Нужно целое число.")


def countdown(seconds: int, what: str) -> None:
    say(f"\n{what}: {seconds} с. Старт через:")
    for i in (3, 2, 1):
        say(f"    {i}...")
        time.sleep(1.0)
    say("    ПОШЛА ЗАПИСЬ\n")


# ── состояние ────────────────────────────────────────────────────────────────

class Progress:
    """Состояние мастера: что уже сделано. Пишется после каждого шага."""

    def __init__(self, path: Path = PROGRESS):
        self.path = path
        self.data = {"done": [], "started": None, "notes": [], "runs": {}}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                say(f"[WARN] progress.json не прочитан ({e}) — начинаем с нуля")
        if not self.data.get("started"):
            self.data["started"] = datetime.now().isoformat(timespec="seconds")

    def is_done(self, key: str) -> bool:
        return key in self.data["done"]

    def mark(self, key: str, info=None) -> None:
        if key not in self.data["done"]:
            self.data["done"].append(key)
        if info is not None:
            self.data["runs"][key] = info
        self.save()

    def note(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.data["notes"].append(f"{stamp} {text}")
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ── прогон ───────────────────────────────────────────────────────────────────

def make_cfg(**overrides) -> AppConfig:
    cfg = AppConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"AppConfig не знает поле {k!r}")
        setattr(cfg, k, v)
    return cfg


def run_capture(cfg: AppConfig) -> bool:
    """Один прогон приложения. False = прогон упал (сессия продолжается)."""
    from app.runner import AppRunner
    try:
        AppRunner(cfg).run()
        return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        say(f"\n[ОШИБКА] Прогон не удался: {type(e).__name__}: {e}")
        say("         Прогресс сохранён — шаг можно повторить с --resume.")
        return False


def rel(path) -> str:
    """Путь относительно корня репозитория; абсолютный, если он вне корня."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_csv(path: Path) -> list:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, fields: list, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a" if exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_gt(path: Path, frames: list, obj_present: int, danger_true: int,
             dist_true_m=None) -> None:
    """gt_<config_id>.csv на все кадры конфигурации.

    В статической серии сцена не меняется, поэтому эталон один на всю запись и
    размечать руками нечего — значения проставляются на каждый кадр.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GT_FIELDS)
        w.writeheader()
        for fr in frames:
            w.writerow({
                "frame": fr,
                "obj_present": int(obj_present),
                "danger_true": int(danger_true),
                "dist_true_m": ("" if dist_true_m is None
                                else f"{float(dist_true_m):.4f}"),
            })


def recorded_frames(recording_dir: Path) -> list:
    return sorted(int(p.stem.split("_")[1])
                  for p in (Path(recording_dir) / "frames").glob("frame_*.npz"))


def load_scenarios() -> dict:
    data = yaml.safe_load((ROOT / "scenarios.yaml").read_text(encoding="utf-8"))
    defaults = data.get("defaults", {}) or {}
    out = {}
    for sc in data.get("scenarios", []) or []:
        merged = dict(defaults)
        merged.update(sc)
        out[merged["scenario_id"]] = merged
    return out


# ── этап calib ───────────────────────────────────────────────────────────────

def stage_calib(progress: Progress) -> None:
    head("ЭТАП CALIB — проверка калибровки")
    calib_dir = CAPTURES / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings = []

    extr_path = ROOT / "extrinsic.json"
    plane_path = ROOT / "table_plane.json"

    # ── extrinsic ────────────────────────────────────────────────────────────
    extr = {}
    if extr_path.exists():
        extr = json.loads(extr_path.read_text(encoding="utf-8"))
        mtime = datetime.fromtimestamp(extr_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        rms = float(extr.get("rms_px", float("nan")))
        n_pts = int(extr.get("n_points", 0))
        n_inl = int(extr.get("n_inliers", 0))
        say(f"\nextrinsic.json — {mtime:%Y-%m-%d %H:%M} (возраст {age_days} дн.)")
        say(f"  RMS репроекции : {rms:.2f} px")
        say(f"  Точек          : {n_pts}")
        say(f"  Инлайеров      : {n_inl}")
        say(f"  Z-разброс      : {float(extr.get('z_span_mm', 0)):.0f} мм")
        ho = extr.get("holdout")
        if ho:
            say(f"  Hold-out (LOO) : {ho['rms_px']:.2f} px = {ho['rms_mm']:.1f} мм "
                f"(медиана {ho['median_mm']:.1f}, максимум {ho['max_mm']:.1f} мм)")
        else:
            say("  Hold-out       : НЕ ПОСЧИТАН (калибровка старая)")
            warnings.append("В extrinsic.json нет hold-out: нужна перекалибровка, "
                            "иначе точность пересчёта в статье указать нечем.")
        if rms != rms or rms > 5.0:
            warnings.append(f"RMS {rms:.2f} px > 5 px — заявленный порог не выполнен.")
            say(f"\n  [!] RMS {rms:.2f} px выше порога 5 px.")
        if n_pts < 30:
            warnings.append(f"Точек калибровки {n_pts} < 30 — выборка мала.")
            say(f"  [!] Точек {n_pts} < 30.")
        if age_days > 30:
            warnings.append(f"Калибровке {age_days} дней.")
    else:
        say("\n[!] extrinsic.json отсутствует.")
        warnings.append("extrinsic.json отсутствует.")

    if warnings:
        say("\n  Перекалибровка обязательна: все метрические выводы (расстояние,")
        say("  d_safe) верны ровно настолько, насколько точен пересчёт камера→база.")
    if ask_yes("Запустить calibrate_click_fk.py сейчас?", default=bool(warnings)):
        say("\nОткроется окно. Кликай по суставам, следуй подсказкам скрипта.")
        wait_enter("Готов начать калибровку?")
        subprocess.run([sys.executable, str(ROOT / "calibrate_click_fk.py")],
                       cwd=str(ROOT))
        if extr_path.exists():
            extr = json.loads(extr_path.read_text(encoding="utf-8"))
            warnings.append("Калибровка перезапущена оператором в этой сессии.")

    # ── плоскость стола ──────────────────────────────────────────────────────
    plane = {}
    if plane_path.exists():
        plane = json.loads(plane_path.read_text(encoding="utf-8"))
        say(f"\ntable_plane.json — снята {plane.get('created', '?')}")
        say(f"  ROI рабочей зоны (px) : {plane.get('roi_px')}")
        say(f"  gate_min_m            : {plane.get('gate_min_m')}")
        say(f"  gate_max_m            : {plane.get('gate_max_m')}")
        say("\n  ВНИМАНИЕ. gate_min_m из этого файла ПЕРЕЗАПИСЫВАЕТ")
        say("  collision_obj_plane_height_min_m в конфиге (vision/collision_service.py,")
        say("  загрузка плоскости). Это рабочий порог высоты объекта: предмет ниже")
        say("  него системой не выделяется вообще. Пересъёмка плоскости меняет")
        say("  чувствительность детекции — после неё серию Э1 по высоте надо")
        say("  переснимать, иначе минимальная различимая высота будет от другого порога.")
        gate_min = plane.get("gate_min_m")
        if gate_min is not None:
            say(f"\n  Сейчас порог высоты = {float(gate_min) * 1000:.0f} мм.")
    else:
        say("\n[!] table_plane.json отсутствует — детекция откатится на цвет-гейт.")
        warnings.append("table_plane.json отсутствует.")

    if ask_yes("Переснять плоскость стола (calibrate_table_plane.py)?", default=False):
        wait_enter("Убери все предметы со стола. Готово?")
        subprocess.run([sys.executable, str(ROOT / "calibrate_table_plane.py")],
                       cwd=str(ROOT))
        if plane_path.exists():
            plane = json.loads(plane_path.read_text(encoding="utf-8"))
            warnings.append("Плоскость стола переснята — порог высоты мог измениться.")

    # ── копии и отчёт ────────────────────────────────────────────────────────
    for src in (extr_path, plane_path):
        if src.exists():
            shutil.copy2(src, calib_dir / f"{src.stem}_{stamp}{src.suffix}")

    report = calib_dir / "report.md"
    ho = (extr or {}).get("holdout") or {}
    lines = [
        "# Калибровка — итоговые числа",
        "",
        f"Снято мастером: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Пересчёт камера→база (extrinsic.json)",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| RMS репроекции | {float(extr.get('rms_px', float('nan'))):.2f} px |",
        f"| Точек | {int(extr.get('n_points', 0))} |",
        f"| Инлайеров | {int(extr.get('n_inliers', 0))} |",
        f"| Z-разброс точек | {float(extr.get('z_span_mm', 0)):.0f} мм |",
    ]
    if ho:
        lines += [
            f"| **Hold-out (leave-one-out)** | **{ho['rms_px']:.2f} px = "
            f"{ho['rms_mm']:.1f} мм** |",
            f"| Hold-out, медиана | {ho['median_mm']:.1f} мм |",
            f"| Hold-out, максимум | {ho['max_mm']:.1f} мм |",
            f"| Точек в hold-out | {ho['n']} |",
        ]
        lines += [
            "",
            "Hold-out — заявляемая точность пересчёта: каждая точка по очереди",
            "исключается из подгонки, поза решается по остальным, ошибка меряется",
            "на исключённой. RMS выше считается на тех же точках, на которых",
            "решалась задача, и потому систематически оптимистична.",
            f"Перевод в миллиметры: err_mm = err_px · z / fx ({ho.get('method', '')}).",
        ]
    else:
        lines += ["", "> Hold-out не посчитан: extrinsic.json получен до того, как",
                  "> эта проверка появилась. Перекалибруйся, иначе точность",
                  "> пересчёта в статье указать нечем."]
    lines += [
        "",
        "## Плоскость стола (table_plane.json)",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Снята | {plane.get('created', '—')} |",
        f"| ROI рабочей зоны, px | {plane.get('roi_px', '—')} |",
        f"| **gate_min_m (рабочий порог высоты)** | "
        f"**{plane.get('gate_min_m', '—')} м** |",
        f"| gate_max_m | {plane.get('gate_max_m', '—')} м |",
        f"| RANSAC, порог | {plane.get('ransac_dist_m', '—')} м |",
        "",
        "`gate_min_m` из этого файла перезаписывает `collision_obj_plane_height_min_m`",
        "при загрузке плоскости, то есть является ДЕЙСТВУЮЩИМ порогом высоты объекта.",
        "Предмет ниже него не выделяется вообще. Пересъёмка плоскости меняет",
        "чувствительность детекции: после неё серию Э1 по высоте нужно переснять.",
    ]
    if warnings:
        lines += ["", "## Предупреждения", ""]
        lines += [f"- {w}" for w in warnings]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\n[OK] Отчёт калибровки: {report}")
    for w in warnings:
        progress.note(f"calib: {w}")
    progress.mark("calib", {"warnings": warnings, "stamp": stamp})


# ── этап e1 ──────────────────────────────────────────────────────────────────

def stage_e1(progress: Progress, resume: bool) -> None:
    head("ЭТАП E1 — статическая серия (точность восприятия)")
    say(f"Конфигураций: {len(E1_CONFIGS)}, запись {E1_DURATION_S} с на каждую.")
    say("Робот НЕПОДВИЖЕН, управление выключено. Дамп масок каждые "
        f"{E1_MASK_EVERY_N} кадров.")
    say("\nБлок расстояний идёт по ОДНОМУ объекту в кадре: только так измеренное")
    say("min_dist_m однозначно относится к тому предмету, который ты померил")
    say("линейкой. В остальных блоках объектов несколько, и там метрика — доля")
    say("обнаруженных, эталон расстояния не нужен.")
    wait_enter("Убедись, что робот стоит и управление выключено. Готов?")

    for i, conf in enumerate(E1_CONFIGS, 1):
        key = f"e1:{conf['config_id']}"
        if resume and progress.is_done(key):
            say(f"\n[{i}/{len(E1_CONFIGS)}] {conf['config_id']} — уже сделано, пропуск")
            continue
        head(f"[{i}/{len(E1_CONFIGS)}] {conf['config_id']}  ({conf['block']})")
        say(f"\nЧТО СДЕЛАТЬ:\n    {conf['setup']}")
        if conf["metric"] == "distance":
            say("\n    Метрика: ошибка расстояния. В кадре должен быть РОВНО один предмет.")
        else:
            say(f"\n    Метрика: доля обнаруженных из {conf['n_objects']} предметов.")
        wait_enter("Предметы расставлены?")

        measured = []
        for k, nom in enumerate(conf["nominal_mm"], 1):
            v = ask_float(f"Измеренное линейкой расстояние №{k} (номинал {nom} мм), мм")
            measured.append(v)
        n_real = ask_int("Сколько предметов реально в кадре", conf["n_objects"])

        run_dir = CAPTURES / "e1" / conf["config_id"]
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        cfg = make_cfg(
            scenario_id=conf["config_id"],
            enable_robot_control=False,
            run_duration_s=float(E1_DURATION_S),
            enable_decision_log=True,
            decision_log_path=str(run_dir / "decisions.csv"),
            enable_objects_log=True,
            objects_log_path=str(run_dir / "objects.csv"),
            enable_recording=True,
            recording_path=str(run_dir / "recording"),
            dump_masks_every_n=E1_MASK_EVERY_N,
            masks_dump_path=str(run_dir / "masks"),
            perf_log_path=str(run_dir / "perf_log.csv"),
            show_o3d_window=False,
        )
        countdown(E1_DURATION_S, "Запись конфигурации")
        ok = run_capture(cfg)

        frames = recorded_frames(run_dir / "recording")
        say(f"\nЗаписано кадров: {len(frames)}")
        # Эталон расстояния пишем ТОЛЬКО там, где объект один: при нескольких
        # предметах min_dist_m относится к ближайшему из них, и сопоставлять его
        # с конкретным замером не с чем — там метрикой служит доля обнаруженных.
        known = [v for v in measured if v is not None]
        nearest_m = min(known) / 1000.0 if known else None
        dist_true_m = nearest_m if conf["metric"] == "distance" else None
        danger_true = int(nearest_m is not None and nearest_m <= DANGER_M)
        gt_path = CAPTURES / f"gt_{conf['config_id']}.csv"
        write_gt(gt_path, frames, obj_present=1 if n_real > 0 else 0,
                 danger_true=danger_true, dist_true_m=dist_true_m)
        say(f"Эталон: {gt_path.name} "
            f"(danger_true={danger_true}, dist_true_m="
            f"{'—' if dist_true_m is None else f'{dist_true_m:.3f} м'})")

        append_row(CAPTURES / "e1_index.csv", E1_INDEX_FIELDS, {
            "config_id": conf["config_id"],
            "block": conf["block"],
            "metric": conf["metric"],
            "n_objects_true": n_real,
            "nominal_mm": ";".join(str(v) for v in conf["nominal_mm"]),
            "measured_mm": ";".join("" if v is None else f"{v:g}" for v in measured),
            "height_mm": conf.get("height_mm", ""),
            "lighting": conf.get("lighting", "normal"),
            "object_type": conf.get("object_type", "cube"),
            "arm_pose": conf.get("arm_pose", ""),
            "duration_s": E1_DURATION_S,
            "n_frames": len(frames),
            "path": str(run_dir),
        })
        progress.mark(key, {"ok": ok, "n_frames": len(frames),
                            "measured_mm": measured, "n_objects_true": n_real})
        if not ok:
            progress.note(f"e1: прогон {conf['config_id']} завершился с ошибкой")
        if i < len(E1_CONFIGS) and not ask_yes("Продолжить со следующей конфигурацией?"):
            say("Пауза. Продолжить: --stage e1 --resume")
            return
    say("\n[OK] Серия Э1 завершена.")


# ── этап e2 ──────────────────────────────────────────────────────────────────

def stage_e2(progress: Progress, resume: bool) -> None:
    head("ЭТАП E2 — динамическая серия (сценарии)")
    scenarios = load_scenarios()
    todo = [sid for sid in scenarios if sid.split("_")[0] in E2_SCENARIOS]
    todo.sort()
    say(f"Сценариев: {len(todo)}, повторов: {E2_REPEATS}, прогон {E2_DURATION_S} с.")
    say("Робот гоняет цикл, управление ВКЛЮЧЕНО. Скорость берётся из сценария.")
    wait_enter("Зона свободна, аварийный стоп под рукой. Готов?")

    for sid in todo:
        sc = scenarios[sid]
        for run in range(1, E2_REPEATS + 1):
            key = f"e2:{sid}:{run}"
            if resume and progress.is_done(key):
                say(f"\n{sid} повтор {run} — уже сделано, пропуск")
                continue
            head(f"{sid}  —  повтор {run}/{E2_REPEATS}")
            say(f"\nСЦЕНАРИЙ: {sc.get('description', '')}")
            say(f"    препятствие      : {sc.get('obstacle')}")
            say(f"    исходная дистанция: {sc.get('start_distance_m')} м")
            say(f"    скорость цикла   : {sc.get('robot_ctrl_speed')} %")
            say(f"    освещение        : {sc.get('lighting')}")
            say(f"    перекрытие       : {sc.get('occlusion')}")
            say(f"    ожидается DANGER : {sc.get('danger_expected')}")
            wait_enter("Препятствие расставлено по описанию?")

            run_dir = CAPTURES / "scenarios" / sid / f"run_{run:02d}_3d"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            cfg = make_cfg(
                scenario_id=sid,
                enable_robot_control=True,
                robot_ctrl_speed=int(sc.get("robot_ctrl_speed", 30)),
                run_duration_s=float(E2_DURATION_S),
                enable_decision_log=True,
                decision_log_path=str(run_dir / "decisions.csv"),
                enable_objects_log=True,
                objects_log_path=str(run_dir / "objects.csv"),
                enable_stop_log=True,
                stop_log_path=str(run_dir / "stop_events.csv"),
                enable_recording=True,
                recording_path=str(run_dir / "recording"),
                dump_masks_every_n=E1_MASK_EVERY_N,
                masks_dump_path=str(run_dir / "masks"),
                perf_log_path=str(run_dir / "perf_log.csv"),
                show_o3d_window=False,
            )
            countdown(E2_DURATION_S, "Прогон сценария")
            ok = run_capture(cfg)

            contact = "1" if ask_yes("Был ли физический контакт руки с препятствием?",
                                     default=False) else "0"
            normal = ask_yes("Прогон завершился штатно (без сбоев и вмешательств)?")
            stop_rows = read_csv(run_dir / "stop_events.csv")
            dec_rows = read_csv(run_dir / "decisions.csv")
            n_danger = sum(1 for r in dec_rows if r.get("collision_level") == "DANGER")
            # Ответ оператора про контакт — эталон, дописываем его в stop_events.
            if stop_rows:
                for r in stop_rows:
                    r["contact"] = contact
                with open(run_dir / "stop_events.csv", "w", newline="",
                          encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(stop_rows[0].keys()))
                    w.writeheader()
                    w.writerows(stop_rows)
            say(f"\nСобытий DANGER: {len(stop_rows)}, кадров DANGER: {n_danger}")

            append_row(CAPTURES / "scenarios_index.csv", SCENARIOS_INDEX_FIELDS, {
                "scenario_id": sid,
                "run": run,
                "contour": "3D",
                "obstacle": sc.get("obstacle", ""),
                "start_distance_m": sc.get("start_distance_m", ""),
                "robot_ctrl_speed": sc.get("robot_ctrl_speed", ""),
                "lighting": sc.get("lighting", ""),
                "occlusion": sc.get("occlusion", ""),
                "danger_expected": sc.get("danger_expected", ""),
                "n_stop_events": len(stop_rows),
                "n_danger_frames": n_danger,
                "contact": contact,
                "path": str(run_dir),
            })
            progress.mark(key, {"ok": ok, "normal": normal, "contact": contact,
                                "n_stop_events": len(stop_rows),
                                "n_danger_frames": n_danger})
            if not normal:
                why = ask("Что пошло не так (одной строкой)")
                progress.note(f"e2 {sid} run {run}: {why}")
            say("\nРазметка Э2 делается офлайн: tools/annotate.py по этой записи,")
            say("отмечать только моменты смены состояния (метка действует вперёд).")
            if not ask_yes("Продолжить?"):
                say("Пауза. Продолжить: --stage e2 --resume")
                return
    say("\n[OK] Серия Э2 завершена.")


# ── этап e3 ──────────────────────────────────────────────────────────────────

def stage_e3(progress: Progress, resume: bool) -> None:
    head("ЭТАП E3 — серия скорости (время реакции)")
    scenarios = load_scenarios()
    say(f"Уровни скорости: {[lv['speed'] for lv in E3_LEVELS]} %, "
        f"нужно {E3_EPISODES_PER_LEVEL} эпизодов на уровень.")
    say("Эпизод = один проход руки мимо объекта; за прогон 30 с их выходит 2-3,")
    say("поэтому эпизоды считаются ПО ФАКТУ, а не по числу прогонов.")
    say("\nЗАПИСЬ КАДРОВ В ЭТОЙ СЕРИИ ВЫКЛЮЧЕНА: здесь меряется время реакции,")
    say("и оно не должно измеряться под дисковой нагрузкой. Эталон тут — твой")
    say("ответ про контакт.")
    wait_enter("Кубик на траектории как в S04. Готов?")

    for lv in E3_LEVELS:
        sid = lv["scenario_id"]
        sc = scenarios.get(sid)
        if sc is None:
            say(f"[!] Сценарий {sid} не найден в scenarios.yaml — уровень пропущен")
            progress.note(f"e3: нет сценария {sid}")
            continue
        head(f"Уровень скорости {lv['speed']}%  ({sid})")
        if lv["speed"] >= 70:
            say("\n  !!! ВНИМАНИЕ. На этой скорости КОНТАКТ ВЕРОЯТЕН.")
            say("  На траектории должен стоять ЛЁГКИЙ предмет: пенопласт или")
            say("  пустая коробка. Тяжёлый предмет на этой скорости опасен и")
            say("  для руки, и для стенда.")
            if not ask_yes("Подтверждаешь, что стоит лёгкий предмет?", default=False):
                say("  Уровень пропущен.")
                progress.note(f"e3: уровень {lv['speed']}% пропущен (нет лёгкого предмета)")
                continue

        episodes = int(progress.data["runs"].get(f"e3:{sid}:episodes", 0)) if resume else 0
        run = int(progress.data["runs"].get(f"e3:{sid}:runs", 0)) if resume else 0
        target = E3_EPISODES_PER_LEVEL
        while episodes < target:
            run += 1
            head(f"{sid} — прогон {run} (эпизодов набрано {episodes}/{target})")
            wait_enter("Предмет на месте, зона свободна. Готов?")
            run_dir = CAPTURES / "speed" / sid / f"run_{run:02d}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            cfg = make_cfg(
                scenario_id=sid,
                enable_robot_control=True,
                robot_ctrl_speed=int(sc.get("robot_ctrl_speed", lv["speed"])),
                run_duration_s=float(E3_DURATION_S),
                enable_decision_log=True,
                decision_log_path=str(run_dir / "decisions.csv"),
                enable_objects_log=True,
                objects_log_path=str(run_dir / "objects.csv"),
                enable_stop_log=True,
                stop_log_path=str(run_dir / "stop_events.csv"),
                enable_recording=False,      # серия времени: диск не нагружаем
                dump_masks_every_n=0,
                perf_log_path=str(run_dir / "perf_log.csv"),
                show_o3d_window=False,
            )
            countdown(E3_DURATION_S, "Прогон на скорости "
                      f"{sc.get('robot_ctrl_speed', lv['speed'])}%")
            run_capture(cfg)

            stop_rows = read_csv(run_dir / "stop_events.csv")
            say(f"\nСистема зафиксировала срабатываний DANGER: {len(stop_rows)}")
            n_ep = ask_int("Сколько проходов руки мимо объекта было в этом прогоне",
                           len(stop_rows) or 2)
            for k in range(1, n_ep + 1):
                contact = "1" if ask_yes(f"  Эпизод {k}: был контакт?",
                                         default=False) else "0"
                sr = stop_rows[k - 1] if k - 1 < len(stop_rows) else {}
                if sr:
                    sr["contact"] = contact
                append_row(CAPTURES / "speed_index.csv", E3_INDEX_FIELDS, {
                    "scenario_id": sid,
                    "speed_pct": sc.get("robot_ctrl_speed", lv["speed"]),
                    "run": run,
                    "episode": k,
                    "contact": contact,
                    "min_dist_m_at_danger": sr.get("min_dist_m_at_danger", ""),
                    "total_reaction_ms": sr.get("total_reaction_ms", ""),
                    "path": str(run_dir),
                })
                episodes += 1
            if stop_rows:
                with open(run_dir / "stop_events.csv", "w", newline="",
                          encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(stop_rows[0].keys()))
                    w.writeheader()
                    w.writerows(stop_rows)

            progress.data["runs"][f"e3:{sid}:episodes"] = episodes
            progress.data["runs"][f"e3:{sid}:runs"] = run
            progress.save()

            # Остаточная дистанция у нуля — статистика хвоста, её надо добирать.
            dists = [float(r["min_dist_m_at_danger"]) for r in stop_rows
                     if r.get("min_dist_m_at_danger")]
            if episodes >= target and dists and min(dists) < E3_TOPUP_MARGIN_M:
                say(f"\n  Минимальная остаточная дистанция {min(dists) * 1000:.1f} мм —")
                say(f"  ближе {E3_TOPUP_MARGIN_M * 1000:.0f} мм к нулю. Это хвост")
                say("  распределения, по одному-двум эпизодам его не оценить.")
                if ask_yes("Добрать ещё 5 эпизодов на этом уровне?", default=True):
                    target += 5
                    progress.note(f"e3 {sid}: добор эпизодов "
                                  f"(min остаточная {min(dists) * 1000:.1f} мм)")
            if episodes < target and not ask_yes("Продолжить этот уровень?"):
                say("Пауза. Продолжить: --stage e3 --resume")
                return
        progress.mark(f"e3:{sid}", {"episodes": episodes, "runs": run})
    say("\n[OK] Серия Э3 завершена.")


# ── этап report ──────────────────────────────────────────────────────────────

def _detection_rate() -> list:
    """Доля обнаруженных объектов по конфигурациям Э1 с несколькими предметами.

    compute_metrics такого не считает (там метрика по кадрам, а не по числу
    объектов), поэтому считаем здесь: сколько объектов система видела в кадре
    против того, сколько их реально стояло.
    """
    out = []
    for row in read_csv(CAPTURES / "e1_index.csv"):
        n_true = int(row.get("n_objects_true", 0) or 0)
        if n_true <= 0:
            continue
        dec = read_csv(Path(row["path"]) / "decisions.csv")
        if not dec:
            continue
        seen = [int(r.get("n_objects", 0) or 0) for r in dec]
        out.append({
            "config_id": row["config_id"],
            "block": row["block"],
            "n_objects_true": n_true,
            "n_frames": len(seen),
            "mean_detected": round(sum(seen) / len(seen), 3),
            "detection_rate": round(sum(seen) / (len(seen) * n_true), 4),
            "frames_all_detected_pct": round(
                100.0 * sum(1 for s in seen if s >= n_true) / len(seen), 2),
            "height_mm": row.get("height_mm", ""),
            "lighting": row.get("lighting", ""),
            "object_type": row.get("object_type", ""),
        })
    return out


def _run_metrics(label: str, dirs: list, gt_paths: list, out_dir: Path) -> dict:
    """Запустить compute_metrics по набору прогонов. Возвращает сводку."""
    perf = [str(Path(d) / "decisions.csv") for d in dirs
            if (Path(d) / "decisions.csv").exists()]
    objs = [str(Path(d) / "objects.csv") for d in dirs
            if (Path(d) / "objects.csv").exists()]
    stops = [str(Path(d) / "stop_events.csv") for d in dirs
             if (Path(d) / "stop_events.csv").exists()]
    if not perf:
        return {"label": label, "skipped": "нет decisions.csv"}
    cmd = [sys.executable, str(ROOT / "tools" / "compute_metrics.py"),
           "--label", label, "--out-dir", str(out_dir), "--perf", *perf]
    if objs:
        cmd += ["--objects", *objs]
    if stops:
        cmd += ["--stop-events", *stops]
    if gt_paths:
        cmd += ["--gt", *[str(p) for p in gt_paths]]
    idx = CAPTURES / "scenarios_index.csv"
    if idx.exists():
        cmd += ["--scenarios-index", str(idx)]
    say(f"\n$ compute_metrics --label {label}  ({len(perf)} прогонов)")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        say(f"[!] compute_metrics вернул {r.returncode}")
        say(r.stdout[-2000:])
        say(r.stderr[-2000:])
        return {"label": label, "error": r.stderr[-500:] or "см. вывод"}
    warns = [ln.strip() for ln in r.stdout.splitlines() if "[WARN]" in ln]
    if not gt_paths:
        # Серия без разметки (Э3 меряет время, а не обнаружение) — жалоба на
        # отсутствие пересечения с GT тут ожидаема и ни о чём не говорит.
        warns = [w for w in warns if "GT" not in w]
    for w in warns:
        say(f"    {w}")
    # metrics_<label>.csv лежит «длинным» видом: строка = metric,value.
    csv_path = out_dir / f"metrics_{label}.csv"
    summary = {r["metric"]: r["value"] for r in read_csv(csv_path)
               if r.get("metric")}
    return {"label": label, "summary": summary, "warnings": warns,
            "n_runs": len(perf)}


def stage_report(progress: Progress) -> None:
    head("ЭТАП REPORT — метрики и сводный отчёт")
    metrics_dir = CAPTURES / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results = []

    e1_rows = read_csv(CAPTURES / "e1_index.csv")
    if e1_rows:
        gt = [CAPTURES / f"gt_{r['config_id']}.csv" for r in e1_rows]
        gt = [p for p in gt if p.exists()]
        results.append(_run_metrics("E1_static", [r["path"] for r in e1_rows],
                                    gt, metrics_dir))

    e2_rows = read_csv(CAPTURES / "scenarios_index.csv")
    if e2_rows:
        gt = sorted(CAPTURES.glob("gt_S*.csv"))
        results.append(_run_metrics("E2_dynamic", [r["path"] for r in e2_rows],
                                    gt, metrics_dir))

    e3_rows = read_csv(CAPTURES / "speed_index.csv")
    if e3_rows:
        dirs = sorted({r["path"] for r in e3_rows})
        results.append(_run_metrics("E3_speed", dirs, [], metrics_dir))

    # Доля обнаруженных (Э1, многообъектные конфигурации)
    det = _detection_rate()
    if det:
        path = metrics_dir / "e1_detection_rate.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(det[0].keys()))
            w.writeheader()
            w.writerows(det)
        say(f"[OUT] {path}")

    # ── EXPERIMENT_REPORT.md ─────────────────────────────────────────────────
    from app.recorder import git_info
    commit, branch = git_info(ROOT)
    extr = {}
    if (ROOT / "extrinsic.json").exists():
        extr = json.loads((ROOT / "extrinsic.json").read_text(encoding="utf-8"))
    plane = {}
    if (ROOT / "table_plane.json").exists():
        plane = json.loads((ROOT / "table_plane.json").read_text(encoding="utf-8"))
    ho = extr.get("holdout") or {}
    ho_txt = f"{ho['rms_mm']:.1f} мм (LOO, n={ho['n']})" if ho else "—"

    L = [
        "# Отчёт по экспериментам",
        "",
        f"Сформирован: {datetime.now():%Y-%m-%d %H:%M}",
        f"Сессия начата: {progress.data.get('started', '?')}",
        "",
        "## Версия кода и параметры",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Ветка | `{branch}` |",
        f"| Коммит | `{commit[:12]}` |",
        f"| Порог DANGER | {DANGER_M} м |",
        f"| Порог WARN | {AppConfig.collision_warn_dist_m} м |",
        f"| RMS калибровки | {extr.get('rms_px', '—')} px |",
        f"| Точек / инлайеров | {extr.get('n_points', '—')} / "
        f"{extr.get('n_inliers', '—')} |",
        f"| Hold-out калибровки | {ho_txt} |",
        f"| Порог высоты (gate_min_m) | {plane.get('gate_min_m', '—')} м |",
        f"| ROI рабочей зоны | {plane.get('roi_px', '—')} |",
        "",
    ]
    L += ["## Что запускалось", "", "| Серия | Конфигураций/прогонов | Записано |",
          "|---|---|---|",
          f"| Э1, статика | {len(e1_rows)} конфигураций | "
          f"{sum(int(r.get('n_frames', 0) or 0) for r in e1_rows)} кадров |",
          f"| Э2, динамика | {len(e2_rows)} прогонов | "
          f"{sum(int(r.get('n_danger_frames', 0) or 0) for r in e2_rows)} кадров DANGER |",
          f"| Э3, скорость | {len(e3_rows)} эпизодов | запись кадров выключена |",
          ""]

    for res in results:
        L += [f"## Метрики: {res['label']}", ""]
        if res.get("skipped"):
            L += [f"> Пропущено: {res['skipped']}", ""]
            continue
        if res.get("error"):
            L += [f"> Ошибка расчёта: {res['error']}", ""]
            continue
        s = res.get("summary", {})
        keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                ("fp_rate_frames_pct", "Ложные срабатывания, % кадров"),
                ("fn_rate_frames_pct", "Пропуски, % кадров"),
                ("dist_mae_m", "MAE расстояния, м"),
                ("dist_rmse_m", "RMSE расстояния, м"),
                ("detect_latency_ms_median", "Задержка детекции, мс (медиана)"),
                ("stop_time_ms_median", "Время команды стопа, мс (медиана)"),
                ("total_reaction_ms_median", "Полная реакция, мс (медиана)"),
                ("total_reaction_ms_p95", "Полная реакция, мс (p95)"),
                ("min_safe_dist_m_min", "Минимальная безопасная дистанция, м")]
        L += ["| Метрика | Значение |", "|---|---|"]
        for k, ru in keys:
            if s.get(k) not in (None, ""):
                L.append(f"| {ru} | {s[k]} |")
        L += [f"| Прогонов в наборе | {res.get('n_runs', 0)} |", ""]
        L += [f"Подробно: `captures/metrics/metrics_{res['label']}.md`", ""]

    if det:
        L += ["## Доля обнаруженных объектов (Э1, многообъектные конфигурации)", "",
              "| Конфигурация | Блок | Объектов | Доля обнаруженных | "
              "Кадров со всеми |", "|---|---|---|---|---|"]
        for d in det:
            L.append(f"| {d['config_id']} | {d['block']} | {d['n_objects_true']} | "
                     f"{d['detection_rate']:.3f} | {d['frames_all_detected_pct']:.1f}% |")
        L.append("")

    heights = [r for r in e1_rows if r.get("height_mm")]
    if heights:
        L += ["## Минимальная различимая высота объекта", "",
              f"Рабочий порог высоты (gate_min_m) = {plane.get('gate_min_m', '—')} м. "
              "Предмет ниже него не выделяется в принципе.", "",
              "| Высота, мм | Конфигурация | Кадров |", "|---|---|---|"]
        for r in sorted(heights, key=lambda x: int(x["height_mm"])):
            L.append(f"| {r['height_mm']} | {r['config_id']} | {r.get('n_frames', '')} |")
        L += ["", "Числа обнаружения по этим конфигурациям — в "
              "`metrics_E1_static.md` и `e1_detection_rate.csv`.", "",
              f"Замеры этого блока сделаны на расстоянии 80 мм, а это ровно порог "
              f"DANGER ({DANGER_M} м), поэтому вердикт DANGER/SAFE здесь пограничный "
              "и в выводы о высоте не идёт. Читать надо долю обнаруженных: высота, "
              "начиная с которой она перестаёт быть нулевой, и есть минимальная "
              "различимая высота объекта.", ""]

    notes = progress.data.get("notes", [])
    warns = [w for res in results for w in res.get("warnings", [])]
    L += ["## Предупреждения и пропущенные шаги", ""]
    if notes or warns:
        L += [f"- {n}" for n in notes] + [f"- {w}" for w in warns]
    else:
        L += ["Нет."]
    L += ["", "## Сырые данные", "",
          "Кадры (`captures/**/frames/`) в репозиторий не попадают — по ним",
          "считается абляция офлайн через `source_mode='playback'`. Они должны",
          "остаться на диске.", ""]

    report = CAPTURES / "EXPERIMENT_REPORT.md"
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    say(f"\n[OUT] {report}")

    # ── manifest.json ────────────────────────────────────────────────────────
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "branch": branch,
        "commit": commit,
        "calibration": {
            "extrinsic": rel(CAPTURES / "calibration"),
            "rms_px": extr.get("rms_px"),
            "holdout_mm": ho.get("rms_mm"),
            "gate_min_m": plane.get("gate_min_m"),
        },
        "series": {
            "e1": [{"config_id": r["config_id"], "path": r["path"],
                    "n_frames": r.get("n_frames"), "metric": r.get("metric")}
                   for r in e1_rows],
            "e2": [{"scenario_id": r["scenario_id"], "run": r["run"],
                    "path": r["path"], "n_stop_events": r.get("n_stop_events")}
                   for r in e2_rows],
            "e3": [{"scenario_id": r["scenario_id"], "speed_pct": r.get("speed_pct"),
                    "episode": r.get("episode"), "path": r["path"]}
                   for r in e3_rows],
        },
        "metrics": sorted(rel(p) for p in metrics_dir.glob("*")),
        "figures": sorted(rel(p) for p in (CAPTURES / "figures").glob("*.png")),
        "reports": [rel(report)],
        "notes": notes,
    }
    mpath = CAPTURES / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    say(f"[OUT] {mpath}")
    progress.mark("report")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Мастер сбора экспериментальных данных (docs/EXPERIMENTS.md)")
    ap.add_argument("--stage", required=True,
                    choices=["calib", "e1", "e2", "e3", "report"])
    ap.add_argument("--resume", action="store_true",
                    help="Продолжить с места остановки (captures/progress.json)")
    args = ap.parse_args()

    CAPTURES.mkdir(parents=True, exist_ok=True)
    progress = Progress()
    say(f"Состояние сессии: {PROGRESS}")
    if args.resume:
        say(f"Уже выполнено шагов: {len(progress.data['done'])}")

    try:
        if args.stage == "calib":
            stage_calib(progress)
        elif args.stage == "e1":
            stage_e1(progress, args.resume)
        elif args.stage == "e2":
            stage_e2(progress, args.resume)
        elif args.stage == "e3":
            stage_e3(progress, args.resume)
        elif args.stage == "report":
            stage_report(progress)
    except KeyboardInterrupt:
        say("\n\nПрервано оператором. Прогресс сохранён.")
        say(f"Продолжить: python tools/run_experiments.py --stage {args.stage} --resume")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
