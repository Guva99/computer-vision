# Валидация системы: сбор метрик для статьи

Инфраструктура закрывает замечание рецензента: «заявлено обнаружение
препятствий и предотвращение столкновений, а измерена только
производительность конвейера». Все новые механизмы — за флагами
конфигурации; **по умолчанию всё выключено, поведение системы не меняется**.
Алгоритмы детекции не менялись (только возврат данных, хуки и логирование).

## Конвейер валидации (порядок работы)

```
0. Перекалибровка extrinsic  →  1. Прогоны по матрице сценариев (запись+логи)
→  2. Разметка ground truth  →  3. Реплей второго метода на тех же записях
→  4. Расчёт метрик и таблиц для статьи
```

---

## 0. Перекалибровка внешних параметров (ОБЯЗАТЕЛЬНО ПЕРВОЙ)

Текущий `extrinsic.json`: RMS 27 px, 6 инлайеров из 20 — метрические выводы
недостоверны, перевод камера→база может ошибаться на 5–10 см.

```bash
python calibrate_click_fk.py            # цель: >=30 точек, RMS < 5 px
```

Что нового в скрипте:
- итеративная отбраковка выбросов (порог `--outlier-px`, по умолчанию 8 px);
- отчёт качества: RMS, инлайеры, распределение остатков (px и мм), Z-разброс,
  покрытие кадра;
- **hold-out валидация**: часть точек исключается из подгонки, ошибка на них
  печатается в миллиметрах (`err_mm ≈ err_px · z / fx`) — прямой ответ на
  требование «ошибка оценки расстояния»;
- при RMS > 5 px или инлайеров < 15 — громкое предупреждение и запрос
  подтверждения перед сохранением;
- клавиша `v` — проверка уже сохранённого `extrinsic.json` на свежих
  контрольных точках (не участвовавших в подгонке);
- история всех попыток — `captures/calibration_log.json`.

**В статью:** ошибка hold-out в мм из отчёта = экспериментальная точность
пересчёта координат камера→робот.

## 1. Прогоны по матрице сценариев (на стенде)

Матрица описана в `scenarios.yaml` (тип препятствия, дистанция, скорость,
освещение, перекрытие, ожидание DANGER, число повторов). Раннер ждёт
подтверждения оператора перед каждым повтором (препятствие ставится вручную).

```bash
pip install pyyaml                       # один раз
python tools/run_scenarios.py --contour 3d
python tools/run_scenarios.py --contour 2d
python tools/run_scenarios.py --contour 3d --only S04_cube_on_path --repeats 3
```

Каждый прогон автоматически включает: покадровый лог решений (Задача 1), лог
остановов (Задача 2), запись RGB-D + углов (Задача 3), `scenario_id` во всех
логах. Артефакты: `captures/scenarios/<sid>/run_<k>_<contour>/`, сводка —
`captures/scenarios_index.csv`.

Ручной запуск с флагами (без раннера) — в `realsense_io.py: AppConfig`:
`enable_decision_log`, `enable_stop_log`, `enable_recording`, `scenario_id`;
в 2D — те же константы в `cv_2d_method/src/constants/config.py` или CLI
`cv_2d_method/main.py --enable-decision-log --enable-stop-log ...`.

## 2. Разметка ground truth (оффлайн)

```bash
python tools/annotate.py --recording captures/scenarios/S04_cube_on_path/run_01_3d/recording
```

Метки: `obj_present`, `danger_true`, `dist_true_m` (ввод с клавиатуры по
физическому замеру или измерение по двум кликам через карту глубины); маски
препятствия/манипулятора для IoU (клавиши `m`/`n`). Метка действует вперёд до
следующего изменения — размечать нужно только смены состояния.
Выход: `captures/gt_<scenario_id>.csv`, `captures/gt_masks/<scenario_id>/`.
Импорт из внешних инструментов: `--import-labelme <dir>` / `--import-cvat <xml>`.

## 3. Оффлайн-реплей (сравнение 2D и 3D на одних данных)

```bash
# 3D-контур по записи (робот не нужен; управление принудительно отключено)
python -c "from realsense_io import AppConfig; from app.runner import AppRunner; AppRunner(AppConfig(source_mode='playback', playback_path='captures/scenarios/S04_cube_on_path/run_01_3d/recording', enable_robot_control=False, use_robot_kinematics=False, enable_decision_log=True, perf_log_path='captures/replay_3d/perf_log.csv', objects_csv_path='captures/replay_3d/objects.csv', scenario_id='S04_cube_on_path')).run()"

# 2D-контур по той же записи
cd cv_2d_method
python main.py --no-robot --playback ../captures/scenarios/S04_cube_on_path/run_01_3d/recording --enable-decision-log --perf-log ../captures/replay_2d/perf_log.csv --objects-log ../captures/replay_2d/objects.csv --scenario-id S04_cube_on_path
```

Углы A1..A6 воспроизводятся из `joints.csv` записи (нужны 3D-контуру для FK
и маски манипулятора). Дренаж очереди кадров в playback отсутствует по
построению (кадры отдаются строго последовательно).

## 4. Расчёт метрик (без камеры, робота и pyrealsense2)

```bash
python tools/compute_metrics.py --label 3D \
    --perf captures/scenarios/*/run_*_3d/perf_log.csv \
    --objects captures/scenarios/*/run_*_3d/objects.csv \
    --stop-events captures/scenarios/*/run_*_3d/stop_events.csv \
    --gt captures/gt_*.csv \
    --masks captures/masks/3d --gt-masks captures/gt_masks/S04_cube_on_path \
    --scenarios-index captures/scenarios_index.csv

python tools/compute_metrics.py --label 2D --perf ... (логи 2D)

# сводная таблица «2D против 3D»
python tools/compute_metrics.py --compare captures/metrics/metrics_3D.csv captures/metrics/metrics_2D.csv
```

Считает: precision/recall/F1 по DANGER (по кадрам и по эпизодам), % ложных
остановов (на кадр и на прогон), % пропусков, MAE/RMSE/max ошибки расстояния,
IoU масок, задержки (detect/stop/total: mean/median/p95/max), минимальную
безопасную дистанцию `d_safe = d_danger − v_звена · t_reaction`, разбивку по
факторам сценария и сводку FN по причинам отбраковки (`reject_reason`).
Экспорт: CSV + Markdown (`captures/metrics/`) — готово для вставки в статью.

## Дамп масок для IoU (при необходимости)

3D: `dump_masks_every_n=10` в AppConfig → `captures/masks/3d/`.
2D: `DUMP_MASKS_EVERY_N=10` в конфиге → `captures/masks/2d/`.
Имена файлов совпадают с gt_masks (`frame_%06d_{obstacle,manip}.png`).

## Benchmark-режим (честные замеры производительности)

3D: `benchmark_mode=True` в AppConfig — принудительно отключает
`show_o3d_window` (~60 мс/кадр), `perf_window`, `perf_overlay`,
debug-мозаику и дамп масок.
2D: `BENCHMARK_MODE=True` — отключает окно мониторинга, окно облака точек
и дамп масок.

## Важные примечания для текста статьи

1. **`cv_2d_method/run_iter0…3.py` и `iter0…3/*.csv` — синтетические данные**
   (в шапках скриптов: «ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. Реальные кадры НЕ
   используются»). Их назначение — оценка вычислительной сложности конвейера
   облака точек. Они НЕ являются экспериментальной валидацией и не должны
   смешиваться с метриками Задач 1–8.

2. **Пороги — два разных параметра, не путать:**
   - Формула (10) статьи — расстояние до FK-сфер хвата
     `d_sph = max(0, min‖p−c‖ − ρ)`: реализация
     `collision.py: min_distance_to_robot()` (строки ~561–583), радиусы ρ —
     `collision_link_radii_m` / `collision_gripper_radius_m`
     (`realsense_io.py`). Единицы — **метры** (кадр камеры).
   - Классификация (формула 13) — `collision_danger_dist_m` и
     `collision_warn_dist_m` (`realsense_io.py:218–221`), тоже метры.
   - Отдельно в 2D-контуре есть своя верификация по глубине: допуск ±0.08 м
     (`gripper_detector.py:412` — жёстко зашит; `obstacle_detector.py:26`
     `height_tolerance = 0.08`) и `obstacle_threshold = 0.60 м` — это
     параметры 2D-метода, не связанные с `collision_danger_dist_m`.

3. **Расхождение статьи и кода:** в ARTICLE_CONTEXT.md / формуле (13)
   указан порог DANGER **0.05 м**, а в коде сейчас
   `collision_danger_dist_m = 0.08` (поднят коммитом «fix stable vision»,
   плюс добавлен гистерезис выхода `collision_danger_exit_dist_m = 0.10`).
   В тексте статьи значение нужно обновить (или обосновать выбор 0.08 м
   запасом на время реакции).

## Что остаётся на человеке

- физическая расстановка препятствий по матрице сценариев;
- эталонные замеры расстояний (рулетка/штангенциркуль) для `dist_true_m`;
- прогоны на стенде (камера + KUKA) и перекалибровка (Задача 0);
- разметка записей в `tools/annotate.py`.
