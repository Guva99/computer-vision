# Валидация системы: сбор метрик для статьи

Инфраструктура закрывает замечание рецензента: «заявлено обнаружение
препятствий и предотвращение столкновений, а измерена только
производительность конвейера». Все механизмы сбора данных — за флагами
`AppConfig` (`realsense_io.py`); **по умолчанию всё выключено, поведение
системы и частота кадров не меняются**. Логика детекции, пороги и решающее
правило не трогались: добавлены только возврат данных и логирование.

> Замер накладных расходов (12 кадров × 12 повторов, `PerceptionPipeline.process`):
> HEAD 55.6 мс/кадр, текущий код с выключенными флагами 55.3 мс/кадр — разница
> внутри разброса (σ ≈ 3 мс). С включёнными логами 56.1 мс/кадр.

## Как это запускать

Весь стенд ведёт мастер — он печатает, что расставить, ждёт подтверждения,
отсчитывает время записи и складывает артефакты туда, где их ждёт аналитика:

```bash
python tools/run_experiments.py --stage calib    # проверка калибровки
python tools/run_experiments.py --stage e1       # статика
python tools/run_experiments.py --stage e2       # динамика по сценариям
python tools/run_experiments.py --stage e3       # серия скорости
python tools/run_experiments.py --stage report   # метрики и отчёт
python tools/run_experiments.py --stage e2 --resume   # продолжить после перерыва
```

Состояние сессии — `captures/progress.json`, серию можно прервать и продолжить.
Методология этапов — в [EXPERIMENTS.md](EXPERIMENTS.md).

```
0. Перекалибровка extrinsic  →  1. Прогоны серий (запись + логи)
→  2. Разметка ground truth  →  3. Реплей записей (абляция)
→  4. Расчёт метрик и таблиц для статьи
```

---

## 0. Перекалибровка внешних параметров (ОБЯЗАТЕЛЬНО ПЕРВОЙ)

Текущий `extrinsic.json`: **RMS 21.3 px, 12 точек, 7 инлайеров** — заявленный
порог (RMS < 5 px, >= 30 точек) не выполнен ни по одному пункту, метрические
выводы на этой калибровке недостоверны.

```bash
python calibrate_click_fk.py            # цель: >=30 точек, RMS < 5 px
```

Скрипт печатает и сохраняет в `extrinsic.json`:
- RMS репроекции, число точек и инлайеров, Z-разброс;
- **hold-out (leave-one-out)**: каждая точка по очереди исключается из
  подгонки, поза решается по остальным, ошибка меряется на исключённой и
  переводится в миллиметры как `err_mm = err_px · z / fx`. Поле `holdout`
  в JSON: `rms_px`, `rms_mm`, `median_mm`, `max_mm`, `n`.

Разница принципиальная: RMS считается на тех же точках, на которых решалась
задача, и систематически оптимистична; hold-out показывает ошибку на НОВОЙ
точке. **В статью идёт hold-out в мм** — это и есть заявляемая точность
пересчёта камера→база.

`--stage calib` печатает эти числа, копирует `extrinsic.json` и
`table_plane.json` в `captures/calibration/` с отметкой времени и пишет
`captures/calibration/report.md`.

### Порог высоты объекта — следствие калибровки плоскости

`gate_min_m` из `table_plane.json` **перезаписывает**
`collision_obj_plane_height_min_m` при загрузке плоскости
(`vision/collision_service.py`, `_load_table_plane`). Сейчас это 0.04 м, а не
0.025 м из конфига. Это ДЕЙСТВУЮЩИЙ порог: предмет ниже него не выделяется
вообще. Пересъёмка плоскости меняет чувствительность детекции — после неё
серию Э1 по высоте надо переснимать. `--stage calib` печатает текущее значение
и предупреждает об этом.

## 1. Прогоны серий (на стенде)

Матрица описана в `scenarios.yaml`. Э2 идёт по S01–S08 (3 повтора, прогон 30 с),
S09/S10 относятся к Э3 (серия скорости) и в переборе динамики не участвуют.

Мастер включает на каждый прогон: покадровый лог решений, лог объектов, лог
остановов, запись RGB-D + углов, дамп масок, `scenario_id` во всех логах.
Артефакты: `captures/scenarios/<sid>/run_<k>_3d/`, сводка —
`captures/scenarios_index.csv`.

Ручной запуск без мастера — флаги в `realsense_io.py: AppConfig`:
`enable_decision_log`, `enable_objects_log`, `enable_stop_log`,
`enable_recording`, `dump_masks_every_n`, `scenario_id`, `run_duration_s`.

### Как меряется время реакции

`stop_events.csv` пишет четыре метки монотонных часов (`time.perf_counter`):
`t_frame` (кадр получен), `t_decision` (решение перешло в DANGER после
дебаунса), `t_command` (запись `$OV_PRO=0` вернула управление), `t_stopped`
(движение прекратилось).

`t_stopped` — **детектируемая, а не измеряемая величина**: прямого «скорость
TCP = 0» контроллер не отдаёт. Отдельный поток по своему соединению опрашивает
состояние и ловит момент остановки; источник выбирается автоматически —
`$VEL_ACT`, иначе `$PRO_STATE`, иначе фолбэк `$AXIS_ACT` (углы перестали
меняться). Что реально использовалось и с каким разрешением, пишется в те же
строки: `stop_detect_source` и `t_stopped_precision_ms` (фактический период
опроса, контроллер отвечает ~60 мс). Эту погрешность надо указать в статье, а
не выдавать оценку за прямое измерение.

В Э3 запись кадров выключена: задержки не должны измеряться под дисковой
нагрузкой (~46 МБ/с при 30 FPS).

## 2. Разметка ground truth (оффлайн)

```bash
python tools/annotate.py --recording captures/scenarios/S04_cube_on_path/run_01_3d/recording
```

Метки: `obj_present`, `danger_true`, `dist_true_m`; маски препятствия и
манипулятора для IoU (клавиши `m`/`n`). Метка действует вперёд до следующего
изменения — размечать нужно только смены состояния, примерно два клика на
эпизод. Выход: `captures/gt_<scenario_id>.csv`, `captures/gt_masks/<sid>/`.
Импорт из внешних инструментов: `--import-labelme <dir>` / `--import-cvat <xml>`.

**Для Э1 разметка не нужна**: эталон вводится при расстановке (замер линейкой),
мастер сам пишет `captures/gt_<config_id>.csv` на все кадры конфигурации.
Эталон расстояния пишется только там, где в кадре ОДИН объект — иначе
`min_dist_m` относится к ближайшему из нескольких и сопоставлять его с
конкретным замером не с чем; в многообъектных блоках метрика другая (доля
обнаруженных, `captures/metrics/e1_detection_rate.csv`).

## 3. Оффлайн-реплей (абляция без стенда)

```bash
python -c "from realsense_io import AppConfig; from app.runner import AppRunner; AppRunner(AppConfig(source_mode='playback', playback_path='captures/scenarios/S04_cube_on_path/run_01_3d/recording', enable_decision_log=True, decision_log_path='captures/replay_3d/decisions.csv', enable_objects_log=True, objects_log_path='captures/replay_3d/objects.csv', scenario_id='S04_cube_on_path')).run()"
```

Кадры отдаются строго последовательно из записи, углы A1..A6 берутся из
`joints.csv` по номеру кадра, `enable_robot_control` принудительно `False`.
Номер кадра берётся из ИМЕНИ ФАЙЛА, а не из счётчика цикла: в записи бывают
пропуски, и только имя гарантирует, что логи реплея сошьются с той же
разметкой GT. Так пересчитываются метрики после правки порогов — без
повторной съёмки.

## 4. Расчёт метрик (без камеры, робота и pyrealsense2)

`--stage report` делает это сам по каждой серии и собирает
`captures/EXPERIMENT_REPORT.md` и `captures/manifest.json`. Вручную:

```bash
python tools/compute_metrics.py --label 3D \
    --perf captures/scenarios/*/run_*_3d/decisions.csv \
    --objects captures/scenarios/*/run_*_3d/objects.csv \
    --stop-events captures/scenarios/*/run_*_3d/stop_events.csv \
    --gt captures/gt_*.csv \
    --masks captures/scenarios/S04_cube_on_path/run_01_3d/masks \
    --gt-masks captures/gt_masks/S04_cube_on_path \
    --scenarios-index captures/scenarios_index.csv
```

Флаг называется `--perf` по историческим причинам, но читает он схему
`decisions.csv` (`frame`, `scenario_id`, `collision_level`, `min_dist_m`).

Считает: precision/recall/F1 по DANGER (по кадрам и по эпизодам), % ложных
остановов (на кадр и на прогон), % пропусков, MAE/RMSE/max ошибки расстояния,
IoU масок, задержки (detect/stop/total: mean/median/p95/max), минимальную
безопасную дистанцию `d_safe = d_danger − v_звена · t_reaction`, разбивку по
факторам сценария и сводку FN по причинам отбраковки (`reject_reason`).
Экспорт: CSV + Markdown (`captures/metrics/`).

## Дамп масок для IoU

`dump_masks_every_n=30` в AppConfig, каталог — `masks_dump_path` (по умолчанию
`<recording_path>/masks`). Имена файлов совпадают с тем, что пишет
`annotate.py` в gt_masks: `frame_%06d_{obstacle,manip}.png`.

## Причины отбраковки объектов (`objects.csv`)

Строка на каждого рассмотренного кандидата. Пустая причина = объект дошёл до
оценки расстояния (заполнены `dist_m` и `level`). Непустая — какой этап снял:

| Причина | Где | Что означает |
|---|---|---|
| `area` | `collision.py` | площадь компоненты вне `[min_area, max_area]` |
| `shape` | `collision.py` | не компактный блоб (заполненность / вытянутость) |
| `border` | `collision.py` | прилип к кромке кадра |
| `pts3d` | `collision.py` | мало валидных 3D-точек |
| `keepz` | `collision.py` | мало точек после отсечения по глубине |
| `span` | `collision.py` | размах высот p90−p10 меньше `plane_min_span_m` |
| `arm_reject` | `collision_service.py` | прилип к руке и совпал с ней по глубине |
| `not_confirmed` | `collision_service.py` | трек не набрал `persist_frames` |

По ним `compute_metrics` сводит FN-кадры: видно, объект не выделился вовсе,
был отбракован гейтом или дистанция посчиталась завышенной.

## Важные примечания для текста статьи

1. **`cv_2d_method/run_iter0…3.py` и `iter0…3/*.csv` — синтетические данные**
   (в шапках скриптов: «ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. Реальные кадры НЕ
   используются»). Их назначение — оценка вычислительной сложности конвейера
   облака точек. Они НЕ являются экспериментальной валидацией.

2. **Пороги — два разных параметра, не путать:**
   - Формула (10) статьи — расстояние до FK-сфер хвата
     `d_sph = max(0, min‖p−c‖ − ρ)`: реализация
     `collision.py: min_distance_to_robot()`, радиусы ρ —
     `collision_link_radii_m` / `collision_gripper_radius_m`
     (`realsense_io.py`). Единицы — **метры** (кадр камеры).
   - Классификация (формула 13) — `collision_danger_dist_m` и
     `collision_warn_dist_m` (`realsense_io.py`), тоже метры.
   - Отдельно в 2D-контуре есть своя верификация по глубине: допуск ±0.08 м
     (`gripper_detector.py:412`; `obstacle_detector.py:26`
     `height_tolerance = 0.08`) и `obstacle_threshold = 0.60 м` — это
     параметры 2D-метода, не связанные с `collision_danger_dist_m`.

3. **Расхождение статьи и кода:** в ARTICLE_CONTEXT.md / формуле (13)
   указан порог DANGER **0.05 м**, а в коде сейчас
   `collision_danger_dist_m = 0.08` (плюс гистерезис выхода
   `collision_danger_exit_dist_m = 0.10`). В тексте статьи значение нужно
   обновить или обосновать выбор 0.08 м запасом на время реакции.

## Что остаётся на человеке

- физическая расстановка препятствий по описаниям мастера;
- эталонные замеры расстояний (линейка/штангенциркуль) при расстановке Э1;
- ответы про контакт после прогонов Э2 и Э3;
- прогоны на стенде (камера + KUKA) и перекалибровка;
- разметка записей Э2 в `tools/annotate.py` (только смены состояния).

## Чего в этой ветке НЕТ

`tools/run_scenarios.py`, режим `benchmark_mode`, реплей 2D-контура по общей
записи и сравнение «2D против 3D» одной командой существовали в планах ветки
`feature/validation`, которая в `origin` не попала. `compute_metrics.py
--compare` работает, но принимает уже готовые сводки обоих контуров.
