from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


@dataclass
class AppConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_min_m: float = 0.15
    depth_max_m: float = 1.8
    use_robot_kinematics: bool = True
    robot_ip: str = "192.168.17.2"
    robot_port: int = 7000
    # real-time: не блокировать цикл на чтении робота
    robot_read_timeout_s: float = 0.2    # макс. ожидание ответа $AXIS_ACT (сек); контроллер под нагрузкой отвечает ~60мс
    robot_read_every_n: int = 2          # читать углы раз в N кадров (углы меняются плавно)
    o3d_render_every_n: int = 3          # обновлять тяжёлый 3D-рендер раз в N кадров
    show_o3d_window: bool = False        # 3D-окно Open3D (~60 мс/кадр). True = для скринов в статью
    # Мониторинг ресурсов ПК (CPU/RAM/GPU) для анализа нагрузки в статье
    perf_monitor: bool = True            # снимать характеристики и писать CSV
    perf_overlay: bool = True            # показывать CPU/RAM/GPU на кадре
    perf_window: bool = True             # отдельное окно с графиками (как в 2D-программе)
    perf_gpu_every_n: int = 15           # снимать GPU раз в N кадров (счётчик дороговат)
    perf_log_path: str = "captures/perf_log.csv"
    # Толстый «капсульный» FK-скелет (обтягивает тело руки по радиусам звеньев)
    fk_skeleton_thick: bool = True       # False = тонкая ось 2px
    fk_skeleton_alpha: float = 0.45      # прозрачность толстого скелета (рука просвечивает)
    fk_skeleton_thick_scale: float = 0.7 # множитель толщины (1.0 = полный радиус коллизии)
    output_dir: str = "captures"
    # Детекция хвата на RGB (тёмный инструмент, светлый фон)
    show_gripper_overlay: bool = False
    gripper_roi_y_fraction: float = 0.32  # искать только ниже этой линии (доля высоты кадра)
    gripper_gray_max: int = 95  # чем светлее хват — поднимите (например 110–130)
    gripper_min_area_px: int = 450
    arm_gray_min: int = 158  # белый корпус руки (хват + сустав над ним)
    near_arm_dilate_px: int = 72
    gripper_touch_arm_dilate_px: int = 32
    gripper_max_depth_delta_m: float = 0.12
    gripper_fallback_max_x_fraction: float = 0.52  # если маска руки слабая — не искать справа (блоки)
    arm_mask_min_area_px: int = 5000
    gripper_bbox_smooth_alpha: float = 0.35
    gripper_hold_miss_frames: int = 12
    gripper_track_iou_weight: float = 85.0
    gripper_track_center_weight: float = 70.0
    gripper_track_max_jump_px: int = 180
    gripper_tip_max_dist_px: int = 140
    gripper_tip_max_dist_frac: float = 0.24
    gripper_tip_allow_above_px: int = 20
    # Сустав (белое звено) сразу над хватом
    show_wrist_joint_overlay: bool = True
    wrist_band_height_px: int = 90  # узко — только зона над хватом, не весь предплечье
    wrist_band_pad_px: int = 48
    wrist_bridge_w: int = 15
    wrist_bridge_h: int = 55
    wrist_min_area_px: int = 180
    wrist_max_area_px: int = 32000
    wrist_use_near_arm: bool = True
    wrist_gray_min: int = 138  # ниже arm_gray_min — белый цилиндр у шва
    wrist_valid_dilate_px: int = 11
    wrist_valid_dilate_iters: int = 4
    wrist_fallback_without_near_arm: bool = True
    wrist_anchor_max_below_grip_px: int = 14
    wrist_min_bbox_height_px: int = 16
    wrist_min_bbox_width_px: int = 8
    wrist_max_link_span_px: int = 100
    wrist_bbox_smooth_alpha: float = 0.3  # 0 = только контур; иначе сглаженный прямоугольник зоны
    wrist_hold_miss_frames: int = 20  # сколько кадров без детекта держать последнюю зону
    wrist_use_static_roi: bool = False  # одно положение: задать доли кадра и включить
    wrist_roi_x0_frac: float = 0.12
    wrist_roi_y0_frac: float = 0.28
    wrist_roi_x1_frac: float = 0.62
    wrist_roi_y1_frac: float = 0.72
    wrist_max_dx_from_grip_px: int = 42
    wrist_max_dx_from_grip_frac: float = 0.95
    wrist_min_overlap_arm_px: int = 120
    wrist_band_height_from_gripper: float = 2.0
    wrist_ema_clamp_dx_px: int = 44
    wrist_ema_clamp_dx_frac: float = 1.0
    # FK-guided ROI
    use_fk_roi: bool = True           # включить FK-guided ROI; False = старый режим
    fk_grip_roi_radius_px: int = 60   # радиус ROI вокруг проекции A6 (фланец)
    fk_wrist_roi_radius_px: int = 70  # радиус ROI вокруг проекции A5 (запястье)
    # Длина хвата от фланца (м); 0 = только фланец; >0 = добавляет точку TCP на конце хвата
    # 0.137 = совпадает с длиной инструмента в калибровке (extrinsic.json) → скелет тянется до кончика хвата
    fk_gripper_length_m: float = 0.137
    extrinsic_path: str = "extrinsic.json"
    use_robot_self_filter: bool = True
    robot_link_radius_px: int = 16
    robot_mask_dilate_px: int = 7
    robot_mask_depth_eps_m: float = 0.09
    robot_mask_min_pixels: int = 10
    manipulator_hold_frames: int = 3  # было 8; меньше = маска руки меньше отстаёт при движении
    show_robot_mask_overlay: bool = False  # жёлтый контур манипулятора убран — оставляем только фиолетовый FK-скелет
    robot_mask_overlay_min_area_px: int = 1200
    use_manipulator_only_mode: bool = True
    manipulator_arm_gray_min: int = 150
    manipulator_seed_dilate_px: int = 19
    # seed-gate parameters
    manipulator_max_seed_dist_px: int = 110
    manipulator_min_seed_overlap_ratio: float = 0.01
    manipulator_max_component_area_frac: float = 0.30
    manipulator_table_seed_exclude_dilate_px: int = 61
    seed_gate_protect_dilate_px: int = 19
    # adaptive mask parameters
    mask_adaptive_window_px: int = 41
    mask_adaptive_k_sigma: float = 0.35
    # component filter parameters
    component_filter_min_area_px: int = 240
    component_filter_min_compactness: float = 0.05
    component_filter_reject_bottom_frac: float = 0.92
    strict_no_table_mode: bool = True
    strict_table_guard_px: int = 8
    strict_bottom_seed_overlap_min: float = 0.06
    front_view_smart_cut_enabled: bool = True
    front_view_cut_offset_px: int = 10
    front_view_seed_connectivity_min_area_px: int = 120
    # Merge dark gripper into manipulator_mask (single yellow contour)
    manipulator_include_gripper: bool = True
    gripper_merge_gray_max: int = 130
    gripper_merge_roi_radius_px: int = 75
    gripper_merge_roi_extend_down_px: int = 90
    gripper_merge_attach_dilate_px: int = 27
    gripper_merge_max_depth_delta_m: float = 0.12
    gripper_merge_min_area_px: int = 60
    gripper_merge_close_px: int = 15
    gripper_merge_allow_no_depth: bool = True
    detect_gripper_from_manipulator_mask: bool = False
    gripper_from_mask_gray_max: int = 110
    gripper_from_mask_dark_quantile: float = 0.42
    gripper_from_mask_edge_canny_lo: int = 45
    gripper_from_mask_edge_canny_hi: int = 130
    gripper_from_mask_edge_dilate_px: int = 3
    gripper_from_mask_depth_band_m: float = 0.07
    gripper_from_mask_min_area_px: int = 120
    gripper_from_mask_max_area_px: int = 8000
    gripper_from_mask_near_dilate_px: int = 19
    gripper_from_mask_score_edge_density_w: float = 2.0
    gripper_from_mask_score_boundary_overlap_w: float = 2.4
    gripper_from_mask_score_depth_consistency_w: float = 1.6
    gripper_from_mask_score_dist_to_fk_w: float = 0.022
    gripper_from_mask_score_jump_w: float = 0.018
    table_depth_m: float = 0.0        # глубина стола (м); 0 = авто-оценка
    table_depth_margin_m: float = 0.05  # запас ниже руки для отсечения стола
    table_min_y_frac: float = 0.45
    table_min_area_px: int = 6000
    table_protect_dilate_px: int = 25
    table_min_width_frac: float = 0.35
    table_max_height_frac: float = 0.60
    show_table_candidate_debug: bool = False
    # Collision: FK spheres vs scene objects (camera frame, metres)
    enable_collision: bool = True
    collision_link_radii_m: tuple = (0.16, 0.13, 0.11, 0.095, 0.075, 0.065)
    collision_gripper_radius_m: float = 0.05  # радиус сфер пальцев хвата (J6→TCP)
    collision_spheres_per_link: int = 8
    collision_voxel_m: float = 0.01
    collision_ransac_dist_m: float = 0.005
    collision_min_height_above_plane_m: float = 0.012
    collision_depth_table_pct: float = 85.0
    collision_depth_table_min_height_m: float = 0.02
    collision_dbscan_eps_m: float = 0.035
    # ── ADAPTIVE ε (ВАК §3, формулы 5–6) ──────────────────────────────────────
    # CLAUDE: весь блок "ADAPTIVE ε" можно удалить безопасно —
    #   в collision.py ищи функцию _estimate_adaptive_eps() и связанный блок
    #   внутри detect_scene_objects(). Флаг по умолчанию False → поведение
    #   не меняется пока не включишь явно.
    collision_dbscan_adaptive_eps: bool = False   # True = адаптивный ε по k-distance
    collision_dbscan_adaptive_k: int = 10         # k для kNN-оценки (= min_points)
    collision_dbscan_adaptive_kappa: float = 1.2  # масштаб: ε = κ · median(k-dist)
    collision_dbscan_adaptive_eps_min: float = 0.020  # нижняя граница clip, м
    collision_dbscan_adaptive_eps_max: float = 0.060  # верхняя граница clip, м
    # ── END ADAPTIVE ε ─────────────────────────────────────────────────────────
    collision_dbscan_min_points: int = 20
    collision_object_min_points: int = 30
    collision_object_max_extent_m: float = 0.8
    collision_warn_dist_m: float = 0.12
    # DANGER: порог поднят 0.05→0.08 — конвейер систематически завышает дистанцию
    # (эрозии масок + перцентиль), плюс нужен запас на реакцию KUKA.
    collision_danger_dist_m: float = 0.08
    # Гистерезис DANGER: войдя в DANGER, остаёмся в нём пока d <= exit-порога.
    # Раньше срабатывает, не дребезжит на границе.
    collision_danger_exit_dist_m: float = 0.10
    # Защёлка потерянного DANGER (безопасность SSM/ISO-TS-15066): если опасный
    # объект ПРОПАЛ из детекции (мигание/перекрытие рукой), держим DANGER ещё N
    # кадров, а не сбрасываем сразу. Иначе робот рвётся вперёд к «исчезнувшей»
    # башне и бьёт её. Absence of detection != absence of object.
    collision_danger_lost_hold_frames: int = 5
    # Фильтр временной устойчивости: объект засчитывается только если виден
    # >=N кадров подряд примерно в одном месте. Мигающие протечки руки (шум
    # глубины на белом пластике) скачут по кадру → отсекаются; кубик стабилен.
    collision_persist_frames: int = 8      # сколько кадров подряд нужно подтверждение
    # Fast-path: кандидат, уже находящийся в WARN-зоне руки, подтверждается за
    # 2 кадра — ложный WARN дешевле пропущенного DANGER при быстром сближении.
    collision_persist_frames_near: int = 2
    # 0.05→0.10: при частичном перекрытии объекта рукой центроид сдвигается;
    # малый порог рвал трек ровно в момент сближения → объект «исчезал» на 8 кадров.
    collision_track_match_m: float = 0.10  # порог сопоставления объекта между кадрами (м)
    collision_track_max_miss: int = 3      # удалять трек-КАНДИДАТ после N пропусков
    # Подтверждённый трек (стабильный объект) держится дольше кандидата: рука,
    # перекрывшая объект на пару кадров, не роняет его. 15→8: длинный коастинг
    # плодил призраков (бокс на старом месте) — безопасность держит защёлка
    # DANGER (collision_danger_lost_hold_frames), а дубли дедупит склейка боксов.
    collision_track_max_miss_confirmed: int = 8
    collision_focus_parts: tuple = ("gripper", "wrist", "arm")
    # Hybrid collision: body distance from manipulator-mask cloud (measured),
    # gripper from FK sphere (no depth on dark metal). Voxel for body cloud.
    collision_body_voxel_m: float = 0.02
    # Воксель прореживания точек ОБЪЕКТА перед расчётом дистанции. Полный
    # пиксельный кластер = тысячи точек → min_dist O(N·K) дорого (главный вклад
    # в scene после морфологии). 1 см: ошибка дистанции <= 1 см при DANGER 8 см.
    collision_obj_dist_voxel_m: float = 0.01
    # Robustness: use a low percentile of nearest distances (not a single
    # closest point) + erode masks to drop noisy boundary pixels.
    collision_dist_percentile: float = 2.0   # 0 = strict min
    collision_body_erode_px: int = 5
    collision_obj_erode_px: int = 3
    collision_log_path: str = "captures/collisions.log"
    collision_log_every_n: int = 30
    show_collision_overlay: bool = True
    collision_use_2d_detection: bool = True
    # 150→250: мелкие блобы шума глубины/дрейфа плоскости на пустом столе давали
    # фантомные объекты. Реальные кубики (~40-60px → 1600-3600px) сильно больше.
    collision_obj_min_area_px: int = 250
    # Склейка перекрывающихся боксов: фрагменты одного кубика (дыры глубины на
    # гранях дробят компоненту) объединяются в один объект. Слияние точек лишь
    # уточняет дистанцию (ближайшая точка сохраняется) → безопасно.
    collision_obj_merge_overlap: bool = True
    collision_obj_merge_iou: float = 0.15       # порог IoU для слияния
    collision_obj_merge_contain: float = 0.4    # ИЛИ доля меньшего бокса внутри большего
    collision_obj_max_area_frac: float = 0.25
    # 10→25: объекты, прилипшие к кромке кадра (профиль ограждения), отсекаются
    collision_obj_exclude_border_px: int = 25
    collision_obj_drop_border: bool = True
    collision_obj_min_points_3d: int = 15
    collision_obj_open_px: int = 3
    collision_obj_close_px: int = 5
    collision_obj_use_near_manip: bool = True
    collision_obj_near_manip_dilate_px: int = 130
    collision_obj_max_depth_behind_arm_m: float = 0.20
    # Исключение ореола руки: вычитаем раздутую маску манипулятора из кандидатов,
    # чтобы белые края/детали самой руки не считались «чужим объектом» вплотную.
    # ВАЖНО: большое значение «съедает» кубик при подходе руки → контакт не доходит
    # до 0 → нет DANGER. Держим маленьким (только тонкий край смешанных пикселей).
    collision_obj_manip_exclude_dilate_px: int = 28
    # Reclaim: зона исключения нужна только для ДЕТЕКЦИИ (не считать ореол руки
    # объектом). Для ДИСТАНЦИИ возвращаем компоненту её пиксели, съеденные зоной
    # исключения (дилатация компоненты ∩ маска без исключения) — иначе измеренное
    # расстояние завышается на ширину зоны и DANGER не наступает вовремя.
    collision_obj_reclaim_near_arm: bool = True
    # ── Плоскостной гейт детекции (замена цвет-гейта) ─────────────────────────
    # Объект = высота над КАЛИБРОВАННОЙ плоскостью стола в [min,max] м. Плоскость
    # берётся из table_plane.json (calibrate_table_plane.py), меряется в кадре
    # КАМЕРЫ и НЕ зависит от T_cr. Цвето-независимо: кубик любого цвета/материала
    # виден, потому что физически возвышается над столом. При отсутствии json
    # (или plane_gate=False) детекция откатывается на цвет-гейт ниже.
    collision_obj_plane_gate: bool = True
    table_plane_path: str = "table_plane.json"
    collision_obj_plane_height_min_m: float = 0.025  # ниже — шум глубины стола
    collision_obj_plane_height_max_m: float = 0.30
    # Гейт вертикальной структуры против фантомов на пустом столе (ДРЕЙФ-
    # НЕЗАВИСИМ): объект принимается, только если размах высот его точек
    # p90−p10 >= min_span. Кубик (видны верх+грани) даёт размах ~1.5-4см, плоский
    # шум/дрейф плоскости — <1см. Порог по абсолютной высоте не годится (дрейф
    # поднимает шум). 0 = выкл. Тесно? Снизь до 0.012 или переснимай плоскость.
    collision_obj_plane_min_span_m: float = 0.015
    # ── ROI рабочей зоны стола (пиксели кадра) ────────────────────────────────
    # Детекция объектов идёт ТОЛЬКО внутри этого прямоугольника. Камера и стол
    # неподвижны → зона фиксирована в кадре и НЕ зависит от T_cr. Убирает провод,
    # базу робота, фон за столом и дальние выбросы «на пустом месте» за один шаг.
    # Приоритет у roi_px из table_plane.json (calibrate_table_plane.py, тот же
    # прямоугольник, которым отмечали стол). Дефолт ниже — грубый фолбэк (640×480),
    # покрывает стол и отсекает верх кадра с базой/проводом; подстрой калибровкой.
    collision_obj_roi_gate: bool = True
    collision_obj_roi_px: tuple = (30, 245, 615, 475)  # (x0, y0, x1, y1)
    # Robot self-filter: дорисовать FK-капсулу руки в маску исключения ПЕРЕД
    # детекцией. При высотном гейте рука (тоже выше стола) с дырами в глубинной
    # маске (тёмный металл запястья/хвата, синий шланг) иначе становится ложным
    # объектом → ложный DANGER. FK-геометрии глубина не нужна. Части: сегменты
    # скелета, чьи сферы растеризуются (gripper=хват J6→TCP, wrist=запястье).
    collision_fk_self_filter: bool = True
    # ВСЕ звенья скелета, не только хват+запястье: база и корпус робота тоже
    # выше стола, а глубинная маска их местами не закрывает (тёмный металл базы,
    # синий шланг, тень, логотип) → высотный гейт метил КОРПУС как объект →
    # ложный DANGER «arm 0.0Xm» на самой руке (робот стоит). Капсула питает
    # только маску ИСКЛЮЧЕНИЯ детекции; дистанция до тела — по сырой маске.
    collision_fk_self_filter_parts: tuple = (
        "gripper", "wrist", "link4", "link3", "link2", "link1",
    )
    # Запас на погрешность extrinsic (RMS ~27px). Больше — надёжнее закрывает
    # руку, но может «съесть» кубик вплотную к хвату. 10px ≈ 2-3 см у стола.
    collision_fk_self_filter_pad_px: int = 10
    # Очистка облака РУКИ от протечек маски на стол: настоящая рука выше стола,
    # а точки на уровне стола рядом с рукой (тень/тёмные пиксели, попавшие в
    # маску) дают ложную близость к столовым объектам → ложный DANGER «arm».
    # Отсекаем точки тела ниже min_height над калиброванной плоскостью.
    collision_body_plane_clean: bool = True
    collision_body_plane_min_height_m: float = 0.03
    # Очистка облака РУКИ от ЦВЕТНЫХ протечек: рука KUKA белая/серая (низкая
    # насыщенность), яркий кубик частично проходит порог белого сегментатора →
    # его пиксели в облаке тела дают ложный DANGER «arm 0.02м» на кубике вдали от
    # руки. Выкидываем насыщенные (S>=sat_max) пиксели из тела. Реальную руку
    # (десатурирована) не трогает; T_cr-независимо.
    collision_body_color_clean: bool = True
    collision_body_sat_max: int = 70
    # ДИАГНОСТИКА (временно): печатать, какой гейт отбраковал объект у руки.
    # [REJECT2D] — стадия детекции, [CONFIRM] — стадия трекера. Выключить после.
    collision_obj_debug_reject: bool = True
    # OBJECT PERMANENCE: держать подтверждённый объект в ПАМЯТИ (координаты стола),
    # пока его место не окажется ЯВНО пустым. Под рукой (перекрытие FK-капсулой)
    # объект удерживается — детекция сквозь руку не нужна. Это отвязывает
    # безопасность от «видим ли объект именно в этот кадр» и убирает мёртвую зону
    # у руки (где гейт площади ронял объект → ложный SAFE). Отсутствие детекции ≠
    # отсутствие объекта: удаляем только когда видим голый стол на его месте.
    collision_object_permanence: bool = True
    collision_perm_match_m: float = 0.06     # сопоставление память↔детекция (объект статичен)
    # Радиус сопоставления растёт с дальностью: шум глубины RealSense увеличивается
    # с z, у дальнего объекта центроид прыгает > фиксированных 6 см → память не
    # узнаёт его и заводит ВТОРУЮ запись на тот же кубик → лишний бокс.
    collision_perm_match_depth_k: float = 0.03  # +3 см на метр дальности
    # Дедуп памяти: слить записи-дубли (вложенные/перекрытые боксы на одном
    # объекте). Ловим по перекрытию 2D-боксов + 3D-близости, оставляем полную.
    collision_perm_dedup: bool = True
    collision_perm_dedup_iou: float = 0.10
    collision_perm_dedup_contain: float = 0.30
    collision_perm_dedup_m: float = 0.15     # 3D-санитар: не сливать разные кубики
    collision_perm_clear_frames: int = 10    # кадров «видно и пусто» до удаления
    collision_perm_window_px: int = 9        # полуокно репроекции центроида для тестов
    collision_perm_occ_frac: float = 0.4     # доля окна под рукой → «перекрыт» (держим)
    # Держать САМУЮ ПОЛНУЮ рамку/точки объекта: рука отъедает ближний край →
    # детекция даёт меньший кусок; вместо усадки держим полный (рамка стабильна,
    # дистанция по ближней грани → DANGER не запаздывает). Медленный decay
    # принимает стойко уменьшившийся объект; кратковременную усадку (рука прошла)
    # игнорирует. Обновляется, когда объект снова виден целиком.
    collision_perm_keep_fullest: bool = True
    # Вертикальный close маски объекта: сшить швы между кубиками СТОПКИ по
    # вертикали (узкий по X, высокий по Y) — не склеивает соседние объекты по
    # горизонтали. Помогает боксу тянуться на всю высоту башни. 0 = выкл.
    collision_obj_vclose_px: int = 15
    # Цветовой фильтр объектов (ФОЛБЭК, если нет плоскости): брать только
    # контрастные пятна (светлые ИЛИ насыщенные), тёмный стол отсекается.
    collision_obj_color_gate: bool = True
    collision_obj_bright_min: int = 200   # белые объекты (светлее стола)
    collision_obj_sat_min: int = 60       # цветные объекты (HSV saturation)
    # Исключить синий оттенок (пневмошланг робота) из насыщенной ветки фильтра.
    collision_obj_exclude_blue: bool = True
    collision_obj_blue_hue_lo: int = 90   # OpenCV HSV hue 0..180
    collision_obj_blue_hue_hi: int = 140
    # Консолидация фрагментов в цельный блоб + фильтр формы (отсекает шланг/шум).
    collision_obj_consolidate_px: int = 11
    collision_obj_min_fill_ratio: float = 0.32  # area / bbox_area
    collision_obj_max_aspect: float = 4.5       # max(w,h)/min(w,h)
    # ── Workspace-фильтр: игнорировать объекты вне рабочей зоны ──────────────
    # Границы в БАЗЕ РОБОТА (м), замерены measure_workspace.py по 4 углам стола
    # (2026-07-21, TCP на уровне стола, запас ±3 см). Убирает ложные объекты на
    # профиле ограждения/стенках. Z-верх поднят до ~35 см над столом.
    # Калибровка T_cr слабая (RMS 27 px, 6 инлайеров) — перевод камера→база
    # может ошибаться на 5-10 см. Бокс шире замера везде, КРОМЕ стороны профиля
    # (x_lo): там граница тонкая, иначе фильтр перестаёт резать профиль.
    # По console-выводу [WS DBG] (координаты отброшенных объектов) бокс можно
    # подстроить точнее.
    # ВЫКЛ (2026-07-29): фильтр на слабой T_cr (6 инлайеров, RMS 27px) съедал
    # реальные кубики у границы бокса, а протечки руки у базы пропускал. При
    # высотном гейте фон и так режется плоскостью — бокс не нужен. Вернуть True
    # только после пересъёмки extrinsic.
    collision_workspace_filter: bool = False
    collision_workspace_x_m: tuple = (-0.45, 0.52)
    collision_workspace_y_m: tuple = (-0.03, 0.60)
    collision_workspace_z_m: tuple = (-0.05, 0.50)
    # ── Анти-протечка маски руки ──────────────────────────────────────────────
    # Кандидат, прилипший к маске руки И совпадающий с ней по глубине (±delta),
    # — это пиксели самой руки/шланга/тени, а не объект: трек не создаётся.
    # Реальный объект на столе глубже руки, пока она не опустилась к нему.
    collision_obj_arm_depth_reject: bool = True
    collision_obj_arm_depth_delta_m: float = 0.03
    # Дебаунс остановки: стоп робота после N подряд DANGER-кадров.
    # Одиночные выбросы (1-2 кадра) больше не дёргают робота (~0.2 с при 15 FPS).
    collision_stop_confirm_frames: int = 3
    # ── Управление роботом: цикл движения + стоп по коллизии ──────────────────
    # Отдельно от use_robot_kinematics (то — только ЧТЕНИЕ углов $AXIS_ACT).
    # Здесь — ЗАПИСЬ движений (ptp) и мягкая остановка через $OV_PRO.
    enable_robot_control: bool = True        # True = гонять цикл + тормозить по коллизии
    # Уровень коллизии, при котором тормозим (SAFE/WARN/DANGER). DANGER = только вплотную.
    collision_stop_level: str = "DANGER"
    robot_ctrl_speed: int = 30               # % override для движения (и resume)
    robot_home_position: tuple = (450, 0, 600, 180, 0, 180)
    robot_cycle_left: tuple = (350, 200, 600, 180, 0, 180)
    robot_cycle_right: tuple = (350, -200, 600, 180, 0, 180)
    robot_base: int = 1
    robot_tool: int = 1
    # Пост-обработка глубины RealSense: заполнить дырки на глянцевых/цветных
    # кубиках (иначе верх стопки без глубины → бокс не на всю высоту И коллизия
    # «не видит» верх). Только spatial + hole_filling (per-frame): temporal смазал
    # бы движущуюся руку, disparity-transform ломается на уже выровненной глубине.
    # Настройка сенсора глубины: больше ИЗМЕРЕННЫХ точек на кубиках (пресет
    # High Density + полная мощность лазера). В отличие от hole_filling ничего не
    # выдумывает — просто плотнее меряет, поэтому маски крупнее и стабильнее.
    depth_sensor_tune: bool = True
    depth_high_density: bool = True
    depth_laser_power: float = -1.0   # -1 = максимум устройства, 0 = не трогать
    # ВЫКЛ по умолчанию: hole_filling заливал дырки стопки ДАЛЬНЕЙ глубиной (фон)
    # → высота стопки падала ниже гейта → объект пропадал из детекции. Плотный
    # valid был обманкой (объект залит фоном). Держим сырую глубину.
    depth_filters_enabled: bool = False
    depth_spatial_holes_fill: int = 2   # 0..5 — агрессивность заполнения в spatial
    depth_hole_filling: bool = True     # отдельный hole_filling_filter (добивка)
    depth_hole_filling_mode: int = 1    # 0=слева 1=дальний 2=ближний сосед
    # Метрики качества: пер-кадровый лог решений для tools/compute_metrics.py.
    # ВЫКЛ по умолчанию — детекцию и производительность не затрагивает.
    # Включи на время прогона сценария из scenarios.yaml, потом размечай GT
    # (tools/annotate.py) и считай проценты (P/R/F1, ложные стопы, MAE дистанции).
    enable_decision_log: bool = False
    decision_log_path: str = "captures/decisions.csv"
    scenario_id: str = ""   # id сценария из scenarios.yaml (попадает в лог)

    # ── Сбор данных для экспериментальной валидации (docs/EXPERIMENTS.md) ─────
    # Мастер серий: tools/run_experiments.py. ВСЕ флаги ниже по умолчанию ВЫКЛ:
    # при выключенных флагах ни один объект не создаётся, в горячем цикле не
    # появляется ни одной лишней ветки с работой — поведение и FPS те же.
    #
    # Источник кадров: 'camera' (RealSense) или 'playback' (запись с диска).
    # Playback нужен, чтобы пересчитывать метрики и делать абляцию без съёмки.
    source_mode: str = "camera"          # camera | playback
    playback_path: str = ""              # каталог записи (там frames/, joints.csv, meta.json)
    playback_fps: float = 0.0            # 0 = максимально быстро; иначе троттлинг
    # A1. Рекордер: кадры + углы + метаданные. Запись идёт в ФОНОВОМ потоке —
    # диск не должен тормозить контур безопасности (решение о стопе важнее кадра).
    enable_recording: bool = False
    recording_path: str = "captures/recording"
    recording_queue_max: int = 120       # кадров в очереди writer'а (~185 МБ ОЗУ)
    recording_put_timeout_s: float = 0.5  # дольше ждать очередь нельзя → кадр теряем
    recording_compress: bool = False     # npz со сжатием: втрое меньше, но грузит CPU
    # Только углы, без кадров. Нужно там, где меряется время реакции (Э3):
    # запись кадров даёт ~7 МБ/с и искажает замер, а тормозной путь считается по
    # joints.csv через FK — без углов его не получить вовсе.
    recording_joints_only: bool = False
    # A3. Лог остановов: задержки detect/stop/total по событию DANGER.
    enable_stop_log: bool = False
    stop_log_path: str = "captures/stop_events.csv"
    stop_poll_interval_s: float = 0.02   # желаемый период опроса контроллера
    stop_poll_timeout_s: float = 5.0     # сколько ждать остановки, потом сдаёмся
    stop_still_eps_deg: float = 0.05     # углы «не меняются», если max|Δ| меньше
    stop_still_samples: int = 3          # столько подряд «тихих» опросов = стоп
    # A4. Лог объектов: строка на КАЖДОГО рассмотренного кандидата с причиной
    # отбраковки (area/shape/border/pts3d/keepz/span/arm_reject/not_confirmed).
    enable_objects_log: bool = False
    objects_log_path: str = "captures/objects.csv"
    # A5. Дамп масок для IoU: имена совпадают с tools/annotate.py (gt_masks).
    dump_masks_every_n: int = 0          # 0 = не сохранять
    masks_dump_path: str = ""            # пусто → <recording_path>/masks
    # A6. Снимки оверлея для статьи по клавише p.
    figures_dir: str = "captures/figures"
    # Ограничение длительности прогона, сек (0 = до нажатия q). Нужно мастеру
    # серий: запись сценария должна длиться ровно столько, сколько объявлено.
    run_duration_s: float = 0.0
    # Debug
    show_debug_masks: bool = True  # нажмите d в окне для toggle или выставьте True здесь


class RealSenseCamera:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align_to_color = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.001
        # Фильтры глубины (создаём один раз): применяются ПОСЛЕ align в
        # get_aligned_frames. Заполняют дырки на кубиках → полные маски объектов.
        self._depth_filters = self._build_depth_filters(config)
        # Рекордер (A1) обязан писать глубину ДО постобработки — иначе на записи
        # нельзя переиграть сами фильтры при абляции. Копию сырой глубины делаем
        # только когда запись включена: иначе это лишний memcpy 600 КБ на кадр.
        self.keep_raw_depth = bool(getattr(config, "enable_recording", False))
        self.last_raw_depth = None

    @staticmethod
    def _build_depth_filters(config) -> list:
        if not bool(getattr(config, "depth_filters_enabled", True)):
            return []
        filters = []
        spatial = rs.spatial_filter()
        try:
            spatial.set_option(
                rs.option.holes_fill, int(getattr(config, "depth_spatial_holes_fill", 2))
            )
        except Exception:
            pass
        filters.append(spatial)
        if bool(getattr(config, "depth_hole_filling", True)):
            hole = rs.hole_filling_filter()
            try:
                hole.set_option(
                    rs.option.holes_fill, int(getattr(config, "depth_hole_filling_mode", 1))
                )
            except Exception:
                pass
            filters.append(hole)
        return filters

    def start(self) -> None:
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        rs_cfg.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
        self.profile = self.pipeline.start(rs_cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self._tune_depth_sensor(depth_sensor)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _tune_depth_sensor(self, depth_sensor) -> None:
        """Поднять ЗАПОЛНЕННОСТЬ глубины штатными ручками сенсора.

        Кубики глянцевые/цветные — сенсор отдаёт глубину в основном с верхних
        граней, маски получаются мелкими и мигают. Пресет High Density и полная
        мощность лазера дают больше ИЗМЕРЕННЫХ точек (в отличие от hole_filling,
        который выдумывает глубину из соседей и топил стопку фоном).
        """
        cfg = self.config
        if not bool(getattr(cfg, "depth_sensor_tune", True)):
            return
        if bool(getattr(cfg, "depth_high_density", True)):
            try:
                depth_sensor.set_option(rs.option.visual_preset,
                                        float(rs.rs400_visual_preset.high_density))
                print("[RS] visual_preset = High Density")
            except Exception as e:
                print(f"[RS] visual_preset не поддержан: {e}")
        lp = float(getattr(cfg, "depth_laser_power", -1.0))
        if lp != 0.0:
            try:
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                val = rng.max if lp < 0 else max(rng.min, min(lp, rng.max))
                depth_sensor.set_option(rs.option.laser_power, val)
                print(f"[RS] laser_power = {val:.0f} (max {rng.max:.0f})")
            except Exception as e:
                print(f"[RS] laser_power не поддержан: {e}")

    def stop(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def get_aligned_frames(self):
        # Транзиентные сбои RealSense (align/USB/тайминг) не должны ронять приложение —
        # при ошибке возвращаем None, цикл пропустит кадр и попробует снова.
        try:
            frames = self.pipeline.wait_for_frames()
            # Сбросить очередь до самого нового кадра, чтобы не копилась задержка:
            # если обработка просела, берём актуальный момент, а не прошлое.
            # Дренаж ДО ПУСТОТЫ (poll не блокирует): при FPS < ~4 фиксированный
            # лимит не успевал вычерпать очередь → задержка росла до 2-4 сек.
            # Теперь лаг = времени одной обработки, а не накопленного бэклога.
            for _ in range(240):  # верхняя страховка от вечного цикла (8 сек @30fps)
                newer = self.pipeline.poll_for_frames()
                if not newer:
                    break
                frames = newer
            aligned_frames = self.align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None
            if self.keep_raw_depth:
                self.last_raw_depth = np.asanyarray(depth_frame.get_data()).copy()
            # Пост-обработка глубины (spatial + hole_filling): заполнить дырки на
            # кубиках. После align — чтобы не терять выравнивание к цвету. При сбое
            # фильтра берём сырую глубину (не роняем кадр).
            if self._depth_filters:
                try:
                    _df = depth_frame
                    for _f in self._depth_filters:
                        _df = _f.process(_df)
                    depth_frame = _df
                except Exception as _fe:
                    print(f"[WARN] depth filter failed, raw depth: {_fe}")
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
            return color, depth, intrinsics, self.depth_scale
        except RuntimeError as e:
            # частый случай: "Error occured during execution of the processing block"
            print(f"[WARN] RealSense frame skipped: {e}")
            return None
