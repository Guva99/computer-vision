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
    # ── Покадровый лог решений (валидация, Задача 1) ─────────────────────────
    # При True в perf_log.csv добавляются колонки collision_level, n_objects,
    # min_dist_m, latency_ms, robot_paused, scenario_id, а детализация по
    # объектам пишется в objects_csv_path. False = прежний формат CSV.
    enable_decision_log: bool = False
    objects_csv_path: str = "captures/objects.csv"
    scenario_id: str = ""                # проставляется сценарным раннером
    # ── Лог событий останова (валидация, Задача 2) ───────────────────────────
    # Фиксирует t_danger при stop_movement() и t_stop, когда угловые скорости
    # всех суставов ($AXIS_ACT) ниже порога N кадров подряд → stop_events.csv.
    enable_stop_log: bool = False
    stop_events_csv_path: str = "captures/stop_events.csv"
    stop_speed_eps_deg_s: float = 0.5       # порог «робот стоит» (град/с)
    stop_confirm_frames_stopped: int = 3    # кадров подряд ниже порога
    # ── Рекордер и оффлайн-реплей (валидация, Задача 3) ──────────────────────
    # enable_recording: синхронная запись RGB-D + углов A1..A6 + таймстемпов
    # (формат app/recorder.py: npz по кадрам + joints.csv + meta.json).
    # source_mode="playback": кадры и углы берутся из записи (playback_path),
    # управление роботом принудительно отключается (безопасность).
    enable_recording: bool = False
    recording_path: str = ""             # ""= авто captures/recordings/rec_<ts>
    recording_compress: bool = False     # True = npz со сжатием (медленнее)
    source_mode: str = "live"            # "live" | "playback"
    playback_path: str = ""              # каталог записи для реплея
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
    collision_track_max_miss: int = 3      # удалять трек после N пропусков
    collision_focus_parts: tuple = ("gripper", "wrist", "arm")
    # Hybrid collision: body distance from manipulator-mask cloud (measured),
    # gripper from FK sphere (no depth on dark metal). Voxel for body cloud.
    collision_body_voxel_m: float = 0.02
    # Robustness: use a low percentile of nearest distances (not a single
    # closest point) + erode masks to drop noisy boundary pixels.
    collision_dist_percentile: float = 2.0   # 0 = strict min
    collision_body_erode_px: int = 5
    collision_obj_erode_px: int = 3
    collision_log_path: str = "captures/collisions.log"
    collision_log_every_n: int = 30
    show_collision_overlay: bool = True
    collision_use_2d_detection: bool = True
    collision_obj_min_area_px: int = 150
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
    # Цветовой фильтр объектов: брать только контрастные пятна (светлые ИЛИ
    # насыщенные по цвету), тёмный десатурированный стол отсекается.
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
    collision_workspace_filter: bool = True
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
    # Debug
    show_debug_masks: bool = False  # нажмите d в окне для toggle или выставьте True здесь


class RealSenseCamera:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align_to_color = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.001

    def start(self) -> None:
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        rs_cfg.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
        self.profile = self.pipeline.start(rs_cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

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
            for _ in range(8):  # ограниченный дренаж (без бесконечного цикла)
                newer = self.pipeline.poll_for_frames()
                if not newer:
                    break
                frames = newer
            aligned_frames = self.align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
            return color, depth, intrinsics, self.depth_scale
        except RuntimeError as e:
            # частый случай: "Error occured during execution of the processing block"
            print(f"[WARN] RealSense frame skipped: {e}")
            return None
