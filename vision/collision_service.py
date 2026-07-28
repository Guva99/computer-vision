"""
Сервис обнаружения коллизий: объекты сцены ↔ манипулятор.

Hybrid-схема: расстояние тела руки берётся из измеренного облака манипулятора
(без зависимости от T_cr), хват — из FK-сфер (тёмный металл не даёт глубины).
Ведёт журнал и управляет AABB-боксами в окне Open3D.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import collision as collision_mod
from realsense_io import AppConfig
from segmentation import split_cloud_by_mask


@dataclass
class CollisionFrame:
    scene_objects: list = field(default_factory=list)
    scene_obj_mask_2d: Optional[np.ndarray] = None
    scene_objects_mask: Optional[np.ndarray] = None
    results: list = field(default_factory=list)
    worst_level: str = "SAFE"
    worst_focus: object = None
    # Отбракованные кандидаты с причиной (Задача 8, анализ отказов);
    # заполняется только при enable_decision_log.
    rejects: list = field(default_factory=list)


class CollisionService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.logger = None
        if cfg.enable_collision:
            self.logger = collision_mod.CollisionLogger(
                cfg.collision_log_path, cfg.collision_log_every_n
            )
        self._bbox_geoms: list = []
        self._tracks: list = []  # временная устойчивость объектов (анти-мигание протечек руки)
        self._danger_hold = False  # гистерезис DANGER (анти-дребезг на границе порога)
        self._danger_lost_frames = 0  # защёлка: сколько кадров держим DANGER после потери объекта
        self._T_rc = None  # кэш inv(T_cr) для workspace-фильтра (база ← камера)
        self._ws_dbg_count = 0  # троттлинг диагностики workspace-фильтра
        self._table_plane = None  # кэш плоскости стола (a,b,c,d), стол статичен
        self._plane_age = 10 ** 9  # кадров с последней подгонки (форс на 1-м кадре)

    def _location_is_free(self, track, z_img, arm_mask, intrinsics):
        """Положительное свидетельство, что место объекта СЕЙЧАС пусто.

        True только если камера видит стол/фон ЗА объектом (глубина в окне у
        его центра больше z_объекта + margin). Окклюзия рукой, отсутствие
        валидной глубины или объект всё ещё виден спереди → False (не «пусто»,
        объект остаётся — безопасная асимметрия).
        """
        if z_img is None or intrinsics is None:
            return False
        c = np.asarray(track["centroid"], dtype=np.float64)
        z_obj = float(track.get("z_obj", c[2]))
        if c[2] <= 1e-4 or z_obj <= 1e-4:
            return False
        fx = float(intrinsics["fx"]); fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"]); cy = float(intrinsics["cy"])
        u = int(round(fx * c[0] / c[2] + cx))
        v = int(round(fy * c[1] / c[2] + cy))
        h, w = z_img.shape
        win = int(getattr(self.cfg, "collision_permanence_window_px", 7))
        u0, u1 = max(0, u - win), min(w, u + win + 1)
        v0, v1 = max(0, v - win), min(h, v + win + 1)
        if u1 <= u0 or v1 <= v0:
            return False
        # Рука перекрывает место объекта → окклюзия, не «пусто».
        if arm_mask is not None and np.count_nonzero(arm_mask[v0:v1, u0:u1]) > 0:
            return False
        zwin = z_img[v0:v1, u0:u1]
        zwin = zwin[np.isfinite(zwin)]
        if zwin.size < 3:
            return False  # нет валидной глубины → нет свидетельства
        z_now = float(np.median(zwin))
        margin = float(getattr(self.cfg, "collision_permanence_clear_margin_m", 0.05))
        return z_now > z_obj + margin

    def _confirm_objects(self, scene_objects, body_ds, gripper_spheres,
                         reject_log=None, z_img=None, arm_mask=None,
                         intrinsics=None):
        """Оставить только объекты, стабильно видимые >=N кадров в одном месте.
        Мигающие протечки руки (случайные места) отсекаются, кубик проходит.

        Асимметрия в сторону безопасности:
        - подтверждённый трек НЕ выпадает из оценки при кратких пропусках
          (miss <= max_miss) — рука, перекрывшая объект, не «гасит» его;
          используются последние известные точки (объект на столе статичен);
        - fast-path: кандидат, уже находящийся в WARN-зоне руки, подтверждается
          за collision_persist_frames_near кадров вместо полных N — ложный WARN
          дешевле пропущенного DANGER при быстром сближении.

        Анти-протечка (против ложных DANGER от кусков самой руки/шланга/тени).
        Протечка = кандидат, по ГЛУБИНЕ неотличимый от руки (_looks_like_arm):
        - глубина совпадает с рукой рядом (delta < thr) — пиксели руки;
        - у руки рядом нет валидной глубины (delta=inf, тёмный металл хвата) И
          кандидат геометрически лежит на FK-сферах хвата — блики лопатки;
          реальный объект ПЕРЕД хватом отстоит от сфер и протечкой не считается;
        - fast-path запрещён трекам, РОДИВШИМСЯ вплотную к руке: протечки всегда
          рождаются у руки, а реальный объект подтверждается заранее, пока рука
          далеко, и в fast-path не нуждается;
        - трек, родившийся у руки, статичный И похожий на руку по глубине,
          не подтверждается (leak_static). Башня вплотную к суставу с ДРУГОЙ
          глубиной под критерий не попадает — она подтверждается и держится;
        - подтверждённый трек статичного объекта coast-ится дольше
          (collision_track_max_miss_confirmed) — мигание 2D-детекции при
          прилипании к маске руки не роняет уже подтверждённый объект;
        - object permanence (collision_obstacle_permanence): подтверждённый
          объект держится как препятствие, пока камера не увидит, что место
          пусто (глубина за объектом), а не N кадров. z_img/arm_mask/intrinsics
          нужны для этой проверки свободного пространства.
        """
        cfg = self.cfg
        match_m = cfg.collision_track_match_m
        thr = float(getattr(cfg, "collision_obj_arm_depth_delta_m", 0.03))
        nodepth_flag = bool(getattr(cfg, "collision_obj_arm_nodepth_reject", True))
        grip_leak_m = float(getattr(cfg, "collision_leak_grip_dist_m", 0.06))

        def _looks_like_arm(o):
            """Причина-протечка ('arm_depth_reject'/'arm_nodepth_reject') или None.

            None = по глубине это реальный объект, а не пиксели руки.
            """
            if not getattr(o, "touches_arm", False):
                return None
            delta = float(getattr(o, "arm_depth_delta_m", float("inf")))
            if delta < thr:
                return "arm_depth_reject"
            if np.isfinite(delta) or not nodepth_flag:
                return None
            # Глубину руки рядом измерить нельзя (тёмный металл хвата):
            # различаем по FK — протечка лежит НА сферах хвата (dist ~ 0),
            # реальный объект перед хватом отстоит от них. Без FK-сфер
            # консервативно считаем протечкой (капсулы в маске тоже нет).
            if not gripper_spheres:
                return "arm_nodepth_reject"
            d_g, _, _ = collision_mod.min_distance_to_robot(gripper_spheres, o.points)
            if d_g <= grip_leak_m:
                return "arm_nodepth_reject"
            return None

        used = set()
        for obj in scene_objects:
            c = np.asarray(obj.centroid, dtype=np.float64)
            best_i, best_d = -1, match_m
            for i, t in enumerate(self._tracks):
                if i in used:
                    continue
                d = float(np.linalg.norm(c - t["centroid"]))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                t = self._tracks[best_i]
                t["centroid"] = c
                t["z_obj"] = float(c[2])  # актуальная опорная глубина объекта
                t["hits"] = min(t["hits"] + 1, 9999)
                t["miss"] = 0
                t["obj"] = obj
                used.add(best_i)
            else:
                # Анти-протечка: кандидат, по глубине неотличимый от руки,
                # трек не создаёт — это пиксели руки, а не объект.
                if bool(getattr(cfg, "collision_obj_arm_depth_reject", True)):
                    reason = _looks_like_arm(obj)
                    if reason is not None:
                        if reject_log is not None:
                            reject_log.append({"reject_reason": reason})
                        continue
                self._tracks.append({
                    "centroid": c, "z_obj": float(c[2]), "hits": 1, "miss": 0,
                    "obj": obj, "age": 0, "clear_count": 0,
                    "born_touching": bool(getattr(obj, "touches_arm", False)),
                    "born_centroid": c.copy(),
                })
                used.add(len(self._tracks) - 1)
        # Устаревание треков. Подтверждённый объект (object permanence) держится
        # НЕ по счётчику пропусков, а пока камера не увидит, что место пусто:
        # отсутствие детекции ≠ отсутствие объекта. НЕподтверждённый трек
        # (мигающий шум/протечка) по-прежнему снимается по max_miss.
        need = int(cfg.collision_persist_frames)
        max_miss = int(cfg.collision_track_max_miss)
        max_miss_conf = int(getattr(cfg, "collision_track_max_miss_confirmed", 15))
        perm_on = bool(getattr(cfg, "collision_obstacle_permanence", True))
        clear_need = int(getattr(cfg, "collision_permanence_clear_frames", 8))
        max_age = int(getattr(cfg, "collision_permanence_max_age_frames", 900))
        survivors = []
        for i, t in enumerate(self._tracks):
            matched = i in used
            if not matched:
                t["miss"] += 1
            confirmed_track = t["hits"] >= need
            if perm_on and confirmed_track:
                # Постоянное препятствие: снимается только при положительном
                # свидетельстве «пусто» clear_need кадров подряд, либо по
                # жёсткому пределу возраста (страховка от устаревших треков).
                t["age"] = t.get("age", 0) + 1
                if matched:
                    t["clear_count"] = 0
                elif self._location_is_free(t, z_img, arm_mask, intrinsics):
                    t["clear_count"] = t.get("clear_count", 0) + 1
                else:
                    t["clear_count"] = 0
                if t["clear_count"] >= clear_need or t["age"] >= max_age:
                    continue  # снять препятствие
                survivors.append(t)
            elif confirmed_track:
                # permanence выключен → прежнее поведение (coast по max_miss_conf)
                if t["miss"] <= max_miss_conf:
                    survivors.append(t)
            else:
                # неподтверждённый кандидат
                if t["miss"] <= max_miss:
                    survivors.append(t)
        self._tracks = survivors

        need_near = int(getattr(cfg, "collision_persist_frames_near", 2))
        warn_m = float(cfg.collision_warn_dist_m)
        leak_static = bool(getattr(cfg, "collision_leak_static_reject", True))
        move_m = float(getattr(cfg, "collision_leak_static_move_m", 0.04))
        confirmed = []
        for t in self._tracks:
            # Статичная протечка: родилась у руки, не сместилась И по глубине
            # неотличима от руки. Реальный объект с другой глубиной (башня у
            # сустава) сюда не попадает — подтверждается обычным порядком.
            # Уже подтверждённый (постоянный) объект под этот фильтр не идёт —
            # он прошёл подтверждение и держится object-permanence.
            if (
                leak_static
                and t["hits"] < need
                and t.get("born_touching", False)
                and float(
                    np.linalg.norm(t["centroid"] - t.get("born_centroid", t["centroid"]))
                ) < move_m
                and _looks_like_arm(t["obj"]) is not None
            ):
                if reject_log is not None:
                    reject_log.append({"reject_reason": "leak_static"})
                continue
            if t["hits"] >= need:
                # Coasting: включая miss>0 — последние известные точки объекта.
                confirmed.append(t["obj"])
                continue
            if (
                t["miss"] == 0
                and need_near <= t["hits"] < need
                and not t.get("born_touching", False)
            ):
                # Fast-path только для кандидатов уже рядом с рукой.
                d = collision_mod.min_dist_point_to_cloud(t["obj"].points, body_ds)
                d_g, _, _ = collision_mod.min_distance_to_robot(
                    gripper_spheres, t["obj"].points
                )
                if min(d, d_g) <= warn_m:
                    confirmed.append(t["obj"])
                    continue
            # Кандидат виден в этом кадре, но подтверждение ещё не набрано —
            # для анализа отказов (Задача 8) это причина "not_confirmed".
            if reject_log is not None and t["miss"] == 0:
                reject_log.append({"reject_reason": "not_confirmed"})
        # Пере-нумерация: coasting может вернуть объект со «старым» obj_id,
        # совпадающим со свежим — а results ищутся по obj_id.
        for k, obj in enumerate(confirmed):
            obj.obj_id = k
        return confirmed

    def evaluate(self, masks, fk_projector, points, colors, valid_flat,
                 color_bgr, intrinsics, joint_angles, frame_count, vis) -> CollisionFrame:
        cfg = self.cfg
        cf = CollisionFrame(
            scene_obj_mask_2d=np.zeros(color_bgr.shape[:2], dtype=np.uint8),
            scene_objects_mask=np.zeros(color_bgr.shape[:2], dtype=np.uint8),
        )
        if not (
            cfg.enable_collision
            and len(points) > 0
            and np.count_nonzero(masks.manipulator) > 40
        ):
            return cf

        # Сбор причин отбраковки (Задача 8) — только при включённом логе решений
        reject_log = (cf.rejects
                      if bool(getattr(cfg, "enable_decision_log", False))
                      else None)

        # FK-сферы строим ДО детекции: капсула хвата нужна в маске исключения.
        spheres = []
        if joint_angles is not None and fk_projector.T_cr is not None and len(joint_angles) >= 6:
            spheres = fk_projector.build_spheres(joint_angles)
        gripper_spheres = [s for s in spheres if s.part == "gripper"]

        # FK-капсула хвата в маске исключения: лопатка/кронштейн — тёмный металл
        # без валидной глубины, цвет/глубинная маска руки их пропускает, и их
        # блики рождали ложные «объекты» вплотную к телу (стойкий DANGER arm).
        # Дорисовываем хват геометрией FK (глубина не нужна); для ДИСТАНЦИИ
        # хват по-прежнему представлен gripper-сферами в hybrid-оценке.
        manip_for_detect = masks.manipulator
        if bool(getattr(cfg, "collision_fk_gripper_mask", True)) and gripper_spheres:
            capsule = collision_mod.build_gripper_capsule_mask(
                gripper_spheres,
                intrinsics,
                color_bgr.shape[:2],
                pad_px=int(getattr(cfg, "collision_fk_gripper_pad_px", 8)),
            )
            if np.count_nonzero(capsule) > 0:
                manip_for_detect = cv2.bitwise_or(masks.manipulator, capsule)

        # Плоскость стола для высотного гейта детекции (Уровень 1). Стол
        # статичен → подгоняем раз в N кадров и кэшируем. Фитим по точкам сцены
        # БЕЗ руки (рука сместила бы плоскость). При неудаче держим прошлую.
        if bool(getattr(cfg, "collision_obj_plane_gate", True)):
            refit_n = int(getattr(cfg, "collision_plane_refit_every_n", 30))
            self._plane_age += 1
            if self._table_plane is None or self._plane_age >= refit_n:
                arm_sel = (masks.manipulator > 0).reshape(-1)[valid_flat]
                scene_pts = points[~arm_sel] if np.any(arm_sel) else points
                plane = collision_mod.fit_table_plane(
                    scene_pts,
                    voxel_m=float(getattr(cfg, "collision_plane_voxel_m", 0.02)),
                    dist_thresh_m=float(getattr(cfg, "collision_plane_ransac_dist_m", 0.008)),
                    min_inliers=int(getattr(cfg, "collision_plane_min_inliers", 50)),
                )
                if plane is not None:
                    self._table_plane = plane
                    self._plane_age = 0

        if cfg.collision_use_2d_detection:
            cf.scene_objects, cf.scene_obj_mask_2d = collision_mod.detect_scene_objects_2d(
                points,
                colors,
                valid_flat,
                manip_for_detect,
                color_bgr.shape[:2],
                cfg,
                color_bgr=color_bgr,
                reject_log=reject_log,
                table_plane=self._table_plane,
            )
        else:
            _, (scene_pts, scene_col) = split_cloud_by_mask(
                points, colors, valid_flat, manip_for_detect
            )
            cf.scene_objects = collision_mod.detect_scene_objects(
                scene_pts, scene_col, cfg
            )
        # Workspace-фильтр: объект вне рабочей зоны (в базе робота) — не объект
        # для коллизий (профиль ограждения, стенки, фон за столом).
        if (
            bool(getattr(cfg, "collision_workspace_filter", False))
            and fk_projector.T_cr is not None
            and cf.scene_objects
        ):
            if self._T_rc is None:
                self._T_rc = np.linalg.inv(fk_projector.T_cr)
            x_lo, x_hi = cfg.collision_workspace_x_m
            y_lo, y_hi = cfg.collision_workspace_y_m
            z_lo, z_hi = cfg.collision_workspace_z_m
            kept = []
            dropped = []
            for obj in cf.scene_objects:
                c = np.asarray(obj.centroid, dtype=np.float64)
                p = self._T_rc @ np.array([c[0], c[1], c[2], 1.0])
                if (x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi
                        and z_lo <= p[2] <= z_hi):
                    kept.append(obj)
                else:
                    dropped.append(p[:3])
                    if reject_log is not None:
                        reject_log.append({"reject_reason": "workspace_filter"})
            cf.scene_objects = kept
            # Диагностика тюнинга бокса: куда реально попадают отброшенные
            # объекты в базе робота (печать раз в ~30 кадров с отбросом).
            self._ws_dbg_count += 1
            if dropped and self._ws_dbg_count % 30 == 1:
                coords = ", ".join(
                    f"({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f})" for q in dropped
                )
                print(f"[WS DBG] dropped base-coords: {coords} | kept={len(kept)}")
        # Hybrid: body distance from measured manipulator cloud (no T_cr
        # dependence), gripper from FK sphere (dark metal has no depth);
        # spheres/gripper_spheres построены выше, до детекции.
        # Erode the arm mask before sampling: boundary pixels share depth
        # with adjacent objects/table and cause false near-zero distances.
        body_mask = masks.manipulator
        if cfg.collision_body_erode_px > 1:
            k_be = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.collision_body_erode_px, cfg.collision_body_erode_px),
            )
            eroded_arm = cv2.erode(masks.manipulator, k_be)
            if np.count_nonzero(eroded_arm) > 40:
                body_mask = eroded_arm
        manip_sel = (body_mask > 0).reshape(-1)[valid_flat]
        manip_pts = points[manip_sel] if np.any(manip_sel) else points[:0]
        body_ds = collision_mod._voxel_down(manip_pts, cfg.collision_body_voxel_m)
        # Фильтр устойчивости: отсечь мигающие протечки руки, оставить стабильные.
        # body_ds/gripper_spheres нужны для fast-path (кандидат уже в WARN-зоне).
        # z_img — карта глубины по пикселям (NaN вне valid) для проверки
        # свободного пространства object-permanence: видно ли стол ЗА объектом.
        if cfg.collision_persist_frames > 1:
            z_img = None
            if bool(getattr(cfg, "collision_obstacle_permanence", True)):
                h_img, w_img = color_bgr.shape[:2]
                if valid_flat.size == h_img * w_img:
                    z_flat = np.full(h_img * w_img, np.nan, dtype=np.float64)
                    z_flat[valid_flat] = points[:, 2]
                    z_img = z_flat.reshape(h_img, w_img)
            cf.scene_objects = self._confirm_objects(
                cf.scene_objects, body_ds, gripper_spheres,
                reject_log=reject_log,
                z_img=z_img, arm_mask=masks.manipulator, intrinsics=intrinsics,
            )
        # Растеризация точек объектов нужна только для debug-мозаики
        # (Python-цикл по точкам — дорого). Строим лишь когда показываем debug.
        if cfg.show_debug_masks:
            cf.scene_objects_mask = collision_mod.build_scene_objects_image_mask(
                cf.scene_objects, intrinsics, color_bgr.shape[:2]
            )
        if cf.scene_objects and (len(body_ds) > 0 or gripper_spheres):
            (
                cf.results,
                cf.worst_level,
                cf.worst_focus,
            ) = collision_mod.evaluate_collisions_hybrid(
                body_ds,
                gripper_spheres,
                cf.scene_objects,
                cfg.collision_warn_dist_m,
                cfg.collision_danger_dist_m,
                cfg.collision_focus_parts,
                cfg.collision_body_voxel_m,
                cfg.collision_dist_percentile,
            )
        # Гистерезис DANGER: вход по danger_dist_m, выход только когда минимальная
        # дистанция превысит danger_exit_dist_m — раньше срабатывает и не дребезжит.
        if cf.worst_level == "DANGER":
            self._danger_hold = True
            self._danger_lost_frames = 0
        elif self._danger_hold:
            d_min = min((r.min_dist_m for r in cf.results), default=float("inf"))
            if not np.isfinite(d_min):
                # Объект, вызвавший DANGER, пропал из детекции (мигание 2D /
                # окклюзия рукой). Безопасность (SSM, ISO/TS 15066): не отпускаем
                # DANGER мгновенно — держим ещё N кадров, иначе робот рвётся
                # вперёд к объекту, который на самом деле никуда не делся.
                hold_n = int(getattr(cfg, "collision_danger_lost_hold_frames", 5))
                self._danger_lost_frames += 1
                if self._danger_lost_frames <= hold_n:
                    cf.worst_level = "DANGER"
                else:
                    self._danger_hold = False
            elif d_min <= float(getattr(cfg, "collision_danger_exit_dist_m", 0.10)):
                cf.worst_level = "DANGER"
                self._danger_lost_frames = 0
            else:
                self._danger_hold = False
        if self.logger is not None:
            self.logger.update(
                frame_count, cf.worst_level, cf.worst_focus
            )
        # 3D-боксы коллизий — только при открытом окне Open3D
        if vis is not None:
            for g in self._bbox_geoms:
                try:
                    vis.remove_geometry(g, reset_bounding_box=False)
                except Exception:
                    pass
            self._bbox_geoms.clear()
            for obj in cf.scene_objects:
                res = next(
                    (r for r in cf.results if r.obj_id == obj.obj_id), None
                )
                level = res.level if res is not None else "SAFE"
                if level == "SAFE" and not cfg.show_collision_overlay:
                    continue
                ls = collision_mod.make_o3d_aabb(obj.aabb_min, obj.aabb_max, level)
                vis.add_geometry(ls, reset_bounding_box=False)
                self._bbox_geoms.append(ls)
        return cf
