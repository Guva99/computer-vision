"""
Сервис обнаружения коллизий: объекты сцены ↔ манипулятор.

Hybrid-схема: расстояние тела руки берётся из измеренного облака манипулятора
(без зависимости от T_cr), хват — из FK-сфер (тёмный металл не даёт глубины).
Ведёт журнал и управляет AABB-боксами в окне Open3D.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    table_roi: Optional[tuple] = None  # ROI рабочей зоны (px) для отрисовки границы


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
        self._known: list = []   # object-permanence: объекты в памяти (координаты стола)
        self._danger_hold = False  # гистерезис DANGER (анти-дребезг на границе порога)
        self._danger_lost = 0      # счётчик кадров удержания DANGER после пропажи объекта
        self._T_rc = None  # кэш inv(T_cr) для workspace-фильтра (база ← камера)
        self._ws_dbg_count = 0  # троттлинг диагностики workspace-фильтра
        # Калиброванная плоскость стола (кадр камеры) для высотного гейта детекции.
        # Мерится один раз (calibrate_table_plane.py), не зависит от T_cr.
        self._table_roi = None  # ROI рабочей зоны (x0,y0,x1,y1 px), заполнит загрузчик
        self._table_plane = self._load_table_plane()
        self._resolve_table_roi()

    def _resolve_table_roi(self):
        """ROI рабочей зоны в пикселях: из table_plane.json (roi_px) или конфига.
        Вне ROI (провод, база робота, фон) детекция не идёт. Камера неподвижна →
        зона фиксирована, от T_cr не зависит."""
        if not bool(getattr(self.cfg, "collision_obj_roi_gate", False)):
            self._table_roi = None
            return
        # json-ROI из калибровки имеет приоритет; иначе — дефолт из конфига.
        if self._table_roi is None:
            self._table_roi = tuple(getattr(self.cfg, "collision_obj_roi_px", ()))
        if self._table_roi:
            print(f"[COLLISION] ROI рабочей зоны (px): {tuple(self._table_roi)}")

    def _load_table_plane(self):
        """Загрузить (a,b,c,d) из table_plane.json или None (→ фолбэк на цвет)."""
        if not bool(getattr(self.cfg, "collision_obj_plane_gate", False)):
            return None
        path = Path(getattr(self.cfg, "table_plane_path", "table_plane.json"))
        if not path.exists():
            print(f"[COLLISION] table_plane.json не найден ({path}) — цвет-гейт (фолбэк). "
                  f"Запусти calibrate_table_plane.py.")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            plane = np.asarray(data["plane"], dtype=np.float64).ravel()[:4]
            # Границы гейта из калибровки имеют приоритет над дефолтами конфига.
            if "gate_min_m" in data:
                self.cfg.collision_obj_plane_height_min_m = float(data["gate_min_m"])
            if "gate_max_m" in data:
                self.cfg.collision_obj_plane_height_max_m = float(data["gate_max_m"])
            # ROI рабочей зоны из калибровки (приоритет над дефолтом конфига).
            if "roi_px" in data and data["roi_px"]:
                self._table_roi = tuple(int(v) for v in data["roi_px"][:4])
            print(f"[COLLISION] Плоскость стола загружена: n=({plane[0]:+.2f},"
                  f"{plane[1]:+.2f},{plane[2]:+.2f}) d={plane[3]:+.3f}, гейт "
                  f"{self.cfg.collision_obj_plane_height_min_m*100:.1f}.."
                  f"{self.cfg.collision_obj_plane_height_max_m*100:.0f} см")
            return plane
        except Exception as e:
            print(f"[COLLISION] Ошибка чтения table_plane.json: {e} — цвет-гейт (фолбэк)")
            return None

    def _confirm_objects(self, scene_objects, body_ds, gripper_spheres):
        """Оставить только объекты, стабильно видимые >=N кадров в одном месте.
        Мигающие протечки руки (случайные места) отсекаются, кубик проходит.

        Асимметрия в сторону безопасности:
        - подтверждённый трек НЕ выпадает из оценки при кратких пропусках
          (miss <= max_miss) — рука, перекрывшая объект, не «гасит» его;
          используются последние известные точки (объект на столе статичен);
        - fast-path: кандидат, уже находящийся в WARN-зоне руки, подтверждается
          за collision_persist_frames_near кадров вместо полных N — ложный WARN
          дешевле пропущенного DANGER при быстром сближении.

        Анти-протечка (против ложных DANGER от кусков самой руки/шланга/тени):
        - кандидат, прилипший к маске руки И совпадающий с ней по глубине,
          трек не создаёт — это пиксели руки, а не объект;
        - fast-path запрещён трекам, РОДИВШИМСЯ вплотную к руке: протечки всегда
          рождаются у руки, а реальный объект подтверждается заранее, пока рука
          далеко, и в fast-path не нуждается.
        """
        cfg = self.cfg
        match_m = cfg.collision_track_match_m
        used = set()
        n_in = len(scene_objects)       # диагностика: сколько сырых детекций пришло
        n_arm_reject = 0                # сколько отбраковано как «протечка руки»
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
                t["hits"] = min(t["hits"] + 1, 9999)
                t["miss"] = 0
                t["obj"] = obj
                used.add(best_i)
            else:
                # Анти-протечка: прилипший к руке и совпадающий с ней по глубине
                # кандидат — пиксели самой руки; трек не создаём.
                if (
                    bool(getattr(cfg, "collision_obj_arm_depth_reject", True))
                    and getattr(obj, "touches_arm", False)
                    and getattr(obj, "arm_depth_delta_m", float("inf"))
                    < float(getattr(cfg, "collision_obj_arm_depth_delta_m", 0.03))
                ):
                    n_arm_reject += 1
                    continue
                self._tracks.append({
                    "centroid": c, "hits": 1, "miss": 0, "obj": obj,
                    "born_touching": bool(getattr(obj, "touches_arm", False)),
                })
                used.add(len(self._tracks) - 1)
        # состарить несопоставленные треки
        for i, t in enumerate(self._tracks):
            if i not in used:
                t["miss"] += 1
        # Подтверждённый трек (стабильный объект) coast-ится дольше кандидата:
        # рука, перекрывшая башню на несколько кадров, не должна её «убивать».
        need = int(cfg.collision_persist_frames)
        max_miss = int(cfg.collision_track_max_miss)
        max_miss_conf = int(getattr(cfg, "collision_track_max_miss_confirmed", 15))
        self._tracks = [
            t for t in self._tracks
            if t["miss"] <= (max_miss_conf if t["hits"] >= need else max_miss)
        ]
        need_near = int(getattr(cfg, "collision_persist_frames_near", 2))
        warn_m = float(cfg.collision_warn_dist_m)
        confirmed = []
        for t in self._tracks:
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
        # Пере-нумерация: coasting может вернуть объект со «старым» obj_id,
        # совпадающим со свежим — а results ищутся по obj_id.
        for k, obj in enumerate(confirmed):
            obj.obj_id = k
        # Диагностика потери объекта у руки: печатаем, когда были сырые детекции,
        # но ни одна не подтвердилась (или что-то отбраковано как протечка руки).
        if bool(getattr(cfg, "collision_obj_debug_reject", False)):
            self._confirm_dbg_count = getattr(self, "_confirm_dbg_count", 0) + 1
            n_touch = sum(1 for o in scene_objects if getattr(o, "touches_arm", False))
            if n_in > 0 and (len(confirmed) == 0 or n_arm_reject > 0
                             or self._confirm_dbg_count % 5 == 0):
                print(
                    f"[CONFIRM] in={n_in} touch_arm={n_touch} "
                    f"arm_reject={n_arm_reject} tracks={len(self._tracks)} "
                    f"confirmed={len(confirmed)} need={need}"
                )
        return confirmed

    @staticmethod
    def _obj_area(obj):
        """Мера «полноты» наблюдения: площадь 2D-бокса (px), иначе число точек."""
        bb = getattr(obj, "bbox_2d", None)
        if bb is not None:
            return float(int(bb[2]) * int(bb[3]))
        pts = getattr(obj, "points", None)
        return float(len(pts)) if pts is not None else 0.0

    def _dedup_known(self):
        """Слить записи-дубли в памяти (два бокса на одном объекте).

        Дальний объект шумит по глубине → центроид прыгает дальше радиуса
        сопоставления → память заводит ВТОРУЮ запись на тот же кубик. Старая при
        этом не удаляется: место не под рукой, а стол там не пустой (объект-то
        стоит) → «держим» бесконечно. Ловим такие пары по перекрытию 2D-боксов
        (вложенные рамки) + 3D-близости, оставляем более полную запись.
        """
        cfg = self.cfg
        if not bool(getattr(cfg, "collision_perm_dedup", True)) or len(self._known) < 2:
            return
        iou_t = float(getattr(cfg, "collision_perm_dedup_iou", 0.10))
        con_t = float(getattr(cfg, "collision_perm_dedup_contain", 0.30))
        max_m = float(getattr(cfg, "collision_perm_dedup_m", 0.15))
        keep: list = []
        for k in self._known:
            dup_of = -1
            for j, kept in enumerate(keep):
                if float(np.linalg.norm(k["centroid"] - kept["centroid"])) > max_m:
                    continue  # разные объекты — не трогаем
                bb_a = getattr(k.get("obj"), "bbox_2d", None)
                bb_b = getattr(kept.get("obj"), "bbox_2d", None)
                if bb_a is not None and bb_b is not None:
                    iou, cont = collision_mod._box_overlap(bb_a, bb_b)
                    if iou >= iou_t or cont >= con_t:
                        dup_of = j
                        break
                else:
                    dup_of = j  # боксов нет — решает 3D-близость
                    break
            if dup_of < 0:
                keep.append(k)
                continue
            # оставить более ПОЛНУЮ запись; свежесть (empty) берём лучшую из пары
            fresh = min(k.get("empty", 0), keep[dup_of].get("empty", 0))
            if k.get("area", 0.0) > keep[dup_of].get("area", 0.0):
                keep[dup_of] = k
            keep[dup_of]["empty"] = fresh
        self._known = keep

    def _update_permanence(self, observations, occ_mask, points, valid_flat,
                           intrinsics, image_shape):
        """Object permanence: держать подтверждённые объекты в ПАМЯТИ (координаты
        стола), пока их место не окажется ЯВНО пустым.

        Мёртвая зона у руки возникала так: рука отъедала пиксели объекта, остаток
        < min_area → детекция роняла его → objs=0 → ложный SAFE ровно когда рука
        ближе всего. Здесь объект НЕ исчезает: пока его место перекрыто рукой
        (occ_mask = маска руки + FK-капсула), удерживаем его; уровень DANGER/WARN
        считается геометрией как обычно (рука рядом → DANGER держится, робот стоит).

        Удаляем объект ТОЛЬКО когда видим на его месте голый стол N кадров подряд
        (отсутствие детекции ≠ отсутствие объекта). Под рукой не стареем к удалению.
        """
        cfg = self.cfg
        match_m = float(getattr(cfg, "collision_perm_match_m", 0.06))
        clear_frames = int(getattr(cfg, "collision_perm_clear_frames", 10))
        win = int(getattr(cfg, "collision_perm_window_px", 9))
        occ_frac = float(getattr(cfg, "collision_perm_occ_frac", 0.4))
        keep_fullest = bool(getattr(cfg, "collision_perm_keep_fullest", True))
        h, w = int(image_shape[0]), int(image_shape[1])
        fx = float(intrinsics["fx"]); fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"]); cy = float(intrinsics["cy"])
        occ_bool = (occ_mask > 0) if occ_mask is not None else None

        def _occluded(c):
            """Перекрыт ли центроид рукой: доля окна репроекции под occ-маской."""
            z = float(c[2])
            if z <= 1e-4 or occ_bool is None:
                return False
            u = int(round(fx * float(c[0]) / z + cx))
            v = int(round(fy * float(c[1]) / z + cy))
            x0 = max(0, u - win); x1 = min(w, u + win + 1)
            y0 = max(0, v - win); y1 = min(h, v + win + 1)
            if x1 <= x0 or y1 <= y0:
                return False
            return float(np.mean(occ_bool[y0:y1, x0:x1])) >= occ_frac

        # 1) сопоставить наблюдения этого кадра с известными по 3D-центроиду
        used = set()
        match_k = float(getattr(cfg, "collision_perm_match_depth_k", 0.03))
        for obj in observations:
            c = np.asarray(obj.centroid, dtype=np.float64)
            # Радиус сопоставления растёт с дальностью: шум глубины ~z, у дальнего
            # объекта центроид прыгает дальше фиксированного порога → иначе память
            # заводит вторую запись на тот же кубик (лишний бокс).
            best_i, best_d = -1, match_m + match_k * max(0.0, float(c[2]))
            for i, k in enumerate(self._known):
                if i in used:
                    continue
                d = float(np.linalg.norm(c - k["centroid"]))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                k = self._known[best_i]
                # Держим ПОЛНУЮ рамку ТОЛЬКО пока объект перекрыт рукой (усадка — от
                # руки). Виден целиком → всегда берём текущую детекцию: само-
                # коррекция, никакой залипшей широкой рамки из старого кадра.
                new_area = self._obj_area(obj)
                if (not keep_fullest) or (not _occluded(c)) or new_area >= k.get("area", 0.0):
                    k["obj"] = obj; k["area"] = new_area; k["centroid"] = c
                # else: под рукой и новый меньше — держим полный сохранённый obj
                k["empty"] = 0
                used.add(best_i)
            else:
                self._known.append({
                    "obj": obj, "centroid": c, "empty": 0,
                    "area": self._obj_area(obj),
                })
                used.add(len(self._known) - 1)

        # 2) несопоставленные известные: держать под рукой, удалять по пустому столу
        # Карта высоты над столом (ленивая — только если есть кого проверять).
        h_img = None
        if (
            self._table_plane is not None and points is not None
            and valid_flat is not None
            and any(i not in used for i in range(len(self._known)))
        ):
            pa, pb, pc, pd = (float(v) for v in np.asarray(self._table_plane).ravel()[:4])
            hh = points[:, 0] * pa + points[:, 1] * pb + points[:, 2] * pc + pd
            h_img = np.full(h * w, np.nan, dtype=np.float32)
            h_img[valid_flat] = hh.astype(np.float32)
            h_img = h_img.reshape(h, w)
        gate_min = float(getattr(cfg, "collision_obj_plane_height_min_m", 0.04))

        survivors = []
        for i, k in enumerate(self._known):
            if i in used:
                survivors.append(k)
                continue
            c = k["centroid"]
            z = float(c[2])
            held = True  # безопасный дефолт: не смогли проверить место → держим
            if z > 1e-4:
                u = int(round(fx * float(c[0]) / z + cx))
                v = int(round(fy * float(c[1]) / z + cy))
                x0 = max(0, u - win); x1 = min(w, u + win + 1)
                y0 = max(0, v - win); y1 = min(h, v + win + 1)
                if x1 > x0 and y1 > y0:
                    occluded = (
                        occ_bool is not None
                        and float(np.mean(occ_bool[y0:y1, x0:x1])) >= occ_frac
                    )
                    if occluded:
                        k["empty"] = 0  # под рукой — держим, не стареем к удалению
                    elif h_img is not None:
                        wh = h_img[y0:y1, x0:x1]
                        wh = wh[np.isfinite(wh)]
                        if wh.size >= 5 and float(np.median(wh)) < gate_min:
                            k["empty"] += 1  # видно И пусто → считаем к удалению
                            if k["empty"] >= clear_frames:
                                held = False  # объект реально убрали
                        else:
                            k["empty"] = 0   # видно, но объект ещё там (или шум) — держим
            if held:
                survivors.append(k)
        self._known = survivors
        # 2.5) слить дубли памяти (шум глубины у дальних объектов → 2 записи)
        self._dedup_known()

        # 3) выдать объекты (наблюдённые + удержанные), пере-нумеровать под results
        out = [k["obj"] for k in self._known if k.get("obj") is not None]
        for idx, o in enumerate(out):
            o.obj_id = idx
        return out

    def evaluate(self, masks, fk_projector, points, colors, valid_flat,
                 color_bgr, intrinsics, joint_angles, frame_count, vis) -> CollisionFrame:
        cfg = self.cfg
        cf = CollisionFrame(
            scene_obj_mask_2d=np.zeros(color_bgr.shape[:2], dtype=np.uint8),
            scene_objects_mask=np.zeros(color_bgr.shape[:2], dtype=np.uint8),
        )
        cf.table_roi = self._table_roi  # граница рабочей зоны для оверлея
        if not (
            cfg.enable_collision
            and len(points) > 0
            and np.count_nonzero(masks.manipulator) > 40
        ):
            return cf

        # FK-сферы строим ДО детекции: нужны и для self-filter капсулы (закрыть
        # тёмный металл руки, который высотный гейт иначе примет за объект), и
        # для гибридной дистанции ниже.
        spheres = []
        if joint_angles is not None and fk_projector.T_cr is not None and len(joint_angles) >= 6:
            spheres = fk_projector.build_spheres(joint_angles)

        # Self-filter: дорисовать FK-капсулу руки в маску ИСКЛЮЧЕНИЯ для детекции.
        # Дистанцию до тела ниже меряем по СЫРОЙ глубинной маске (masks.manipulator),
        # капсула — только чтобы рука не попадала в кандидаты-объекты.
        manip_for_detect = masks.manipulator
        if bool(getattr(cfg, "collision_fk_self_filter", False)) and spheres:
            parts = tuple(getattr(cfg, "collision_fk_self_filter_parts", ("gripper", "wrist")))
            pad = int(getattr(cfg, "collision_fk_self_filter_pad_px", 10))
            cap = collision_mod.build_gripper_capsule_mask(
                spheres, intrinsics, color_bgr.shape[:2], pad_px=pad, parts=parts
            )
            if np.count_nonzero(cap) > 0:
                manip_for_detect = cv2.bitwise_or(masks.manipulator, cap)

        if cfg.collision_use_2d_detection:
            cf.scene_objects, cf.scene_obj_mask_2d = collision_mod.detect_scene_objects_2d(
                points,
                colors,
                valid_flat,
                manip_for_detect,
                color_bgr.shape[:2],
                cfg,
                color_bgr=color_bgr,
                table_plane=self._table_plane,
                roi_rect=self._table_roi,
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
        # dependence), gripper from FK sphere (dark metal has no depth).
        gripper_spheres = [s for s in spheres if s.part == "gripper"]
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
        # Очистка облака руки от ЦВЕТНЫХ протечек: реальная рука KUKA белая/серая
        # (низкая насыщенность), а яркий кубик (жёлтый/зелёный/дерево) частично
        # проходит порог белого сегментатора → его пиксели попадают в облако тела
        # → «до руки» ≈ 0 → ложный DANGER «arm» на кубике вдали от руки. Выкидываем
        # насыщенные пиксели из тела (T_cr-независимо, реальную руку не трогает).
        if (
            bool(getattr(cfg, "collision_body_color_clean", True))
            and color_bgr is not None
            and np.any(manip_sel)
        ):
            sat = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
            sat_manip = sat.reshape(-1)[valid_flat][manip_sel]
            sat_max = int(getattr(cfg, "collision_body_sat_max", 70))
            keep_desat = sat_manip < sat_max
            if np.count_nonzero(keep_desat) > 20:
                manip_pts = manip_pts[keep_desat]
        # Очистка облака руки от протечек маски на стол по калиброванной плоскости:
        # реальная рука выше стола; точки на уровне стола рядом с рукой — протечки,
        # дающие фантомную близость к столовым объектам (ложный DANGER «arm»).
        if (
            self._table_plane is not None
            and len(manip_pts) > 0
            and bool(getattr(cfg, "collision_body_plane_clean", True))
        ):
            pa, pb, pc, pd = (float(v) for v in np.asarray(self._table_plane).ravel()[:4])
            body_h = manip_pts[:, 0] * pa + manip_pts[:, 1] * pb + manip_pts[:, 2] * pc + pd
            keep_body = body_h > float(getattr(cfg, "collision_body_plane_min_height_m", 0.03))
            if np.count_nonzero(keep_body) > 20:
                manip_pts = manip_pts[keep_body]
        body_ds = collision_mod._voxel_down(manip_pts, cfg.collision_body_voxel_m)
        # Фильтр устойчивости: отсечь мигающие протечки руки, оставить стабильные.
        # body_ds/gripper_spheres нужны для fast-path (кандидат уже в WARN-зоне).
        if cfg.collision_persist_frames > 1:
            cf.scene_objects = self._confirm_objects(
                cf.scene_objects, body_ds, gripper_spheres
            )
        # Object permanence: удержать объект в памяти сквозь перекрытие рукой
        # (мёртвая зона у руки, где гейт площади ронял объект → ложный SAFE).
        # occ_mask = manip_for_detect (рука + FK-капсула) — тест «под рукой».
        if bool(getattr(cfg, "collision_object_permanence", True)):
            cf.scene_objects = self._update_permanence(
                cf.scene_objects, manip_for_detect, points, valid_flat,
                intrinsics, color_bgr.shape[:2],
            )
        # Дедуп финальных объектов: слить перекрывающиеся боксы (фрагменты кубика
        # + призраки-коастинг трекера на старом месте). После трекера, чтобы
        # ловить оба источника дублей. Слияние точек лишь уточняет дистанцию.
        if bool(getattr(cfg, "collision_obj_merge_overlap", True)) and len(cf.scene_objects) > 1:
            cf.scene_objects = collision_mod._merge_overlapping_objects(
                cf.scene_objects,
                float(getattr(cfg, "collision_obj_merge_iou", 0.15)),
                float(getattr(cfg, "collision_obj_merge_contain", 0.4)),
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
            self._danger_lost = 0
        elif self._danger_hold:
            d_min = min((r.min_dist_m for r in cf.results), default=float("inf"))
            if np.isfinite(d_min):
                if d_min <= float(getattr(cfg, "collision_danger_exit_dist_m", 0.10)):
                    cf.worst_level = "DANGER"
                    self._danger_lost = 0
                else:
                    self._danger_hold = False
                    self._danger_lost = 0
            else:
                # Опасный объект ПРОПАЛ (мигание/перекрытие рукой), а не отъехал:
                # держим DANGER ещё N кадров — робот не рвётся к «исчезнувшей» башне.
                lost_hold = int(getattr(cfg, "collision_danger_lost_hold_frames", 5))
                self._danger_lost += 1
                if self._danger_lost <= lost_hold:
                    cf.worst_level = "DANGER"
                else:
                    self._danger_hold = False
                    self._danger_lost = 0
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
