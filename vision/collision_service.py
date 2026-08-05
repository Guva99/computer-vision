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
        self._T_rc = None  # кэш inv(T_cr) для workspace-фильтра (база ← камера)
        self._ws_dbg_count = 0  # троттлинг диагностики workspace-фильтра
        # Калиброванная плоскость стола (кадр камеры) для высотного гейта детекции.
        # Мерится один раз (calibrate_table_plane.py), не зависит от T_cr.
        self._table_plane = self._load_table_plane()

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
        self._tracks = [t for t in self._tracks if t["miss"] <= cfg.collision_track_max_miss]

        need = int(cfg.collision_persist_frames)
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
        body_ds = collision_mod._voxel_down(manip_pts, cfg.collision_body_voxel_m)
        # Фильтр устойчивости: отсечь мигающие протечки руки, оставить стабильные.
        # body_ds/gripper_spheres нужны для fast-path (кандидат уже в WARN-зоне).
        if cfg.collision_persist_frames > 1:
            cf.scene_objects = self._confirm_objects(
                cf.scene_objects, body_ds, gripper_spheres
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
        elif self._danger_hold:
            d_min = min((r.min_dist_m for r in cf.results), default=float("inf"))
            if d_min <= float(getattr(cfg, "collision_danger_exit_dist_m", 0.10)):
                cf.worst_level = "DANGER"
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
