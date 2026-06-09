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


class CollisionService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.logger = None
        if cfg.enable_collision:
            self.logger = collision_mod.CollisionLogger(
                cfg.collision_log_path, cfg.collision_log_every_n
            )
        self._bbox_geoms: list = []

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

        if cfg.collision_use_2d_detection:
            cf.scene_objects, cf.scene_obj_mask_2d = collision_mod.detect_scene_objects_2d(
                points,
                colors,
                valid_flat,
                masks.manipulator,
                color_bgr.shape[:2],
                cfg,
                color_bgr=color_bgr,
            )
        else:
            _, (scene_pts, scene_col) = split_cloud_by_mask(
                points, colors, valid_flat, masks.manipulator
            )
            cf.scene_objects = collision_mod.detect_scene_objects(
                scene_pts, scene_col, cfg
            )
        # Растеризация точек объектов нужна только для debug-мозаики
        # (Python-цикл по точкам — дорого). Строим лишь когда показываем debug.
        if cfg.show_debug_masks:
            cf.scene_objects_mask = collision_mod.build_scene_objects_image_mask(
                cf.scene_objects, intrinsics, color_bgr.shape[:2]
            )
        spheres = []
        if joint_angles is not None and fk_projector.T_cr is not None and len(joint_angles) >= 6:
            spheres = fk_projector.build_spheres(joint_angles)
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
        if cf.scene_objects and (len(manip_pts) > 0 or gripper_spheres):
            (
                cf.results,
                cf.worst_level,
                cf.worst_focus,
            ) = collision_mod.evaluate_collisions_hybrid(
                manip_pts,
                gripper_spheres,
                cf.scene_objects,
                cfg.collision_warn_dist_m,
                cfg.collision_danger_dist_m,
                cfg.collision_focus_parts,
                cfg.collision_body_voxel_m,
                cfg.collision_dist_percentile,
            )
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
