"""
Сервис отрисовки: оверлеи (хват/запястье/маска робота), FK-скелет, ROI-круги,
текстовый HUD (углы, режим, число точек, статус коллизий) и debug-мозаика масок.
Stateless — берёт всё из переданного контекста кадра.
"""
import numpy as np
import cv2

import collision as collision_mod
from gripper_detection import (
    build_debug_mosaic,
    draw_fk_projections,
    draw_gripper_overlay,
    draw_robot_mask_overlay,
    draw_roi_circle,
    draw_wrist_overlay,
    draw_wrist_zone_rect,
    _gray_valid_near_arm,
    _valid_relaxed_mask,
)
from realsense_io import AppConfig

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SKELETON_NAMES = ["Base", "J1", "J2", "J3", "J4", "J5(W)", "J6(G)", "TCP"]


class OverlayRenderer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def draw_gripper(self, overlay, gripper_contour) -> None:
        if self.cfg.show_gripper_overlay and gripper_contour is not None:
            draw_gripper_overlay(overlay, gripper_contour, color_bgr=(0, 165, 255))

    def draw_robot_mask(self, overlay, masks) -> None:
        cfg = self.cfg
        if cfg.show_robot_mask_overlay and (
            np.count_nonzero(masks.manipulator) > 0 or np.count_nonzero(masks.robot) > 0
        ):
            draw_robot_mask_overlay(
                overlay,
                masks.manipulator if np.count_nonzero(masks.manipulator) > 0 else masks.robot,
                color_bgr=(0, 255, 255),
                thickness=2,
                min_area_px=cfg.robot_mask_overlay_min_area_px,
            )

    def draw_wrist(self, overlay, wrist_draw) -> None:
        if wrist_draw is None:
            return
        if wrist_draw.kind == "rect":
            draw_wrist_zone_rect(overlay, wrist_draw.rect)
        elif wrist_draw.kind == "contour":
            draw_wrist_overlay(overlay, wrist_draw.contour)

    def draw_hud(self, overlay, joint_angles, fkf, link_thickness_px, n_points,
                 cf, intrinsics, color_shape) -> None:
        cfg = self.cfg
        if joint_angles:
            y = 18
            cv2.putText(
                overlay,
                f"A1={joint_angles[0]:.1f} A2={joint_angles[1]:.1f} A3={joint_angles[2]:.1f}",
                (10, y),
                _FONT,
                0.3,
                (255, 255, 255),
                1,
            )
            y += 30
            cv2.putText(
                overlay,
                f"A4={joint_angles[3]:.1f} A5={joint_angles[4]:.1f} A6={joint_angles[5]:.1f}",
                (10, y),
                _FONT,
                0.3,
                (255, 255, 255),
                1,
            )
            y += 42

        # Draw FK projections if available — фиолетовый скелет p_cam
        if fkf.fk_uvs is not None:
            draw_fk_projections(overlay, fkf.fk_uvs,
                                names=_SKELETON_NAMES,
                                color_bgr=(200, 0, 255),
                                draw_skeleton=True,
                                link_thickness_px=link_thickness_px,
                                skeleton_alpha=cfg.fk_skeleton_alpha,
                                point_indices=[len(fkf.fk_uvs) - 1])  # только TCP
        if fkf.grip_roi_uv is not None:
            draw_roi_circle(overlay, fkf.grip_roi_uv, cfg.fk_grip_roi_radius_px,
                            color_bgr=(0, 255, 255), label="Grip ROI")
        if fkf.wrist_roi_uv is not None:
            draw_roi_circle(overlay, fkf.wrist_roi_uv, cfg.fk_wrist_roi_radius_px,
                            color_bgr=(255, 0, 255), label="Wrist ROI")
        # Mode indicator
        mode_txt = "FK-ROI" if fkf.use_fk else "legacy"
        cv2.putText(overlay, f"mode: {mode_txt}", (overlay.shape[1] - 130, 20),
                    _FONT, 0.45, (200, 200, 0), 1)
        y_status = 80 if joint_angles else 28
        cv2.putText(
            overlay,
            f"points={n_points} (raw)",
            (10, y_status),
            _FONT,
            0.33,
            (0, 255, 0),
            1,
        )
        line_y = y_status + 26
        if cfg.show_gripper_overlay:
            gtxt = "gripper: OFF"
            cv2.putText(
                overlay,
                gtxt,
                (10, line_y),
                _FONT,
                0.5,
                (0, 255, 255),
                2,
            )
            line_y += 24
        # if cfg.show_wrist_joint_overlay:
        #     if cfg.use_manipulator_only_mode:
        #         wtxt = "wrist: OFF"
        #     else:
        #         wrist_ok = wrist_contour is not None or (
        #             cfg.wrist_bbox_smooth_alpha > 0.0 and wrist_bbox_ema is not None
        #         )
        #         wtxt = "wrist: OK" if wrist_ok else "wrist: --"
        #     cv2.putText(
        #         overlay,
        #         wtxt,
        #         (10, line_y),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (255, 0, 255),
        #         2,
        #     )
        #     line_y += 24
        if cfg.enable_collision:
            n_obj = len(cf.scene_objects)
            part_txt = cf.worst_focus.part if cf.worst_focus else "--"
            if cf.worst_focus is not None:
                dist_txt = f"{cf.worst_focus.min_dist_m:.2f}m"
            else:
                dist_txt = "--"
            coll_color = collision_mod.LEVEL_COLORS_BGR.get(
                cf.worst_level, (200, 200, 200)
            )
            cv2.putText(
                overlay,
                f"COLLISION: {cf.worst_level} ({part_txt} {dist_txt}) objs={n_obj}",
                (10, line_y),
                _FONT,
                0.3,
                coll_color,
                1,
            )
            if cfg.show_collision_overlay:
                # Граница рабочей зоны ROI: всё вне неё в детекцию не идёт —
                # видно, какой объект внутри зоны, а какой отсекается краем.
                roi = getattr(cf, "table_roi", None)
                if roi:
                    rx0, ry0, rx1, ry1 = (int(v) for v in roi)
                    cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), (255, 255, 0), 1)
                    cv2.putText(overlay, "WORK ZONE", (rx0 + 3, ry0 + 14),
                                _FONT, 0.4, (255, 255, 0), 1)
                # (серый контур кольца-зоны убран — мешал виду)
                for obj in cf.scene_objects:
                    res = next(
                        (r for r in cf.results if r.obj_id == obj.obj_id), None
                    )
                    level = res.level if res is not None else "SAFE"
                    part = res.part if res is not None else "--"
                    dist_m = res.min_dist_m if res is not None else float("inf")
                    # Тугой 2D-bbox компоненты (если есть) точнее проекции 3D-AABB:
                    # нет глубинной инфляции и смещения. Фолбэк — проекция AABB.
                    rect = getattr(obj, "bbox_2d", None)
                    if rect is None:
                        rect = collision_mod.project_aabb_to_image(
                            obj.aabb_min, obj.aabb_max, intrinsics, color_shape
                        )
                    if rect is None:
                        continue
                    label = f"#{obj.obj_id} {part} {dist_m:.2f}m"
                    _h = getattr(obj, "height_above_table_m", float("nan"))
                    if _h == _h:  # not NaN — высота над столом для диагностики фантомов
                        label += f" h{_h * 100:.0f}"
                    collision_mod.draw_object_box(overlay, rect, level, label=label)

    def build_debug_mosaic(self, color_bgr, depth, depth_scale, masks, gripper_debug, cf):
        cfg = self.cfg
        _dbg_gray, _dbg_valid, _dbg_near_arm, _dbg_arm_cx, _dbg_arm_mask = _gray_valid_near_arm(
            color_bgr, depth, depth_scale,
            cfg.depth_min_m, cfg.depth_max_m,
            cfg.arm_gray_min, cfg.near_arm_dilate_px,
        )
        _dbg_valid_soft = _valid_relaxed_mask(_dbg_valid, cfg.wrist_valid_dilate_px, cfg.wrist_valid_dilate_iters)
        _grip_cand = gripper_debug.get(
            "gripper_cand", np.zeros_like(masks.manipulator)
        )
        return build_debug_mosaic({
            "manipulator": masks.manipulator,
            "gripper_cand": _grip_cand,
            "obj_mask": cf.scene_obj_mask_2d,
            "scene_objects": cf.scene_objects_mask,
            "table_candidate": masks.table if cfg.show_table_candidate_debug else np.zeros_like(masks.manipulator),
            "valid": _dbg_valid.astype(np.uint8) * 255,
        })
