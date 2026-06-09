"""
Сервис сегментации манипулятора.

Строит маски робота/стола/манипулятора (FK-self-filter + цвет/глубина), держит
temporal-hold последней хорошей маски и резолвит пересечение со столом по
protect-зоне «робот-приоритет». Низкоуровневые алгоритмы — в gripper_detection.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gripper_detection import (
    build_manipulator_mask,
    build_robot_depth_mask,
    build_table_mask,
)
from realsense_io import AppConfig


@dataclass
class Masks:
    """Набор бинарных масок на один кадр (uint8, 0/255)."""
    robot: np.ndarray
    manipulator: np.ndarray
    table: np.ndarray


class ManipulatorSegmenter:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._last_good_manipulator_mask: Optional[np.ndarray] = None
        self._manipulator_miss_frames = 0

    def empty_masks(self, color_shape) -> Masks:
        h, w = color_shape[:2]
        return Masks(
            robot=np.zeros((h, w), dtype=np.uint8),
            manipulator=np.zeros((h, w), dtype=np.uint8),
            table=np.zeros((h, w), dtype=np.uint8),
        )

    # ── приватные обёртки над алгоритмами (устраняют дублирование cfg-аргументов) ──
    def _table_mask(self, depth, depth_scale, robot_seed_mask) -> np.ndarray:
        cfg = self.cfg
        return build_table_mask(
            depth=depth,
            depth_scale=depth_scale,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
            robot_seed_mask=robot_seed_mask,
            protect_mask=robot_seed_mask,
            table_depth_m=cfg.table_depth_m,
            table_depth_margin_m=cfg.table_depth_margin_m,
            seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
            protect_dilate_px=cfg.table_protect_dilate_px,
            min_y_frac=cfg.table_min_y_frac,
            min_area_px=cfg.table_min_area_px,
            min_width_frac=cfg.table_min_width_frac,
            max_height_frac=cfg.table_max_height_frac,
        )

    def _manipulator_mask(self, color_bgr, depth, depth_scale, robot_seed_mask,
                          table_mask, grip_roi_uv, front_view_mode, gripper_debug) -> np.ndarray:
        cfg = self.cfg
        return build_manipulator_mask(
            color_bgr=color_bgr,
            depth=depth,
            depth_scale=depth_scale,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
            robot_seed_mask=robot_seed_mask,
            arm_gray_min=cfg.manipulator_arm_gray_min,
            seed_dilate_px=cfg.manipulator_seed_dilate_px,
            table_depth_m=cfg.table_depth_m,
            table_depth_margin_m=cfg.table_depth_margin_m,
            max_seed_dist_px=cfg.manipulator_max_seed_dist_px,
            min_seed_overlap_ratio=cfg.manipulator_min_seed_overlap_ratio,
            max_component_area_frac=cfg.manipulator_max_component_area_frac,
            table_seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
            table_mask=table_mask,
            mask_adaptive_window_px=cfg.mask_adaptive_window_px,
            mask_adaptive_k_sigma=cfg.mask_adaptive_k_sigma,
            component_min_area_px=cfg.component_filter_min_area_px,
            component_min_compactness=cfg.component_filter_min_compactness,
            component_reject_bottom_frac=cfg.component_filter_reject_bottom_frac,
            strict_no_table_mode=cfg.strict_no_table_mode,
            strict_table_guard_px=cfg.strict_table_guard_px,
            strict_bottom_seed_overlap_min=cfg.strict_bottom_seed_overlap_min,
            grip_roi_uv=grip_roi_uv,
            front_view_mode=front_view_mode,
            front_view_smart_cut_enabled=cfg.front_view_smart_cut_enabled,
            front_view_cut_offset_px=cfg.front_view_cut_offset_px,
            front_view_seed_connectivity_min_area_px=cfg.front_view_seed_connectivity_min_area_px,
            include_gripper=cfg.manipulator_include_gripper,
            gripper_merge_gray_max=cfg.gripper_merge_gray_max,
            gripper_merge_roi_radius_px=cfg.gripper_merge_roi_radius_px,
            gripper_merge_attach_dilate_px=cfg.gripper_merge_attach_dilate_px,
            gripper_merge_max_depth_delta_m=cfg.gripper_merge_max_depth_delta_m,
            gripper_merge_min_area_px=cfg.gripper_merge_min_area_px,
            gripper_merge_close_px=cfg.gripper_merge_close_px,
            gripper_merge_roi_extend_down_px=cfg.gripper_merge_roi_extend_down_px,
            gripper_merge_allow_no_depth=cfg.gripper_merge_allow_no_depth,
            debug_out=gripper_debug,
        )

    def build_fk_masks(self, masks: Masks, fkf, depth, depth_scale, color_bgr,
                       gripper_debug) -> Masks:
        """FK-self-filter: маска робота по проекции скелета + стол + манипулятор.

        Вызывается только при cfg.use_robot_self_filter и наличии fk_uvs.
        """
        cfg = self.cfg
        masks.robot = build_robot_depth_mask(
            depth=depth,
            depth_scale=depth_scale,
            joint_uvs=fkf.fk_uvs,
            joint_depths_m=fkf.joint_depths_m,
            link_radius_px=cfg.robot_link_radius_px,
            dilate_px=cfg.robot_mask_dilate_px,
            depth_eps_m=cfg.robot_mask_depth_eps_m,
        )
        masks.table = self._table_mask(depth, depth_scale, masks.robot)
        if np.count_nonzero(masks.robot) > cfg.robot_mask_min_pixels:
            masks.manipulator = self._manipulator_mask(
                color_bgr, depth, depth_scale, masks.robot, masks.table,
                fkf.grip_roi_uv, fkf.front_view_mode, gripper_debug,
            )
        return masks

    def fallback_manipulator_only(self, masks: Masks, fkf, depth, depth_scale,
                                  color_bgr, gripper_debug) -> Masks:
        """В manipulator-only режиме всегда пытаемся построить маску руки
        (даже при слабой FK-проекции) через fallback central-component логику."""
        cfg = self.cfg
        if cfg.use_manipulator_only_mode and np.count_nonzero(masks.manipulator) < 40:
            if np.count_nonzero(masks.table) < 40:
                masks.table = self._table_mask(depth, depth_scale, masks.robot)
            masks.manipulator = self._manipulator_mask(
                color_bgr, depth, depth_scale, masks.robot, masks.table,
                fkf.grip_roi_uv, fkf.front_view_mode, gripper_debug,
            )
        return masks

    def hold_and_resolve(self, masks: Masks) -> Masks:
        """Temporal-hold последней хорошей маски + резолвер non-overlap со столом."""
        cfg = self.cfg
        # Temporal hold: если текущий кадр потерял маску — держим последнюю
        # хорошую маску до cfg.manipulator_hold_frames кадров.
        if np.count_nonzero(masks.manipulator) > 40:
            self._last_good_manipulator_mask = masks.manipulator.copy()
            self._manipulator_miss_frames = 0
        elif self._last_good_manipulator_mask is not None:
            self._manipulator_miss_frames += 1
            if self._manipulator_miss_frames <= cfg.manipulator_hold_frames:
                masks.manipulator = self._last_good_manipulator_mask.copy()
            else:
                self._last_good_manipulator_mask = None
                self._manipulator_miss_frames = 0

        # Resolver: enforce non-overlap with robot-priority protect zone.
        protect_zone = np.zeros_like(masks.robot)
        protect_src = cv2.bitwise_or(masks.robot, masks.manipulator)
        if np.count_nonzero(protect_src) > 0:
            k_pr = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.seed_gate_protect_dilate_px, cfg.seed_gate_protect_dilate_px),
            )
            protect_zone = cv2.dilate(protect_src, k_pr, iterations=1)

        if np.count_nonzero(masks.table) > 0:
            masks.table[protect_zone > 0] = 0
            masks.manipulator[(masks.table > 0) & (protect_zone == 0)] = 0
        if np.count_nonzero(masks.manipulator) < 40 and np.count_nonzero(masks.robot) > 40:
            k_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            masks.manipulator = cv2.dilate(masks.robot, k_fb, iterations=1)
            masks.manipulator[(masks.table > 0) & (protect_zone == 0)] = 0
        return masks
