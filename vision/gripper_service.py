"""
Сервис детекции хвата (gripper) и запястья (wrist) на RGB-D.

Выбирает ветку детекции (FK-guided ROI / robot-mask / legacy), сглаживает bbox
по EMA и держит последнее положение N кадров. Возвращает контур хвата и
draw-спецификацию запястья (саму отрисовку делает OverlayRenderer).
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gripper_detection import (
    detect_gripper_contour,
    detect_gripper_from_manipulator_mask,
    detect_gripper_in_roi,
    detect_gripper_with_robot_mask,
    detect_wrist_above_gripper,
    detect_wrist_in_roi,
    detect_wrist_with_robot_mask,
)
from realsense_io import AppConfig


def bbox_to_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
        dtype=np.int32,
    )


@dataclass
class WristDraw:
    """Инструкция отрисовки запястья для OverlayRenderer."""
    kind: str  # "rect" | "contour"
    rect: Optional[tuple] = None
    contour: Optional[np.ndarray] = None


class GripperWristService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._gripper_bbox_ema: Optional[np.ndarray] = None
        self._gripper_miss_frames = 0
        self._wrist_bbox_ema: Optional[np.ndarray] = None
        self._wrist_miss_frames = 0

    @property
    def _prev_gripper_bbox(self):
        if self._gripper_bbox_ema is None:
            return None
        return tuple(int(round(v)) for v in self._gripper_bbox_ema)

    @property
    def _prev_wrist_bbox(self):
        if self._wrist_bbox_ema is None:
            return None
        return tuple(int(round(v)) for v in self._wrist_bbox_ema)

    # ─────────────────────────────── GRIPPER ───────────────────────────────
    def detect_gripper(self, fkf, masks, color_bgr, depth, depth_scale):
        cfg = self.cfg
        gripper_contour = None  # gripper detection disabled by user request (по умолчанию)
        need_gripper_contour = (
            cfg.show_gripper_overlay
            or cfg.detect_gripper_from_manipulator_mask
        )

        if need_gripper_contour:
            if fkf.use_fk and fkf.grip_roi_uv is not None:
                # ---- FK-guided 2D ROI ----
                if (
                    cfg.detect_gripper_from_manipulator_mask
                    and np.count_nonzero(masks.manipulator) > 120
                ):
                    gripper_contour, _grip_mask = detect_gripper_from_manipulator_mask(
                        color_bgr=color_bgr,
                        depth=depth,
                        depth_scale=depth_scale,
                        depth_min_m=cfg.depth_min_m,
                        depth_max_m=cfg.depth_max_m,
                        roi_center_uv=fkf.grip_roi_uv,
                        roi_radius=cfg.fk_grip_roi_radius_px,
                        manipulator_mask=masks.manipulator,
                        gray_max=cfg.gripper_from_mask_gray_max,
                        min_area_px=cfg.gripper_from_mask_min_area_px,
                        max_area_px=cfg.gripper_from_mask_max_area_px,
                        near_mask_dilate_px=cfg.gripper_from_mask_near_dilate_px,
                    )
                else:
                    gripper_contour = None
                if cfg.use_robot_self_filter and np.count_nonzero(masks.robot) > 120:
                    if gripper_contour is None:
                        gripper_contour, _grip_mask = detect_gripper_with_robot_mask(
                            color_bgr, depth, depth_scale,
                            cfg.depth_min_m, cfg.depth_max_m,
                            roi_center_uv=fkf.grip_roi_uv,
                            roi_radius=cfg.fk_grip_roi_radius_px,
                            robot_mask=masks.manipulator if np.count_nonzero(masks.manipulator) > 0 else masks.robot,
                            gray_max=cfg.gripper_gray_max,
                            min_area_px=cfg.gripper_min_area_px,
                            prev_gripper_bbox=self._prev_gripper_bbox,
                            track_max_jump_px=cfg.gripper_track_max_jump_px,
                            track_iou_weight=cfg.gripper_track_iou_weight,
                            track_center_weight=cfg.gripper_track_center_weight,
                        )
                    if gripper_contour is None:
                        gripper_contour, _grip_mask = detect_gripper_in_roi(
                            color_bgr, depth, depth_scale,
                            cfg.depth_min_m, cfg.depth_max_m,
                            roi_center_uv=fkf.grip_roi_uv,
                            roi_radius=cfg.fk_grip_roi_radius_px,
                            gray_max=cfg.gripper_gray_max,
                            min_area_px=cfg.gripper_min_area_px,
                            arm_gray_min=cfg.arm_gray_min,
                            near_arm_dilate_px=cfg.near_arm_dilate_px,
                            gripper_max_depth_delta_m=cfg.gripper_max_depth_delta_m,
                            prev_gripper_bbox=self._prev_gripper_bbox,
                            track_max_jump_px=cfg.gripper_track_max_jump_px,
                            track_iou_weight=cfg.gripper_track_iou_weight,
                            track_center_weight=cfg.gripper_track_center_weight,
                        )
                else:
                    gripper_contour, _grip_mask = detect_gripper_in_roi(
                        color_bgr, depth, depth_scale,
                        cfg.depth_min_m, cfg.depth_max_m,
                        roi_center_uv=fkf.grip_roi_uv,
                        roi_radius=cfg.fk_grip_roi_radius_px,
                        gray_max=cfg.gripper_gray_max,
                        min_area_px=cfg.gripper_min_area_px,
                        arm_gray_min=cfg.arm_gray_min,
                        near_arm_dilate_px=cfg.near_arm_dilate_px,
                        gripper_max_depth_delta_m=cfg.gripper_max_depth_delta_m,
                        prev_gripper_bbox=self._prev_gripper_bbox,
                        track_max_jump_px=cfg.gripper_track_max_jump_px,
                        track_iou_weight=cfg.gripper_track_iou_weight,
                        track_center_weight=cfg.gripper_track_center_weight,
                    )
            else:
                # ---- Legacy colour-based gripper detection ----
                gripper_contour, _grip_mask = detect_gripper_contour(
                    color_bgr, depth, depth_scale,
                    cfg.depth_min_m, cfg.depth_max_m,
                    cfg.gripper_roi_y_fraction,
                    cfg.gripper_gray_max,
                    cfg.gripper_min_area_px,
                    arm_gray_min=cfg.arm_gray_min,
                    near_arm_dilate_px=cfg.near_arm_dilate_px,
                    gripper_touch_arm_dilate_px=cfg.gripper_touch_arm_dilate_px,
                    gripper_max_depth_delta_m=cfg.gripper_max_depth_delta_m,
                    gripper_fallback_max_x_fraction=cfg.gripper_fallback_max_x_fraction,
                    arm_mask_min_area_px=cfg.arm_mask_min_area_px,
                    prev_gripper_bbox=self._prev_gripper_bbox,
                    track_iou_weight=cfg.gripper_track_iou_weight,
                    track_center_weight=cfg.gripper_track_center_weight,
                    track_max_jump_px=cfg.gripper_track_max_jump_px,
                    gripper_tip_max_dist_px=cfg.gripper_tip_max_dist_px,
                    gripper_tip_max_dist_frac=cfg.gripper_tip_max_dist_frac,
                    gripper_tip_allow_above_px=cfg.gripper_tip_allow_above_px,
                    table_depth_m=cfg.table_depth_m,
                    table_depth_margin_m=cfg.table_depth_margin_m,
                )

        if cfg.gripper_bbox_smooth_alpha > 0.0:
            if gripper_contour is not None:
                gx, gy, gw, gh = cv2.boundingRect(gripper_contour)
                gv = np.array([gx, gy, gw, gh], dtype=np.float64)
                ga = cfg.gripper_bbox_smooth_alpha
                if self._gripper_bbox_ema is None:
                    self._gripper_bbox_ema = gv.copy()
                else:
                    self._gripper_bbox_ema = ga * gv + (1.0 - ga) * self._gripper_bbox_ema
                self._gripper_miss_frames = 0
            elif self._gripper_bbox_ema is not None:
                self._gripper_miss_frames += 1
                if self._gripper_miss_frames > cfg.gripper_hold_miss_frames:
                    self._gripper_bbox_ema = None
                    self._gripper_miss_frames = 0
            if gripper_contour is None and self._gripper_bbox_ema is not None:
                gi, gj, gwi, ghi = [int(round(t)) for t in self._gripper_bbox_ema]
                if gwi > 2 and ghi > 2:
                    gripper_contour = bbox_to_contour(gi, gj, gwi, ghi)
        return gripper_contour

    # ──────────────────────────────── WRIST ────────────────────────────────
    def detect_wrist(self, fkf, masks, color_bgr, depth, depth_scale,
                     gripper_contour) -> Optional[WristDraw]:
        cfg = self.cfg
        wrist_contour = None
        if not (cfg.show_wrist_joint_overlay and not cfg.use_manipulator_only_mode):
            return None

        if fkf.use_fk and fkf.wrist_roi_uv is not None:
            # ---- FK-guided 2D ROI (independent of gripper) ----
            if cfg.use_robot_self_filter and np.count_nonzero(masks.robot) > 120:
                wrist_contour, _wrist_mask = detect_wrist_with_robot_mask(
                    color_bgr, depth, depth_scale,
                    cfg.depth_min_m, cfg.depth_max_m,
                    roi_center_uv=fkf.wrist_roi_uv,
                    roi_radius=cfg.fk_wrist_roi_radius_px,
                    robot_mask=masks.manipulator if np.count_nonzero(masks.manipulator) > 0 else masks.robot,
                    wrist_gray_min=cfg.wrist_gray_min,
                    min_area_px=cfg.wrist_min_area_px,
                    max_area_px=cfg.wrist_max_area_px,
                    wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                    wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                    prev_wrist_bbox=self._prev_wrist_bbox,
                    track_max_jump_px=cfg.gripper_track_max_jump_px,
                    track_iou_weight=cfg.gripper_track_iou_weight,
                    track_center_weight=cfg.gripper_track_center_weight,
                )
                if wrist_contour is None:
                    wrist_contour, _wrist_mask = detect_wrist_in_roi(
                        color_bgr, depth, depth_scale,
                        cfg.depth_min_m, cfg.depth_max_m,
                        roi_center_uv=fkf.wrist_roi_uv,
                        roi_radius=cfg.fk_wrist_roi_radius_px,
                        wrist_gray_min=cfg.wrist_gray_min,
                        min_area_px=cfg.wrist_min_area_px,
                        max_area_px=cfg.wrist_max_area_px,
                        wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                        wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                        prev_wrist_bbox=self._prev_wrist_bbox,
                        track_max_jump_px=cfg.gripper_track_max_jump_px,
                        track_iou_weight=cfg.gripper_track_iou_weight,
                        track_center_weight=cfg.gripper_track_center_weight,
                    )
            else:
                wrist_contour, _wrist_mask = detect_wrist_in_roi(
                    color_bgr, depth, depth_scale,
                    cfg.depth_min_m, cfg.depth_max_m,
                    roi_center_uv=fkf.wrist_roi_uv,
                    roi_radius=cfg.fk_wrist_roi_radius_px,
                    wrist_gray_min=cfg.wrist_gray_min,
                    min_area_px=cfg.wrist_min_area_px,
                    max_area_px=cfg.wrist_max_area_px,
                    wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                    wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                    prev_wrist_bbox=self._prev_wrist_bbox,
                    track_max_jump_px=cfg.gripper_track_max_jump_px,
                    track_iou_weight=cfg.gripper_track_iou_weight,
                    track_center_weight=cfg.gripper_track_center_weight,
                )
        else:
            # ---- Legacy wrist detection above gripper ----
            wrist_contour, _wrist_mask = detect_wrist_above_gripper(
                color_bgr, depth, depth_scale,
                cfg.depth_min_m, cfg.depth_max_m,
                gripper_contour,
                arm_gray_min=cfg.arm_gray_min,
                near_arm_dilate_px=cfg.near_arm_dilate_px,
                wrist_band_height_px=cfg.wrist_band_height_px,
                wrist_band_pad_px=cfg.wrist_band_pad_px,
                wrist_bridge_w=cfg.wrist_bridge_w,
                wrist_bridge_h=cfg.wrist_bridge_h,
                wrist_min_area_px=cfg.wrist_min_area_px,
                wrist_max_area_px=cfg.wrist_max_area_px,
                wrist_use_near_arm=cfg.wrist_use_near_arm,
                wrist_gray_min=cfg.wrist_gray_min,
                wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                wrist_fallback_without_near_arm=cfg.wrist_fallback_without_near_arm,
                wrist_anchor_max_below_grip_px=cfg.wrist_anchor_max_below_grip_px,
                wrist_min_bbox_height_px=cfg.wrist_min_bbox_height_px,
                wrist_min_bbox_width_px=cfg.wrist_min_bbox_width_px,
                wrist_max_link_span_px=cfg.wrist_max_link_span_px,
                wrist_use_static_roi=cfg.wrist_use_static_roi,
                wrist_roi_x0_frac=cfg.wrist_roi_x0_frac,
                wrist_roi_y0_frac=cfg.wrist_roi_y0_frac,
                wrist_roi_x1_frac=cfg.wrist_roi_x1_frac,
                wrist_roi_y1_frac=cfg.wrist_roi_y1_frac,
                wrist_max_dx_from_grip_px=cfg.wrist_max_dx_from_grip_px,
                wrist_max_dx_from_grip_frac=cfg.wrist_max_dx_from_grip_frac,
                wrist_min_overlap_arm_px=cfg.wrist_min_overlap_arm_px,
                wrist_band_height_from_gripper=cfg.wrist_band_height_from_gripper,
                table_depth_m=cfg.table_depth_m,
                table_depth_margin_m=cfg.table_depth_margin_m,
            )

        if cfg.wrist_bbox_smooth_alpha > 0.0:
            if wrist_contour is not None:
                wx, wy, ww, wh = cv2.boundingRect(wrist_contour)
                v = np.array([wx, wy, ww, wh], dtype=np.float64)
                a = cfg.wrist_bbox_smooth_alpha
                if self._wrist_bbox_ema is None:
                    self._wrist_bbox_ema = v.copy()
                else:
                    self._wrist_bbox_ema = a * v + (1.0 - a) * self._wrist_bbox_ema
                self._wrist_miss_frames = 0
            elif self._wrist_bbox_ema is not None:
                self._wrist_miss_frames += 1
                if self._wrist_miss_frames > cfg.wrist_hold_miss_frames:
                    self._wrist_bbox_ema = None
                    self._wrist_miss_frames = 0
            if self._wrist_bbox_ema is not None:
                xi, yi, wi, hi = [int(round(t)) for t in self._wrist_bbox_ema]
                return WristDraw(kind="rect", rect=(xi, yi, wi, hi))
            return None
        elif wrist_contour is not None:
            return WristDraw(kind="contour", contour=wrist_contour)
        return None
