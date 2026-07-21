"""
Конвейер восприятия одного кадра.

Связывает сервисы зрения и манипулятора в фиксированную последовательность стадий
(cloud → robot → o3d → masks → scene → draw), идентичную исходному циклу main().
Владеет сервисами и их per-frame состоянием; возвращает готовый overlay, debug-мозаику,
облако для сохранения и тайминги стадий.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pointcloud_pipeline import build_cloud_arrays
from realsense_io import AppConfig
from robot.services.fk_service import FkFrame, FkProjector
from robot.services.joint_reader import JointAngleReader
from vision.collision_service import CollisionService
from vision.gripper_service import GripperWristService
from vision.overlay_service import OverlayRenderer
from vision.segmentation_service import ManipulatorSegmenter
from vision.visualizer import CloudVisualizer


@dataclass
class FrameResult:
    overlay: np.ndarray
    debug_mosaic: Optional[np.ndarray]
    points: np.ndarray
    colors: np.ndarray
    collision_level: str = "SAFE"  # SAFE/WARN/DANGER — сигнал для остановки робота
    stage_times: dict = field(default_factory=dict)
    # ── данные для валидации (Задача 1): только возврат наружу, логика не менялась ──
    t_capture: float = 0.0        # perf_counter на момент захвата кадра (из раннера)
    t_decision: float = 0.0       # perf_counter сразу после вычисления коллизии
    min_dist_m: float = float("nan")  # минимальная дистанция рука↔объект по кадру
    n_objects: int = 0            # число подтверждённых объектов сцены
    objects: list = field(default_factory=list)  # [{obj_id, part, dist_m, level}, ...]
    joint_angles: Optional[tuple] = None  # углы A1..A6 на этом кадре (для лога останова)
    manipulator_mask: Optional[np.ndarray] = None  # маска руки (для дампа масок, Задача 6)


class PerceptionPipeline:
    def __init__(self, cfg: AppConfig, visualizer: CloudVisualizer):
        self.cfg = cfg
        self.visualizer = visualizer
        self.joint_reader = JointAngleReader(cfg).connect()
        self.fk_projector = FkProjector(cfg)
        self.segmenter = ManipulatorSegmenter(cfg)
        self.gripper_wrist = GripperWristService(cfg)
        self.collision_service = CollisionService(cfg)
        self.overlay = OverlayRenderer(cfg)

    def close(self) -> None:
        self.joint_reader.close()

    def process(self, frame, frame_count: int, t_capture: float = 0.0) -> FrameResult:
        cfg = self.cfg
        color_bgr, depth, intrinsics, depth_scale = frame
        st = {}

        # ── stage: cloud ──
        _t1 = time.perf_counter()
        points, colors, valid_flat = build_cloud_arrays(
            color_bgr, depth, intrinsics, depth_scale, cfg.depth_min_m, cfg.depth_max_m
        )
        st["cloud"] = time.perf_counter() - _t1

        if frame_count == 1 and len(points) > 0:
            print(f"[INFO] Points: {len(points)}")
            print(f"[INFO] Bounds (m): X[{points[:, 0].min():.3f},{points[:, 0].max():.3f}] "
                  f"Y[{points[:, 1].min():.3f},{points[:, 1].max():.3f}] "
                  f"Z[{points[:, 2].min():.3f},{points[:, 2].max():.3f}]")

        # ── stage: robot (неблокирующее чтение углов) ──
        _t2 = time.perf_counter()
        joint_angles = self.joint_reader.read(frame_count)
        st["robot"] = time.perf_counter() - _t2

        # ── stage: o3d render ──
        _t3 = time.perf_counter()
        self.visualizer.render(points, colors, frame_count)
        st["o3d"] = time.perf_counter() - _t3

        # ── stage: masks ──
        _ck_a = time.perf_counter()
        overlay = color_bgr.copy()
        masks = self.segmenter.empty_masks(color_bgr.shape)
        gripper_debug = {}
        fkf = FkFrame()  # defaults: use_fk=False, всё None/False

        if self.fk_projector.base_use_fk(joint_angles):
            try:
                fkf = self.fk_projector.project(joint_angles, intrinsics, color_bgr.shape)
                if cfg.use_robot_self_filter and fkf.fk_uvs is not None:
                    self.segmenter.build_fk_masks(
                        masks, fkf, depth, depth_scale, color_bgr, gripper_debug
                    )
            except Exception as e:
                print(f"[WARN] FK projection failed: {e}")
                fkf.use_fk = False

        self.segmenter.fallback_manipulator_only(
            masks, fkf, depth, depth_scale, color_bgr, gripper_debug
        )
        self.segmenter.hold_and_resolve(masks)
        _ck_masks = time.perf_counter()
        st["masks"] = _ck_masks - _ck_a

        # ── stage: scene + collision ──
        cf = self.collision_service.evaluate(
            masks, self.fk_projector, points, colors, valid_flat,
            color_bgr, intrinsics, joint_angles, frame_count, self.visualizer.vis,
        )
        # Момент принятия решения о коллизии — для latency_ms (Задача 1)
        t_decision = time.perf_counter()
        _ck_scene = t_decision
        st["scene"] = _ck_scene - _ck_masks

        # ── stage: draw (детекция хвата/запястья + рисование) ──
        gripper_contour = self.gripper_wrist.detect_gripper(
            fkf, masks, color_bgr, depth, depth_scale
        )
        self.overlay.draw_gripper(overlay, gripper_contour)
        self.overlay.draw_robot_mask(overlay, masks)
        wrist_draw = self.gripper_wrist.detect_wrist(
            fkf, masks, color_bgr, depth, depth_scale, gripper_contour
        )
        self.overlay.draw_wrist(overlay, wrist_draw)

        link_thickness_px = None
        if (fkf.fk_uvs is not None and cfg.fk_skeleton_thick and joint_angles is not None
                and self.fk_projector.T_cr is not None and len(joint_angles) >= 6):
            link_thickness_px = self.fk_projector.link_thickness_px(joint_angles, intrinsics)
        self.overlay.draw_hud(
            overlay,
            joint_angles=joint_angles,
            fkf=fkf,
            link_thickness_px=link_thickness_px,
            n_points=len(points),
            cf=cf,
            intrinsics=intrinsics,
            color_shape=color_bgr.shape[:2],
        )
        _ck_draw = time.perf_counter()
        st["draw"] = _ck_draw - _ck_scene

        debug_mosaic = None
        if cfg.show_debug_masks:
            debug_mosaic = self.overlay.build_debug_mosaic(
                color_bgr, depth, depth_scale, masks, gripper_debug, cf
            )

        return FrameResult(
            overlay=overlay,
            debug_mosaic=debug_mosaic,
            points=points,
            colors=colors,
            collision_level=cf.worst_level,
            stage_times=st,
            t_capture=t_capture,
            t_decision=t_decision,
            min_dist_m=min(
                (r.min_dist_m for r in cf.results), default=float("nan")
            ),
            n_objects=len(cf.scene_objects),
            objects=[
                {"obj_id": r.obj_id, "part": r.part,
                 "dist_m": r.min_dist_m, "level": r.level}
                for r in cf.results
            ],
            joint_angles=joint_angles,
            manipulator_mask=masks.manipulator,
        )
