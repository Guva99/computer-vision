from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json
import time

import cv2
import numpy as np
import open3d as o3d

from gripper_detection import (
    build_debug_mosaic,
    build_manipulator_mask,
    build_robot_depth_mask,
    build_table_mask,
    detect_gripper_contour,
    detect_gripper_from_manipulator_mask,
    detect_gripper_in_roi,
    detect_gripper_with_robot_mask,
    detect_wrist_above_gripper,
    detect_wrist_in_roi,
    detect_wrist_with_robot_mask,
    draw_robot_mask_overlay,
    draw_fk_projections,
    draw_gripper_overlay,
    draw_roi_circle,
    draw_wrist_overlay,
    draw_wrist_zone_rect,
)
import collision as collision_mod
from kuka_fk import fk_joints, joints_to_uvs
from pointcloud_pipeline import build_cloud_arrays, to_open3d_cloud
from realsense_io import AppConfig, RealSenseCamera
from segmentation import split_cloud_by_mask

try:
    from robot.drivers.openshowvar import OpenShowVar

    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("[WARNING] Robot modules not found")


def get_external_objects_cloud(points, colors):
    return to_open3d_cloud(points, colors)


def save_full_cloud(output_dir, points, colors):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"cloud_raw_{stamp}.ply"
    o3d.io.write_point_cloud(str(path), to_open3d_cloud(points, colors))
    return path


def parse_joint_angles(axis_string):
    try:
        cleaned = axis_string.replace(",", " ").replace("{", "").replace("}", "").replace("E6AXIS:", "")
        parts = cleaned.split()
        angles = []
        for i, part in enumerate(parts):
            if part.startswith("A") and len(part) == 2 and part[1].isdigit():
                if i + 1 < len(parts):
                    try:
                        angles.append(float(parts[i + 1]))
                    except ValueError:
                        pass
        return tuple(angles[:6]) if len(angles) >= 6 else None
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}, data: {axis_string[:100]}")
        return None


def bbox_to_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
        dtype=np.int32,
    )


def load_extrinsic(path: str) -> Optional[np.ndarray]:
    """Load 4×4 camera-from-robot extrinsic matrix from JSON file."""
    p = Path(path)
    if not p.exists():
        print(f"[INFO] extrinsic.json not found at '{path}' — FK-guided ROI disabled.")
        return None
    try:
        data = json.loads(p.read_text())
        T_cr = np.array(data["T_cr"], dtype=np.float64)
        assert T_cr.shape == (4, 4), "T_cr must be 4×4"
        print(f"[OK] Loaded extrinsic matrix from '{path}'  (RMS was {data.get('rms_px', '?'):.2f} px)")
        return T_cr
    except Exception as e:
        print(f"[WARN] Failed to load extrinsic '{path}': {e}")
        return None


def _uv_in_frame(uv: Optional[tuple], w: int, h: int) -> bool:
    if uv is None:
        return False
    u, v = uv
    return (0 <= int(u) < int(w)) and (0 <= int(v) < int(h))

def main():
    cfg = AppConfig()

    # Load extrinsic calibration for FK-guided ROI
    T_cr: Optional[np.ndarray] = None
    if cfg.use_fk_roi:
        T_cr = load_extrinsic(cfg.extrinsic_path)
        if T_cr is None:
            print("[INFO] Falling back to legacy colour-only detection.")

    robot = None
    if ROBOT_AVAILABLE and cfg.use_robot_kinematics:
        print("\n" + "=" * 60)
        print("RAW POINT CLOUD + gripper outline (vision) + robot angles")
        print("=" * 60)
        try:
            robot = OpenShowVar(ip=cfg.robot_ip, port=cfg.robot_port,
                                read_timeout=cfg.robot_read_timeout_s)
            if robot.can_connect:
                print(f"[OK] Robot read: {cfg.robot_ip}:{cfg.robot_port}")
            else:
                robot = None
                print(f"[WARN] Robot not reachable")
        except Exception as e:
            print(f"[ERROR] {e}")
            robot = None
    else:
        print("\n[INFO] Camera only (robot angles off)")

    camera = RealSenseCamera(cfg)
    # 3D-окно Open3D отключаемо: cfg.show_o3d_window=False → vis=None (экономит ~60 мс/кадр).
    # Для скриншотов в статью поставь show_o3d_window=True в AppConfig.
    pcd = o3d.geometry.PointCloud()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis = None
    if cfg.show_o3d_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Raw point cloud (RGB colors, no filters)", width=1200, height=800)
        vis.add_geometry(pcd)
        vis.add_geometry(coordinate_frame)

    frame_count = 0
    last_joint_angles = None  # последние успешно прочитанные углы (для real-time без блокировки)
    # Тайминг-проба: суммы по стадиям, печать средних раз в N кадров
    _t_acc = {"grab": 0.0, "cloud": 0.0, "robot": 0.0, "o3d": 0.0,
              "masks": 0.0, "scene": 0.0, "draw": 0.0, "total": 0.0}
    _t_report_every = 30
    gripper_bbox_ema: Optional[np.ndarray] = None
    gripper_miss_frames = 0
    wrist_bbox_ema: Optional[np.ndarray] = None
    wrist_miss_frames = 0
    last_good_manipulator_mask: Optional[np.ndarray] = None
    manipulator_miss_frames = 0
    collision_logger = None
    collision_bbox_geoms: list = []
    if cfg.enable_collision:
        collision_logger = collision_mod.CollisionLogger(
            cfg.collision_log_path, cfg.collision_log_every_n
        )
    print("\n[INFO] Starting camera (raw depth, no post-processing)...")
    try:
        camera.start()
        print("[INFO] Camera OK. Keys: q=quit, s=save PLY\n")

        while True:
            _t0 = time.perf_counter()
            frame = camera.get_aligned_frames()
            if frame is None:
                continue
            _t_grab = time.perf_counter() - _t0

            frame_count += 1
            color_bgr, depth, intrinsics, depth_scale = frame

            _t1 = time.perf_counter()
            points, colors, valid_flat = build_cloud_arrays(
                color_bgr, depth, intrinsics, depth_scale, cfg.depth_min_m, cfg.depth_max_m
            )
            _t_cloud = time.perf_counter() - _t1

            if frame_count == 1 and len(points) > 0:
                print(f"[INFO] Points: {len(points)}")
                print(f"[INFO] Bounds (m): X[{points[:, 0].min():.3f},{points[:, 0].max():.3f}] "
                      f"Y[{points[:, 1].min():.3f},{points[:, 1].max():.3f}] "
                      f"Z[{points[:, 2].min():.3f},{points[:, 2].max():.3f}]")

            # Read robot angles WITHOUT blocking the camera loop:
            #  • short socket timeout (cfg.robot_read_timeout_s) — recv не висит
            #  • only every N frames — углы меняются плавно
            #  • on timeout/None — reuse last known angles (last_joint_angles)
            _t2 = time.perf_counter()
            if robot and (frame_count % cfg.robot_read_every_n == 0):
                try:
                    raw = robot.read("$AXIS_ACT", debug=False)
                    if raw is not None:
                        parsed = parse_joint_angles(raw.decode())
                        if parsed is not None:
                            last_joint_angles = parsed
                except Exception:
                    pass  # timeout / busy controller → keep last_joint_angles
            joint_angles = last_joint_angles
            _t_robot = time.perf_counter() - _t2

            # Open3D рендер — тяжёлый; обновляем раз в N кадров (1-й кадр всегда).
            # vis is None когда окно выключено (cfg.show_o3d_window=False).
            _t3 = time.perf_counter()
            if vis is not None and (frame_count == 1 or (frame_count % cfg.o3d_render_every_n == 0)):
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                if frame_count == 1:
                    vis.reset_view_point(True)
                    print("[INFO] Open3D view reset")
                vis.poll_events()
                vis.update_renderer()
            _t_o3d = time.perf_counter() - _t3

            _ck_a = time.perf_counter()  # ── timing: start of detection ──
            overlay = color_bgr.copy()
            gripper_contour = None  # gripper detection disabled by user request
            wrist_contour = None

            # ----------------------------------------------------------------
            # Decide whether to use FK-guided ROI or legacy mode
            # ----------------------------------------------------------------
            fk_uvs = None  # list[7]: base + J1..J6
            grip_roi_uv: Optional[tuple] = None   # ROI center for A6 (flange)
            wrist_roi_uv: Optional[tuple] = None  # ROI center for A5
            front_view_mode = False
            robot_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            manipulator_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            table_candidate = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            gripper_debug: Dict[str, np.ndarray] = {}
            scene_objects = []
            scene_objects_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            scene_obj_mask_2d = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            collision_results = []
            collision_worst_level = "SAFE"
            collision_worst_focus = None

            use_fk = (cfg.use_fk_roi and T_cr is not None and joint_angles is not None
                      and len(joint_angles) >= 6)

            if use_fk:
                try:
                    fk_uvs = joints_to_uvs(joint_angles, T_cr, intrinsics,
                                           gripper_length_m=cfg.fk_gripper_length_m)
                    # fk_uvs[6] = J6 flange, fk_uvs[7] = gripper tip (если fk_gripper_length_m>0)
                    # grip_roi_uv = последняя точка (tip если есть, иначе flange)
                    h_img, w_img = color_bgr.shape[:2]
                    _grip_idx = 7 if (cfg.fk_gripper_length_m > 0.0 and len(fk_uvs) > 7) else 6
                    grip_roi_uv = fk_uvs[_grip_idx] if _uv_in_frame(fk_uvs[_grip_idx], w_img, h_img) else None
                    wrist_roi_uv = fk_uvs[5] if _uv_in_frame(fk_uvs[5], w_img, h_img) else None
                    if grip_roi_uv is not None:
                        gu, _gv = grip_roi_uv
                        front_view_mode = abs(float(gu) - 0.5 * float(w_img)) < (0.22 * float(w_img))
                    if cfg.use_robot_self_filter and fk_uvs is not None:
                        fk_pos_robot = fk_joints(joint_angles, gripper_length_m=cfg.fk_gripper_length_m)
                        joint_depths_m = []
                        for p in fk_pos_robot:
                            p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
                            p_cam = T_cr @ p_h
                            joint_depths_m.append(float(p_cam[2]) if p_cam[2] > 0.0 else None)
                        robot_mask = build_robot_depth_mask(
                            depth=depth,
                            depth_scale=depth_scale,
                            joint_uvs=fk_uvs,
                            joint_depths_m=joint_depths_m,
                            link_radius_px=cfg.robot_link_radius_px,
                            dilate_px=cfg.robot_mask_dilate_px,
                            depth_eps_m=cfg.robot_mask_depth_eps_m,
                        )
                        table_candidate = build_table_mask(
                            depth=depth,
                            depth_scale=depth_scale,
                            depth_min_m=cfg.depth_min_m,
                            depth_max_m=cfg.depth_max_m,
                            robot_seed_mask=robot_mask,
                            protect_mask=robot_mask,
                            table_depth_m=cfg.table_depth_m,
                            table_depth_margin_m=cfg.table_depth_margin_m,
                            seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
                            protect_dilate_px=cfg.table_protect_dilate_px,
                            min_y_frac=cfg.table_min_y_frac,
                            min_area_px=cfg.table_min_area_px,
                            min_width_frac=cfg.table_min_width_frac,
                            max_height_frac=cfg.table_max_height_frac,
                        )
                        if np.count_nonzero(robot_mask) > cfg.robot_mask_min_pixels:
                            manipulator_mask = build_manipulator_mask(
                                color_bgr=color_bgr,
                                depth=depth,
                                depth_scale=depth_scale,
                                depth_min_m=cfg.depth_min_m,
                                depth_max_m=cfg.depth_max_m,
                                robot_seed_mask=robot_mask,
                                arm_gray_min=cfg.manipulator_arm_gray_min,
                                seed_dilate_px=cfg.manipulator_seed_dilate_px,
                                table_depth_m=cfg.table_depth_m,
                                table_depth_margin_m=cfg.table_depth_margin_m,
                                max_seed_dist_px=cfg.manipulator_max_seed_dist_px,
                                min_seed_overlap_ratio=cfg.manipulator_min_seed_overlap_ratio,
                                max_component_area_frac=cfg.manipulator_max_component_area_frac,
                                table_seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
                                table_mask=table_candidate,
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
                except Exception as e:
                    print(f"[WARN] FK projection failed: {e}")
                    use_fk = False

            # In manipulator-only mode, always try to build a manipulator mask
            # (even when FK projection is weak), using fallback central component logic.
            if cfg.use_manipulator_only_mode and np.count_nonzero(manipulator_mask) < 40:
                if np.count_nonzero(table_candidate) < 40:
                    table_candidate = build_table_mask(
                        depth=depth,
                        depth_scale=depth_scale,
                        depth_min_m=cfg.depth_min_m,
                        depth_max_m=cfg.depth_max_m,
                        robot_seed_mask=robot_mask,
                        protect_mask=robot_mask,
                        table_depth_m=cfg.table_depth_m,
                        table_depth_margin_m=cfg.table_depth_margin_m,
                        seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
                        protect_dilate_px=cfg.table_protect_dilate_px,
                        min_y_frac=cfg.table_min_y_frac,
                        min_area_px=cfg.table_min_area_px,
                        min_width_frac=cfg.table_min_width_frac,
                        max_height_frac=cfg.table_max_height_frac,
                    )
                manipulator_mask = build_manipulator_mask(
                    color_bgr=color_bgr,
                    depth=depth,
                    depth_scale=depth_scale,
                    depth_min_m=cfg.depth_min_m,
                    depth_max_m=cfg.depth_max_m,
                    robot_seed_mask=robot_mask,
                    arm_gray_min=cfg.manipulator_arm_gray_min,
                    seed_dilate_px=cfg.manipulator_seed_dilate_px,
                    table_depth_m=cfg.table_depth_m,
                    table_depth_margin_m=cfg.table_depth_margin_m,
                    max_seed_dist_px=cfg.manipulator_max_seed_dist_px,
                    min_seed_overlap_ratio=cfg.manipulator_min_seed_overlap_ratio,
                    max_component_area_frac=cfg.manipulator_max_component_area_frac,
                    table_seed_exclude_dilate_px=cfg.manipulator_table_seed_exclude_dilate_px,
                    table_mask=table_candidate,
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

            # Temporal hold: если текущий кадр потерял маску — держим последнюю
            # хорошую маску до cfg.manipulator_hold_frames кадров.
            if np.count_nonzero(manipulator_mask) > 40:
                last_good_manipulator_mask = manipulator_mask.copy()
                manipulator_miss_frames = 0
            elif last_good_manipulator_mask is not None:
                manipulator_miss_frames += 1
                if manipulator_miss_frames <= cfg.manipulator_hold_frames:
                    manipulator_mask = last_good_manipulator_mask.copy()
                else:
                    last_good_manipulator_mask = None
                    manipulator_miss_frames = 0

            # Resolver: enforce non-overlap with robot-priority protect zone.
            protect_zone = np.zeros_like(robot_mask)
            protect_src = cv2.bitwise_or(robot_mask, manipulator_mask)
            if np.count_nonzero(protect_src) > 0:
                k_pr = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (cfg.seed_gate_protect_dilate_px, cfg.seed_gate_protect_dilate_px),
                )
                protect_zone = cv2.dilate(protect_src, k_pr, iterations=1)

            if np.count_nonzero(table_candidate) > 0:
                table_candidate[protect_zone > 0] = 0
                manipulator_mask[(table_candidate > 0) & (protect_zone == 0)] = 0
            if np.count_nonzero(manipulator_mask) < 40 and np.count_nonzero(robot_mask) > 40:
                k_fb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                manipulator_mask = cv2.dilate(robot_mask, k_fb, iterations=1)
                manipulator_mask[(table_candidate > 0) & (protect_zone == 0)] = 0

            _ck_masks = time.perf_counter()  # ── timing: masks done ──
            if (
                cfg.enable_collision
                and len(points) > 0
                and np.count_nonzero(manipulator_mask) > 40
            ):
                if cfg.collision_use_2d_detection:
                    scene_objects, scene_obj_mask_2d = collision_mod.detect_scene_objects_2d(
                        points,
                        colors,
                        valid_flat,
                        manipulator_mask,
                        color_bgr.shape[:2],
                        cfg,
                        color_bgr=color_bgr,
                    )
                else:
                    _, (scene_pts, scene_col) = split_cloud_by_mask(
                        points, colors, valid_flat, manipulator_mask
                    )
                    scene_objects = collision_mod.detect_scene_objects(
                        scene_pts, scene_col, cfg
                    )
                # Растеризация точек объектов нужна только для debug-мозаики
                # (Python-цикл по точкам — дорого). Строим лишь когда показываем debug.
                if cfg.show_debug_masks:
                    scene_objects_mask = collision_mod.build_scene_objects_image_mask(
                        scene_objects, intrinsics, color_bgr.shape[:2]
                    )
                spheres = []
                if joint_angles is not None and T_cr is not None and len(joint_angles) >= 6:
                    spheres = collision_mod.build_link_spheres(
                        joint_angles,
                        T_cr,
                        cfg.collision_link_radii_m,
                        cfg.collision_spheres_per_link,
                    )
                # Hybrid: body distance from measured manipulator cloud (no T_cr
                # dependence), gripper from FK sphere (dark metal has no depth).
                gripper_spheres = [s for s in spheres if s.part == "gripper"]
                # Erode the arm mask before sampling: boundary pixels share depth
                # with adjacent objects/table and cause false near-zero distances.
                body_mask = manipulator_mask
                if cfg.collision_body_erode_px > 1:
                    k_be = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (cfg.collision_body_erode_px, cfg.collision_body_erode_px),
                    )
                    eroded_arm = cv2.erode(manipulator_mask, k_be)
                    if np.count_nonzero(eroded_arm) > 40:
                        body_mask = eroded_arm
                manip_sel = (body_mask > 0).reshape(-1)[valid_flat]
                manip_pts = points[manip_sel] if np.any(manip_sel) else points[:0]
                if scene_objects and (len(manip_pts) > 0 or gripper_spheres):
                    (
                        collision_results,
                        collision_worst_level,
                        collision_worst_focus,
                    ) = collision_mod.evaluate_collisions_hybrid(
                        manip_pts,
                        gripper_spheres,
                        scene_objects,
                        cfg.collision_warn_dist_m,
                        cfg.collision_danger_dist_m,
                        cfg.collision_focus_parts,
                        cfg.collision_body_voxel_m,
                        cfg.collision_dist_percentile,
                    )
                if collision_logger is not None:
                    collision_logger.update(
                        frame_count, collision_worst_level, collision_worst_focus
                    )
                # 3D-боксы коллизий — только при открытом окне Open3D
                if vis is not None:
                    for g in collision_bbox_geoms:
                        try:
                            vis.remove_geometry(g, reset_bounding_box=False)
                        except Exception:
                            pass
                    collision_bbox_geoms.clear()
                    for obj in scene_objects:
                        res = next(
                            (r for r in collision_results if r.obj_id == obj.obj_id), None
                        )
                        level = res.level if res is not None else "SAFE"
                        if level == "SAFE" and not cfg.show_collision_overlay:
                            continue
                        ls = collision_mod.make_o3d_aabb(obj.aabb_min, obj.aabb_max, level)
                        vis.add_geometry(ls, reset_bounding_box=False)
                        collision_bbox_geoms.append(ls)

            _ck_scene = time.perf_counter()  # ── timing: scene+collision done ──
            need_gripper_contour = (
                cfg.show_gripper_overlay
                or cfg.detect_gripper_from_manipulator_mask
            )

            if need_gripper_contour:
                if use_fk and grip_roi_uv is not None:
                    # ---- FK-guided 2D ROI ----
                    if (
                        cfg.detect_gripper_from_manipulator_mask
                        and np.count_nonzero(manipulator_mask) > 120
                    ):
                        gripper_contour, _grip_mask = detect_gripper_from_manipulator_mask(
                            color_bgr=color_bgr,
                            depth=depth,
                            depth_scale=depth_scale,
                            depth_min_m=cfg.depth_min_m,
                            depth_max_m=cfg.depth_max_m,
                            roi_center_uv=grip_roi_uv,
                            roi_radius=cfg.fk_grip_roi_radius_px,
                            manipulator_mask=manipulator_mask,
                            gray_max=cfg.gripper_from_mask_gray_max,
                            min_area_px=cfg.gripper_from_mask_min_area_px,
                            max_area_px=cfg.gripper_from_mask_max_area_px,
                            near_mask_dilate_px=cfg.gripper_from_mask_near_dilate_px,
                        )
                    else:
                        gripper_contour = None
                    if cfg.use_robot_self_filter and np.count_nonzero(robot_mask) > 120:
                        if gripper_contour is None:
                            gripper_contour, _grip_mask = detect_gripper_with_robot_mask(
                                color_bgr, depth, depth_scale,
                                cfg.depth_min_m, cfg.depth_max_m,
                                roi_center_uv=grip_roi_uv,
                                roi_radius=cfg.fk_grip_roi_radius_px,
                                robot_mask=manipulator_mask if np.count_nonzero(manipulator_mask) > 0 else robot_mask,
                                gray_max=cfg.gripper_gray_max,
                                min_area_px=cfg.gripper_min_area_px,
                                prev_gripper_bbox=(
                                    tuple(int(round(v)) for v in gripper_bbox_ema)
                                    if gripper_bbox_ema is not None else None
                                ),
                                track_max_jump_px=cfg.gripper_track_max_jump_px,
                                track_iou_weight=cfg.gripper_track_iou_weight,
                                track_center_weight=cfg.gripper_track_center_weight,
                            )
                        if gripper_contour is None:
                            gripper_contour, _grip_mask = detect_gripper_in_roi(
                                color_bgr, depth, depth_scale,
                                cfg.depth_min_m, cfg.depth_max_m,
                                roi_center_uv=grip_roi_uv,
                                roi_radius=cfg.fk_grip_roi_radius_px,
                                gray_max=cfg.gripper_gray_max,
                                min_area_px=cfg.gripper_min_area_px,
                                arm_gray_min=cfg.arm_gray_min,
                                near_arm_dilate_px=cfg.near_arm_dilate_px,
                                gripper_max_depth_delta_m=cfg.gripper_max_depth_delta_m,
                                prev_gripper_bbox=(
                                    tuple(int(round(v)) for v in gripper_bbox_ema)
                                    if gripper_bbox_ema is not None else None
                                ),
                                track_max_jump_px=cfg.gripper_track_max_jump_px,
                                track_iou_weight=cfg.gripper_track_iou_weight,
                                track_center_weight=cfg.gripper_track_center_weight,
                            )
                    else:
                        gripper_contour, _grip_mask = detect_gripper_in_roi(
                            color_bgr, depth, depth_scale,
                            cfg.depth_min_m, cfg.depth_max_m,
                            roi_center_uv=grip_roi_uv,
                            roi_radius=cfg.fk_grip_roi_radius_px,
                            gray_max=cfg.gripper_gray_max,
                            min_area_px=cfg.gripper_min_area_px,
                            arm_gray_min=cfg.arm_gray_min,
                            near_arm_dilate_px=cfg.near_arm_dilate_px,
                            gripper_max_depth_delta_m=cfg.gripper_max_depth_delta_m,
                            prev_gripper_bbox=(
                                tuple(int(round(v)) for v in gripper_bbox_ema)
                                if gripper_bbox_ema is not None else None
                            ),
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
                        prev_gripper_bbox=(
                            tuple(int(round(v)) for v in gripper_bbox_ema)
                            if gripper_bbox_ema is not None else None
                        ),
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
                    if gripper_bbox_ema is None:
                        gripper_bbox_ema = gv.copy()
                    else:
                        gripper_bbox_ema = ga * gv + (1.0 - ga) * gripper_bbox_ema
                    gripper_miss_frames = 0
                elif gripper_bbox_ema is not None:
                    gripper_miss_frames += 1
                    if gripper_miss_frames > cfg.gripper_hold_miss_frames:
                        gripper_bbox_ema = None
                        gripper_miss_frames = 0
                if gripper_contour is None and gripper_bbox_ema is not None:
                    gi, gj, gwi, ghi = [int(round(t)) for t in gripper_bbox_ema]
                    if gwi > 2 and ghi > 2:
                        gripper_contour = bbox_to_contour(gi, gj, gwi, ghi)
            if cfg.show_gripper_overlay and gripper_contour is not None:
                draw_gripper_overlay(overlay, gripper_contour, color_bgr=(0, 165, 255))

            if cfg.show_robot_mask_overlay and (np.count_nonzero(manipulator_mask) > 0 or np.count_nonzero(robot_mask) > 0):
                draw_robot_mask_overlay(
                    overlay,
                    manipulator_mask if np.count_nonzero(manipulator_mask) > 0 else robot_mask,
                    color_bgr=(0, 255, 255),
                    thickness=2,
                    min_area_px=cfg.robot_mask_overlay_min_area_px,
                )
            if cfg.show_wrist_joint_overlay and not cfg.use_manipulator_only_mode:
                if use_fk and wrist_roi_uv is not None:
                    # ---- FK-guided 2D ROI (independent of gripper) ----
                    if cfg.use_robot_self_filter and np.count_nonzero(robot_mask) > 120:
                        wrist_contour, _wrist_mask = detect_wrist_with_robot_mask(
                            color_bgr, depth, depth_scale,
                            cfg.depth_min_m, cfg.depth_max_m,
                            roi_center_uv=wrist_roi_uv,
                            roi_radius=cfg.fk_wrist_roi_radius_px,
                            robot_mask=manipulator_mask if np.count_nonzero(manipulator_mask) > 0 else robot_mask,
                            wrist_gray_min=cfg.wrist_gray_min,
                            min_area_px=cfg.wrist_min_area_px,
                            max_area_px=cfg.wrist_max_area_px,
                            wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                            wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                            prev_wrist_bbox=(
                                tuple(int(round(v)) for v in wrist_bbox_ema)
                                if wrist_bbox_ema is not None else None
                            ),
                            track_max_jump_px=cfg.gripper_track_max_jump_px,
                            track_iou_weight=cfg.gripper_track_iou_weight,
                            track_center_weight=cfg.gripper_track_center_weight,
                        )
                        if wrist_contour is None:
                            wrist_contour, _wrist_mask = detect_wrist_in_roi(
                                color_bgr, depth, depth_scale,
                                cfg.depth_min_m, cfg.depth_max_m,
                                roi_center_uv=wrist_roi_uv,
                                roi_radius=cfg.fk_wrist_roi_radius_px,
                                wrist_gray_min=cfg.wrist_gray_min,
                                min_area_px=cfg.wrist_min_area_px,
                                max_area_px=cfg.wrist_max_area_px,
                                wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                                wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                                prev_wrist_bbox=(
                                    tuple(int(round(v)) for v in wrist_bbox_ema)
                                    if wrist_bbox_ema is not None else None
                                ),
                                track_max_jump_px=cfg.gripper_track_max_jump_px,
                                track_iou_weight=cfg.gripper_track_iou_weight,
                                track_center_weight=cfg.gripper_track_center_weight,
                            )
                    else:
                        wrist_contour, _wrist_mask = detect_wrist_in_roi(
                            color_bgr, depth, depth_scale,
                            cfg.depth_min_m, cfg.depth_max_m,
                            roi_center_uv=wrist_roi_uv,
                            roi_radius=cfg.fk_wrist_roi_radius_px,
                            wrist_gray_min=cfg.wrist_gray_min,
                            min_area_px=cfg.wrist_min_area_px,
                            max_area_px=cfg.wrist_max_area_px,
                            wrist_valid_dilate_px=cfg.wrist_valid_dilate_px,
                            wrist_valid_dilate_iters=cfg.wrist_valid_dilate_iters,
                            prev_wrist_bbox=(
                                tuple(int(round(v)) for v in wrist_bbox_ema)
                                if wrist_bbox_ema is not None else None
                            ),
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
                        if wrist_bbox_ema is None:
                            wrist_bbox_ema = v.copy()
                        else:
                            wrist_bbox_ema = a * v + (1.0 - a) * wrist_bbox_ema
                        wrist_miss_frames = 0
                    elif wrist_bbox_ema is not None:
                        wrist_miss_frames += 1
                        if wrist_miss_frames > cfg.wrist_hold_miss_frames:
                            wrist_bbox_ema = None
                            wrist_miss_frames = 0
                    if wrist_bbox_ema is not None:
                        xi, yi, wi, hi = [int(round(t)) for t in wrist_bbox_ema]
                        draw_wrist_zone_rect(overlay, (xi, yi, wi, hi))
                elif wrist_contour is not None:
                    draw_wrist_overlay(overlay, wrist_contour)

            if joint_angles:
                y = 18
                cv2.putText(
                    overlay,
                    f"A1={joint_angles[0]:.1f} A2={joint_angles[1]:.1f} A3={joint_angles[2]:.1f}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )
                y += 30
                cv2.putText(
                    overlay,
                    f"A4={joint_angles[3]:.1f} A5={joint_angles[4]:.1f} A6={joint_angles[5]:.1f}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )
                y += 42

            # Draw FK projections if available — фиолетовый скелет p_cam
            if fk_uvs is not None:
                draw_fk_projections(overlay, fk_uvs,
                                    names=["Base", "J1", "J2", "J3", "J4", "J5(W)", "J6(G)", "TCP"],
                                    color_bgr=(200, 0, 255),
                                    draw_skeleton=True)
            if grip_roi_uv is not None:
                draw_roi_circle(overlay, grip_roi_uv, cfg.fk_grip_roi_radius_px,
                                color_bgr=(0, 255, 255), label="Grip ROI")
            if wrist_roi_uv is not None:
                draw_roi_circle(overlay, wrist_roi_uv, cfg.fk_wrist_roi_radius_px,
                                color_bgr=(255, 0, 255), label="Wrist ROI")
            # Mode indicator
            mode_txt = "FK-ROI" if use_fk else "legacy"
            cv2.putText(overlay, f"mode: {mode_txt}", (overlay.shape[1] - 130, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            y_status = 80 if joint_angles else 28
            cv2.putText(
                overlay,
                f"points={len(points)} (raw)",
                (10, y_status),
                cv2.FONT_HERSHEY_SIMPLEX,
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
                    cv2.FONT_HERSHEY_SIMPLEX,
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
                n_obj = len(scene_objects)
                part_txt = collision_worst_focus.part if collision_worst_focus else "--"
                if collision_worst_focus is not None:
                    dist_txt = f"{collision_worst_focus.min_dist_m:.2f}m"
                else:
                    dist_txt = "--"
                coll_color = collision_mod.LEVEL_COLORS_BGR.get(
                    collision_worst_level, (200, 200, 200)
                )
                cv2.putText(
                    overlay,
                    f"COLLISION: {collision_worst_level} ({part_txt} {dist_txt}) objs={n_obj}",
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    coll_color,
                    1,
                )
                if cfg.show_collision_overlay:
                    # Серое кольцо-зона рисуется только при включённом debug — это
                    # повторный дорогой dilate, не нужный для обычной работы.
                    if (cfg.show_debug_masks
                            and getattr(cfg, "collision_obj_use_near_manip", True)
                            and np.count_nonzero(manipulator_mask) > 0):
                        ring_px = int(getattr(cfg, "collision_obj_near_manip_dilate_px", 90))
                        near_ring = collision_mod.build_near_manip_ring_mask(
                            manipulator_mask, ring_px
                        )
                        contours, _ = cv2.findContours(
                            near_ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(overlay, contours, -1, (120, 120, 120), 1)
                    for obj in scene_objects:
                        res = next(
                            (r for r in collision_results if r.obj_id == obj.obj_id), None
                        )
                        level = res.level if res is not None else "SAFE"
                        part = res.part if res is not None else "--"
                        dist_m = res.min_dist_m if res is not None else float("inf")
                        rect = collision_mod.project_aabb_to_image(
                            obj.aabb_min, obj.aabb_max, intrinsics, color_bgr.shape[:2]
                        )
                        if rect is None:
                            continue
                        label = f"#{obj.obj_id} {part} {dist_m:.2f}m"
                        collision_mod.draw_object_box(overlay, rect, level, label=label)
            _ck_draw = time.perf_counter()  # ── timing: gripper+overlay drawing done ──
            cv2.imshow("RGB", overlay)

            if cfg.show_debug_masks:
                from gripper_detection import _gray_valid_near_arm, _valid_relaxed_mask
                _dbg_gray, _dbg_valid, _dbg_near_arm, _dbg_arm_cx, _dbg_arm_mask = _gray_valid_near_arm(
                    color_bgr, depth, depth_scale,
                    cfg.depth_min_m, cfg.depth_max_m,
                    cfg.arm_gray_min, cfg.near_arm_dilate_px,
                )
                _dbg_valid_soft = _valid_relaxed_mask(_dbg_valid, cfg.wrist_valid_dilate_px, cfg.wrist_valid_dilate_iters)
                _grip_cand = gripper_debug.get(
                    "gripper_cand", np.zeros_like(manipulator_mask)
                )
                mosaic = build_debug_mosaic({
                    "manipulator": manipulator_mask,
                    "gripper_cand": _grip_cand,
                    "obj_mask": scene_obj_mask_2d,
                    "scene_objects": scene_objects_mask,
                    "table_candidate": table_candidate if cfg.show_table_candidate_debug else np.zeros_like(manipulator_mask),
                    "valid": _dbg_valid.astype(np.uint8) * 255,
                })
                cv2.imshow("Debug masks", mosaic)

            # ── timing probe: накопить и печатать средние раз в N кадров ──
            _t_total = time.perf_counter() - _t0
            _t_acc["grab"] += _t_grab
            _t_acc["cloud"] += _t_cloud
            _t_acc["robot"] += _t_robot
            _t_acc["o3d"] += _t_o3d
            _t_acc["masks"] += (_ck_masks - _ck_a)
            _t_acc["scene"] += (_ck_scene - _ck_masks)
            _t_acc["draw"] += (_ck_draw - _ck_scene)
            _t_acc["total"] += _t_total
            if frame_count % _t_report_every == 0:
                n = _t_report_every
                fps = 1.0 / (_t_acc["total"] / n) if _t_acc["total"] > 0 else 0.0
                print(
                    f"[TIMING avg/{n}f] "
                    f"grab={_t_acc['grab']/n*1e3:5.1f}  "
                    f"cloud={_t_acc['cloud']/n*1e3:5.1f}  "
                    f"robot={_t_acc['robot']/n*1e3:5.1f}  "
                    f"o3d={_t_acc['o3d']/n*1e3:5.1f}  "
                    f"masks={_t_acc['masks']/n*1e3:5.1f}  "
                    f"scene={_t_acc['scene']/n*1e3:5.1f}  "
                    f"draw={_t_acc['draw']/n*1e3:5.1f}  "
                    f"total={_t_acc['total']/n*1e3:5.1f} ms  "
                    f"({fps:.1f} FPS)"
                )
                for k in _t_acc:
                    _t_acc[k] = 0.0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_full_cloud(cfg.output_dir, points, colors)
                print(f"Saved: {path}")
    finally:
        camera.stop()
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()
        if robot:
            try:
                robot.sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
