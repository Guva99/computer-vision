"""
FK-сервис манипулятора: проекция forward-kinematics KUKA в кадр камеры.

Инкапсулирует:
 • загрузку extrinsic-матрицы T_cr (camera-from-robot);
 • проекцию суставов base+J1..J6(+TCP) в пиксели и ROI-центры хвата/запястья;
 • глубины суставов в кадре камеры (для self-filter маски робота);
 • построение коллизионных сфер звеньев;
 • толщину «капсульного» FK-скелета в пикселях.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import collision as collision_mod
from kuka_fk import fk_joints, joints_to_uvs
from realsense_io import AppConfig


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


@dataclass
class FkFrame:
    """Результат FK-проекции на один кадр."""
    use_fk: bool = False
    fk_uvs: Optional[list] = None          # list[7..8]: base + J1..J6 (+ TCP)
    grip_roi_uv: Optional[tuple] = None    # ROI center for A6 (flange/TCP)
    wrist_roi_uv: Optional[tuple] = None   # ROI center for A5 (wrist)
    front_view_mode: bool = False
    joint_depths_m: Optional[list] = None  # глубины суставов в кадре камеры (м)


class FkProjector:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        # Load extrinsic calibration for FK-guided ROI
        self.T_cr: Optional[np.ndarray] = None
        if cfg.use_fk_roi:
            self.T_cr = load_extrinsic(cfg.extrinsic_path)
            if self.T_cr is None:
                print("[INFO] Falling back to legacy colour-only detection.")

    def base_use_fk(self, joint_angles) -> bool:
        cfg = self.cfg
        return (cfg.use_fk_roi and self.T_cr is not None and joint_angles is not None
                and len(joint_angles) >= 6)

    def project(self, joint_angles, intrinsics, color_shape) -> FkFrame:
        """Спроецировать FK-скелет в кадр и вычислить ROI-центры и глубины суставов."""
        cfg = self.cfg
        T_cr = self.T_cr
        h_img, w_img = color_shape[:2]
        fk_uvs = joints_to_uvs(joint_angles, T_cr, intrinsics,
                               gripper_length_m=cfg.fk_gripper_length_m)
        # fk_uvs[6] = J6 flange, fk_uvs[7] = gripper tip (если fk_gripper_length_m>0)
        # grip_roi_uv = последняя точка (tip если есть, иначе flange)
        _grip_idx = 7 if (cfg.fk_gripper_length_m > 0.0 and len(fk_uvs) > 7) else 6
        grip_roi_uv = fk_uvs[_grip_idx] if _uv_in_frame(fk_uvs[_grip_idx], w_img, h_img) else None
        wrist_roi_uv = fk_uvs[5] if _uv_in_frame(fk_uvs[5], w_img, h_img) else None
        front_view_mode = False
        if grip_roi_uv is not None:
            gu, _gv = grip_roi_uv
            front_view_mode = abs(float(gu) - 0.5 * float(w_img)) < (0.22 * float(w_img))
        joint_depths_m = None
        if cfg.use_robot_self_filter and fk_uvs is not None:
            fk_pos_robot = fk_joints(joint_angles, gripper_length_m=cfg.fk_gripper_length_m)
            joint_depths_m = []
            for p in fk_pos_robot:
                p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
                p_cam = T_cr @ p_h
                joint_depths_m.append(float(p_cam[2]) if p_cam[2] > 0.0 else None)
        return FkFrame(
            use_fk=True,
            fk_uvs=fk_uvs,
            grip_roi_uv=grip_roi_uv,
            wrist_roi_uv=wrist_roi_uv,
            front_view_mode=front_view_mode,
            joint_depths_m=joint_depths_m,
        )

    def build_spheres(self, joint_angles):
        cfg = self.cfg
        return collision_mod.build_link_spheres(
            joint_angles,
            self.T_cr,
            cfg.collision_link_radii_m,
            cfg.collision_spheres_per_link,
            gripper_length_m=cfg.fk_gripper_length_m,
            gripper_radius_m=cfg.collision_gripper_radius_m,
        )

    def link_thickness_px(self, joint_angles, intrinsics) -> List[float]:
        """Толщина каждого звена в пикселях = проекция радиуса звена на глубине:
           radius_px = fx * radius_m / z_cam  → объёмный «капсульный» скелет.
        """
        cfg = self.cfg
        fk_pos_robot = fk_joints(joint_angles, gripper_length_m=cfg.fk_gripper_length_m)
        radii = cfg.collision_link_radii_m
        fx = float(intrinsics["fx"])
        link_thickness_px: List[float] = []
        for idx, p in enumerate(fk_pos_robot):
            p_cam = self.T_cr @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
            z = float(p_cam[2])
            r_m = radii[min(idx, len(radii) - 1)] if len(radii) > 0 else 0.05
            if z > 1e-3:
                link_thickness_px.append(2.0 * fx * r_m / z * cfg.fk_skeleton_thick_scale)
            else:
                link_thickness_px.append(4.0)
        return link_thickness_px
