"""
Forward kinematics for KUKA KR 4 R600 (Agilus).
DH parameters derived from the KR4 R600 dimensional drawing.

We use the **standard DH** convention:
    T_i = Rot_z(theta_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(alpha_i)

KR 4 R600 DH parameters (home position = all angles 0°):
  Joint |   a (mm) |   d (mm) |  alpha (deg) |  theta_offset (deg)
  ------+----------+----------+--------------+--------------------
    1   |    0     |  330     |  -90         |   0
    2   |  290     |    0     |    0         |   0
    3   |   20     |    0     |  -90         |  -90   (mechanical offset)
    4   |    0     |  310     |   90         |  180   (mechanical offset)
    5   |    0     |    0     |  -90         |   0
    6   |    0     |   75     |    0         |   0

  Axis inversion from machine.dat ($AXIS_DIR):
  Axes 1, 4, 5, 6 are inverted (multiplied by -1).

  a2=290 мм (плечо), d4=310 мм (предплечье) — из чертежа KR4 R600.
  a2+d4 = 600 мм ≈ рабочий радиус R600.

All lengths in millimetres; we convert to metres internally.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Инверсия осей из machine.dat: $AXIS_DIR[i] = -1 для осей 1, 4, 5, 6
_AXIS_DIR = np.array([-1, 1, 1, -1, -1, -1], dtype=np.float64)

# ---------------------------------------------------------------------------
# DH table  (a_mm, d_mm, alpha_deg, theta_offset_deg)
# ---------------------------------------------------------------------------
_DH = np.array(
    [
        #   a      d      alpha   theta_offset
        [   0.0,  330.0,  -90.0,    0.0],   # J1
        [ 290.0,    0.0,    0.0,    0.0],   # J2
        [  20.0,    0.0,  -90.0,  -90.0],   # J3
        [   0.0,  310.0,   90.0,  180.0],   # J4
        [   0.0,    0.0,  -90.0,    0.0],   # J5
        [   0.0,   75.0,    0.0,    0.0],   # J6
    ],
    dtype=np.float64,
)
# Convert mm → m
_DH[:, :2] /= 1000.0


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Standard DH homogeneous transformation matrix (4×4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct,  -st * ca,  st * sa,  a * ct],
            [st,   ct * ca, -ct * sa,  a * st],
            [0.0,  sa,       ca,       d     ],
            [0.0,  0.0,      0.0,      1.0   ],
        ],
        dtype=np.float64,
    )


def fk_joints(
    angles_deg: Tuple[float, ...],
    gripper_length_m: float = 0.0,
) -> List[np.ndarray]:
    """
    Compute 3-D positions of the robot base + 6 joints in the robot base frame.

    Parameters
    ----------
    angles_deg : sequence of 6 floats
        Joint angles A1..A6 in degrees (as read from KUKA controller).
    gripper_length_m : float
        Length of the gripper/tool from the flange along the tool Z-axis (metres).
        0.0 = only flange. Positive = gripper tip added as 8th point.

    Returns
    -------
    List of 7 (or 8) np.ndarray shape (3,) in metres:
        [base_origin, J1, J2, J3, J4, J5, J6/flange, (gripper_tip if >0)]
    """
    if len(angles_deg) < 6:
        raise ValueError(f"Need 6 joint angles, got {len(angles_deg)}")

    T = np.eye(4, dtype=np.float64)
    positions = [T[:3, 3].copy()]  # base origin

    for i in range(6):
        a, d, alpha_deg, theta_off_deg = _DH[i]
        theta = np.deg2rad(_AXIS_DIR[i] * angles_deg[i] + theta_off_deg)
        alpha = np.deg2rad(alpha_deg)
        T = T @ _dh_matrix(a, d, alpha, theta)
        positions.append(T[:3, 3].copy())

    # Optionally append gripper tip (TCP offset along flange Z-axis)
    if gripper_length_m > 0.0:
        tip_h = T @ np.array([0.0, 0.0, gripper_length_m, 1.0])
        positions.append(tip_h[:3].copy())

    return positions  # len == 7 or 8


def fk_tcp(
    angles_deg: Tuple[float, ...],
    tcp_offset_m: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (position_3d, orientation_4x4) of the TCP in the robot base frame.

    Parameters
    ----------
    tcp_offset_m : (3,) array — tool-centre-point offset in flange frame (metres).
                   None means flange origin.
    """
    positions = fk_joints(angles_deg)
    # Recompute full transform to get orientation
    T = np.eye(4, dtype=np.float64)
    for i in range(6):
        a, d, alpha_deg, theta_off_deg = _DH[i]
        theta = np.deg2rad(_AXIS_DIR[i] * angles_deg[i] + theta_off_deg)
        alpha = np.deg2rad(alpha_deg)
        T = T @ _dh_matrix(a, d, alpha, theta)

    if tcp_offset_m is not None:
        p = T @ np.array([*tcp_offset_m, 1.0])
        return p[:3], T
    return T[:3, 3].copy(), T


def project_to_image(
    p3d_robot: np.ndarray,
    T_cr: np.ndarray,
    intrinsics: Dict[str, float],
) -> Optional[Tuple[int, int]]:
    """
    Project a 3-D point in the robot base frame onto the image plane.

    Parameters
    ----------
    p3d_robot : (3,) array — point in robot base frame (metres).
    T_cr : (4,4) array — extrinsic matrix camera-from-robot
                         (transforms robot-frame points into camera frame).
    intrinsics : dict with keys 'fx', 'fy', 'cx', 'cy' (pixels).

    Returns
    -------
    (u, v) pixel coordinates, or None if point is behind camera.
    """
    p_h = np.array([p3d_robot[0], p3d_robot[1], p3d_robot[2], 1.0])
    p_cam = T_cr @ p_h  # (4,)
    zc = p_cam[2]
    if zc <= 1e-4:
        return None
    u = intrinsics["fx"] * p_cam[0] / zc + intrinsics["cx"]
    v = intrinsics["fy"] * p_cam[1] / zc + intrinsics["cy"]
    return int(round(u)), int(round(v))


def joints_to_uvs(
    angles_deg: Tuple[float, ...],
    T_cr: np.ndarray,
    intrinsics: Dict[str, float],
    gripper_length_m: float = 0.0,
) -> List[Optional[Tuple[int, int]]]:
    """
    Convenience: compute FK for all positions (base + 6 joints + optional gripper tip)
    and project each to image coordinates.

    Returns list of length 7 (or 8 if gripper_length_m > 0):
        [base_uv, J1_uv, …, J6_uv, (gripper_tip_uv)]
    Each entry is (u, v) or None if behind camera.
    """
    positions = fk_joints(angles_deg, gripper_length_m=gripper_length_m)
    return [project_to_image(p, T_cr, intrinsics) for p in positions]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    angles = (-59.884, -26.499, 120.533, -0.854, -90.858, 2.948)

    # TCP с инструментом
    tool_offset = np.array([0.0, 0.0, 0.13693])
    tcp_pos, tcp_T = fk_tcp(angles, tcp_offset_m=tool_offset)
    print(f"X = {tcp_pos[0] * 1000:.3f} mm")
    print(f"Y = {tcp_pos[1] * 1000:.3f} mm")
    print(f"Z = {tcp_pos[2] * 1000:.3f} mm")

    # Визуализация скелета
    joints = fk_joints(angles, gripper_length_m=0.13693)
    pts = np.array(joints) * 1000  # в мм

    labels = ["Base", "J1", "J2", "J3", "J4", "J5", "J6", "TCP"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="royalblue", linewidth=2, markersize=6)

    for i, lbl in enumerate(labels[:len(pts)]):
        ax.text(pts[i, 0], pts[i, 1], pts[i, 2], f"  {lbl}", fontsize=8)

    ax.scatter(*pts[0], color="green", s=80, zorder=5, label="Base")
    ax.scatter(*pts[-1], color="red", s=80, zorder=5, label="TCP")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("KUKA KR4 R600 — fk_joints skeleton")
    ax.legend()

    plt.tight_layout()
    plt.show()
