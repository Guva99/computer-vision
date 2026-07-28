"""
FK collision spheres vs scene objects (camera frame, metres).
Uses kuka_fk, open3d, numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d

from kuka_fk import fk_joints
from pointcloud_pipeline import filter_cloud, to_open3d_cloud

# Throttled debug counters
_dbg_call_count = 0
_dbg2d_call_count = 0


@dataclass
class LinkSphere:
    center: np.ndarray  # (3,) camera frame, metres
    radius: float
    part: str


@dataclass
class SceneObject:
    obj_id: int
    points: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    # Признаки протечки маски руки (заполняются в detect_scene_objects_2d):
    # блоб прилип к маске руки + разница его глубины с локальной глубиной руки.
    touches_arm: bool = False
    arm_depth_delta_m: float = float("inf")


@dataclass
class CollisionResult:
    obj_id: int
    min_dist_m: float
    level: str
    part: str


LEVEL_COLORS_BGR = {
    "SAFE": (0, 200, 0),
    "WARN": (0, 220, 255),
    "DANGER": (0, 0, 255),
}

LEVEL_ORDER = {"SAFE": 0, "WARN": 1, "DANGER": 2}


def _link_part_name(link_index: int) -> str:
    if link_index == 5:
        return "gripper"
    if link_index == 4:
        return "wrist"
    return f"link{link_index + 1}"


def build_link_spheres(
    joint_angles: Sequence[float],
    T_cr: np.ndarray,
    link_radii_m: Sequence[float],
    spheres_per_link: int = 8,
    gripper_length_m: float = 0.0,
    gripper_radius_m: float = 0.05,
) -> List[LinkSphere]:
    """Collision spheres along DH links in camera frame.

    При gripper_length_m>0 добавляется сегмент J6(фланец)→TCP(кончик хвата),
    помеченный 'gripper' — чтобы пальцы хвата были покрыты сферами и контакт
    с объектом читался корректно (а не как расстояние до фланца).
    """
    positions_robot = fk_joints(tuple(joint_angles[:6]), gripper_length_m=gripper_length_m)
    spheres: List[LinkSphere] = []
    n_per = max(2, int(spheres_per_link))

    # Сегменты: 6 звеньев робота + (опц.) хват J6→TCP
    segments = []  # (i0, i1, radius_m, part)
    for link_i in range(6):
        r = float(link_radii_m[link_i]) if link_i < len(link_radii_m) else 0.08
        segments.append((link_i, link_i + 1, r, _link_part_name(link_i)))
    if gripper_length_m > 0.0 and len(positions_robot) > 7:
        segments.append((6, 7, float(gripper_radius_m), "gripper"))

    for i0, i1, radius, part in segments:
        p0 = np.asarray(positions_robot[i0], dtype=np.float64)
        p1 = np.asarray(positions_robot[i1], dtype=np.float64)
        h0 = np.array([p0[0], p0[1], p0[2], 1.0], dtype=np.float64)
        h1 = np.array([p1[0], p1[1], p1[2], 1.0], dtype=np.float64)
        c0 = (T_cr @ h0)[:3]
        c1 = (T_cr @ h1)[:3]
        if c0[2] <= 1e-4 and c1[2] <= 1e-4:
            continue
        for j in range(n_per):
            t = j / (n_per - 1) if n_per > 1 else 0.5
            center = c0 + t * (c1 - c0)
            if center[2] <= 1e-4:
                continue
            spheres.append(LinkSphere(center=center.astype(np.float64), radius=radius, part=part))
    return spheres


def _filter_scene_by_depth(
    pts: np.ndarray,
    col: np.ndarray,
    depth_pct_table: float = 85.0,
    min_height_m: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep points closer to the camera than the estimated table/background depth.
    Avoids RANSAC picking the frame wall as the dominant plane.
    """
    if len(pts) == 0:
        return pts, col
    z = pts[:, 2]
    z_table = float(np.percentile(z, float(depth_pct_table)))
    keep = z < (z_table - float(min_height_m))
    return pts[keep], col[keep]


def fit_table_plane(
    points: np.ndarray,
    voxel_m: float = 0.02,
    dist_thresh_m: float = 0.008,
    min_inliers: int = 50,
) -> Optional[np.ndarray]:
    """RANSAC-плоскость стола в кадре камеры → (a,b,c,d), |n|=1.

    Стол — доминирующая плоскость сцены; RANSAC её и находит. Нормаль
    ориентируется так, чтобы объекты (ближе к камере, меньший z) имели
    ПОЛОЖИТЕЛЬНУЮ высоту над плоскостью: height(p)=a·x+b·y+c·z+d > 0.
    Возвращает None, если плоскость не найдена или это, вероятно, стена
    (нормаль почти перпендикулярна лучу зрения, |c| мало).
    """
    if len(points) < min_inliers:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud = cloud.voxel_down_sample(voxel_size=float(voxel_m))
    if len(cloud.points) < 50:
        return None
    try:
        model, inliers = cloud.segment_plane(
            distance_threshold=float(dist_thresh_m),
            ransac_n=3,
            num_iterations=200,
        )
    except Exception:
        return None
    if len(inliers) < 50:
        return None
    a, b, c, d = (float(x) for x in model)
    n = (a * a + b * b + c * c) ** 0.5
    if n < 1e-9:
        return None
    a, b, c, d = a / n, b / n, c / n, d / n
    # Ориентация: объекты ближе к камере (меньший z) → нужен c<0, тогда
    # уменьшение z увеличивает height. Стол получает height≈0.
    if c > 0:
        a, b, c, d = -a, -b, -c, -d
    # Санити: стол смотрит примерно на камеру (|c| большое). Стена/профиль
    # ограждения (вертикаль) имеет |c|~0 — такую «плоскость» отвергаем.
    if abs(c) < 0.5:
        return None
    return np.array([a, b, c, d], dtype=np.float64)


def plane_height_2d(
    points: np.ndarray,
    valid_flat: np.ndarray,
    plane: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Карта высоты над плоскостью стола по пикселям (NaN вне valid), метры."""
    h, w = int(image_shape[0]), int(image_shape[1])
    height_flat = np.full(h * w, np.nan, dtype=np.float64)
    a, b, c, d = (float(x) for x in plane)
    hgt = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    height_flat[valid_flat] = hgt
    return height_flat.reshape(h, w)


# ── ADAPTIVE ε (ВАК §3, формулы 5–6) ──────────────────────────────────────
# CLAUDE: чтобы убрать — удали функцию _estimate_adaptive_eps() ниже
#   и блок "# [ADAPTIVE ε]" внутри detect_scene_objects().
#   Флаг в AppConfig: collision_dbscan_adaptive_eps (realsense_io.py).
def _estimate_adaptive_eps(
    cloud: "o3d.geometry.PointCloud",
    k: int,
    kappa: float,
    eps_min: float,
    eps_max: float,
) -> float:
    """
    Оценка адаптивного ε для DBSCAN по k-distance графику (Ester et al., 1996).

    Физическое обоснование: шаг между соседними точками RealSense растёт
    линейно с дальностью (Δ ≈ z / f_x), поэтому фиксированный ε приводит
    к избыточному дроблению близких кластеров и потере дальних.
    Оценка по медиане расстояний до k-го соседа даёт ε, адаптированный
    к текущей плотности облака кадра.

    Формула (5):  ε = κ · median_p { ||p − p^(k)||₂ }
    Формула (6):  ε ← clip(ε, ε_min, ε_max)
    """
    pts_arr = np.asarray(cloud.points)
    n = len(pts_arr)
    if n < k + 1:
        return float(np.clip(0.035, eps_min, eps_max))

    tree = o3d.geometry.KDTreeFlann(cloud)
    k_dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        # search_knn возвращает (count, indices, squared_distances)
        _, _, sq_dists = tree.search_knn_vector_3d(cloud.points[i], k + 1)
        # sq_dists[0] — сама точка (расстояние 0), sq_dists[k] — k-й сосед
        k_dists[i] = float(sq_dists[k]) ** 0.5

    eps = float(kappa * np.median(k_dists))
    eps = float(np.clip(eps, eps_min, eps_max))
    return eps
# ── END ADAPTIVE ε ─────────────────────────────────────────────────────────


def detect_scene_objects(
    scene_pts: np.ndarray,
    scene_col: np.ndarray,
    cfg,
) -> List[SceneObject]:
    """Voxel downsample -> depth filter (no RANSAC) -> DBSCAN clusters."""
    global _dbg_call_count
    n_scene = len(scene_pts)

    if len(scene_pts) < max(cfg.collision_object_min_points, cfg.collision_dbscan_min_points):
        _dbg_call_count += 1
        if _dbg_call_count % 60 == 1:
            print(f"[COLLISION DBG] scene_pts={n_scene} -> too few, skip")
        return []

    pts, col = filter_cloud(
        scene_pts,
        scene_col,
        voxel_size=float(cfg.collision_voxel_m),
        nb_neighbors=20,
        std_ratio=2.0,
        use_statistical_outlier=False,
    )
    n_voxel = len(pts)
    if len(pts) < cfg.collision_object_min_points:
        _dbg_call_count += 1
        if _dbg_call_count % 60 == 1:
            print(f"[COLLISION DBG] scene_pts={n_scene}, after_voxel={n_voxel} -> too few, skip")
        return []

    depth_pct = float(getattr(cfg, "collision_depth_table_pct", 85.0))
    min_h = float(getattr(cfg, "collision_depth_table_min_height_m", 0.02))
    pts, col = _filter_scene_by_depth(pts, col, depth_pct_table=depth_pct, min_height_m=min_h)
    n_depth = len(pts)
    if len(pts) < cfg.collision_dbscan_min_points:
        _dbg_call_count += 1
        if _dbg_call_count % 60 == 1:
            print(
                f"[COLLISION DBG] scene_pts={n_scene}, after_voxel={n_voxel}, "
                f"after_depth_filter={n_depth} -> too few, skip"
            )
        return []

    cloud = to_open3d_cloud(pts, col)

    # [ADAPTIVE ε] ─────────────────────────────────────────────────────────────
    # CLAUDE: удалить этот if-блок чтобы вернуть фиксированный ε
    if bool(getattr(cfg, "collision_dbscan_adaptive_eps", False)):
        _k     = int(getattr(cfg, "collision_dbscan_adaptive_k", 10))
        _kappa = float(getattr(cfg, "collision_dbscan_adaptive_kappa", 1.2))
        _emin  = float(getattr(cfg, "collision_dbscan_adaptive_eps_min", 0.020))
        _emax  = float(getattr(cfg, "collision_dbscan_adaptive_eps_max", 0.060))
        _eps = _estimate_adaptive_eps(cloud, _k, _kappa, _emin, _emax)
        if _dbg_call_count % 60 == 1:
            print(f"[COLLISION DBG] adaptive_eps={_eps:.4f} m")
    else:
        _eps = float(cfg.collision_dbscan_eps_m)
    # [END ADAPTIVE ε] ──────────────────────────────────────────────────────────

    labels = np.array(
        cloud.cluster_dbscan(
            eps=_eps,
            min_points=int(cfg.collision_dbscan_min_points),
            print_progress=False,
        )
    )
    if labels.size == 0:
        _dbg_call_count += 1
        if _dbg_call_count % 60 == 1:
            print(
                f"[COLLISION DBG] scene_pts={n_scene}, after_voxel={n_voxel}, "
                f"after_depth_filter={n_depth}, clusters=0"
            )
        return []

    objects: List[SceneObject] = []
    obj_id = 0
    max_extent = float(cfg.collision_object_max_extent_m)
    min_pts = int(cfg.collision_object_min_points)

    for label in np.unique(labels):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if idx.size < min_pts:
            continue
        obj_pts = pts[idx]
        obj_col = col[idx]
        aabb_min = obj_pts.min(axis=0)
        aabb_max = obj_pts.max(axis=0)
        extent = float(np.max(aabb_max - aabb_min))
        if extent > max_extent:
            continue
        objects.append(
            SceneObject(
                obj_id=obj_id,
                points=obj_pts,
                colors=obj_col,
                centroid=obj_pts.mean(axis=0),
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
        )
        obj_id += 1

    _dbg_call_count += 1
    if _dbg_call_count % 60 == 1:
        n_labels = len([l for l in np.unique(labels) if l >= 0])
        print(
            f"[COLLISION DBG] scene_pts={n_scene}, after_voxel={n_voxel}, "
            f"after_depth_filter={n_depth}, dbscan_labels={n_labels}, objects={len(objects)}"
        )
    return objects


def build_gripper_capsule_mask(
    spheres: List[LinkSphere],
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
    pad_px: int = 8,
    parts: Sequence[str] = ("gripper",),
) -> np.ndarray:
    """FK-капсула хвата, растеризованная в кадр (uint8, 0/255).

    Тёмный металл лопатки не даёт валидной глубины, поэтому цвет/глубинная
    маска руки его пропускает — и его блики/винты рождают ложные «объекты»
    вплотную к руке (ложный DANGER). Геометрии FK глубина не нужна: сферы
    сегмента J6→TCP проецируются в кадр кругами радиусом fx·r/z (+pad_px
    запас на погрешность калибровки), соседние круги соединяются линией.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    prev_uv: Optional[Tuple[int, int]] = None
    prev_r = 0
    for sp in spheres:
        if sp.part not in parts:
            continue
        z = float(sp.center[2])
        if z <= 1e-4:
            prev_uv = None
            continue
        u = int(round(fx * float(sp.center[0]) / z + cx))
        v = int(round(fy * float(sp.center[1]) / z + cy))
        r_px = int(round(fx * float(sp.radius) / z)) + int(pad_px)
        if r_px <= 0:
            prev_uv = None
            continue
        cv2.circle(mask, (u, v), r_px, 255, -1)
        if prev_uv is not None:
            cv2.line(mask, prev_uv, (u, v), 255, thickness=max(1, min(r_px, prev_r) * 2))
        prev_uv = (u, v)
        prev_r = r_px
    return mask


def build_near_manip_ring_mask(
    manipulator_mask: np.ndarray,
    dilate_px: int,
) -> np.ndarray:
    """Ring around manipulator: dilate(manip) minus manip (search zone for foreign objects).

    Большой dilate_px (кольцо ~130px) даёт ядро 261×261 — дорого на полном кадре.
    Кольцо грубое, поэтому дилатацию считаем на уменьшенной в `scale` раз маске
    (NEAREST), это даёт тот же результат на порядок быстрее.
    """
    h, w = manipulator_mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if dilate_px <= 0 or np.count_nonzero(manipulator_mask) == 0:
        return out

    scale = 4 if dilate_px >= 16 else 1
    if scale > 1:
        sw, sh = w // scale, h // scale
        small = cv2.resize(manipulator_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        dpx = max(1, dilate_px // scale)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dpx + 1, 2 * dpx + 1))
        near_small = cv2.dilate(small, k, iterations=1)
        near = cv2.resize(near_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        near = cv2.dilate(manipulator_mask, k, iterations=1)

    ring = (near > 0) & (manipulator_mask == 0)
    out[ring] = 255
    return out


def detect_scene_objects_2d(
    points: np.ndarray,
    colors: np.ndarray,
    valid_flat: np.ndarray,
    manipulator_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cfg,
    color_bgr: Optional[np.ndarray] = None,
    reject_log: Optional[list] = None,
    table_plane: Optional[np.ndarray] = None,
) -> Tuple[List[SceneObject], np.ndarray]:
    """
    Detect scene objects from 2D valid-depth mask minus manipulator (like arm segmentation).
    Maps each component to 3D points via valid_flat for collision distance.

    When `color_bgr` is given and `cfg.collision_obj_color_gate` is True, the
    candidate mask is further restricted to *contrasting* pixels — either bright
    (white box) or colour-saturated (green cube) — so the dark, desaturated
    table surface is rejected instead of being detected as a foreign object.

    `reject_log` (Задача 8, анализ отказов): если передан список, в него
    добавляются записи об отбракованных кандидатах с причиной
    (`reject_reason`: area_min/area_max/shape_filter/frame_border/
    few_points_3d/depth_filter). На логику детекции не влияет.
    """
    global _dbg2d_call_count
    h, w = int(image_shape[0]), int(image_shape[1])
    obj_mask_u8 = np.zeros((h, w), dtype=np.uint8)

    if len(points) == 0 or valid_flat.size != h * w:
        _dbg2d_call_count += 1
        if _dbg2d_call_count % 60 == 1:
            print("[COLLISION DBG] scene_obj2d: invalid input, objects=0")
        return [], obj_mask_u8

    valid_2d = valid_flat.reshape(h, w)
    # Исключаем РАЗДУТУЮ маску руки: тонкий ореол краёв/деталей самой руки (белый
    # пластик KUKA) иначе проходит как «чужой объект» вплотную → ложный DANGER.
    manip_excl = manipulator_mask
    ex_px = int(getattr(cfg, "collision_obj_manip_exclude_dilate_px", 0))
    if ex_px > 0 and np.count_nonzero(manipulator_mask) > 0:
        k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ex_px + 1, 2 * ex_px + 1))
        manip_excl = cv2.dilate(manipulator_mask, k_ex, iterations=1)
    obj_mask = valid_2d & (manip_excl == 0)

    # ── Гейт кандидатов ─────────────────────────────────────────────────────
    # ОСНОВНОЙ критерий — ГЕОМЕТРИЯ: высота над плоскостью стола. Кубик от стола
    # отличает не цвет (дерево/тёмный не проходят цвет-гейт), а то, что он ВЫШЕ
    # стола на 1.5–30 см. Не зависит от цвета, освещения и калибровки T_cr.
    # Цветовой гейт остаётся запасным (когда плоскость не найдена).
    contrast: Optional[np.ndarray] = None
    plane_gate_on = (
        table_plane is not None
        and bool(getattr(cfg, "collision_obj_plane_gate", True))
    )
    if plane_gate_on:
        height2d = plane_height_2d(points, valid_flat, table_plane, (h, w))
        h_min = float(getattr(cfg, "collision_obj_plane_height_min_m", 0.015))
        h_max = float(getattr(cfg, "collision_obj_plane_height_max_m", 0.30))
        # NaN-сравнения дают False → пиксели без глубины отсекаются автоматически.
        plane_mask = (height2d >= h_min) & (height2d <= h_max)
        obj_mask = obj_mask & plane_mask
        contrast = plane_mask  # база для reclaim ниже
        # Синий пневмошланг руки (если пробился) убираем и здесь.
        if (
            color_bgr is not None
            and bool(getattr(cfg, "collision_obj_exclude_blue", True))
            and color_bgr.shape[:2] == (h, w)
        ):
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            hue, sat = hsv[:, :, 0], hsv[:, :, 1]
            hue_lo = int(getattr(cfg, "collision_obj_blue_hue_lo", 90))
            hue_hi = int(getattr(cfg, "collision_obj_blue_hue_hi", 140))
            is_blue = (hue >= hue_lo) & (hue <= hue_hi) & (sat >= 60)
            obj_mask = obj_mask & (~is_blue)
            contrast = contrast & (~is_blue)
    elif (
        color_bgr is not None
        and bool(getattr(cfg, "collision_obj_color_gate", True))
        and color_bgr.shape[:2] == (h, w)
    ):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        bright_min = int(getattr(cfg, "collision_obj_bright_min", 150))
        sat_min = int(getattr(cfg, "collision_obj_sat_min", 60))
        colored = sat >= sat_min
        # Drop the blue pneumatic hose: saturated-but-blue pixels are not objects.
        if bool(getattr(cfg, "collision_obj_exclude_blue", True)):
            hue_lo = int(getattr(cfg, "collision_obj_blue_hue_lo", 90))
            hue_hi = int(getattr(cfg, "collision_obj_blue_hue_hi", 140))
            is_blue = (hue >= hue_lo) & (hue <= hue_hi)
            colored = colored & (~is_blue)
        contrast = (gray >= bright_min) | colored
        obj_mask = obj_mask & contrast

    # Reclaim-база: те же фильтры, но вычитается ТОЛЬКО сырая маска руки (без
    # дилатации). Зона исключения нужна лишь для детекции (ореол руки ≠ объект);
    # для измерения дистанции найденной компоненте возвращаются её пиксели,
    # съеденные зоной, — иначе расстояние завышено на ширину зоны (~3 см) и
    # DANGER не наступает при подходе руки.
    reclaim_base: Optional[np.ndarray] = None
    if bool(getattr(cfg, "collision_obj_reclaim_near_arm", True)) and ex_px > 0:
        reclaim_base = valid_2d & (manipulator_mask == 0)
        if contrast is not None:
            reclaim_base = reclaim_base & contrast

    obj_mask_u8 = (obj_mask.astype(np.uint8) * 255)

    open_px = int(getattr(cfg, "collision_obj_open_px", 3))
    close_px = int(getattr(cfg, "collision_obj_close_px", 5))
    if open_px > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_px, open_px)
        )
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_OPEN, k_open)
    if close_px > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px, close_px)
        )
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_CLOSE, k_close)

    use_near = bool(getattr(cfg, "collision_obj_use_near_manip", True))
    if use_near and np.count_nonzero(manipulator_mask) > 0:
        ring_px = int(getattr(cfg, "collision_obj_near_manip_dilate_px", 130))
        near_ring = build_near_manip_ring_mask(manipulator_mask, ring_px)
        obj_mask_u8[near_ring == 0] = 0

    # Consolidate depth-holed fragments of one object into a solid blob.
    cons_px = int(getattr(cfg, "collision_obj_consolidate_px", 11))
    if cons_px > 1:
        k_cons = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cons_px, cons_px))
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_CLOSE, k_cons)

    z_arm: Optional[float] = None
    if np.count_nonzero(manipulator_mask) > 0:
        arm_sel = (manipulator_mask > 0).reshape(-1)[valid_flat]
        if np.any(arm_sel):
            z_arm = float(np.median(points[arm_sel][:, 2]))
    max_z_behind = float(getattr(cfg, "collision_obj_max_depth_behind_arm_m", 0.20))

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        obj_mask_u8, connectivity=8
    )
    min_area = int(getattr(cfg, "collision_obj_min_area_px", 150))
    max_area = int(float(getattr(cfg, "collision_obj_max_area_frac", 0.25)) * h * w)
    border_px = int(getattr(cfg, "collision_obj_exclude_border_px", 10))
    drop_border = bool(getattr(cfg, "collision_obj_drop_border", True))
    min_pts_3d = int(getattr(cfg, "collision_obj_min_points_3d", 15))
    min_fill = float(getattr(cfg, "collision_obj_min_fill_ratio", 0.32))
    max_aspect = float(getattr(cfg, "collision_obj_max_aspect", 4.5))

    # Карта глубины по пикселям (NaN вне valid) — для признаков протечки руки:
    # сравнение глубины блоба с локальной глубиной руки рядом с ним.
    z_img: Optional[np.ndarray] = None
    if bool(getattr(cfg, "collision_obj_arm_depth_reject", True)):
        z_flat = np.full(h * w, np.nan, dtype=np.float64)
        z_flat[valid_flat] = points[:, 2]
        z_img = z_flat.reshape(h, w)

    objects: List[SceneObject] = []
    obj_id = 0
    n_kept = 0

    def _reject(reason: str, area_px: int) -> None:
        # Сбор причин отбраковки (Задача 8); включается только когда
        # передан reject_log — иначе ноль накладных расходов.
        if reject_log is not None:
            reject_log.append({"reject_reason": reason, "area_px": area_px})

    for lab in range(1, n_comp):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            _reject("area_min" if area < min_area else "area_max", area)
            continue
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        # Shape filter: compact, roughly box-like blobs only (reject hose/noise).
        fill_ratio = float(area) / float(max(1, bw * bh))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        if fill_ratio < min_fill or aspect > max_aspect:
            _reject("shape_filter", area)
            continue
        if drop_border:
            if (
                x0 <= border_px
                or y0 <= border_px
                or (x0 + bw) >= (w - border_px)
                or (y0 + bh) >= (h - border_px)
            ):
                _reject("frame_border", area)
                continue

        comp = labels == lab
        # Reclaim: вернуть компоненте пиксели, съеденные зоной исключения руки
        # (дилатация компоненты на ex_px ∩ reclaim-база). Форма/площадь выше
        # проверялись по усечённой компоненте, а 3D-точки для дистанции берём
        # из полной — ближайший к руке край объекта снова участвует в метрике.
        if reclaim_base is not None:
            k_rc = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ex_px + 1, 2 * ex_px + 1)
            )
            comp_dil = cv2.dilate(comp.astype(np.uint8), k_rc) > 0
            comp = comp | (comp_dil & reclaim_base)
        # Erode the component before sampling 3D points: boundary pixels carry
        # mixed depth (object vs table/arm) and would corrupt the distance.
        erode_px = int(getattr(cfg, "collision_obj_erode_px", 3))
        comp_sample = comp
        if erode_px > 1:
            k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
            eroded = cv2.erode(comp.astype(np.uint8), k_er) > 0
            if np.count_nonzero(eroded.reshape(-1)[valid_flat]) >= min_pts_3d:
                comp_sample = eroded
        sel = comp_sample.reshape(-1)[valid_flat]
        if not np.any(sel):
            _reject("few_points_3d", area)
            continue
        obj_pts = points[sel]
        obj_col = colors[sel]
        if len(obj_pts) < min_pts_3d:
            _reject("few_points_3d", area)
            continue

        z = obj_pts[:, 2]
        z_lo = float(np.percentile(z, 2))
        z_hi = float(np.percentile(z, 98))
        keep_z = (z >= z_lo) & (z <= z_hi)
        if z_arm is not None and max_z_behind > 0:
            keep_z &= obj_pts[:, 2] < (z_arm + max_z_behind)
        if np.count_nonzero(keep_z) < min_pts_3d:
            _reject("depth_filter", area)
            continue
        obj_pts = obj_pts[keep_z]
        obj_col = obj_col[keep_z]

        # Признаки протечки руки: прилип ли блоб к маске руки и совпадает ли
        # его глубина с локальной глубиной руки рядом. Решение об отбраковке
        # принимает CollisionService при создании трека (не здесь), чтобы уже
        # подтверждённый объект не «исчезал» при подходе руки вплотную.
        touches_arm = False
        arm_depth_delta = float("inf")
        if np.count_nonzero(manipulator_mask) > 0:
            t_px = ex_px + 5
            k_t = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * t_px + 1, 2 * t_px + 1)
            )
            near_arm = (cv2.dilate(comp.astype(np.uint8), k_t) > 0) & (
                manipulator_mask > 0
            )
            touches_arm = bool(np.any(near_arm))
            if touches_arm and z_img is not None:
                z_vals = z_img[near_arm]
                z_vals = z_vals[np.isfinite(z_vals)]
                if z_vals.size > 0:
                    arm_depth_delta = abs(
                        float(np.median(obj_pts[:, 2])) - float(np.median(z_vals))
                    )

        aabb_min = obj_pts.min(axis=0)
        aabb_max = obj_pts.max(axis=0)
        objects.append(
            SceneObject(
                obj_id=obj_id,
                points=obj_pts,
                colors=obj_col,
                centroid=obj_pts.mean(axis=0),
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                touches_arm=touches_arm,
                arm_depth_delta_m=arm_depth_delta,
            )
        )
        obj_id += 1
        n_kept += 1

    _dbg2d_call_count += 1
    if _dbg2d_call_count % 60 == 1:
        print(
            f"[COLLISION DBG] scene_obj2d: components={max(0, n_comp - 1)}, "
            f"kept={n_kept}, objects={len(objects)}"
        )
    return objects, obj_mask_u8


def min_distance_to_robot(
    spheres: List[LinkSphere],
    obj_points: np.ndarray,
) -> Tuple[float, str, np.ndarray]:
    """Minimum distance from object point cloud to union of spheres."""
    if len(spheres) == 0 or len(obj_points) == 0:
        return float("inf"), "none", np.zeros(3, dtype=np.float64)

    best_dist = float("inf")
    best_part = "none"
    best_center = np.zeros(3, dtype=np.float64)

    for sp in spheres:
        diff = obj_points - sp.center.reshape(1, 3)
        dists = np.linalg.norm(diff, axis=1) - sp.radius
        d_min = float(np.min(dists))
        if d_min < best_dist:
            best_dist = d_min
            best_part = sp.part
            nearest_idx = int(np.argmin(dists))
            best_center = sp.center.copy()

    return max(0.0, best_dist), best_part, best_center


def _voxel_down(pts: np.ndarray, voxel_m: float) -> np.ndarray:
    """Voxel-downsample a point set (camera frame, metres) for fast distance."""
    if len(pts) == 0 or voxel_m <= 0.0:
        return pts
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    cloud = cloud.voxel_down_sample(voxel_size=float(voxel_m))
    return np.asarray(cloud.points)


def min_dist_point_to_cloud(
    obj_points: np.ndarray,
    body_points: np.ndarray,
    chunk: int = 512,
    percentile: float = 0.0,
) -> float:
    """
    Distance (metres) between an object point set and the robot-body point
    cloud. Both are measured in the camera frame, so this does NOT depend on the
    FK / extrinsic calibration. Chunked to bound memory.

    `percentile` > 0 returns a low percentile of the per-object-point nearest
    distances instead of the single closest pair — robust against a few noisy
    boundary pixels that would otherwise trigger a false DANGER.
    """
    if len(obj_points) == 0 or len(body_points) == 0:
        return float("inf")
    body = body_points[None, :, :]  # (1, K, 3)
    nearest = np.empty(len(obj_points), dtype=np.float64)
    for i in range(0, len(obj_points), chunk):
        ch = obj_points[i:i + chunk][:, None, :]  # (c, 1, 3)
        d = np.linalg.norm(ch - body, axis=2)      # (c, K)
        nearest[i:i + chunk] = d.min(axis=1)
    if percentile <= 0.0:
        return float(nearest.min())
    return float(np.percentile(nearest, percentile))


def evaluate_collisions_hybrid(
    body_points: np.ndarray,
    gripper_spheres: List[LinkSphere],
    objects: List[SceneObject],
    warn_m: float,
    danger_m: float,
    focus_parts: Sequence[str],
    body_voxel_m: float = 0.02,
    dist_percentile: float = 0.0,
) -> Tuple[List[CollisionResult], str, Optional[CollisionResult]]:
    """
    Hybrid collision: distance to the robot BODY is measured from the
    manipulator-mask point cloud (`body_points`, real depth → no T_cr
    dependence), while the GRIPPER — which has no valid depth (dark metal) — is
    represented by FK spheres (`gripper_spheres`). Per object the minimum of the
    two wins, and the part label comes from whichever source is closer.
    """
    body_ds = _voxel_down(body_points, body_voxel_m)

    results: List[CollisionResult] = []
    worst_level = "SAFE"
    worst_focus: Optional[CollisionResult] = None

    for obj in objects:
        d_body = min_dist_point_to_cloud(obj.points, body_ds, percentile=dist_percentile)
        d_grip, part_grip, _ = min_distance_to_robot(gripper_spheres, obj.points)

        if d_grip < d_body:
            dist_m, part = d_grip, part_grip
        else:
            dist_m, part = d_body, "arm"

        level = classify_level(dist_m, warn_m, danger_m)
        res = CollisionResult(obj_id=obj.obj_id, min_dist_m=dist_m, level=level, part=part)
        results.append(res)

        if LEVEL_ORDER[level] > LEVEL_ORDER[worst_level]:
            worst_level = level
        if part in focus_parts:
            if worst_focus is None or LEVEL_ORDER[level] > LEVEL_ORDER[worst_focus.level]:
                worst_focus = res
            elif (
                LEVEL_ORDER[level] == LEVEL_ORDER[worst_focus.level]
                and dist_m < worst_focus.min_dist_m
            ):
                worst_focus = res

    if worst_focus is None and results:
        worst_focus = min(results, key=lambda r: r.min_dist_m)
    return results, worst_level, worst_focus


def classify_level(dist_m: float, warn_m: float, danger_m: float) -> str:
    if dist_m <= danger_m:
        return "DANGER"
    if dist_m <= warn_m:
        return "WARN"
    return "SAFE"


def evaluate_collisions(
    spheres: List[LinkSphere],
    objects: List[SceneObject],
    warn_m: float,
    danger_m: float,
    focus_parts: Sequence[str],
) -> Tuple[List[CollisionResult], str, Optional[CollisionResult]]:
    """Per-object collision; return list, worst overall level, worst focus result."""
    results: List[CollisionResult] = []
    worst_level = "SAFE"
    worst_focus: Optional[CollisionResult] = None

    for obj in objects:
        dist_m, part, _ = min_distance_to_robot(spheres, obj.points)
        level = classify_level(dist_m, warn_m, danger_m)
        res = CollisionResult(obj_id=obj.obj_id, min_dist_m=dist_m, level=level, part=part)
        results.append(res)
        if LEVEL_ORDER[level] > LEVEL_ORDER[worst_level]:
            worst_level = level
        if part in focus_parts:
            if worst_focus is None or LEVEL_ORDER[level] > LEVEL_ORDER[worst_focus.level]:
                worst_focus = res
            elif (
                LEVEL_ORDER[level] == LEVEL_ORDER[worst_focus.level]
                and dist_m < worst_focus.min_dist_m
            ):
                worst_focus = res

    if worst_focus is None and results:
        worst_focus = min(results, key=lambda r: r.min_dist_m)
    return results, worst_level, worst_focus


def project_aabb_to_image(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Project AABB corners to image; return (x, y, w, h) or None."""
    h, w = image_shape[:2]
    corners = np.array(
        [
            [aabb_min[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_min[1], aabb_min[2]],
            [aabb_min[0], aabb_max[1], aabb_min[2]],
            [aabb_max[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_min[1], aabb_max[2]],
            [aabb_min[0], aabb_max[1], aabb_max[2]],
            [aabb_max[0], aabb_max[1], aabb_max[2]],
        ],
        dtype=np.float64,
    )
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    us, vs = [], []
    for p in corners:
        if p[2] <= 1e-4:
            continue
        us.append(fx * p[0] / p[2] + cx)
        vs.append(fy * p[1] / p[2] + cy)
    if not us:
        return None
    x0 = int(max(0, np.floor(min(us))))
    y0 = int(max(0, np.floor(min(vs))))
    x1 = int(min(w - 1, np.ceil(max(us))))
    y1 = int(min(h - 1, np.ceil(max(vs))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def draw_object_box(
    image_bgr: np.ndarray,
    rect: Tuple[int, int, int, int],
    level: str,
    label: str = "",
    thickness: int = 2,
) -> None:
    color = LEVEL_COLORS_BGR.get(level, (200, 200, 200))
    x, y, bw, bh = rect
    cv2.rectangle(image_bgr, (x, y), (x + bw, y + bh), color, thickness)
    if label:
        cv2.putText(
            image_bgr,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def make_o3d_aabb(aabb_min: np.ndarray, aabb_max: np.ndarray, level: str) -> o3d.geometry.LineSet:
    """Axis-aligned box as LineSet for Open3D."""
    aabb = o3d.geometry.AxisAlignedBoundingBox(aabb_min, aabb_max)
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    rgb = np.array(LEVEL_COLORS_BGR.get(level, (128, 128, 128)), dtype=np.float64) / 255.0
    rgb = rgb[::-1]  # BGR -> RGB
    ls.colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(ls.lines), 1)))
    return ls


def build_scene_objects_image_mask(
    objects: List[SceneObject],
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize object points into a binary mask for debug."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    for obj in objects:
        for p in obj.points:
            if p[2] <= 1e-4:
                continue
            u = int(round(fx * p[0] / p[2] + cx))
            v = int(round(fy * p[1] / p[2] + cy))
            if 0 <= u < w and 0 <= v < h:
                mask[v, u] = 255
    if np.count_nonzero(mask) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


class CollisionLogger:
    def __init__(self, log_path: str, every_n_frames: int = 30):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n = max(1, int(every_n_frames))
        self._last_level: Optional[str] = None
        self._frame_counter = 0

    def update(
        self,
        frame_idx: int,
        worst_level: str,
        worst_focus: Optional[CollisionResult],
    ) -> None:
        self._frame_counter += 1
        level_changed = worst_level != self._last_level
        periodic = (self._frame_counter % self.every_n) == 0
        if worst_level == "SAFE" and not level_changed and not periodic:
            self._last_level = worst_level
            return
        if not level_changed and not periodic:
            self._last_level = worst_level
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if worst_focus is not None:
            line = (
                f"{ts} | frame={frame_idx} | part={worst_focus.part} | "
                f"obj={worst_focus.obj_id} | dist={worst_focus.min_dist_m:.3f}m | "
                f"level={worst_focus.level}"
            )
        else:
            line = f"{ts} | frame={frame_idx} | level={worst_level} | no_objects"

        # print(f"[COLLISION] {line}")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._last_level = worst_level
