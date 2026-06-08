"""
Детекция хвата: тёмный инструмент должен быть у белого корпуса руки, не «любая тёмная зона».
1) маска белого манипулятора; 2) расширение (рядом с рукой); 3) тёмные пятна только в этой зоне;
4) выбор контура: не слишком большой, максимально «низко» в кадре (хват у конца руки).
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def build_debug_mosaic(
    masks: Dict[str, np.ndarray],
    tile_w: int = 320,
    tile_h: int = 240,
) -> np.ndarray:
    """Combine named uint8 masks into a labelled colour mosaic for cv2.imshow."""
    items = list(masks.items())
    n = len(items)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, (name, mask) in enumerate(items):
        r, c = divmod(idx, cols)
        y0, x0 = r * tile_h, c * tile_w
        # normalise to 0-255 if needed, convert to BGR
        if mask is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            m = mask.copy()
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            m_resized = cv2.resize(m, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            if m_resized.ndim == 2:
                tile = cv2.cvtColor(m_resized, cv2.COLOR_GRAY2BGR)
            else:
                tile = m_resized
        cv2.putText(tile, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return canvas


def draw_fk_projections(
    image_bgr: np.ndarray,
    joint_uvs: List[Optional[Tuple[int, int]]],
    names: Optional[List[str]] = None,
    radius: int = 7,
    color_bgr: Optional[Tuple[int, int, int]] = None,
    draw_skeleton: bool = True,
) -> None:
    """Draw FK-projected joint positions (and skeleton lines) onto image in-place.

    color_bgr — если задан, все суставы рисуются одним цветом (иначе каждый своим).
    draw_skeleton — рисовать линии между суставами (FK-скелет).
    """
    default_colors = [
        (255, 0, 0), (0, 128, 255), (0, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
    ]
    h, w = image_bgr.shape[:2]

    # Рисуем линии скелета между соседними суставами
    if draw_skeleton:
        for i in range(len(joint_uvs) - 1):
            uv0, uv1 = joint_uvs[i], joint_uvs[i + 1]
            if uv0 is None or uv1 is None:
                continue
            u0, v0 = int(round(uv0[0])), int(round(uv0[1]))
            u1, v1 = int(round(uv1[0])), int(round(uv1[1]))
            if not (0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h):
                continue
            c = color_bgr if color_bgr is not None else default_colors[i % len(default_colors)]
            cv2.line(image_bgr, (u0, v0), (u1, v1), c, 2, cv2.LINE_AA)

    # Рисуем точки суставов поверх линий
    for i, uv in enumerate(joint_uvs):
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        c = color_bgr if color_bgr is not None else default_colors[i % len(default_colors)]
        cv2.circle(image_bgr, (u, v), radius, c, 2, cv2.LINE_AA)
        cv2.circle(image_bgr, (u, v), 3, c, -1)
        label = names[i] if names and i < len(names) else f"J{i}"
        cv2.putText(image_bgr, label, (u + radius + 2, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)


def draw_roi_circle(
    image_bgr: np.ndarray,
    center_uv: Tuple[int, int],
    radius_px: int,
    color_bgr: Tuple[int, int, int] = (200, 200, 0),
    label: str = "",
) -> None:
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    cv2.circle(image_bgr, (u, v), radius_px, color_bgr, 1)
    if label:
        cv2.putText(image_bgr, label, (u + radius_px + 3, v - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)


def build_robot_depth_mask(
    depth: np.ndarray,
    depth_scale: float,
    joint_uvs: List[Optional[Tuple[int, int]]],
    joint_depths_m: Optional[List[Optional[float]]] = None,
    link_radius_px: int = 16,
    dilate_px: int = 7,
    depth_eps_m: float = 0.09,
) -> np.ndarray:
    """
    Build a VDI-lite robot mask from FK joints projected to image.
    Draws capsules between consecutive joints and gates by depth proximity.
    """
    h, w = depth.shape[:2]
    geom = np.zeros((h, w), dtype=np.uint8)
    exp_depth = np.zeros((h, w), dtype=np.float32)

    # joints list is [base, J1, J2, J3, J4, J5, J6]
    for i in range(len(joint_uvs) - 1):
        p0 = joint_uvs[i]
        p1 = joint_uvs[i + 1]
        if p0 is None or p1 is None:
            continue
        u0, v0 = int(p0[0]), int(p0[1])
        u1, v1 = int(p1[0]), int(p1[1])
        if not (0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h):
            continue

        cv2.line(geom, (u0, v0), (u1, v1), 255, thickness=max(2, link_radius_px * 2))
        cv2.circle(geom, (u0, v0), link_radius_px, 255, thickness=-1)
        cv2.circle(geom, (u1, v1), link_radius_px, 255, thickness=-1)

        if joint_depths_m is not None and i < len(joint_depths_m) and (i + 1) < len(joint_depths_m):
            z0 = joint_depths_m[i]
            z1 = joint_depths_m[i + 1]
            if z0 is not None and z1 is not None and z0 > 0.0 and z1 > 0.0:
                z_mid = float(0.5 * (z0 + z1))
                cv2.line(exp_depth, (u0, v0), (u1, v1), z_mid, thickness=max(2, link_radius_px * 2))
                cv2.circle(exp_depth, (u0, v0), link_radius_px, float(z0), thickness=-1)
                cv2.circle(exp_depth, (u1, v1), link_radius_px, float(z1), thickness=-1)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        geom = cv2.dilate(geom, k, iterations=1)

    z_m = depth.astype(np.float32) * depth_scale
    valid = depth > 0
    if np.any(exp_depth > 0):
        close_depth = np.abs(z_m - exp_depth) < float(depth_eps_m)
        out = ((geom > 0) & valid & (close_depth | (exp_depth <= 0))).astype(np.uint8) * 255
    else:
        out = ((geom > 0) & valid).astype(np.uint8) * 255
    return out


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    ua = float(aw * ah + bw * bh) - inter
    return inter / ua if ua > 1e-9 else 0.0


def _bbox_center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + 0.5 * aw, ay + 0.5 * ah
    bcx, bcy = bx + 0.5 * bw, by + 0.5 * bh
    return float(np.hypot(acx - bcx, acy - bcy))


def _largest_plausible_arm_contour(
    mask: np.ndarray, w: int, h: int, max_rel_width: float = 0.82, max_rel_height: float = 0.58
) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 4000:
            break
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > max_rel_width * w or bh > max_rel_height * h:
            continue
        return c
    return None


def _gray_valid_near_arm(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    arm_gray_min: int,
    near_arm_dilate_px: int,
    table_depth_m: float = 0.0,
    table_depth_margin_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """gray, valid, near_arm, arm_cx, arm_mask (заливка белого корпуса; 0 если нет контура).

    table_depth_m  — если > 0, пиксели глубже (table_depth_m - table_depth_margin_m) считаются
                     столом и исключаются из arm_mask, даже если они белые.
                     Если 0 — авто: оцениваем как 95-й перцентиль валидных глубин в нижней трети кадра.
    """
    h, w = color_bgr.shape[:2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)

    # ---- auto table depth estimate ----------------------------------------
    table_z = table_depth_m
    if table_z <= 0.0:
        # Look at the bottom third of the frame for the farthest valid pixels
        y_bot = int(h * 0.67)
        bot_z = z_m[y_bot:, :]
        bot_valid = valid[y_bot:, :]
        bot_vals = bot_z[bot_valid]
        if bot_vals.size > 200:
            table_z = float(np.percentile(bot_vals, 85))
        else:
            table_z = depth_max_m  # no estimate → don't filter

    # ---- build arm mask with depth gate ------------------------------------
    arm_bright = (gray >= arm_gray_min).astype(np.uint8) * 255

    # Exclude pixels at or beyond the table depth
    not_table = (z_m < (table_z - table_depth_margin_m)) | (~valid)
    arm_bright = cv2.bitwise_and(arm_bright, arm_bright,
                                  mask=not_table.astype(np.uint8) * 255)

    arm = cv2.morphologyEx(arm_bright, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    arm = cv2.morphologyEx(arm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    arm_cnt = _largest_plausible_arm_contour(arm, w, h)
    arm_mask = np.zeros((h, w), dtype=np.uint8)
    if arm_cnt is not None:
        cv2.drawContours(arm_mask, [arm_cnt], -1, 255, thickness=-1)
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_arm_dilate_px, near_arm_dilate_px))
        near_arm = cv2.dilate(arm_mask, k_big, iterations=1)
        M = cv2.moments(arm_cnt)
        arm_cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else int(0.38 * w)
    else:
        near_arm = np.ones((h, w), dtype=np.uint8) * 255
        arm_cx = int(0.38 * w)
    return gray, valid.astype(bool), near_arm, arm_cx, arm_mask


def detect_gripper_contour(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_y_fraction: float,
    gray_max: int,
    min_area_px: int,
    arm_gray_min: int = 158,
    near_arm_dilate_px: int = 72,
    max_area_px: int = 12000,
    max_x_fraction: float = 0.78,
    gripper_touch_arm_dilate_px: int = 32,
    gripper_max_depth_delta_m: float = 0.22,
    gripper_fallback_max_x_fraction: float = 0.52,
    arm_mask_min_area_px: int = 5000,
    prev_gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    track_iou_weight: float = 85.0,
    track_center_weight: float = 70.0,
    track_max_jump_px: int = 180,
    gripper_tip_max_dist_px: int = 140,
    gripper_tip_max_dist_frac: float = 0.24,
    gripper_tip_allow_above_px: int = 20,
    table_depth_m: float = 0.0,
    table_depth_margin_m: float = 0.05,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Возвращает (contour, debug_mask_uint8).
    """
    h, w = color_bgr.shape[:2]
    y0 = max(0, min(h - 1, int(h * roi_y_fraction)))
    x_max = max(1, int(w * max_x_fraction))

    gray, valid, near_arm, arm_cx, arm_mask = _gray_valid_near_arm(
        color_bgr, depth, depth_scale, depth_min_m, depth_max_m, arm_gray_min, near_arm_dilate_px,
        table_depth_m=table_depth_m, table_depth_margin_m=table_depth_margin_m,
    )
    z_m = depth.astype(np.float32) * depth_scale

    arm_area = int(np.count_nonzero(arm_mask))
    x_limit = x_max
    if arm_area < arm_mask_min_area_px:
        x_limit = min(x_max, max(1, int(w * gripper_fallback_max_x_fraction)))

    roi = np.zeros((h, w), dtype=bool)
    roi[y0:, :x_limit] = True

    dark = gray < gray_max
    mask = (dark & valid & roi & (near_arm > 0)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    touch_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (gripper_touch_arm_dilate_px * 2 + 1, gripper_touch_arm_dilate_px * 2 + 1),
    )
    arm_z = z_m[(arm_mask > 0) & valid]
    arm_z_med = float(np.median(arm_z)) if arm_z.size > 40 else None
    tip_xy = None
    if arm_area >= arm_mask_min_area_px:
        ys, xs = np.where(arm_mask > 0)
        if ys.size > 0:
            tip_y = int(ys.max())
            tip_band = np.abs(ys - tip_y) <= 3
            if np.any(tip_band):
                tip_x = int(np.median(xs[tip_band]))
                tip_xy = (tip_x, tip_y)

    best = None
    best_score = -1e9
    band = int(0.22 * w)
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px or a > max_area_px:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        curr_bbox = (bx, by, bw, bh)
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if abs(cx - arm_cx) > band:
            continue

        cm = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cm, [c], -1, 255, thickness=-1)
        if arm_area >= arm_mask_min_area_px:
            if not np.any((cv2.dilate(cm, touch_k) > 0) & (arm_mask > 0)):
                continue
            gz = z_m[(cm > 0) & valid]
            if arm_z_med is not None and gz.size > 5:
                if abs(float(np.median(gz)) - arm_z_med) > gripper_max_depth_delta_m:
                    continue
            if tip_xy is not None:
                tip_x, tip_y = tip_xy
                tip_dist_lim = max(float(gripper_tip_max_dist_px), float(w) * float(gripper_tip_max_dist_frac))
                tip_dist = float(np.hypot(cx - tip_x, cy - tip_y))
                if tip_dist > tip_dist_lim:
                    continue
                if cy < tip_y - int(gripper_tip_allow_above_px):
                    continue

        # ниже в кадре = ближе к хвату при типичном виде сверху/сбоку
        score = cy + 0.001 * a
        if tip_xy is not None:
            tip_x, tip_y = tip_xy
            tip_dist = float(np.hypot(cx - tip_x, cy - tip_y))
            tip_dist_lim = max(float(gripper_tip_max_dist_px), float(w) * float(gripper_tip_max_dist_frac))
            score += 35.0 * (1.0 - min(1.0, tip_dist / max(1.0, tip_dist_lim)))
            score += 12.0 * (1.0 if cy >= tip_y else 0.0)
        if prev_gripper_bbox is not None:
            jump_px = _bbox_center_distance(curr_bbox, prev_gripper_bbox)
            if jump_px > float(track_max_jump_px):
                continue
            iou = _bbox_iou(curr_bbox, prev_gripper_bbox)
            jump_norm = jump_px / max(1.0, float(track_max_jump_px))
            score += track_iou_weight * iou - track_center_weight * jump_norm
        if score > best_score:
            best_score = score
            best = c
    return best, mask


def _valid_relaxed_mask(valid: np.ndarray, dilate_px: int, iterations: int) -> np.ndarray:
    """Расширяет зону валидной глубины — на белом пластике часто дыры в depth."""
    v = valid.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    out = cv2.dilate(v, k, iterations=iterations)
    return out > 0


def _pick_wrist_contour(
    h: int,
    w: int,
    gray: np.ndarray,
    valid_soft: np.ndarray,
    near_arm: np.ndarray,
    use_near_arm: bool,
    wrist_gray_min: int,
    band: np.ndarray,
    grip_bridge: np.ndarray,
    wrist_min_area_px: int,
    wrist_max_area_px: int,
    y_top: int,
    wrist_anchor_max_below_grip_px: int,
    wrist_min_bbox_height_px: int,
    wrist_min_bbox_width_px: int,
    wrist_max_link_span_px: int,
    grip_cx: int,
    grip_gw: int,
    wrist_max_dx_from_grip_px: int,
    wrist_max_dx_from_grip_frac: float,
    arm_mask: np.ndarray,
    wrist_min_overlap_arm_px: int,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Выбираем компоненту у хвата: нижняя граница контура (max y) должна быть у верхней
    границы bbox хвата — так отсекаются суставы выше по руке и «полоски».
    """
    bright = (gray >= wrist_gray_min) & valid_soft
    if use_near_arm:
        bright &= near_arm > 0
    white = (bright & band).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    dx_lim = max(wrist_max_dx_from_grip_px, int(wrist_max_dx_from_grip_frac * max(grip_gw, 1)))
    arm_pixels = int(np.count_nonzero(arm_mask))

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_key = (-1e9, -1.0)
    for c in contours:
        a = cv2.contourArea(c)
        if a < wrist_min_area_px or a > wrist_max_area_px:
            continue
        cm = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cm, [c], -1, 255, thickness=-1)
        if not np.any((cm > 0) & (grip_bridge > 0)):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        cx_blob = bx + bw // 2
        if abs(cx_blob - grip_cx) > dx_lim:
            continue
        if arm_pixels > 800:
            on_arm = int(np.count_nonzero((cm > 0) & (arm_mask > 0)))
            if on_arm < wrist_min_overlap_arm_px:
                continue
        ys = c[:, 0, 1]
        top_y = int(ys.min())
        bottom_y = int(ys.max())
        if bottom_y > y_top + wrist_anchor_max_below_grip_px:
            continue
        span = bottom_y - top_y + 1
        if span > wrist_max_link_span_px or span < 4:
            continue
        if bh < wrist_min_bbox_height_px or bw < wrist_min_bbox_width_px:
            continue
        key = (bottom_y, float(a))
        if key > best_key:
            best_key = key
            best = c
    return best, white


def clamp_wrist_bbox_to_grip_column(
    xywh: np.ndarray,
    gripper_contour: np.ndarray,
    img_w: int,
    clamp_dx_px: int,
    clamp_dx_frac: float,
) -> None:
    """Сдвигает bbox по X, чтобы центр не уезжал от колонны над хватом (in-place)."""
    gx, gy, gw, gh = cv2.boundingRect(gripper_contour)
    gcx = gx + gw * 0.5
    lim = max(float(clamp_dx_px), float(clamp_dx_frac) * float(max(gw, 1)))
    wcx = xywh[0] + xywh[2] * 0.5
    if abs(wcx - gcx) > lim:
        xywh[0] = gcx - xywh[2] * 0.5
    xywh[0] = float(np.clip(xywh[0], 0.0, max(0.0, float(img_w - 1) - xywh[2])))


def detect_wrist_above_gripper(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    gripper_contour: Optional[np.ndarray],
    arm_gray_min: int = 158,
    near_arm_dilate_px: int = 72,
    wrist_band_height_px: int = 90,
    wrist_band_pad_px: int = 48,
    wrist_bridge_w: int = 15,
    wrist_bridge_h: int = 55,
    wrist_min_area_px: int = 180,
    wrist_max_area_px: int = 32000,
    wrist_use_near_arm: bool = True,
    wrist_gray_min: int = 138,
    wrist_valid_dilate_px: int = 11,
    wrist_valid_dilate_iters: int = 4,
    wrist_fallback_without_near_arm: bool = True,
    wrist_anchor_max_below_grip_px: int = 14,
    wrist_min_bbox_height_px: int = 16,
    wrist_min_bbox_width_px: int = 8,
    wrist_max_link_span_px: int = 100,
    wrist_use_static_roi: bool = False,
    wrist_roi_x0_frac: float = 0.0,
    wrist_roi_y0_frac: float = 0.0,
    wrist_roi_x1_frac: float = 1.0,
    wrist_roi_y1_frac: float = 1.0,
    wrist_max_dx_from_grip_px: int = 42,
    wrist_max_dx_from_grip_frac: float = 0.95,
    wrist_min_overlap_arm_px: int = 120,
    wrist_band_height_from_gripper: float = 2.0,
    table_depth_m: float = 0.0,
    table_depth_margin_m: float = 0.05,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Белый сустав над хватом: полоса над bbox, яркость wrist_gray_min (ниже arm_gray_min),
    depth через расширенную valid; мост от хвата вверх. При неудаче — без near_arm.
    """
    h, w = color_bgr.shape[:2]
    debug = np.zeros((h, w), dtype=np.uint8)
    empty = np.zeros((h, w), dtype=np.uint8)
    if gripper_contour is None:
        return None, empty

    gray, valid, near_arm, _arm_cx, arm_mask = _gray_valid_near_arm(
        color_bgr, depth, depth_scale, depth_min_m, depth_max_m, arm_gray_min, near_arm_dilate_px,
        table_depth_m=table_depth_m, table_depth_margin_m=table_depth_margin_m,
    )
    valid_soft = _valid_relaxed_mask(valid, wrist_valid_dilate_px, wrist_valid_dilate_iters)
    valid_soft &= depth > 0

    gx, gy, gw, gh = cv2.boundingRect(gripper_contour)
    grip_cx = gx + gw // 2
    y_top = gy
    band_h = max(wrist_band_height_px, int(round(float(gh) * wrist_band_height_from_gripper)))
    y0 = max(0, y_top - band_h)
    y1 = max(0, y_top)
    if y1 <= y0:
        return None, empty

    x0 = max(0, gx - wrist_band_pad_px)
    x1 = min(w, gx + gw + wrist_band_pad_px)

    grip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(grip_mask, [gripper_contour], -1, 255, thickness=-1)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wrist_bridge_w, wrist_bridge_h))
    grip_bridge = cv2.dilate(grip_mask, k_bridge, iterations=2)

    band = np.zeros((h, w), dtype=bool)
    band[y0:y1, x0:x1] = True
    if (
        wrist_use_static_roi
        and 0.0 <= wrist_roi_x0_frac < wrist_roi_x1_frac <= 1.0
        and 0.0 <= wrist_roi_y0_frac < wrist_roi_y1_frac <= 1.0
    ):
        static = np.zeros((h, w), dtype=bool)
        sx0 = int(wrist_roi_x0_frac * w)
        sx1 = int(wrist_roi_x1_frac * w)
        sy0 = int(wrist_roi_y0_frac * h)
        sy1 = int(wrist_roi_y1_frac * h)
        static[sy0:sy1, sx0:sx1] = True
        band &= static

    pick_kw = dict(
        h=h,
        w=w,
        gray=gray,
        valid_soft=valid_soft,
        near_arm=near_arm,
        wrist_gray_min=wrist_gray_min,
        band=band,
        grip_bridge=grip_bridge,
        wrist_min_area_px=wrist_min_area_px,
        wrist_max_area_px=wrist_max_area_px,
        y_top=y_top,
        wrist_anchor_max_below_grip_px=wrist_anchor_max_below_grip_px,
        wrist_min_bbox_height_px=wrist_min_bbox_height_px,
        wrist_min_bbox_width_px=wrist_min_bbox_width_px,
        wrist_max_link_span_px=wrist_max_link_span_px,
        grip_cx=grip_cx,
        grip_gw=gw,
        wrist_max_dx_from_grip_px=wrist_max_dx_from_grip_px,
        wrist_max_dx_from_grip_frac=wrist_max_dx_from_grip_frac,
        arm_mask=arm_mask,
        wrist_min_overlap_arm_px=wrist_min_overlap_arm_px,
    )

    best, white = _pick_wrist_contour(
        use_near_arm=wrist_use_near_arm,
        **pick_kw,
    )
    if best is None and wrist_fallback_without_near_arm and wrist_use_near_arm:
        best, white = _pick_wrist_contour(use_near_arm=False, **pick_kw)

    if best is not None:
        cv2.drawContours(debug, [best], -1, 255, thickness=-1)
    return best, white


def detect_in_3d_sphere(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    center_3d: np.ndarray,
    sphere_radius_m: float = 0.08,
    target: str = "gripper",
    dark_rgb_max: float = 0.35,
    bright_rgb_min: float = 0.55,
    dbscan_eps: float = 0.012,
    dbscan_min_pts: int = 15,
) -> Optional[np.ndarray]:
    """
    Find gripper or wrist joint by searching the 3D point cloud inside a sphere
    centred on the FK-projected joint position (camera frame, metres).

    Parameters
    ----------
    points_xyz   : (N,3) float32/64 — point cloud in camera frame (metres).
    colors_rgb   : (N,3) float32   — RGB colours in [0,1].
    center_3d    : (3,)            — expected 3D position from FK in camera frame.
    sphere_radius_m : search radius in metres (default 8 cm).
    target       : "gripper" → look for dark cluster;
                   "wrist"   → look for bright/white cluster.
    dark_rgb_max : max mean RGB for a point to be considered "dark" (gripper).
    bright_rgb_min : min mean RGB for a point to be considered "bright" (wrist).
    dbscan_eps   : DBSCAN neighbourhood radius in metres.
    dbscan_min_pts : min samples per DBSCAN cluster.

    Returns
    -------
    (3,) centroid of the best cluster, or None if nothing found.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return None

    # 1. Restrict to sphere
    diff = points_xyz - center_3d
    dist2 = np.einsum("ij,ij->i", diff, diff)
    in_sphere = dist2 <= sphere_radius_m ** 2
    if not np.any(in_sphere):
        return None

    pts = points_xyz[in_sphere]
    clr = colors_rgb[in_sphere]

    # 2. Colour filter
    brightness = clr.mean(axis=1)   # (M,)
    if target == "gripper":
        color_mask = brightness < dark_rgb_max
    else:
        color_mask = brightness > bright_rgb_min

    if np.count_nonzero(color_mask) < dbscan_min_pts:
        return None

    pts_filtered = pts[color_mask]
    clr_filtered = clr[color_mask]

    # 3. DBSCAN clustering — find the densest cluster closest to FK centre
    try:
        from sklearn.cluster import DBSCAN as _DBSCAN
        labels = _DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_pts).fit_predict(pts_filtered)
    except ImportError:
        # Fallback without sklearn: just return centroid of colour-filtered pts
        return pts_filtered.mean(axis=0)

    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if not unique_labels:
        return None

    # Pick cluster whose centroid is closest to FK centre
    best_centroid = None
    best_dist = np.inf
    for lbl in unique_labels:
        mask_lbl = labels == lbl
        centroid = pts_filtered[mask_lbl].mean(axis=0)
        d = float(np.linalg.norm(centroid - center_3d))
        if d < best_dist:
            best_dist = d
            best_centroid = centroid

    return best_centroid


def detect_gripper_in_roi(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_center_uv: Tuple[int, int],
    roi_radius: int,
    gray_max: int = 95,
    min_area_px: int = 450,
    max_area_px: int = 12000,
    arm_gray_min: int = 158,
    near_arm_dilate_px: int = 72,
    gripper_max_depth_delta_m: float = 0.15,
    prev_gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    track_max_jump_px: int = 120,
    track_iou_weight: float = 85.0,
    track_center_weight: float = 70.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Find gripper (dark blob) **inside a circular ROI** centred on the FK
    projection of joint A6 (flange).  Independent of arm_mask extent.

    Returns (contour | None, debug_mask_uint8).
    """
    h, w = color_bgr.shape[:2]
    uc, vc = roi_center_uv

    # Circular ROI mask
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (uc, vc), roi_radius, 255, thickness=-1)

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)

    dark = (gray < gray_max).astype(np.uint8) * 255
    mask = cv2.bitwise_and(dark, dark, mask=(valid.astype(np.uint8) * 255))
    mask = cv2.bitwise_and(mask, roi_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    # Reference depth at ROI centre
    roi_z = z_m[max(0, vc - 10):vc + 10, max(0, uc - 10):uc + 10]
    valid_roi_z = roi_z[(roi_z > depth_min_m) & (roi_z < depth_max_m)]
    ref_depth = float(np.median(valid_roi_z)) if valid_roi_z.size > 5 else None

    best = None
    best_score = -1e9
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px or a > max_area_px:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx_c = int(M["m10"] / M["m00"])
        cy_c = int(M["m01"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(c)
        curr_bbox = (bx, by, bw, bh)

        # Depth consistency check
        if ref_depth is not None:
            cm = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cm, [c], -1, 255, thickness=-1)
            gz = z_m[(cm > 0) & valid]
            if gz.size > 5 and abs(float(np.median(gz)) - ref_depth) > gripper_max_depth_delta_m:
                continue

        # Prefer centroid close to ROI centre
        dist_to_roi = float(np.hypot(cx_c - uc, cy_c - vc))
        score = -dist_to_roi + 0.001 * a

        if prev_gripper_bbox is not None:
            jump_px = _bbox_center_distance(curr_bbox, prev_gripper_bbox)
            if jump_px > float(track_max_jump_px):
                continue
            iou = _bbox_iou(curr_bbox, prev_gripper_bbox)
            jump_norm = jump_px / max(1.0, float(track_max_jump_px))
            score += track_iou_weight * iou - track_center_weight * jump_norm

        if score > best_score:
            best_score = score
            best = c

    return best, mask


def detect_wrist_in_roi(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_center_uv: Tuple[int, int],
    roi_radius: int,
    wrist_gray_min: int = 138,
    min_area_px: int = 180,
    max_area_px: int = 32000,
    wrist_valid_dilate_px: int = 11,
    wrist_valid_dilate_iters: int = 4,
    prev_wrist_bbox: Optional[Tuple[int, int, int, int]] = None,
    track_max_jump_px: int = 120,
    track_iou_weight: float = 85.0,
    track_center_weight: float = 70.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Find wrist joint (bright/white blob) **inside a circular ROI** centred on
    the FK projection of joint A5.  Does NOT depend on gripper detection.

    Returns (contour | None, debug_mask_uint8).
    """
    h, w = color_bgr.shape[:2]
    uc, vc = roi_center_uv

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (uc, vc), roi_radius, 255, thickness=-1)

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)
    valid_soft = _valid_relaxed_mask(valid, wrist_valid_dilate_px, wrist_valid_dilate_iters)
    valid_soft &= depth > 0

    bright = ((gray >= wrist_gray_min) & valid_soft).astype(np.uint8) * 255
    mask = cv2.bitwise_and(bright, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_score = -1e9
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_px or a > max_area_px:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx_c = int(M["m10"] / M["m00"])
        cy_c = int(M["m01"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(c)
        curr_bbox = (bx, by, bw, bh)

        dist_to_roi = float(np.hypot(cx_c - uc, cy_c - vc))
        score = -dist_to_roi + 0.001 * a

        if prev_wrist_bbox is not None:
            jump_px = _bbox_center_distance(curr_bbox, prev_wrist_bbox)
            if jump_px > float(track_max_jump_px):
                continue
            iou = _bbox_iou(curr_bbox, prev_wrist_bbox)
            jump_norm = jump_px / max(1.0, float(track_max_jump_px))
            score += track_iou_weight * iou - track_center_weight * jump_norm

        if score > best_score:
            best_score = score
            best = c

    return best, mask


def detect_gripper_with_robot_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_center_uv: Tuple[int, int],
    roi_radius: int,
    robot_mask: np.ndarray,
    gray_max: int = 95,
    min_area_px: int = 450,
    max_area_px: int = 12000,
    prev_gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    track_max_jump_px: int = 120,
    track_iou_weight: float = 85.0,
    track_center_weight: float = 70.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    contour, dbg = detect_gripper_in_roi(
        color_bgr, depth, depth_scale, depth_min_m, depth_max_m,
        roi_center_uv, roi_radius, gray_max=gray_max, min_area_px=min_area_px,
        max_area_px=max_area_px, prev_gripper_bbox=prev_gripper_bbox,
        track_max_jump_px=track_max_jump_px, track_iou_weight=track_iou_weight,
        track_center_weight=track_center_weight,
    )
    if contour is None:
        return None, cv2.bitwise_and(dbg, robot_mask)
    cm = np.zeros_like(robot_mask)
    cv2.drawContours(cm, [contour], -1, 255, thickness=-1)
    if np.count_nonzero((cm > 0) & (robot_mask > 0)) < max(40, int(0.06 * np.count_nonzero(cm))):
        return None, cv2.bitwise_and(dbg, robot_mask)
    return contour, cv2.bitwise_and(dbg, robot_mask)


def detect_gripper_from_manipulator_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_center_uv: Tuple[int, int],
    roi_radius: int,
    manipulator_mask: np.ndarray,
    gray_max: int = 110,
    min_area_px: int = 120,
    max_area_px: int = 8000,
    near_mask_dilate_px: int = 19,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Gripper detection from manipulator mask:
    1) dark pixels
    2) near/in manipulator mask
    3) inside FK A6 circular ROI
    """
    h, w = color_bgr.shape[:2]
    u0, v0 = roi_center_uv

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)

    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi, (int(u0), int(v0)), int(roi_radius), 255, thickness=-1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_mask_dilate_px, near_mask_dilate_px))
    near_man = cv2.dilate(manipulator_mask, k, iterations=1)

    dark = (gray < gray_max).astype(np.uint8) * 255
    cand = cv2.bitwise_and(dark, dark, mask=(valid.astype(np.uint8) * 255))
    cand = cv2.bitwise_and(cand, roi)
    cand = cv2.bitwise_and(cand, near_man)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, cand

    best = None
    best_score = -1e9
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < float(min_area_px) or a > float(max_area_px):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        d = float(np.hypot(cx - float(u0), cy - float(v0)))
        score = -d + 0.001 * a
        if score > best_score:
            best_score = score
            best = c

    return best, cand


def detect_wrist_with_robot_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    roi_center_uv: Tuple[int, int],
    roi_radius: int,
    robot_mask: np.ndarray,
    wrist_gray_min: int = 138,
    min_area_px: int = 180,
    max_area_px: int = 32000,
    wrist_valid_dilate_px: int = 11,
    wrist_valid_dilate_iters: int = 4,
    prev_wrist_bbox: Optional[Tuple[int, int, int, int]] = None,
    track_max_jump_px: int = 120,
    track_iou_weight: float = 85.0,
    track_center_weight: float = 70.0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    contour, dbg = detect_wrist_in_roi(
        color_bgr, depth, depth_scale, depth_min_m, depth_max_m,
        roi_center_uv, roi_radius, wrist_gray_min=wrist_gray_min,
        min_area_px=min_area_px, max_area_px=max_area_px,
        wrist_valid_dilate_px=wrist_valid_dilate_px,
        wrist_valid_dilate_iters=wrist_valid_dilate_iters,
        prev_wrist_bbox=prev_wrist_bbox, track_max_jump_px=track_max_jump_px,
        track_iou_weight=track_iou_weight, track_center_weight=track_center_weight,
    )
    if contour is None:
        return None, cv2.bitwise_and(dbg, robot_mask)
    cm = np.zeros_like(robot_mask)
    cv2.drawContours(cm, [contour], -1, 255, thickness=-1)
    if np.count_nonzero((cm > 0) & (robot_mask > 0)) < max(40, int(0.06 * np.count_nonzero(cm))):
        return None, cv2.bitwise_and(dbg, robot_mask)
    return contour, cv2.bitwise_and(dbg, robot_mask)


def draw_gripper_overlay(
    image_bgr: np.ndarray,
    contour: Optional[np.ndarray],
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    if contour is None:
        return
    cv2.drawContours(image_bgr, [contour], -1, color_bgr, thickness)
    x, y, bw, bh = cv2.boundingRect(contour)
    cv2.putText(
        image_bgr,
        "Gripper",
        (max(0, x), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image_bgr, (x, y), (x + bw, y + bh), color_bgr, 1)


def draw_wrist_zone_rect(
    image_bgr: np.ndarray,
    xywh: Tuple[int, int, int, int],
    color_bgr: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
) -> None:
    """Стабильная зона сустава: один сглаженный прямоугольник без «рваного» контура."""
    x, y, bw, bh = xywh
    if bw <= 0 or bh <= 0:
        return
    cv2.rectangle(image_bgr, (x, y), (x + bw, y + bh), color_bgr, thickness)
    cv2.putText(
        image_bgr,
        "Wrist",
        (max(0, x), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def draw_wrist_overlay(
    image_bgr: np.ndarray,
    contour: Optional[np.ndarray],
    color_bgr: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
) -> None:
    if contour is None:
        return
    cv2.drawContours(image_bgr, [contour], -1, color_bgr, thickness)
    x, y, bw, bh = cv2.boundingRect(contour)
    cv2.putText(
        image_bgr,
        "Wrist",
        (max(0, x), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image_bgr, (x, y), (x + bw, y + bh), color_bgr, 1)


def extract_main_robot_contour(robot_mask: np.ndarray, min_area_px: int = 1200) -> Optional[np.ndarray]:
    """Return the most central plausible robot contour from a binary mask."""
    contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = robot_mask.shape[:2]
    cx_ref = 0.5 * w
    cy_ref = 0.55 * h
    best = None
    best_key = (1e18, 1e18)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area_px):
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        dist2 = (cx - cx_ref) ** 2 + (cy - cy_ref) ** 2
        key = (dist2, -area)
        if key < best_key:
            best_key = key
            best = c
    if best is None:
        return None
    return best


def draw_robot_mask_overlay(
    image_bgr: np.ndarray,
    robot_mask: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    min_area_px: int = 1200,
) -> None:
    """
    Draw yellow contour around the robot from geometric self-filter mask.
    """
    c = extract_main_robot_contour(robot_mask, min_area_px=min_area_px)
    if c is None:
        return
    cv2.drawContours(image_bgr, [c], -1, color_bgr, thickness)
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(
        image_bgr,
        "Manipulator",
        (max(0, x), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def build_table_mask(
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    robot_seed_mask: np.ndarray,
    protect_mask: Optional[np.ndarray] = None,
    table_depth_m: float = 0.0,
    table_depth_margin_m: float = 0.05,
    seed_exclude_dilate_px: int = 61,
    protect_dilate_px: int = 25,
    min_y_frac: float = 0.45,
    min_area_px: int = 6000,
    min_width_frac: float = 0.35,
    max_height_frac: float = 0.60,
) -> np.ndarray:
    """
    Build table/scene mask from depth plane, explicitly excluding FK-seed neighborhood.
    """
    h, w = depth.shape[:2]
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)

    table_z = float(table_depth_m)
    if table_z <= 0.0:
        y_bot = int(h * 0.62)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (seed_exclude_dilate_px, seed_exclude_dilate_px),
        )
        seed_far = cv2.dilate(robot_seed_mask, k, iterations=1)
        bot_vals = z_m[y_bot:, :][valid[y_bot:, :] & (seed_far[y_bot:, :] == 0)]
        if bot_vals.size > 700:
            table_z = float(np.percentile(bot_vals, 55))
        elif bot_vals.size > 250:
            table_z = float(np.percentile(bot_vals, 70))
        else:
            table_z = float(depth_max_m)

    table = valid & (z_m >= (table_z - table_depth_margin_m))
    table[: int(h * float(min_y_frac)), :] = False
    table = table.astype(np.uint8) * 255

    protect = robot_seed_mask.copy()
    if protect_mask is not None and protect_mask.shape == protect.shape:
        protect = cv2.bitwise_or(protect, protect_mask)
    if protect_dilate_px > 1:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_dilate_px, protect_dilate_px))
        protect = cv2.dilate(protect, kd, iterations=1)
    # Soft anti-robot rule: suppress table where robot seed is likely.
    table[protect > 0] = 0

    table = cv2.morphologyEx(table, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    table = cv2.morphologyEx(table, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats((table > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    best_lab = -1
    best_key = (-1, -1.0)
    min_w = int(float(min_width_frac) * float(w))
    max_h = int(float(max_height_frac) * float(h))
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if bw < min_w:
            continue
        if bh > max_h:
            continue
        key = (area, float(bw) / max(1.0, float(bh)))
        if key > best_key:
            best_key = key
            best_lab = lab
    if best_lab >= 0:
        out[lbl == best_lab] = 255
    out[protect > 0] = 0
    return out


def build_manipulator_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
    robot_seed_mask: np.ndarray,
    arm_gray_min: int = 150,
    seed_dilate_px: int = 19,
    table_depth_m: float = 0.0,
    table_depth_margin_m: float = 0.05,
    max_seed_dist_px: int = 110,
    min_seed_overlap_ratio: float = 0.01,
    max_component_area_frac: float = 0.30,
    table_seed_exclude_dilate_px: int = 61,
    table_mask: Optional[np.ndarray] = None,
    mask_adaptive_window_px: int = 41,
    mask_adaptive_k_sigma: float = 0.35,
    component_min_area_px: int = 240,
    component_min_compactness: float = 0.05,
    component_reject_bottom_frac: float = 0.92,
    strict_no_table_mode: bool = True,
    strict_table_guard_px: int = 8,
    strict_bottom_seed_overlap_min: float = 0.06,
    grip_roi_uv: Optional[Tuple[int, int]] = None,
    front_view_mode: bool = False,
    front_view_smart_cut_enabled: bool = True,
    front_view_cut_offset_px: int = 10,
    front_view_seed_connectivity_min_area_px: int = 120,
    include_gripper: bool = True,
    gripper_merge_gray_max: int = 130,
    gripper_merge_roi_radius_px: int = 75,
    gripper_merge_roi_extend_down_px: int = 90,
    gripper_merge_attach_dilate_px: int = 27,
    gripper_merge_max_depth_delta_m: float = 0.12,
    gripper_merge_min_area_px: int = 60,
    gripper_merge_close_px: int = 15,
    gripper_merge_allow_no_depth: bool = True,
    debug_out: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Build a stable manipulator mask using geometric FK seed + bright robot body.
    Keeps only bright connected components overlapping FK seed.
    Optionally merges dark gripper pixels near flange (J6) into the mask.
    """
    h, w = depth.shape[:2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    z_m = depth.astype(np.float32) * depth_scale
    valid = (depth > 0) & (z_m > depth_min_m) & (z_m < depth_max_m)

    # Emergency fallback: if FK seed is weak/empty, use legacy central bright-component logic.
    if np.count_nonzero(robot_seed_mask) < 20:
        table_z = table_depth_m
        if table_z <= 0.0:
            y_bot = int(h * 0.67)
            bot_z = z_m[y_bot:, :]
            bot_valid = valid[y_bot:, :]
            bot_vals = bot_z[bot_valid]
            if bot_vals.size > 200:
                table_z = float(np.percentile(bot_vals, 85))
            else:
                table_z = depth_max_m
        not_table = z_m < (table_z - table_depth_margin_m)
        bright = ((gray >= arm_gray_min) & valid & not_table).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((bright > 0).astype(np.uint8), connectivity=8)
        out = np.zeros((h, w), dtype=np.uint8)
        cx_ref = w * 0.5
        cy_ref = h * 0.55
        best_lab = -1
        best_key = (1e18, -1)
        for lab in range(1, n_lbl):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 800:
                continue
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bh < bw:  # reject wide table/paper-like components
                continue
            x = float(stats[lab, cv2.CC_STAT_LEFT] + 0.5 * stats[lab, cv2.CC_STAT_WIDTH])
            y = float(stats[lab, cv2.CC_STAT_TOP] + 0.5 * stats[lab, cv2.CC_STAT_HEIGHT])
            dist2 = (x - cx_ref) ** 2 + (y - cy_ref) ** 2
            key = (dist2, -area)
            if key < best_key:
                best_key = key
                best_lab = lab
        if best_lab >= 0:
            out[lbl == best_lab] = 255
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        return out

    if seed_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_dilate_px, seed_dilate_px))
        seed = cv2.dilate(robot_seed_mask, k, iterations=1)
    else:
        seed = robot_seed_mask.copy()
    seed_bin = seed > 0

    inv_seed = (~seed_bin).astype(np.uint8)
    dist_from_seed = cv2.distanceTransform(inv_seed, cv2.DIST_L2, 5)
    near_seed = dist_from_seed <= float(max(1, max_seed_dist_px))

    if table_mask is None:
        table_mask = build_table_mask(
            depth=depth,
            depth_scale=depth_scale,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            robot_seed_mask=robot_seed_mask,
            table_depth_m=table_depth_m,
            table_depth_margin_m=table_depth_margin_m,
            seed_exclude_dilate_px=table_seed_exclude_dilate_px,
        )
    table_bin = table_mask > 0
    protect_zone = near_seed | seed_bin
    not_table = (~table_bin) | protect_zone

    # Hard stop for strict_no_table mode:
    # if table is detected, reject manipulator pixels below table-top (except protected robot-seed zone).
    strict_allow = np.ones((h, w), dtype=bool)
    if strict_no_table_mode and np.count_nonzero(table_bin) > 0:
        row_counts = np.count_nonzero(table_bin, axis=1)
        row_thr = max(20, int(0.18 * w))
        rows = np.where(row_counts >= row_thr)[0]
        if rows.size > 0:
            table_top_y = int(rows[0])
            y_cut = int(max(0, table_top_y - int(strict_table_guard_px)))
            strict_allow[y_cut:, :] = False
            strict_allow[protect_zone] = True

    # Adaptive threshold (Cork style): local mean - k*sigma, constrained by arm_gray_min.
    w_ad = int(max(5, mask_adaptive_window_px))
    if (w_ad % 2) == 0:
        w_ad += 1
    gray_f = gray.astype(np.float32)
    mu = cv2.GaussianBlur(gray_f, (w_ad, w_ad), 0)
    sq = cv2.GaussianBlur(gray_f * gray_f, (w_ad, w_ad), 0)
    sigma = np.sqrt(np.maximum(0.0, sq - mu * mu))
    t_map = np.maximum(float(arm_gray_min), mu - float(mask_adaptive_k_sigma) * sigma)
    bright = ((gray_f >= t_map) & valid & not_table & near_seed & strict_allow).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Connected components on bright mask, keep components touching seed
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((bright > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    frame_area = float(h * w)
    for lab in range(1, n_lbl):
        comp = lbl == lab
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(component_min_area_px):
            continue
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        aspect = float(bh) / max(1.0, float(bw))
        touches_bottom = (y0 + bh) >= int(component_reject_bottom_frac * float(h))
        if np.count_nonzero(comp) > 20:
            _cts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = float(cv2.arcLength(max(_cts, key=cv2.contourArea), True)) if _cts else 0.0
        else:
            perimeter = 0.0
        compactness = (4.0 * np.pi * float(area) / max(1.0, perimeter * perimeter)) if perimeter > 1.0 else 0.0
        seed_overlap = float(np.count_nonzero(comp & seed_bin)) / float(max(1, area))
        near_ratio = float(np.count_nonzero(comp & near_seed)) / float(max(1, area))
        dist_med = float(np.median(dist_from_seed[comp])) if np.count_nonzero(comp) > 0 else 1e9

        touches_seed = np.any(comp & seed_bin)
        plausible_shape = (
            (aspect > 0.45)
            and (area < float(max_component_area_frac) * frame_area)
            and (compactness >= float(component_min_compactness))
        )

        if touches_bottom and seed_overlap < float(strict_bottom_seed_overlap_min):
            continue
        if near_ratio < 0.55 and not touches_seed:
            continue
        if (not touches_seed) and (seed_overlap < float(min_seed_overlap_ratio)) and (dist_med > 0.75 * float(max_seed_dist_px)):
            continue

        if touches_seed or plausible_shape:
            out[comp] = 255

    # Always include seed itself to avoid dropping thin links
    out = cv2.bitwise_or(out, robot_seed_mask)
    if strict_no_table_mode and np.count_nonzero(table_bin) > 0:
        out[(table_bin) & (~protect_zone)] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Fallback for weak FK-seed: choose central bright component
    if np.count_nonzero(out) < 1200:
        cx_ref = w * 0.5
        cy_ref = h * 0.55
        best_lab = -1
        best_key = (1e18, -1)
        for lab in range(1, n_lbl):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 800:
                continue
            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bh < bw:  # reject horizontally dominant components (paper/table)
                continue
            x = float(stats[lab, cv2.CC_STAT_LEFT] + 0.5 * stats[lab, cv2.CC_STAT_WIDTH])
            y = float(stats[lab, cv2.CC_STAT_TOP] + 0.5 * stats[lab, cv2.CC_STAT_HEIGHT])
            dist2 = (x - cx_ref) ** 2 + (y - cy_ref) ** 2
            key = (dist2, -area)
            if key < best_key:
                best_key = key
                best_lab = lab
        if best_lab >= 0:
            out = np.zeros((h, w), dtype=np.uint8)
            out[lbl == best_lab] = 255
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    # Final soft non-overlap with table outside protected robot zone.
    out[(table_bin) & (~protect_zone)] = 0

    # Front-view smart-cut: below flange line keep only seed-connected components.
    if (
        front_view_smart_cut_enabled
        and front_view_mode
        and grip_roi_uv is not None
        and np.count_nonzero(out) > 0
    ):
        _, v_cut_ref = grip_roi_uv
        y_cut = int(np.clip(v_cut_ref + int(front_view_cut_offset_px), 0, h - 1))
        n3, lbl3, stats3, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        keep_labels = np.zeros(n3, dtype=bool)
        for lab in range(1, n3):
            area = int(stats3[lab, cv2.CC_STAT_AREA])
            if area < int(front_view_seed_connectivity_min_area_px):
                continue
            comp = (lbl3 == lab)
            if np.count_nonzero(comp & seed_bin) > 0:
                keep_labels[lab] = True
        rows = np.arange(h)[:, None]
        below = rows >= y_cut
        remove = below & (lbl3 > 0) & (~keep_labels[lbl3])
        out[remove] = 0

    # Merge dark gripper into manipulator mask (single contour: arm + gripper).
    # Gripper often has invalid depth (dark metal); do NOT require &valid on dark pixels.
    if include_gripper and np.count_nonzero(out) > 0:
        dark = gray < int(gripper_merge_gray_max)
        no_depth = ~valid
        attach_px = int(max(3, gripper_merge_attach_dilate_px))
        k_attach = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (attach_px, attach_px)
        )
        attach = cv2.dilate(out, k_attach, iterations=1)

        search_zone = np.zeros((h, w), dtype=np.uint8)
        if grip_roi_uv is not None:
            u0, v0 = int(round(grip_roi_uv[0])), int(round(grip_roi_uv[1]))
            r = int(max(8, gripper_merge_roi_radius_px))
            cv2.circle(search_zone, (u0, v0), r, 255, thickness=-1)
            extend = int(max(0, gripper_merge_roi_extend_down_px))
            v1 = int(min(h - 1, v0 + extend))
            x0 = int(max(0, u0 - r))
            x1 = int(min(w - 1, u0 + r))
            cv2.rectangle(search_zone, (x0, v0), (x1, v1), 255, thickness=-1)
        else:
            search_zone = attach.copy()

        cand_bool = dark & (search_zone > 0) & (attach > 0)
        cand = cand_bool.astype(np.uint8) * 255

        ref_z = z_m[(out > 0) & valid]
        if ref_z.size > 20 and gripper_merge_max_depth_delta_m > 0:
            z_ref = float(np.median(ref_z))
            delta = float(gripper_merge_max_depth_delta_m)
            # Inverted depth-gate: reject only valid pixels far from arm (table), keep no-depth gripper.
            reject = valid & (np.abs(z_m - z_ref) > delta)
            cand[reject] = 0
            cand_bool = cand > 0

        if gripper_merge_allow_no_depth:
            no_depth_boost = dark & no_depth & (search_zone > 0) & (attach > 0)
            cand_bool = cand_bool | no_depth_boost
            cand = cand_bool.astype(np.uint8) * 255

        gripper_pixels = np.zeros((h, w), dtype=np.uint8)
        n_g, lbl_g, stats_g, _ = cv2.connectedComponentsWithStats(
            cand_bool.astype(np.uint8), connectivity=8
        )
        out_bin = out > 0
        for lab in range(1, n_g):
            area = int(stats_g[lab, cv2.CC_STAT_AREA])
            if area < int(gripper_merge_min_area_px):
                continue
            comp = lbl_g == lab
            if np.any(comp & out_bin):
                gripper_pixels[comp] = 255

        if debug_out is not None:
            debug_out["gripper_cand"] = cand.copy()
            debug_out["gripper_pixels"] = gripper_pixels.copy()

        if np.count_nonzero(gripper_pixels) > 0:
            out = cv2.bitwise_or(out, gripper_pixels)
            close_px = int(max(3, gripper_merge_close_px))
            if close_px % 2 == 0:
                close_px += 1
            k_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_px, close_px)
            )
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)

    return out
