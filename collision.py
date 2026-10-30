"""
Детекция объектов сцены и оценка коллизий (кадр камеры, метры).

Геометрия, типы и решающее правило живут в `core/`; рисование рамок — в
`vision/overlay_service.py`, боксы Open3D — в `vision/visualizer.py`,
растеризация масок — в `vision/masks.py`. Здесь остаётся сама детекция:
построение маски кандидатов, гейты отбраковки и оценка дистанций.

Запасная 3D-ветвь (DBSCAN по облаку) удалена 2026-10-30 как неработающая:
измерено, что она не обнаруживала препятствие ни на одном кадре S04 и не
укладывалась в 30 минут на S08b. Обоснование и числа — docs/adr/0005.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import (
    CENTROID,
    REPRESENTATIVE,
    box_overlap as _box_overlap,
    min_dist_point_to_cloud,
    min_distance_to_spheres as min_distance_to_robot,
    plane_height,
    voxel_down,
)
from core.kinematics import build_link_spheres, link_part_name as _link_part_name
from core.levels import LEVEL_ORDER, classify_level
from core.types import CollisionResult, DetectionReport, LinkSphere, SceneObject

def build_gripper_capsule_mask(
    spheres: List[LinkSphere],
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
    pad_px: int = 8,
    parts: Sequence[str] = ("gripper",),
) -> np.ndarray:
    """FK-капсула руки, растеризованная в кадр (uint8, 0/255).

    Тёмный металл запястья/хвата и синий шланг не дают валидной глубины, поэтому
    глубинная маска руки их пропускает — а высотный гейт (они выше стола) делает
    из них ложные «объекты» вплотную к руке (ложный DANGER). Геометрии FK глубина
    не нужна: сферы выбранных сегментов (`parts`) проецируются в кадр кругами
    радиусом fx·r/z (+pad_px запас на погрешность калибровки), соседние круги
    соединяются линией → сплошная «колбаса» поверх руки.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    prev_uv: Optional[Tuple[int, int]] = None
    prev_r = 0
    for sp in spheres:
        if sp.part not in parts:
            prev_uv = None
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


def _dilate_mask_in_bbox(
    mask_bool: np.ndarray, x0: int, y0: int, bw: int, bh: int, ksize: int, pad: int
) -> np.ndarray:
    """Дилатация bool-маски ТОЛЬКО в окне bbox+pad.

    Компонента ненулевая лишь внутри своего bbox, поэтому дилатацию достаточно
    считать в маленьком окне вокруг него — результат идентичен полнокадровой
    дилатации, но в разы дешевле (полнокадровое ядро 57–67px × ~15 компонент
    съедало 150–400 мс/кадр). `pad` должен покрывать и разрастание самой
    компоненты, и радиус ядра.
    """
    h, w = mask_bool.shape
    ax0 = max(0, x0 - pad); ay0 = max(0, y0 - pad)
    ax1 = min(w, x0 + bw + pad); ay1 = min(h, y0 + bh + pad)
    out = np.zeros_like(mask_bool)
    if ax1 <= ax0 or ay1 <= ay0:
        return out
    sub = mask_bool[ay0:ay1, ax0:ax1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out[ay0:ay1, ax0:ax1] = cv2.dilate(sub, k) > 0
    return out


def _merge_overlapping_objects(
    objects: List["SceneObject"], iou_thresh: float, contain_thresh: float
) -> List["SceneObject"]:
    """Слить объекты с перекрывающимися 2D-боксами (фрагменты одного кубика).

    Объединяет точки/цвета, пересчитывает центроид/AABB/bbox_2d, признаки протечки
    берёт консервативно (touches_arm=any, arm_depth_delta=min). Слияние точек лишь
    уточняет дистанцию (ближайшая точка сохраняется), поэтому безопасно.
    """
    n = len(objects)
    if n < 2:
        return objects
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [o.bbox_2d for o in objects]
    for i in range(n):
        if boxes[i] is None:
            continue
        for j in range(i + 1, n):
            if boxes[j] is None:
                continue
            iou, cont = _box_overlap(boxes[i], boxes[j])
            if iou >= iou_thresh or cont >= contain_thresh:
                parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == n:
        return objects  # нечего сливать

    merged: List["SceneObject"] = []
    for k, idxs in enumerate(groups.values()):
        if len(idxs) == 1:
            o = objects[idxs[0]]
            o.obj_id = k
            merged.append(o)
            continue
        pts = np.vstack([objects[t].points for t in idxs])
        cols = np.vstack([objects[t].colors for t in idxs])
        xs = [objects[t].bbox_2d[0] for t in idxs if objects[t].bbox_2d is not None]
        ys = [objects[t].bbox_2d[1] for t in idxs if objects[t].bbox_2d is not None]
        xe = [objects[t].bbox_2d[0] + objects[t].bbox_2d[2] for t in idxs if objects[t].bbox_2d is not None]
        ye = [objects[t].bbox_2d[1] + objects[t].bbox_2d[3] for t in idxs if objects[t].bbox_2d is not None]
        bbox = (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)) if xs else None
        heights = [objects[t].height_above_table_m for t in idxs]
        h_val = float(np.nanmax(heights)) if np.any(np.isfinite(heights)) else float("nan")
        merged.append(SceneObject(
            obj_id=k,
            points=pts,
            colors=cols,
            centroid=pts.mean(axis=0),
            aabb_min=pts.min(axis=0),
            aabb_max=pts.max(axis=0),
            touches_arm=any(objects[t].touches_arm for t in idxs),
            arm_depth_delta_m=min(objects[t].arm_depth_delta_m for t in idxs),
            height_above_table_m=h_val,
            bbox_2d=bbox,
        ))
    return merged


def detect_scene_objects_2d(
    points: np.ndarray,
    colors: np.ndarray,
    valid_flat: np.ndarray,
    manipulator_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cfg,
    color_bgr: Optional[np.ndarray] = None,
    table_plane: Optional[np.ndarray] = None,
    roi_rect: Optional[Tuple[int, int, int, int]] = None,
) -> DetectionReport:
    """
    Detect scene objects from 2D valid-depth mask minus manipulator (like arm segmentation).
    Maps each component to 3D points via valid_flat for collision distance.

    When `color_bgr` is given and `cfg.collision_obj_color_gate` is True, the
    candidate mask is further restricted to *contrasting* pixels — either bright
    (white box) or colour-saturated (green cube) — so the dark, desaturated
    table surface is rejected instead of being detected as a foreign object.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    obj_mask_u8 = np.zeros((h, w), dtype=np.uint8)

    if len(points) == 0 or valid_flat.size != h * w:
        return DetectionReport(objects=[], mask=obj_mask_u8)

    valid_2d = valid_flat.reshape(h, w)
    # Исключаем РАЗДУТУЮ маску руки: тонкий ореол краёв/деталей самой руки (белый
    # пластик KUKA) иначе проходит как «чужой объект» вплотную → ложный DANGER.
    manip_excl = manipulator_mask
    ex_px = int(cfg.collision_obj_manip_exclude_dilate_px)
    if ex_px > 0 and np.count_nonzero(manipulator_mask) > 0:
        k_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ex_px + 1, 2 * ex_px + 1))
        manip_excl = cv2.dilate(manipulator_mask, k_ex, iterations=1)
    obj_mask = valid_2d & (manip_excl == 0)

    # Гейт кандидатов — что считать «объектом». Приоритет: ГЕОМЕТРИЯ (высота над
    # калиброванной плоскостью стола, цвето-независимо). Цвет-гейт — фолбэк, если
    # плоскость не задана. `contrast` — итоговая маска «пиксель может быть
    # объектом»; используется ниже в reclaim-базе (обе ветки её заполняют).
    contrast: Optional[np.ndarray] = None
    use_plane = (
        table_plane is not None
        and bool(cfg.collision_obj_plane_gate)
    )
    if use_plane:
        a, b, c, d = (float(v) for v in np.asarray(table_plane).ravel()[:4])
        height = plane_height(points, (a, b, c, d))
        h_img = np.full(h * w, -np.inf, dtype=np.float32)
        h_img[valid_flat] = height.astype(np.float32)
        h_img = h_img.reshape(h, w)
        hmin = float(cfg.collision_obj_plane_height_min_m)
        hmax = float(cfg.collision_obj_plane_height_max_m)
        contrast = (h_img >= hmin) & (h_img <= hmax)
        # Вычесть известный синий пневмошланг робота: он выше стола (проходит
        # высотный гейт), но это деталь робота, а не объект. Это НЕ детекция по
        # цвету — а удаление заведомо известной детали.
        if (
            color_bgr is not None
            and bool(cfg.collision_obj_exclude_blue)
            and color_bgr.shape[:2] == (h, w)
        ):
            hsv_b = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            hue_b, sat_b = hsv_b[:, :, 0], hsv_b[:, :, 1]
            hue_lo = int(cfg.collision_obj_blue_hue_lo)
            hue_hi = int(cfg.collision_obj_blue_hue_hi)
            sat_min = int(cfg.collision_obj_sat_min)
            is_blue = (hue_b >= hue_lo) & (hue_b <= hue_hi) & (sat_b >= sat_min)
            contrast = contrast & (~is_blue)
        obj_mask = obj_mask & contrast
    elif (
        color_bgr is not None
        and bool(cfg.collision_obj_color_gate)
        and color_bgr.shape[:2] == (h, w)
    ):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        bright_min = int(cfg.collision_obj_bright_min)
        sat_min = int(cfg.collision_obj_sat_min)
        colored = sat >= sat_min
        # Drop the blue pneumatic hose: saturated-but-blue pixels are not objects.
        if bool(cfg.collision_obj_exclude_blue):
            hue_lo = int(cfg.collision_obj_blue_hue_lo)
            hue_hi = int(cfg.collision_obj_blue_hue_hi)
            is_blue = (hue >= hue_lo) & (hue <= hue_hi)
            colored = colored & (~is_blue)
        contrast = (gray >= bright_min) | colored
        obj_mask = obj_mask & contrast

    # Reclaim-база: те же фильтры, но вычитается ТОЛЬКО сырая маска руки (без
    # дилатации). Зона исключения нужна лишь для детекции (ореол руки ≠ объект);
    # для измерения дистанции найденной компоненте возвращаются её пиксели,
    # съеденные зоной, — иначе расстояние завышено на ширину зоны (~3 см) и
    # DANGER не наступает при подходе руки.
    # ROI рабочей зоны (пиксельный прямоугольник калиброванного стола): камера
    # неподвижна → зона фиксирована и НЕ зависит от T_cr. Всё вне ROI (провод,
    # база робота, фон за столом, дальние выбросы) в детекцию не попадает.
    roi_bool: Optional[np.ndarray] = None
    if roi_rect is not None:
        rx0, ry0, rx1, ry1 = (int(v) for v in roi_rect)
        rx0 = max(0, min(rx0, w)); rx1 = max(0, min(rx1, w))
        ry0 = max(0, min(ry0, h)); ry1 = max(0, min(ry1, h))
        if rx1 > rx0 and ry1 > ry0:
            roi_bool = np.zeros((h, w), dtype=bool)
            roi_bool[ry0:ry1, rx0:rx1] = True
            obj_mask = obj_mask & roi_bool

    reclaim_base: Optional[np.ndarray] = None
    if bool(cfg.collision_obj_reclaim_near_arm) and ex_px > 0:
        reclaim_base = valid_2d & (manipulator_mask == 0)
        if contrast is not None:
            reclaim_base = reclaim_base & contrast
        if roi_bool is not None:
            reclaim_base = reclaim_base & roi_bool

    obj_mask_u8 = (obj_mask.astype(np.uint8) * 255)

    open_px = int(cfg.collision_obj_open_px)
    close_px = int(cfg.collision_obj_close_px)
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

    # Кольцо near_manip нужно только в цвет-режиме (ограничить поиск зоной у руки).
    # В плоскостном режиме детекция идёт по всему столу — кольцо выключаем.
    use_near = bool(cfg.collision_obj_use_near_manip) and not use_plane
    if use_near and np.count_nonzero(manipulator_mask) > 0:
        ring_px = int(cfg.collision_obj_near_manip_dilate_px)
        near_ring = build_near_manip_ring_mask(manipulator_mask, ring_px)
        obj_mask_u8[near_ring == 0] = 0

    # Consolidate depth-holed fragments of one object into a solid blob.
    cons_px = int(cfg.collision_obj_consolidate_px)
    if cons_px > 1:
        k_cons = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cons_px, cons_px))
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_CLOSE, k_cons)
    # Вертикальный close: сшить швы между кубиками СТОПКИ по вертикали (узкое по
    # X ядро не склеивает соседние объекты по горизонтали) → бокс на всю высоту.
    vclose = int(cfg.collision_obj_vclose_px)
    if vclose > 1:
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vclose))
        obj_mask_u8 = cv2.morphologyEx(obj_mask_u8, cv2.MORPH_CLOSE, k_v)

    z_arm: Optional[float] = None
    if np.count_nonzero(manipulator_mask) > 0:
        arm_sel = (manipulator_mask > 0).reshape(-1)[valid_flat]
        if np.any(arm_sel):
            z_arm = float(np.median(points[arm_sel][:, 2]))
    max_z_behind = float(cfg.collision_obj_max_depth_behind_arm_m)

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        obj_mask_u8, connectivity=8
    )
    min_area = int(cfg.collision_obj_min_area_px)
    max_area = int(float(cfg.collision_obj_max_area_frac) * h * w)
    border_px = int(cfg.collision_obj_exclude_border_px)
    drop_border = bool(cfg.collision_obj_drop_border)
    min_pts_3d = int(cfg.collision_obj_min_points_3d)
    min_fill = float(cfg.collision_obj_min_fill_ratio)
    max_aspect = float(cfg.collision_obj_max_aspect)

    # Карта глубины по пикселям (NaN вне valid) — для признаков протечки руки:
    # сравнение глубины блоба с локальной глубиной руки рядом с ним.
    z_img: Optional[np.ndarray] = None
    if bool(cfg.collision_obj_arm_depth_reject):
        z_flat = np.full(h * w, np.nan, dtype=np.float64)
        z_flat[valid_flat] = points[:, 2]
        z_img = z_flat.reshape(h, w)

    objects: List[SceneObject] = []
    obj_id = 0
    n_kept = 0
    # Диагностика: почему компоненты отбрасываются (какой гейт убил объект у руки).
    _rej = {"area": 0, "shape": 0, "border": 0, "pts3d": 0, "keepz": 0, "span": 0}
    # Пообъектные записи для objects.csv (A4). Пишем только под флагом: иначе это
    # список словарей на каждый блоб каждого кадра — в горячем цикле не нужен.
    _log_cands = bool(cfg.enable_objects_log)
    _cands: List[dict] = []

    def _note(reason: str, area_px, height_m=None) -> None:
        """Записать ОТБРАКОВАННОГО кандидата и гейт, который его убил."""
        _cands.append({
            "obj_id": "",
            "area_px": int(area_px),
            "height_m": height_m,
            "reject_reason": reason,
        })

    for lab in range(1, n_comp):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            _rej["area"] += 1
            if _log_cands:
                _note("area", area)
            continue
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        # Shape filter: compact, roughly box-like blobs only (reject hose/noise).
        fill_ratio = float(area) / float(max(1, bw * bh))
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        if fill_ratio < min_fill or aspect > max_aspect:
            _rej["shape"] += 1
            if _log_cands:
                _note("shape", area)
            continue
        if drop_border:
            if (
                x0 <= border_px
                or y0 <= border_px
                or (x0 + bw) >= (w - border_px)
                or (y0 + bh) >= (h - border_px)
            ):
                _rej["border"] += 1
                if _log_cands:
                    _note("border", area)
                continue

        comp = labels == lab
        # Reclaim: вернуть компоненте пиксели, съеденные зоной исключения руки
        # (дилатация компоненты на ex_px ∩ reclaim-база). Форма/площадь выше
        # проверялись по усечённой компоненте, а 3D-точки для дистанции берём
        # из полной — ближайший к руке край объекта снова участвует в метрике.
        if reclaim_base is not None:
            comp_dil = _dilate_mask_in_bbox(
                comp, x0, y0, bw, bh, 2 * ex_px + 1, ex_px + 1
            )
            comp = comp | (comp_dil & reclaim_base)
        # Erode the component before sampling 3D points: boundary pixels carry
        # mixed depth (object vs table/arm) and would corrupt the distance.
        erode_px = int(cfg.collision_obj_erode_px)
        comp_sample = comp
        if erode_px > 1:
            k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
            eroded = cv2.erode(comp.astype(np.uint8), k_er) > 0
            if np.count_nonzero(eroded.reshape(-1)[valid_flat]) >= min_pts_3d:
                comp_sample = eroded
        sel = comp_sample.reshape(-1)[valid_flat]
        if not np.any(sel):
            _rej["pts3d"] += 1
            if _log_cands:
                _note("pts3d", area)
            continue
        obj_pts = points[sel]
        obj_col = colors[sel]
        if len(obj_pts) < min_pts_3d:
            _rej["pts3d"] += 1
            if _log_cands:
                _note("pts3d", area)
            continue

        z = obj_pts[:, 2]
        z_lo = float(np.percentile(z, 2))
        z_hi = float(np.percentile(z, 98))
        keep_z = (z >= z_lo) & (z <= z_hi)
        if z_arm is not None and max_z_behind > 0:
            keep_z &= obj_pts[:, 2] < (z_arm + max_z_behind)
        if np.count_nonzero(keep_z) < min_pts_3d:
            _rej["keepz"] += 1
            if _log_cands:
                _note("keepz", area)
            continue
        obj_pts = obj_pts[keep_z]
        obj_col = obj_col[keep_z]
        # Прореживание точек объекта для быстрой дистанции: полный пиксельный
        # кластер — тысячи точек, min_dist O(N·K) на кадр дорог. Воксель 1 см даёт
        # ошибку дистанции <= 1 см (порог DANGER 8 см) и консистентен с body 2 см.
        dist_voxel = float(cfg.collision_obj_dist_voxel_m)
        if dist_voxel > 0.0 and len(obj_pts) > min_pts_3d:
            obj_pts_ds = voxel_down(obj_pts, dist_voxel, REPRESENTATIVE)
            if len(obj_pts_ds) >= min_pts_3d:
                obj_pts = obj_pts_ds
                obj_col = np.zeros((len(obj_pts_ds), obj_col.shape[1]), dtype=obj_col.dtype)

        # Признаки протечки руки: прилип ли блоб к маске руки и совпадает ли
        # его глубина с локальной глубиной руки рядом. Решение об отбраковке
        # принимает CollisionService при создании трека (не здесь), чтобы уже
        # подтверждённый объект не «исчезал» при подходе руки вплотную.
        touches_arm = False
        arm_depth_delta = float("inf")
        if np.count_nonzero(manipulator_mask) > 0:
            t_px = ex_px + 5
            near_arm = _dilate_mask_in_bbox(
                comp, x0, y0, bw, bh, 2 * t_px + 1, ex_px + t_px + 1
            ) & (manipulator_mask > 0)
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
        obj_height = float("nan")
        if use_plane:
            hgt = plane_height(obj_pts, (a, b, c, d))
            obj_height = float(np.median(hgt))
            # Гейт вертикальной структуры против фантомов на пустом столе
            # (ДРЕЙФ-НЕЗАВИСИМ): у реального кубика точки тянутся по высоте (видны
            # верхняя грань + боковые), размах p90−p10 большой; у плоского шума
            # или дрейфа плоскости все точки на ~одном уровне → размах мал.
            # Абсолютный порог высоты тут не годится: при дрейфе плоскости шум
            # «поднимается» и проходит, а размах остаётся малым.
            min_span = float(cfg.collision_obj_plane_min_span_m)
            if min_span > 0.0 and len(hgt) >= 5:
                span = float(np.percentile(hgt, 90) - np.percentile(hgt, 10))
                if span < min_span:
                    _rej["span"] += 1
                    if _log_cands:
                        _note("span", area, obj_height)
                    continue
        # Принятые кандидаты здесь НЕ записываем: строку по ним пишет
        # CollisionService уже с финальным obj_id, дистанцией и уровнем — иначе
        # в objects.csv один и тот же объект появлялся бы дважды.
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
                height_above_table_m=obj_height,
                bbox_2d=(x0, y0, bw, bh),
            )
        )
        obj_id += 1
        n_kept += 1

    # Склейка перекрывающихся боксов вынесена ПОСЛЕ трекера (collision_service):
    # там дедуп ловит и фрагменты детекции, и призраков-коастинг трекера, и не
    # дёргает центроид детекции (иначе плодятся дубли-треки).

    # Печать диагностики — задача вызывающего: у него есть номер кадра для
    # троттлинга, а детектор остаётся чистой функцией от входа.
    return DetectionReport(
        objects=objects,
        mask=obj_mask_u8,
        candidates=_cands,
        rejects=dict(_rej),
        n_components=max(0, n_comp - 1),
    )


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
    body_ds = voxel_down(body_points, body_voxel_m, CENTROID)

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
