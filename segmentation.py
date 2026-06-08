import cv2
import numpy as np


def segment_white_robot(color_bgr, brightness_threshold=160, min_area_ratio=0.02):
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros(color_bgr.shape[:2], dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    total_pixels = color_bgr.shape[0] * color_bgr.shape[1]
    if stats[largest_idx, cv2.CC_STAT_AREA] < min_area_ratio * total_pixels:
        return np.zeros(color_bgr.shape[:2], dtype=bool)
    return labels == largest_idx


def build_geometry_robot_mask(points, x_bounds, y_bounds, z_bounds):
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    in_x = (points[:, 0] >= x_bounds[0]) & (points[:, 0] <= x_bounds[1])
    in_y = (points[:, 1] >= y_bounds[0]) & (points[:, 1] <= y_bounds[1])
    in_z = (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1])
    return in_x & in_y & in_z


def temporal_mask_filter(mask, state, alpha, threshold):
    if state is None or state.shape != mask.shape:
        state = mask.astype(np.float32)
    else:
        state = alpha * state + (1.0 - alpha) * mask.astype(np.float32)
    return state > threshold, state


def expand_marker_mask(mask, dilate_px):
    if dilate_px <= 0:
        return mask
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def split_cloud_by_mask(points, colors, valid_flat, manipulator_mask):
    flat_mask = manipulator_mask.reshape(-1)[valid_flat]
    manip_pts = points[flat_mask]
    manip_col = colors[flat_mask]
    scene_pts = points[~flat_mask]
    scene_col = colors[~flat_mask]
    return (manip_pts, manip_col), (scene_pts, scene_col)


def split_cloud_by_point_mask(points, colors, point_mask):
    manip_pts = points[point_mask]
    manip_col = colors[point_mask]
    scene_pts = points[~point_mask]
    scene_col = colors[~point_mask]
    return (manip_pts, manip_col), (scene_pts, scene_col)


def project_markers_to_3d(color_bgr, aruco_dict_id, marker_length_px_min, points, valid_flat, image_shape):
    corners, ids = _detect_aruco(color_bgr, aruco_dict_id)
    seed_indices = []
    for corner in corners:
        pts = corner.astype(np.int32).reshape(-1, 2)
        edge = np.linalg.norm(pts[0] - pts[1])
        if edge < marker_length_px_min:
            continue
        center_u = int(pts[:, 0].mean())
        center_v = int(pts[:, 1].mean())
        if 0 <= center_v < image_shape[0] and 0 <= center_u < image_shape[1]:
            flat_idx = center_v * image_shape[1] + center_u
            if flat_idx < len(valid_flat) and valid_flat[flat_idx]:
                point_idx = np.sum(valid_flat[:flat_idx])
                if point_idx < len(points):
                    seed_indices.append(point_idx)
    return seed_indices


def region_grow_from_seeds(points, seed_indices, radius_m):
    if len(points) == 0 or len(seed_indices) == 0:
        return np.zeros(len(points), dtype=bool)
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    mask = np.zeros(len(points), dtype=bool)
    for seed_idx in seed_indices:
        if seed_idx >= len(points):
            continue
        [_, idx, _] = tree.search_radius_vector_3d(cloud.points[seed_idx], radius_m)
        mask[idx] = True
    return mask


def _detect_aruco(color_bgr, aruco_dict_id):
    if not hasattr(cv2, "aruco"):
        return [], None
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(color_bgr)
    return corners, ids
