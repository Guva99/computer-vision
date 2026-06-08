import numpy as np
import open3d as o3d


def depth_to_color(points, depth_min=0.3, depth_max=1.5):
    """
    Конвертирует глубину точек в цвет (heatmap).
    Синий = близко, Красный = далеко.
    
    Args:
        points: Nx3 array точек в метрах
        depth_min: минимальная глубина (м)
        depth_max: максимальная глубина (м)
    
    Returns:
        Nx3 array цветов (0-1)
    """
    if len(points) == 0:
        return np.zeros((0, 3))
    
    # Z координата = глубина от камеры
    depths = points[:, 2]
    
    # Нормализация глубины в диапазон 0-1
    normalized = (depths - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0, 1)
    
    # Создаем heatmap colormap
    # 0.0 (близко) -> синий [0, 0, 1]
    # 0.5 (средне) -> зеленый [0, 1, 0]
    # 1.0 (далеко) -> красный [1, 0, 0]
    colors = np.zeros((len(points), 3))
    
    # Синий -> Зеленый (0.0 -> 0.5)
    mask1 = normalized <= 0.5
    t = normalized[mask1] * 2.0
    colors[mask1, 2] = 1.0 - t  # Синий уменьшается
    colors[mask1, 1] = t         # Зеленый увеличивается
    
    # Зеленый -> Красный (0.5 -> 1.0)
    mask2 = normalized > 0.5
    t = (normalized[mask2] - 0.5) * 2.0
    colors[mask2, 1] = 1.0 - t  # Зеленый уменьшается
    colors[mask2, 0] = t         # Красный увеличивается
    
    return colors


def build_cloud_arrays(color_bgr, depth, intrinsics, depth_scale, depth_min_m, depth_max_m):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth.astype(np.float32) * depth_scale
    valid = (z > depth_min_m) & (z < depth_max_m)
    x = (u - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.stack((x, y, z), axis=-1)[valid]
    colors_rgb = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    colors = colors_rgb[valid]
    return points, colors, valid.reshape(-1)


def filter_cloud(points, colors, voxel_size, nb_neighbors, std_ratio, use_statistical_outlier=True):
    if len(points) == 0:
        return points, colors
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    if use_statistical_outlier and len(cloud.points) >= nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def to_open3d_cloud(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def remove_table_plane(points, colors, ransac_dist, min_height_above_plane):
    if len(points) < 200:
        return points, colors
    cloud = to_open3d_cloud(points, colors)
    plane_model, inliers = cloud.segment_plane(distance_threshold=ransac_dist, ransac_n=3, num_iterations=250)
    if len(inliers) == 0:
        return points, colors
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float32)
    norm = np.linalg.norm(normal) + 1e-8
    distances = (points @ normal + d) / norm
    object_mask = distances > min_height_above_plane
    return points[object_mask], colors[object_mask]


def keep_dbscan_objects(points, colors, eps_m, min_points, object_min_points):
    if len(points) < max(min_points, object_min_points):
        return points[:0], colors[:0], 0
    cloud = to_open3d_cloud(points, colors)
    labels = np.array(cloud.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return points[:0], colors[:0], 0
    kept = np.zeros(labels.shape[0], dtype=bool)
    clusters = 0
    for label in np.unique(labels):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if idx.size >= object_min_points:
            kept[idx] = True
            clusters += 1
    return points[kept], colors[kept], clusters
