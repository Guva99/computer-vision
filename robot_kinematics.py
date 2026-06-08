# """
# Forward kinematics и collision volumes для KUKA KR6.
# """
# import numpy as np


# class KukaKR6Kinematics:
#     """Упрощенная кинематика для KUKA KR6."""
    
#     def __init__(self):
#         # DH параметры для KUKA KR6 (упрощенные, в мм)
#         # [a, alpha, d, theta_offset]
#         self.dh_params = [
#             [25, -90, 400, 0],      # Joint 1
#             [455, 0, 0, -90],        # Joint 2
#             [35, 90, 0, 90],         # Joint 3
#             [0, -90, 420, 0],        # Joint 4
#             [0, 90, 0, 0],           # Joint 5
#             [0, 0, 80, 0],           # Joint 6 (flange)
#         ]
        
#         # Радиусы для collision volumes (мм) — после калибровки камера↔база можно ужесточить
#         self.link_radii = [160, 130, 110, 95, 75, 65]

#         # Названия звеньев для визуализации
#         self.link_names = ["A1-Base", "A2-Shoulder", "A3-Elbow", "A4-Wrist1", "A5-Wrist2", "A6-Flange"]
        
#     def dh_matrix(self, a, alpha, d, theta):
#         """Строит матрицу DH-преобразования."""
#         ca = np.cos(np.radians(alpha))
#         sa = np.sin(np.radians(alpha))
#         ct = np.cos(np.radians(theta))
#         st = np.sin(np.radians(theta))
        
#         return np.array([
#             [ct, -st*ca, st*sa, a*ct],
#             [st, ct*ca, -ct*sa, a*st],
#             [0, sa, ca, d],
#             [0, 0, 0, 1]
#         ])
    
#     def forward_kinematics(self, joint_angles_deg):
#         """
#         Вычисляет позиции всех звеньев.
        
#         Args:
#             joint_angles_deg: Углы суставов в градусах (A1-A6)
            
#         Returns:
#             List of 4x4 transformation matrices (base -> link_i)
#         """
#         transforms = []
#         T = np.eye(4)
        
#         for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
#             theta = joint_angles_deg[i] + theta_offset
#             T_i = self.dh_matrix(a, alpha, d, theta)
#             T = T @ T_i
#             transforms.append(T.copy())
        
#         return transforms
    
#     def get_link_endpoints(self, joint_angles_deg):
#         """
#         Получает координаты начала и конца каждого звена.
        
#         Returns:
#             List of tuples (start_pos, end_pos) в мм
#         """
#         transforms = self.forward_kinematics(joint_angles_deg)
        
#         endpoints = []
#         prev_pos = np.array([0, 0, 0])
        
#         for T in transforms:
#             pos = T[:3, 3]
#             endpoints.append((prev_pos.copy(), pos.copy()))
#             prev_pos = pos.copy()
        
#         return endpoints
    
#     def build_collision_spheres(self, joint_angles_deg, num_spheres_per_link=10):
#         """
#         Строит сферы вдоль каждого звена для collision detection.
        
#         Returns:
#             List of (center, radius) в мм
#         """
#         endpoints = self.get_link_endpoints(joint_angles_deg)
#         spheres = []
        
#         for i, ((start, end), radius) in enumerate(zip(endpoints, self.link_radii)):
#             # Больше сфер = лучше покрытие при поворотах
#             for j in range(num_spheres_per_link):
#                 t = j / (num_spheres_per_link - 1) if num_spheres_per_link > 1 else 0.5
#                 center = start + t * (end - start)
#                 spheres.append((center, radius))
        
#         return spheres
    
#     def point_in_collision_volumes(self, points_mm, joint_angles_deg, use_workspace_filter=False):
#         """
#         Проверяет какие точки находятся внутри collision volumes манипулятора.

#         ВАЖНО: координаты FK — в системе базы робота (мм), точки RealSense — в системе камеры (м).
#         Без внешней калибровки (4x4 база_робота -> камера) маска по сферам не совпадёт с облаком.
#         Раньше здесь был OR по «рабочей зоне» и цилиндр базы в (0,0) — это ошибочно помечало полсцены.

#         Args:
#             points_mm: Nx3 array точек в миллиметрах (камера)
#             joint_angles_deg: Углы суставов в градусах
#             use_workspace_filter: не используется (оставлен для совместимости API)

#         Returns:
#             Boolean mask (True = точка внутри сфер модели в текущих координатах)
#         """
#         if len(points_mm) == 0:
#             return np.zeros(0, dtype=bool)

#         mask = np.zeros(len(points_mm), dtype=bool)
#         spheres = self.build_collision_spheres(joint_angles_deg)

#         for center, radius in spheres:
#             distances = np.linalg.norm(points_mm - center, axis=1)
#             mask |= distances < radius

#         return mask
    
#     def get_link_masks(self, points_mm, joint_angles_deg, num_spheres_per_link=10):
#         """
#         Получает отдельные маски для каждого звена.
        
#         Returns:
#             List of 6 boolean masks, по одной для каждого звена A1-A6
#         """
#         if len(points_mm) == 0:
#             return [np.zeros(0, dtype=bool) for _ in range(6)]
        
#         endpoints = self.get_link_endpoints(joint_angles_deg)
#         link_masks = []
        
#         for i, ((start, end), radius) in enumerate(zip(endpoints, self.link_radii)):
#             mask = np.zeros(len(points_mm), dtype=bool)
#             # Больше сфер на звено для полного покрытия
#             for j in range(num_spheres_per_link):
#                 t = j / (num_spheres_per_link - 1) if num_spheres_per_link > 1 else 0.5
#                 center = start + t * (end - start)
#                 distances = np.linalg.norm(points_mm - center, axis=1)
#                 mask |= (distances < radius)
#             link_masks.append(mask)
        
#         return link_masks
