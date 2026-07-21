"""
Роутер режима обработки камеры (2D/3D).
"""


class ProcessingModeRouter:
    """
    Выбирает активный режим обработки:
    - 3D (point cloud)
    - 2D (legacy fallback)
    """

    def __init__(self, is_cloud_point: bool = True):
        self.is_cloud_point = is_cloud_point

    def use_3d(self) -> bool:
        """Возвращает True, если активен режим 3D."""
        return self.is_cloud_point
