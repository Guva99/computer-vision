"""
Дымовой тест: все модули приложения импортируются.

Ловит самое частое последствие рефакторинга — поломанный импорт или цикл между
модулями, который на стенде обнаруживается только при запуске прогона (а это
уже потерянное время оператора и, возможно, сорванная серия).

Тяжёлые точки входа (`main.py`, калибровки) не импортируются: они открывают
окна и камеру. Их покрывает регрессия.
"""
import importlib

import pytest

MODULES = [
    # верхний уровень
    "realsense_io",
    "pointcloud_pipeline",
    "kuka_fk",
    "collision",
    "segmentation",
    "gripper_detection",
    # приложение
    "app.console",
    "app.decision_log",
    "app.objects_log",
    "app.perf_monitor",
    "app.pipeline",
    "app.playback",
    "app.recorder",
    "app.runner",
    "app.stop_log",
    # ядро
    "core.geometry",
    "core.kinematics",
    "core.levels",
    "core.projection",
    "core.types",
    # зрение
    "vision.collision_service",
    "vision.gripper_service",
    "vision.overlay_service",
    "vision.segmenters",
    "vision.segmenters.base",
    "vision.segmenters.brightness",
    "vision.masks",
    "vision.table_plane",
    "vision.visualizer",
    # робот
    "robot.cycle_path",
    "robot.drivers.kuka_driver",
    "robot.drivers.openshowvar",
    "robot.services.fk_service",
    "robot.services.joint_reader",
    "robot.services.robot_service",
    # инструменты
    "tools.replay",
    "tools.regress",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_appconfig_constructs():
    """AppConfig создаётся без аргументов и отдаёт ключевые пороги."""
    from realsense_io import AppConfig

    cfg = AppConfig()
    assert cfg.collision_danger_dist_m > 0
    assert cfg.collision_warn_dist_m > cfg.collision_danger_dist_m
    # Гистерезис выхода обязан быть не ниже порога входа, иначе DANGER дребезжит.
    assert cfg.collision_danger_exit_dist_m >= cfg.collision_danger_dist_m
