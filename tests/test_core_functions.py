"""
Юнит-тесты чистых функций восприятия и кинематики.

Это не «тесты ради покрытия»: каждый закрепляет инвариант, который легко
нарушить при рефакторинге и который регрессия на записях покажет уже как
расхождение решений — то есть долго и без объяснения причины.

Главный из них — РАЗНАЯ семантика двух воксельных функций (центроид против
реальной точки). Их легко принять за дубль и слить в одну (шаг S2.1), после
чего дистанции поедут на размер вокселя.
"""
import numpy as np
import pytest

import collision as C
from kuka_fk import fk_joints
from robot.services.joint_reader import parse_joint_angles


# ── классификация уровня (формула 13 статьи) ──────────────────────────────────

@pytest.mark.parametrize("dist, expected", [
    (0.00, "DANGER"),
    (0.08, "DANGER"),   # на пороге — опасность (<=)
    (0.0801, "WARN"),
    (0.12, "WARN"),     # на пороге — предупреждение (<=)
    (0.1201, "SAFE"),
    (1.00, "SAFE"),
    (float("inf"), "SAFE"),
])
def test_classify_level_boundaries(dist, expected):
    assert C.classify_level(dist, warn_m=0.12, danger_m=0.08) == expected


def test_level_order_is_monotonic():
    assert C.LEVEL_ORDER["SAFE"] < C.LEVEL_ORDER["WARN"] < C.LEVEL_ORDER["DANGER"]


# ── пересечение 2D-боксов (дедуп объектов и памяти) ───────────────────────────

def test_box_overlap_identical():
    iou, contain = C._box_overlap((10, 10, 20, 20), (10, 10, 20, 20))
    assert iou == pytest.approx(1.0)
    assert contain == pytest.approx(1.0)


def test_box_overlap_disjoint():
    iou, contain = C._box_overlap((0, 0, 10, 10), (100, 100, 10, 10))
    assert iou == 0.0
    assert contain == 0.0


def test_box_overlap_nested_gives_high_containment_low_iou():
    """Вложенная рамка: IoU маленький, доля меньшего внутри большего — 1.0.

    Именно на этом держится дедуп памяти (`collision_perm_dedup_contain`):
    два бокса на одном объекте опознаются по вложенности, а не по IoU.
    """
    iou, contain = C._box_overlap((0, 0, 100, 100), (40, 40, 10, 10))
    assert contain == pytest.approx(1.0)
    assert iou < 0.05


# ── воксельное прореживание: ДВЕ разные семантики ─────────────────────────────

def test_voxel_down_np_keeps_real_points():
    """numpy-версия обязана возвращать РЕАЛЬНЫЕ точки из входа, не средние."""
    pts = np.array([[0.000, 0.0, 1.0],
                    [0.004, 0.0, 1.0],
                    [0.050, 0.0, 1.0]], dtype=np.float64)
    out = C._voxel_down_np(pts, 0.01)
    assert len(out) == 2
    for p in out:
        assert any(np.allclose(p, q) for q in pts), "вернулась точка, которой не было во входе"


def test_voxel_down_o3d_averages_points():
    """o3d-версия усредняет точки вокселя — центроида во входе нет."""
    pts = np.array([[0.000, 0.0, 1.0],
                    [0.006, 0.0, 1.0]], dtype=np.float64)
    out = C._voxel_down(pts, 0.05)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(0.003, abs=1e-6)


def test_voxel_down_both_passthrough_on_zero_size():
    pts = np.random.rand(10, 3)
    assert len(C._voxel_down(pts, 0.0)) == len(pts)
    assert len(C._voxel_down_np(pts, 0.0)) == len(pts)


# ── дистанции: облако↔облако и облако↔сферы ───────────────────────────────────

def test_min_dist_point_to_cloud_simple():
    obj = np.array([[0.0, 0.0, 1.00]])
    body = np.array([[0.0, 0.0, 1.05], [0.0, 0.0, 1.20]])
    assert C.min_dist_point_to_cloud(obj, body) == pytest.approx(0.05)


def test_min_dist_point_to_cloud_percentile_ignores_single_outlier():
    """Низкий перцентиль обязан игнорировать единичный «прилипший» пиксель.

    Это защита от ложного DANGER на краевом пикселе: одна точка в 1 мм от руки
    не должна определять дистанцию всего объекта.
    """
    obj = np.vstack([np.array([[0.0, 0.0, 1.001]]),            # выброс рядом с телом
                     np.tile(np.array([[0.0, 0.0, 1.20]]), (99, 1))])
    body = np.array([[0.0, 0.0, 1.0]])
    assert C.min_dist_point_to_cloud(obj, body) == pytest.approx(0.001, abs=1e-6)
    assert C.min_dist_point_to_cloud(obj, body, percentile=5.0) > 0.1


def test_min_distance_to_robot_subtracts_radius_and_clamps():
    sphere = C.LinkSphere(center=np.array([0.0, 0.0, 1.0]), radius=0.05, part="gripper")
    far = np.array([[0.0, 0.0, 1.20]])
    dist, part, _ = C.min_distance_to_robot([sphere], far)
    assert dist == pytest.approx(0.15)
    assert part == "gripper"
    # Точка внутри сферы: дистанция не отрицательная, а ноль.
    inside = np.array([[0.0, 0.0, 1.01]])
    assert C.min_distance_to_robot([sphere], inside)[0] == 0.0


def test_min_distance_to_robot_empty_inputs():
    assert C.min_distance_to_robot([], np.zeros((3, 3)))[0] == float("inf")
    sphere = C.LinkSphere(center=np.zeros(3), radius=0.1, part="arm")
    assert C.min_distance_to_robot([sphere], np.zeros((0, 3)))[0] == float("inf")


# ── разбор ответа контроллера KUKA ────────────────────────────────────────────

def test_parse_joint_angles_e6axis():
    raw = "{E6AXIS: A1 -86.93, A2 -32.35, A3 126.78, A4 168.72, A5 93.57, A6 -3.18}"
    angles = parse_joint_angles(raw)
    assert angles == pytest.approx((-86.93, -32.35, 126.78, 168.72, 93.57, -3.18))


def test_parse_joint_angles_rejects_incomplete():
    assert parse_joint_angles("{E6AXIS: A1 10.0, A2 20.0}") is None
    assert parse_joint_angles("мусор") is None


# ── прямая кинематика ─────────────────────────────────────────────────────────

def test_fk_joints_base_at_origin_and_reach_within_limits():
    """База в нуле, фланец в пределах досягаемости KR4 R600 (R = 0.6 м)."""
    pts = fk_joints((0.0, -90.0, 90.0, 0.0, 0.0, 0.0))
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.shape[0] >= 7
    assert np.allclose(pts[0], np.zeros(3), atol=1e-9)
    radius = float(np.linalg.norm(pts[-1][:2]))
    assert radius <= 0.75, f"вылет {radius:.3f} м больше физического для KR4 R600"


def test_fk_joints_a1_rotates_around_base_axis():
    """Поворот A1 не меняет высоту звеньев — вращение вокруг вертикали.

    Это та самая особенность, из-за которой при калибровке нельзя менять только
    A1: глубина точек остаётся закреплённой (см. STAND_NOTES, 30.09).
    """
    q = [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]
    a = np.asarray(fk_joints(tuple(q)), dtype=np.float64)
    q[0] = 45.0
    b = np.asarray(fk_joints(tuple(q)), dtype=np.float64)
    assert np.allclose(a[:, 2], b[:, 2], atol=1e-9)
    assert not np.allclose(a[:, 0], b[:, 0], atol=1e-6)
