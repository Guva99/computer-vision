"""
Путь цикла манипулятора из записанных joint-позиций (A1..A6).

Позиции сняты с контроллера в осях (файл positions_*.json). Текущий канал
управления умеет только декартовы цели (FRAME X,Y,Z,A,B,C), поэтому каждую
joint-позу переводим в декартову через FK (kuka_fk.fk_tcp) — приближённо по
DH-модели KR4 R600.

ВАЖНО: FK даёт позу ФЛАНЦА в базовой системе робота (ROBROOT/WORLD). Значит
управление роботом должно идти с BASE=0 и TOOL=0, иначе координаты не совпадут.
"""
from typing import List, Tuple

import numpy as np

from kuka_fk import fk_tcp

# ── Записанные позиции (A1..A6, градусы). Меняется в основном A1. ──────────────
# home — фиксированная стартовая; затем свип слева-направо через все точки.
_HOME = ("home", (-86.93, -32.35, 126.78, 168.72, 93.57, -3.18))

# Упорядочено по A1 от крайней «право» (-161) до крайней «лево» (-15.7):
# right_4 → … → home → … → left_4. Ping-pong по этому списку = движение
# вправо-влево через все промежуточные точки.
_SWEEP: List[Tuple[str, Tuple[float, ...]]] = [
    ("right_4", (-161.22, -31.54, 126.78, 168.72, 93.57, -3.18)),
    ("right_3", (-131.49, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("right_2", (-111.62, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("right_1", (-95.63, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("home", (-86.93, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("left_1", (-71.63, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("left_2", (-59.38, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("left_3", (-42.25, -32.35, 126.78, 168.72, 93.57, -3.18)),
    ("left_4", (-15.71, -32.35, 126.78, 168.72, 93.57, -3.18)),
]


def _rot_to_kuka_abc(R: np.ndarray) -> Tuple[float, float, float]:
    """Матрица поворота 3x3 → углы KUKA A,B,C (град).

    Конвенция KUKA: R = Rz(A)·Ry(B)·Rx(C) (ZYX, intrinsic).
    """
    A = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    B = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    C = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(A), float(B), float(C)


def joints_to_cartesian(angles_deg: Tuple[float, ...]) -> List[float]:
    """A1..A6 (град) → [X, Y, Z (мм), A, B, C (град)] флаца в базе робота."""
    pos_m, T = fk_tcp(angles_deg)
    x, y, z = (pos_m * 1000.0).tolist()
    a, b, c = _rot_to_kuka_abc(T[:3, :3])
    return [x, y, z, a, b, c]


def build_home_and_sweep() -> Tuple[List[float], List[List[float]]]:
    """Вернуть (home_cartesian, [waypoints_cartesian...]) для цикла.

    home — куда встать при старте; waypoints — упорядоченный список для
    ping-pong-свипа влево-вправо.
    """
    home = joints_to_cartesian(_HOME[1])
    sweep = [joints_to_cartesian(a) for _, a in _SWEEP]
    return home, sweep


if __name__ == "__main__":
    home, sweep = build_home_and_sweep()
    print(f"HOME  A1={_HOME[1][0]:>8.2f}  ->  "
          f"X={home[0]:8.1f} Y={home[1]:8.1f} Z={home[2]:8.1f} "
          f"A={home[3]:7.2f} B={home[4]:7.2f} C={home[5]:7.2f}")
    for (name, ang), wp in zip(_SWEEP, sweep):
        print(f"{name:<8} A1={ang[0]:>8.2f}  ->  "
              f"X={wp[0]:8.1f} Y={wp[1]:8.1f} Z={wp[2]:8.1f} "
              f"A={wp[3]:7.2f} B={wp[4]:7.2f} C={wp[5]:7.2f}")
