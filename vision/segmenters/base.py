"""
Порт сегментации манипулятора: контракт, вход и набор масок.

Зачем порт. Сегментация — самый слабый узел метода: принадлежность пикселя
роботу определяется ЯРКОСТЬЮ (`manipulator_arm_gray_min = 150`), поэтому
светлый предмет рядом с белым KUKA от него неотличим. Измерено на стенде: белая
коробка 40 кадров (5.5 с) целиком лежала в маске руки и как препятствие не
существовала (docs/STAND_NOTES.md, 01.10). Замена этого узла — содержание
следующей работы, и она должна подключаться как ещё одна реализация контракта,
а не правкой конвейера.

Контракт намеренно узкий: кадр плюс подсказки FK → набор масок. Реализация не
знает ни про камеру, ни про цикл приложения, ни про то, кто её вызвал.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:  # typing.Protocol есть с 3.8; на старых интерпретаторах порт станет обычным классом
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object

    def runtime_checkable(cls):
        return cls


@dataclass
class Masks:
    """Набор бинарных масок на один кадр (uint8, 0/255).

    `manipulator` — то, что система считает роботом; именно она вычитается из
    кандидатов в препятствия, поэтому её ошибки дают и ложные срабатывания, и
    пропуски. `robot` — геометрический seed по FK-скелету, `table` — стол.
    """
    robot: np.ndarray
    manipulator: np.ndarray
    table: np.ndarray


@dataclass
class SegmentationInput:
    """Вход сегментации за кадр.

    `fk` — результат проекции FK-скелета (robot/services/fk_service.FkFrame):
    для яркостной реализации это seed и ROI хвата, для нейросетевой — готовые
    подсказки «вот здесь робот». Поэтому подсказки и передаются как данные, а не
    добываются внутри реализации.
    """
    color_bgr: np.ndarray
    depth: np.ndarray
    depth_scale: float
    fk: Any
    debug_out: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmenterPort(Protocol):
    """Контракт сегментатора манипулятора.

    Реализация обязана:
      • вернуть `Masks` с массивами формы кадра и типом uint8;
      • быть устойчивой к отсутствию подсказок FK (`fk.use_fk == False`) —
        на стенде углы суставов приходят не на каждом кадре;
      • не бросать исключения на пустом или шумном кадре: контур безопасности
        не должен падать из-за одного плохого кадра.

    Проверяется контрактным тестом, параметризованным по всем реализациям
    (tests/test_segmenter_contract.py).
    """

    def segment(self, frame: SegmentationInput) -> Masks:
        ...
