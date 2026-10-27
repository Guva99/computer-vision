"""
Реализации сегментации манипулятора и выбор одной из них по конфигу.

Добавить новую (например, категорийно-независимую нейросетевую) — значит:
  1. написать класс с методом `segment(SegmentationInput) -> Masks`;
  2. зарегистрировать его в `_SEGMENTERS` под своим именем;
  3. поставить `segmenter = "<имя>"` в конфиге.

Конвейер при этом не правится — это и есть проверяемое условие OCP
(tests/test_segmenter_contract.py). Прежняя реализация остаётся доступной и
служит фолбэком: если новая не загрузилась или дала пустую маску, контур
безопасности обязан продолжать работать, а не ослепнуть.
"""
from vision.segmenters.base import Masks, SegmentationInput, SegmenterPort
from vision.segmenters.brightness import BrightnessSegmenter

# Имя в конфиге → класс. Имена стабильны: они попадают в мету прогона и в логи,
# по ним потом сверяются результаты в статье.
_SEGMENTERS = {
    "brightness": BrightnessSegmenter,
}

DEFAULT_SEGMENTER = "brightness"


def available_segmenters() -> tuple:
    """Имена зарегистрированных реализаций (для сообщений об ошибке и отчётов)."""
    return tuple(sorted(_SEGMENTERS))


def register_segmenter(name: str, factory) -> None:
    """Зарегистрировать реализацию под именем (используется тестами и плагинами)."""
    _SEGMENTERS[str(name)] = factory


def build_segmenter(cfg) -> SegmenterPort:
    """Создать сегментатор по `cfg.segmenter`.

    Неизвестное имя — ошибка на старте, а не тихий откат на яркостную ветвь:
    опечатка в конфиге не должна оставлять прогон с не тем алгоритмом, по
    которому потом будут отчитываться метрики.
    """
    name = str(getattr(cfg, "segmenter", DEFAULT_SEGMENTER) or DEFAULT_SEGMENTER)
    if name not in _SEGMENTERS:
        raise ValueError(
            f"Неизвестный сегментатор {name!r}. Доступны: {available_segmenters()}"
        )
    return _SEGMENTERS[name](cfg)


__all__ = [
    "Masks",
    "SegmentationInput",
    "SegmenterPort",
    "BrightnessSegmenter",
    "build_segmenter",
    "register_segmenter",
    "available_segmenters",
    "DEFAULT_SEGMENTER",
]
