"""
Контрактный тест порта сегментации (LSP) и проверка OCP.

Параметризован по ВСЕМ зарегистрированным реализациям: добавится нейросетевая —
она автоматически обязана пройти тот же набор. Это и есть смысл контракта:
подменяемость не «по договорённости», а проверяемая.

Отдельно проверяется открытость к расширению: фейковая реализация подключается
ТОЛЬКО через конфиг, без правок `app/pipeline.py`. Если однажды конвейер снова
начнёт знать внутренний протокол конкретной реализации, этот тест упадёт.
"""
import numpy as np
import pytest

from realsense_io import AppConfig
from vision.segmenters import (
    DEFAULT_SEGMENTER,
    Masks,
    SegmentationInput,
    available_segmenters,
    build_segmenter,
    register_segmenter,
)

H, W = 120, 160


class _NoFk:
    """Подсказки FK недоступны — штатная ситуация: углы приходят не каждый кадр."""
    use_fk = False
    fk_uvs = None
    joint_depths_m = None
    grip_roi_uv = None
    front_view_mode = False


def _frame(gray_level: int = 180) -> SegmentationInput:
    """Синтетический кадр: светлый блок в центре на тёмном фоне, глубина ~1 м."""
    color = np.zeros((H, W, 3), dtype=np.uint8)
    color[40:80, 60:100] = gray_level
    depth = np.zeros((H, W), dtype=np.uint16)
    depth[:, :] = 1000          # 1 м при depth_scale=0.001
    depth[40:80, 60:100] = 900  # блок ближе
    return SegmentationInput(color_bgr=color, depth=depth, depth_scale=0.001, fk=_NoFk())


@pytest.fixture(params=available_segmenters())
def segmenter(request):
    cfg = AppConfig(segmenter=request.param)
    return build_segmenter(cfg)


def test_returns_masks_of_frame_shape(segmenter):
    """Контракт: три маски формы кадра, тип uint8, значения 0/255."""
    masks = segmenter.segment(_frame())
    assert isinstance(masks, Masks)
    for name in ("robot", "manipulator", "table"):
        m = getattr(masks, name)
        assert m.shape == (H, W), f"{name}: форма {m.shape}, ожидалась {(H, W)}"
        assert m.dtype == np.uint8, f"{name}: тип {m.dtype}"
        assert set(np.unique(m)).issubset({0, 255}), f"{name}: не бинарная маска"


def test_survives_without_fk_hints(segmenter):
    """Без подсказок FK реализация обязана работать, а не падать."""
    masks = segmenter.segment(_frame())
    assert masks is not None


def test_survives_empty_and_noisy_frame(segmenter):
    """Пустой и шумный кадр не роняют контур безопасности."""
    black = SegmentationInput(color_bgr=np.zeros((H, W, 3), dtype=np.uint8),
                              depth=np.zeros((H, W), dtype=np.uint16),
                              depth_scale=0.001, fk=_NoFk())
    assert segmenter.segment(black) is not None

    rng = np.random.default_rng(0)
    noisy = SegmentationInput(
        color_bgr=rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
        depth=rng.integers(0, 2000, (H, W)).astype(np.uint16),
        depth_scale=0.001, fk=_NoFk())
    assert segmenter.segment(noisy) is not None


def test_repeated_calls_do_not_crash(segmenter):
    """Состояние между кадрами (temporal hold) не ломает повторные вызовы."""
    for _ in range(3):
        segmenter.segment(_frame())


# ── OCP: новая реализация подключается только конфигом ───────────────────────

class _FakeSegmenter:
    """Минимальная реализация порта: всё по нулям. Нужна как раз для проверки,
    что для подключения реализации достаточно конфига."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = 0

    def segment(self, frame: SegmentationInput) -> Masks:
        self.calls += 1
        h, w = frame.color_bgr.shape[:2]
        z = np.zeros((h, w), dtype=np.uint8)
        return Masks(robot=z.copy(), manipulator=z.copy(), table=z.copy())


def test_new_implementation_plugs_in_via_config_only():
    register_segmenter("_fake_for_test", _FakeSegmenter)
    seg = build_segmenter(AppConfig(segmenter="_fake_for_test"))
    assert isinstance(seg, _FakeSegmenter)
    masks = seg.segment(_frame())
    assert masks.manipulator.shape == (H, W)
    assert seg.calls == 1


def test_unknown_segmenter_fails_loudly():
    """Опечатка в имени — ошибка на старте, а не тихий откат на другой алгоритм.

    Иначе прогон пошёл бы с не тем методом, а метрики отчитывались бы как за
    заявленный.
    """
    with pytest.raises(ValueError) as e:
        build_segmenter(AppConfig(segmenter="brigthness"))
    assert "brightness" in str(e.value)


def test_default_is_brightness():
    """Дефолт — исходная реализация: поведение по умолчанию не меняется."""
    assert DEFAULT_SEGMENTER == "brightness"
    assert AppConfig().segmenter == "brightness"
