"""
Архитектурные правила как падающий тест («храповик»).

Текущее состояние зафиксировано в tests/arch_baseline.json. Тест не требует,
чтобы нарушений не было прямо сейчас — он требует, чтобы их не становилось
БОЛЬШЕ. Это и позволяет чистить архитектуру шагами, не останавливая работу, и
при этом не даёт откатиться назад: новый 500-строчный файл, новый
`getattr(cfg, ..., default)` или импорт зрения в слой робота уронят сборку.

После каждого шага плана, который снижает числа, базовая линия переснимается:

    python tools/arch_metrics.py --update

Правила слоёв и пределы размеров — в tools/arch_metrics.py.
"""
import json
from pathlib import Path

import pytest

from tools.arch_metrics import BASELINE, scan

pytestmark = pytest.mark.arch


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip(f"нет базовой линии {BASELINE} — запусти tools/arch_metrics.py --update")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current():
    return scan()


def test_no_new_layer_violations(baseline, current):
    """DIP: слой робота не зависит от зрения, ядро — ни от CV, ни от SDK камеры."""
    allowed = set(baseline["layer_violations"])
    new = sorted(set(current["layer_violations"]) - allowed)
    assert not new, (
        "Появились новые нарушения слоёв:\n  " + "\n  ".join(new) +
        "\nЗависимости идут только вниз (см. docs/REFACTORING.md)."
    )


def test_no_new_oversized_files(baseline, current):
    """SRP: новых файлов-монолитов не появляется, старые не растут."""
    base = baseline["oversized_files"]
    problems = []
    for path, lines in sorted(current["oversized_files"].items()):
        if path not in base:
            problems.append(f"{path}: {lines} строк — новый файл сверх предела")
        elif lines > base[path]:
            problems.append(f"{path}: {base[path]} → {lines} строк (рост)")
    assert not problems, "\n".join(problems)


def test_no_new_oversized_functions(baseline, current):
    """SRP/KISS: функции не растут и новых монстров не появляется."""
    base = baseline["oversized_functions"]
    problems = []
    for key, (lines, args) in sorted(current["oversized_functions"].items()):
        if key not in base:
            problems.append(f"{key}: {lines} строк, {args} арг — новая функция сверх предела")
            continue
        b_lines, b_args = base[key]
        if lines > b_lines or args > b_args:
            problems.append(
                f"{key}: {b_lines} строк/{b_args} арг → {lines}/{args} (рост)")
    assert not problems, "\n".join(problems)


def test_getattr_cfg_defaults_do_not_grow(baseline, current):
    """DRY: дефолт параметра живёт в одном месте — в AppConfig.

    `getattr(cfg, "x", 150)` создаёт второй источник правды. На 2026-10-20 таких
    мест 127, и 16 из них УЖЕ разошлись со значением в конфиге. Шаг S1.1 сводит
    их к нулю; до тех пор число обязано только убывать.
    """
    assert current["getattr_cfg_sites"] <= baseline["getattr_cfg_sites"], (
        f"getattr(cfg, ..., default): было {baseline['getattr_cfg_sites']}, "
        f"стало {current['getattr_cfg_sites']}. Новые дефолты в коде не добавляем — "
        f"значение берётся из AppConfig."
    )


def test_bool_flags_do_not_grow(baseline, current):
    """KISS: бюджет сложности. Новый флаг = новая непроверенная ветвь поведения.

    55 флагов — это 3.6e16 формально возможных конфигураций, из которых измерена
    ровно одна. Флаг добавляется только вместе с регрессией или тестом,
    покрывающим его ветвь (см. «Бюджет сложности» в docs/REFACTORING.md).
    """
    assert current["bool_flags"] <= baseline["bool_flags"], (
        f"булевых флагов AppConfig: было {baseline['bool_flags']}, "
        f"стало {current['bool_flags']}"
    )


def test_core_layer_contract_when_it_appears():
    """Когда появится core/ — он обязан импортироваться без cv2/o3d/SDK.

    Пока каталога нет, тест пропускается. После шага S2.1 он начнёт реально
    проверять границу ядра.
    """
    core = Path(__file__).resolve().parents[1] / "core"
    if not core.is_dir():
        pytest.skip("core/ ещё не выделен (шаг S2.1)")
    import subprocess
    import sys
    code = (
        "import sys, importlib, pathlib;"
        "[importlib.import_module('core.' + p.stem)"
        " for p in pathlib.Path('core').glob('*.py') if p.stem != '__init__'];"
        "bad = [m for m in ('cv2', 'open3d', 'pyrealsense2') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(core.parent),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert not out.stdout.strip(), f"core/ подтянул запрещённые модули: {out.stdout.strip()}"
