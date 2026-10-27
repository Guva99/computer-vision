"""
Общая настройка тестов.

Корень проекта кладём в sys.path, чтобы тесты импортировали модули так же, как
их импортирует приложение (`import collision`, `from app.pipeline import ...`),
без установки пакета.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
