"""
Измерение архитектурных правил: запрещённые импорты, размеры, дубли дефолтов.

Принцип, который нельзя проверить скриптом, через три месяца размывается —
особенно когда правку делает уставший человек или сессия, не читавшая
docs/REFACTORING.md. Поэтому правила слоёв и пределы размеров измеряются здесь,
а tests/test_architecture.py превращает их в падающий тест.

Схема «храповик» (ratchet): текущие нарушения записаны в tests/arch_baseline.json
как разрешённый максимум. Тест падает, если нарушений стало БОЛЬШЕ или появилось
НОВОЕ — то есть рефакторинг может идти постепенно, но откатиться назад нельзя.
По мере выполнения шагов плана базовая линия переснимается и только уменьшается.

    python tools/arch_metrics.py            # показать текущее состояние
    python tools/arch_metrics.py --update   # переснять tests/arch_baseline.json
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "arch_baseline.json"

# Каталоги, которые не относятся к коду проекта.
SKIP_PARTS = {".venv", "cv_2d_method", "__pycache__", "captures", "downloads",
              ".git", "tests", "legacy"}

# ── правила слоёв целевой архитектуры (docs/REFACTORING.md) ───────────────────
# Слой → что ему запрещено импортировать. Проверяется по префиксу пути файла.
LAYER_FORBIDDEN = {
    # Ядро домена: чистый numpy. Ни CV, ни SDK камеры, ни верхних слоёв —
    # именно это делает его тестируемым без железа и дисплея.
    "core": ["cv2", "open3d", "pyrealsense2", "realsense_io", "collision",
             "gripper_detection", "vision", "robot", "app", "adapters"],
    # Манипулятор не должен зависеть от зрения (сейчас fk_service тянет collision)
    # и от модуля камеры (он тянет pyrealsense2 на верхнем уровне).
    "robot": ["vision", "app", "adapters", "collision", "gripper_detection",
              "pyrealsense2", "realsense_io", "open3d"],
    # Зрение не знает про цикл приложения.
    "vision": ["app", "pyrealsense2"],
    # Приложение не трогает SDK камеры напрямую — только через адаптер.
    "app": ["pyrealsense2"],
}

MAX_FILE_LINES = 400
MAX_FUNC_LINES = 60
MAX_FUNC_ARGS = 5

_GETATTR_CFG = re.compile(
    r'getattr\(\s*(?:self\.)?(?:cfg|config)\s*,\s*["\'][a-z_][a-z0-9_]*["\']\s*,'
)


def iter_py_files():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if SKIP_PARTS & set(rel.parts):
            continue
        yield rel, path


def layer_of(rel: Path) -> str | None:
    """Слой файла по его пути (верхнеуровневые модули слоя не имеют)."""
    first = rel.parts[0]
    return first if first in LAYER_FORBIDDEN else None


def imported_modules(tree: ast.AST) -> set:
    """Верхнеуровневые имена импортируемых модулей (`a.b.c` → `a`)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # относительный импорт внутри пакета
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def scan() -> dict:
    """Снять все метрики за один проход по файлам."""
    violations, file_sizes, func_offenders = [], {}, {}
    getattr_sites = 0

    for rel, path in iter_py_files():
        text = io.open(path, encoding="utf-8", errors="replace").read()
        lines = text.count("\n") + 1
        if lines > MAX_FILE_LINES:
            file_sizes[rel.as_posix()] = lines
        getattr_sites += len(_GETATTR_CFG.findall(text))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        layer = layer_of(rel)
        if layer:
            for mod in imported_modules(tree) & set(LAYER_FORBIDDEN[layer]):
                violations.append(f"{rel.as_posix()} -> {mod}")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            n_lines = (node.end_lineno or node.lineno) - node.lineno
            n_args = len(node.args.args) + len(node.args.kwonlyargs)
            if n_lines > MAX_FUNC_LINES or n_args > MAX_FUNC_ARGS:
                func_offenders[f"{rel.as_posix()}:{node.name}"] = [n_lines, n_args]

    return {
        "layer_violations": sorted(violations),
        "oversized_files": dict(sorted(file_sizes.items())),
        "oversized_functions": dict(sorted(func_offenders.items())),
        "getattr_cfg_sites": getattr_sites,
        "bool_flags": count_bool_flags(),
    }


def count_bool_flags() -> int:
    """Булевы переключатели в AppConfig: каждый — непроверенная ветвь поведения."""
    src = io.open(ROOT / "realsense_io.py", encoding="utf-8").read()
    if "class AppConfig" not in src:
        return 0
    body = src.split("class AppConfig")[1].split("\nclass ")[0]
    return len(re.findall(r"^\s{4}[a-z_][a-z0-9_]*\s*:\s*bool\s*=", body, re.M))


def main() -> int:
    ap = argparse.ArgumentParser(description="Архитектурные метрики проекта")
    ap.add_argument("--update", action="store_true",
                    help="переснять tests/arch_baseline.json (только уменьшая!)")
    args = ap.parse_args()

    snap = scan()
    print(f"Нарушений слоёв          : {len(snap['layer_violations'])}")
    for v in snap["layer_violations"]:
        print(f"   {v}")
    print(f"Файлов > {MAX_FILE_LINES} строк        : {len(snap['oversized_files'])}")
    for k, n in snap["oversized_files"].items():
        print(f"   {n:5d}  {k}")
    print(f"Функций > {MAX_FUNC_LINES} строк или > {MAX_FUNC_ARGS} арг: "
          f"{len(snap['oversized_functions'])}")
    for k, (ln, na) in list(snap["oversized_functions"].items())[:10]:
        print(f"   {ln:4d} строк {na:3d} арг  {k}")
    if len(snap["oversized_functions"]) > 10:
        print(f"   ... ещё {len(snap['oversized_functions']) - 10}")
    print(f"getattr(cfg, ..., default): {snap['getattr_cfg_sites']}")
    print(f"Булевых флагов AppConfig  : {snap['bool_flags']}")

    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"\nБазовая линия обновлена: {BASELINE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
