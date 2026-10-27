# Проверка перед коммитом: архитектурные правила, юниты, регрессия решений.
#
# Гоняется после КАЖДОГО шага рефакторинга (docs/REFACTORING.md). Если упало —
# правка меняет поведение или ломает правила слоёв, и её нужно откатить, а не
# «дочинить потом».
#
#   .\scripts\check.ps1            # быстрые тесты + регрессия на 2 записях (~1 мин)
#   .\scripts\check.ps1 -Full      # + полный golden-набор из 8 записей (~4.5 мин)
#   .\scripts\check.ps1 -FastOnly  # только быстрые тесты (~3 с)

param(
    [switch]$Full,
    [switch]$FastOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Вывод приложения — UTF-8: иначе русские сообщения падают при редиректе в cp1251.
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "`n=== Архитектурные метрики ===" -ForegroundColor Cyan
& $py tools/arch_metrics.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`n=== Тесты (быстрые) ===" -ForegroundColor Cyan
& $py -m pytest -m "not regression"
if ($LASTEXITCODE -ne 0) { Write-Host "ТЕСТЫ УПАЛИ" -ForegroundColor Red; exit 1 }

if ($FastOnly) { Write-Host "`nГотово (только быстрые)." -ForegroundColor Green; exit 0 }

Write-Host "`n=== Регрессия решений (быстрый набор) ===" -ForegroundColor Cyan
& $py -m pytest -m regression
if ($LASTEXITCODE -ne 0) { Write-Host "РЕГРЕССИЯ РАСХОДИТСЯ" -ForegroundColor Red; exit 1 }

if ($Full) {
    Write-Host "`n=== Регрессия: полный golden-набор ===" -ForegroundColor Cyan
    & $py tools/regress.py
    if ($LASTEXITCODE -ne 0) { Write-Host "РЕГРЕССИЯ РАСХОДИТСЯ" -ForegroundColor Red; exit 1 }
}

Write-Host "`nВсё зелёное." -ForegroundColor Green
