"""
Консольный вывод: принудительный UTF-8 для stdout/stderr.

Зачем: сообщения приложения на русском и со стрелками («→»), а при редиректе
вывода в файл Python 3.9 на Windows берёт кодировку локали (cp1251) вместо
UTF-8 консоли. Любой прогон вида `python main.py > log.txt` падал на первом же
таком print с UnicodeEncodeError — то есть именно тогда, когда лог нужнее всего
(серийные прогоны, запуск из мастера, автоматическая регрессия).

Вызывается первым делом в точках входа. Ничего не делает, если поток уже в
UTF-8, и молча отступает, если переконфигурировать нельзя (поток подменён,
не TextIO) — вывод важен, но ронять из-за него прогон нельзя.
"""
import sys


def setup_console() -> None:
    """Перевести stdout/stderr в UTF-8 с заменой непредставимых символов."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
