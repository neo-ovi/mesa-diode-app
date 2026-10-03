#!/usr/bin/env python3
"""Точка входа GUI-симулятора мезадиода Ge.

Запуск:  python scripts/run_simulator.py   (ключи: --check — проверка окружения, --version)
Сборка в один файл (Windows .exe, Linux): см. mesa_diode/simulator/README.md
"""

import os
import sys
from pathlib import Path

# Корень репозитория — в путь импорта: скрипт запускается из любого каталога.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _keep_matplotlib_cache():
    """Собранная программа (PyInstaller) кладёт кэш шрифтов matplotlib во временный
    каталог и при каждом запуске заново перебирает шрифты системы (секунды).
    Пути к своим шрифтам matplotlib хранит относительно своего каталога данных,
    поэтому кэш можно держать постоянным: ~/.cache/mesa-simulator/matplotlib
    (Windows — %LOCALAPPDATA%\\mesa-simulator\\matplotlib)."""
    if not getattr(sys, "frozen", False):
        return
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    folder = base / "mesa-simulator" / "matplotlib"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return                       # не создать — остаётся временный каталог PyInstaller
    os.environ["MPLCONFIGDIR"] = str(folder)


_keep_matplotlib_cache()

from mesa_diode.simulator.app import main  # noqa: E402 — после настройки пути и кэша matplotlib

if __name__ == "__main__":
    main()
