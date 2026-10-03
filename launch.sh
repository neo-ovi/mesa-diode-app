#!/usr/bin/env bash
# Linux: запуск из исходников с авто-обновлением кода и данных (git pull) —
# то же, что launch.bat в Windows; логика — в scripts/dev_launch.py.
# Нужны системные пакеты (Linux Mint, Ubuntu, Debian):
#   sudo apt install python3-venv python3-tk git
# Запуск:  ./launch.sh   (или: bash launch.sh)
# Готовая программа одним файлом, без Python и git, — packaging/build_linux.sh.
cd "$(dirname "$0")" || exit 1
exec python3 scripts/dev_launch.py "$@"
