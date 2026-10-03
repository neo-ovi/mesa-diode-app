#!/usr/bin/env python3
"""Локальный запуск симулятора с авто-синком кода и данных.

Что делает, по порядку:
  1. Читает MESA_DATA_DIR из .env (см. .env.example) — путь к каталогу с
     экспериментальными данными на диске.
  2. Обновляет (git pull) этот репозиторий и каталог из MESA_DATA_DIR.
     Если где-то есть несохранённые изменения или история разошлась
     (не fast-forward) — тот репозиторий пропускается с предупреждением,
     ничего не перезаписывается и не сливается автоматически.
  3. Создаёт .venv, если его ещё нет, и ставит/обновляет зависимости.
  4. Запускает симулятор.

Важно про приватность: этот скрипт живёт в публичном репозитории и
намеренно не содержит ни имени, ни адреса каталога с данными — только
переменная MESA_DATA_DIR, которую он читает из вашего локального .env
(этот файл в .gitignore и никогда не коммитится).

Запуск:  python scripts/dev_launch.py
Либо просто дважды кликнуть launch.bat (Windows) или запустить ./launch.sh
(Linux) в корне репозитория. В Linux нужны системные пакеты python3-venv и
python3-tk (см. LINUX_PACKAGES).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
LINUX_PACKAGES = "sudo apt install python3-venv python3-tk"   # Linux Mint, Ubuntu, Debian


def linux_hint(text: str) -> str:
    """Подсказка про системные пакеты — только в Linux."""
    return f"{text}\nВ Linux Mint / Ubuntu / Debian: {LINUX_PACKAGES}" if sys.platform.startswith("linux") else text


def venv_python() -> Path:
    """Путь к python внутри .venv (по-разному на Windows и Linux/macOS)."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def read_env_var(name: str) -> str | None:
    """Читает переменную из файла .env в корне репозитория (без сторонних
    зависимостей — на этом этапе venv ещё может не существовать, а значит
    python-dotenv ещё недоступен)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip() or None

    return None


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def git_pull_safe(repo_dir: Path, label: str) -> None:
    """Обновляет один git-репозиторий, если это безопасно.

    Ничего не перезаписывает и не сливает автоматически: при локальных
    изменениях или разошедшейся истории просто предупреждает и выходит,
    оставляя разбираться вручную.
    """
    if not (repo_dir / ".git").is_dir():
        print(f"[{label}] пропущено: {repo_dir} — это не git-репозиторий")
        return

    status = run("git", "status", "--porcelain", cwd=repo_dir)
    if status.stdout.strip():
        print(f"[{label}] есть несохранённые изменения — pull пропущен, "
              f"обновите вручную")
        return

    pull = run("git", "pull", "--ff-only", cwd=repo_dir)
    if pull.returncode != 0:
        print(f"[{label}] pull не выполнен (возможно, история разошлась):")
        print(pull.stderr.strip() or pull.stdout.strip())
        return

    print(f"[{label}] обновлено: {pull.stdout.strip() or 'уже актуально'}")


def ensure_venv() -> None:
    if venv_python().is_file():
        return

    print("Создаю .venv...")
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0:
        shutil.rmtree(VENV_DIR, ignore_errors=True)    # недоделанный .venv без pip — убрать
        sys.exit(linux_hint("Не удалось создать .venv."))


def check_tkinter() -> None:
    """Окна программы — tkinter; venv берёт его из системного Python (в Linux —
    отдельный пакет python3-tk, через pip он не ставится)."""
    result = subprocess.run([str(venv_python()), "-c", "import tkinter"], capture_output=True)
    if result.returncode != 0:
        sys.exit(linux_hint("Нет модуля tkinter — без него окно программы не открыть."))


def install_requirements() -> None:
    print("Устанавливаю зависимости...")
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)]
    )
    if result.returncode != 0:
        sys.exit("Не удалось установить зависимости")


def launch_simulator() -> None:
    print("Запускаю симулятор...")
    subprocess.run([str(venv_python()), "-m", "mesa_diode.simulator.app"], cwd=REPO_ROOT)


def main() -> None:
    data_dir = read_env_var("MESA_DATA_DIR")
    if not data_dir:
        sys.exit(
            "MESA_DATA_DIR не задан. Скопируйте .env.example в .env и "
            "укажите путь к каталогу с данными."
        )

    git_pull_safe(REPO_ROOT, "код")
    git_pull_safe(Path(data_dir), "данные")

    ensure_venv()
    install_requirements()
    check_tkinter()
    launch_simulator()


if __name__ == "__main__":
    main()
