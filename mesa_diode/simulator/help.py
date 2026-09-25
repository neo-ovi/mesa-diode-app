# -*- coding: utf-8 -*-
"""Открытие методички из меню «Справка» (ТЗ §8.1).

Методичка в этом репозитории не хранится. Путь ищется по порядку:
  1) параметр "metodichka_path" в ~/.mesa_diode/config.json;
  2) переменная окружения MESA_DIODE_DOCS — каталог с METODICHKA.pdf/.docx;
  3) каталог docs/ в каталоге данных: $MESA_DATA_DIR/docs/ (переменная
     из .env, см. mesa_diode/config.py).
Если файл не найден, программа просит указать его и запоминает путь в
конфиге. Файл открывается системным приложением.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from mesa_diode import config

CONFIG_PATH = Path.home() / ".mesa_diode" / "config.json"
CONFIG_KEY = "metodichka_path"
DOCS_ENV = "MESA_DIODE_DOCS"
KINDS = {"pdf": "METODICHKA.pdf", "docx": "METODICHKA.docx"}


def _read_config(config_path):
    try:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _from_location(location, kind):
    """Файл нужного вида по заданному месту: сам файл, файл рядом с ним или в каталоге."""
    path = Path(location).expanduser()
    if path.is_file() and path.suffix.lower() == f".{kind}":
        return path
    folder = path if path.is_dir() else path.parent
    candidate = folder / KINDS[kind]
    return candidate if candidate.is_file() else None


def find_metodichka(kind="pdf", config_path=CONFIG_PATH, environ=None):
    """Путь к METODICHKA.pdf / .docx или None (без исключений)."""
    environ = os.environ if environ is None else environ
    locations = []
    configured = _read_config(config_path).get(CONFIG_KEY)
    if configured:
        locations.append(configured)
    if environ.get(DOCS_ENV):
        locations.append(environ[DOCS_ENV])
    if environ.get(config.ENV_VAR):
        locations.append(Path(environ[config.ENV_VAR]) / "docs")
    for location in locations:
        try:
            found = _from_location(location, kind)
        except OSError:
            continue
        if found:
            return found
    return None


def save_metodichka_path(path, config_path=CONFIG_PATH):
    config_path = Path(config_path)
    data = _read_config(config_path)
    data[CONFIG_KEY] = str(Path(path))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def open_with_system(path):
    """Открывает файл системным приложением. Возвращает текст ошибки или None."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606 — штатный способ Windows
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except (OSError, AttributeError) as e:
        return str(e)
    return None


def open_metodichka(parent, kind="pdf"):
    """Ищет и открывает методичку; при неудаче показывает понятный диалог."""
    from tkinter import filedialog, messagebox

    path = find_metodichka(kind)
    if path is None:
        messagebox.showinfo(
            "Методичка",
            "Методичка хранится в приватном репозитории данных. Укажите файл "
            "METODICHKA.pdf или METODICHKA.docx.", parent=parent)
        chosen = filedialog.askopenfilename(
            parent=parent, title="Файл методички",
            filetypes=[("Методичка", "*.pdf *.docx"), ("Все файлы", "*.*")])
        if not chosen:
            return
        save_metodichka_path(chosen)
        path = _from_location(chosen, kind) or Path(chosen)
    error = open_with_system(path)
    if error:
        messagebox.showerror("Методичка", f"Не удалось открыть {path}:\n{error}", parent=parent)
