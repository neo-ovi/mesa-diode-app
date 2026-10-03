# -*- coding: utf-8 -*-
"""Открытие методички из меню «Справка» (ТЗ §8.1).

Методичка в этом репозитории не хранится. Путь ищется по порядку:
  1) параметр "metodichka_path" в ~/.mesa_diode/config.json;
  2) переменная окружения MESA_DIODE_DOCS — каталог с METODICHKA.pdf/.docx;
  3) каталог docs/ в каталоге данных: $MESA_DATA_DIR/docs/ (переменная
     из .env, см. mesa_diode/config.py);
  4) каталог docs/ в папке с данными, выбранной в программе
     («Настройки → Папка с данными»).
Если файл не найден, программа просит указать его и запоминает путь в
конфиге. Файл открывается системным приложением (desktop.open_path).
"""

import os
from pathlib import Path

from mesa_diode import config
from mesa_diode.simulator import desktop

CONFIG_KEY = "metodichka_path"
DOCS_ENV = "MESA_DIODE_DOCS"
KINDS = {"pdf": "METODICHKA.pdf", "docx": "METODICHKA.docx"}


def _from_location(location, kind):
    """Файл нужного вида по заданному месту: сам файл, файл рядом с ним или в каталоге."""
    path = Path(location).expanduser()
    if path.is_file() and path.suffix.lower() == f".{kind}":
        return path
    folder = path if path.is_dir() else path.parent
    candidate = folder / KINDS[kind]
    return candidate if candidate.is_file() else None


def find_metodichka(kind="pdf", config_path=None, environ=None):
    """Путь к METODICHKA.pdf / .docx или None (без исключений)."""
    environ = os.environ if environ is None else environ
    locations = []
    configured = config.read_user_config(config_path).get(CONFIG_KEY)
    if configured:
        locations.append(configured)
    if environ.get(DOCS_ENV):
        locations.append(environ[DOCS_ENV])
    if environ.get(config.ENV_VAR):
        locations.append(Path(environ[config.ENV_VAR]) / "docs")
    saved = config.saved_data_dir(config_path)
    if saved:
        locations.append(saved / "docs")
    for location in locations:
        try:
            found = _from_location(location, kind)
        except OSError:
            continue
        if found:
            return found
    return None


def save_metodichka_path(path, config_path=None):
    config.write_user_config(CONFIG_KEY, str(Path(path)), config_path)


def open_with_system(path):
    """Открывает файл системным приложением. Возвращает текст ошибки или None."""
    return desktop.open_path(path)


def open_metodichka(parent, kind="pdf"):
    """Ищет и открывает методичку; при неудаче показывает понятный диалог."""
    from tkinter import filedialog, messagebox

    path = find_metodichka(kind)
    if path is None:
        messagebox.showinfo(
            "Методичка",
            "Методичка хранится в приватном репозитории данных (папка docs). Укажите файл "
            "METODICHKA.pdf или METODICHKA.docx — или всю папку с данными: "
            "«Настройки → Папка с данными».", parent=parent)
        chosen = filedialog.askopenfilename(
            parent=parent, title="Файл методички",
            filetypes=desktop.file_types([("Методичка", "*.pdf *.docx"), ("Все файлы", "*.*")]))
        if not chosen:
            return
        save_metodichka_path(chosen)
        path = _from_location(chosen, kind) or Path(chosen)
    error = open_with_system(path)
    if error:
        messagebox.showerror("Методичка", f"Не удалось открыть {path}:\n{error}", parent=parent)
