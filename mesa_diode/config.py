"""Граница между публичным кодом и приватными данными.

Весь доступ к экспериментальным данным идёт только через этот модуль.
Пути и параметры образцов не хранятся в коде — они приходят из каталога
с данными. Где он, программа узнаёт по порядку:
  1) переменная окружения MESA_DATA_DIR — из окружения или из файла .env
     (в текущем каталоге, а у собранной программы — ещё и рядом с ней);
  2) папка, выбранная в программе («Настройки → Папка с данными»), —
     хранится в ~/.mesa_diode/config.json на компьютере пользователя.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
if getattr(sys, "frozen", False):        # собранная программа: .env рядом с исполняемым файлом
    load_dotenv(Path(sys.executable).resolve().parent / ".env")

ENV_VAR = "MESA_DATA_DIR"
CONFIG_PATH = Path.home() / ".mesa_diode" / "config.json"
DATA_DIR_KEY = "data_dir"

_NOT_SET = f"""\
Папка с данными не задана.

Экспериментальные данные не входят в этот репозиторий. Укажите папку с
данными в программе («Настройки → Папка с данными») или скопируйте
.env.example в .env и укажите путь в переменной {ENV_VAR}, например:

    {ENV_VAR}=D:\\Projects\\mesa-diode\\data
"""


def read_user_config(config_path=None) -> dict:
    """Пути, выбранные в программе (~/.mesa_diode/config.json), или {}."""
    try:
        data = json.loads(Path(config_path or CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_user_config(key, value, config_path=None) -> None:
    """Записать один ключ в ~/.mesa_diode/config.json, остальные сохраняются."""
    config_path = Path(config_path or CONFIG_PATH)
    data = read_user_config(config_path)
    data[key] = value
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def saved_data_dir(config_path=None):
    """Папка с данными, выбранная в программе, или None."""
    value = read_user_config(config_path).get(DATA_DIR_KEY)
    return Path(value).expanduser() if value else None


def save_data_dir(path, config_path=None) -> None:
    """Запомнить папку с данными (действует сразу, если не задана MESA_DATA_DIR)."""
    write_user_config(DATA_DIR_KEY, str(Path(path).expanduser()), config_path)


def data_dir_setting(environ=None, config_path=None):
    """(путь или None, откуда он: "env" — MESA_DATA_DIR, "settings" — выбор в программе)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_VAR)
    if raw:
        return Path(raw).expanduser(), "env"
    saved = saved_data_dir(config_path)
    return (saved, "settings") if saved else (None, None)


def data_dir() -> Path:
    """Корень каталога с данными."""
    path, source = data_dir_setting()
    if path is None:
        raise RuntimeError(_NOT_SET)
    if not path.is_dir():
        if source == "env":
            raise RuntimeError(f"{ENV_VAR} указывает на несуществующий каталог: {path}")
        raise RuntimeError(f"Папка с данными не найдена: {path}. Укажите её заново: "
                           "«Настройки → Папка с данными».")

    return path


def sample_dir(sample_id: str) -> Path:
    """Каталог конкретного образца."""
    path = data_dir() / "samples" / sample_id
    if not path.is_dir():
        available = sorted(p.name for p in (data_dir() / "samples").glob("*") if p.is_dir())
        raise FileNotFoundError(
            f"Образец {sample_id!r} не найден в {path.parent}. Доступны: {available}"
        )

    return path
