"""Граница между публичным кодом и приватными данными.

Весь доступ к экспериментальным данным идёт только через этот модуль.
Пути и параметры образцов не хранятся в коде — они приходят из каталога,
на который указывает переменная окружения MESA_DATA_DIR.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ENV_VAR = "MESA_DATA_DIR"

_NOT_SET = f"""\
Переменная {ENV_VAR} не задана.

Экспериментальные данные не входят в этот репозиторий. Скопируйте
.env.example в .env и укажите путь к локальному каталогу с данными,
например:

    {ENV_VAR}=D:\\Projects\\mesa-diode\\data
"""


def data_dir() -> Path:
    """Корень каталога с данными."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        raise RuntimeError(_NOT_SET)

    path = Path(raw).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{ENV_VAR} указывает на несуществующий каталог: {path}")

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
