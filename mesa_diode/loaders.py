"""Чтение паспортов образцов и файлов измерений из каталога данных."""

import csv

import numpy as np
import yaml

from mesa_diode.config import sample_dir


def load_meta(sample_id: str) -> dict:
    """Паспорт образца из samples/<id>/meta.yaml."""
    path = sample_dir(sample_id) / "meta.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Паспорт образца не найден: {path}")

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _measurement_path(sample_id: str, kind: str):
    """Путь к файлу измерения, объявленному в секции data_files паспорта."""
    meta = load_meta(sample_id)
    files = meta.get("data_files") or {}

    if kind not in files:
        raise KeyError(f"В паспорте образца {sample_id} нет записи data_files.{kind}")

    relative = files[kind]
    if relative is None:
        raise FileNotFoundError(
            f"Данные {kind!r} для образца {sample_id} ещё не переданы "
            f"(data_files.{kind} = null в meta.yaml)"
        )

    return sample_dir(sample_id) / relative


def load_columns(path, *column_names):
    """Именованные колонки из CSV с заголовком — в виде массивов numpy."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Файл пуст: {path}")

    missing = [name for name in column_names if name not in rows[0]]
    if missing:
        raise ValueError(f"В {path} нет колонок {missing}. Есть: {list(rows[0])}")

    return tuple(
        np.array([float(row[name]) for row in rows], dtype=float) for name in column_names
    )


def load_iv(sample_id: str):
    """ВАХ образца: массивы напряжения (В) и тока (А)."""
    return load_columns(_measurement_path(sample_id, "iv"), "voltage_V", "current_A")


def load_cv(sample_id: str):
    """C–V-характеристика: массивы напряжения (В) и ёмкости (Ф)."""
    return load_columns(_measurement_path(sample_id, "cv"), "voltage_V", "capacitance_F")


def load_xrd(sample_id: str):
    """Кривая качания: отстройка (угл. сек) и интенсивность."""
    return load_columns(_measurement_path(sample_id, "xrd"), "omega_arcsec", "intensity")
