# -*- coding: utf-8 -*-
"""Стили точек для наложения нескольких экспериментальных наборов на график.

Каждому набору — своя пара (маркер, цвет), в стиле Origin: разные наборы
должны быть различимы даже на чёрно-белой распечатке. Цвета не пересекаются
с цветами модельных кривых (синий у ВАХ, оранжево-красный у ВФХ).
"""

MAX_DATASETS = 5

_MARKER_COLOR_CYCLE = [
    ("o", "black"),
    ("s", "#d62728"),   # красный
    ("^", "#2ca02c"),   # зелёный
    ("D", "#9467bd"),   # фиолетовый
    ("*", "#8c564b"),   # коричневый
]

assert len(_MARKER_COLOR_CYCLE) == MAX_DATASETS


def dataset_style(index: int) -> tuple[str, str]:
    """Маркер и цвет для набора под номером ``index`` (с нуля)."""
    return _MARKER_COLOR_CYCLE[index % len(_MARKER_COLOR_CYCLE)]


def remaining_slots(current_count: int, max_datasets: int = MAX_DATASETS) -> int:
    """Сколько ещё наборов можно добавить, не превышая лимит."""
    return max(0, max_datasets - current_count)
