"""Проверки чистой логики стилей и лимита наборов данных (без Tkinter)."""

import pytest

from mesa_diode.simulator.style import MAX_DATASETS, dataset_style, remaining_slots


def test_max_datasets_is_five():
    assert MAX_DATASETS == 5


def test_dataset_style_returns_distinct_pairs_up_to_the_limit():
    styles = [dataset_style(i) for i in range(MAX_DATASETS)]
    assert len(set(styles)) == MAX_DATASETS


def test_dataset_style_cycles_after_the_limit():
    assert dataset_style(0) == dataset_style(MAX_DATASETS)
    assert dataset_style(1) == dataset_style(MAX_DATASETS + 1)


def test_remaining_slots_when_empty():
    assert remaining_slots(0) == MAX_DATASETS


def test_remaining_slots_when_partially_full():
    assert remaining_slots(2) == MAX_DATASETS - 2


def test_remaining_slots_when_full():
    assert remaining_slots(MAX_DATASETS) == 0


def test_remaining_slots_never_negative_even_over_limit():
    assert remaining_slots(MAX_DATASETS + 3) == 0


@pytest.mark.parametrize("max_datasets", [1, 3, 10])
def test_remaining_slots_respects_custom_limit(max_datasets):
    assert remaining_slots(0, max_datasets=max_datasets) == max_datasets
