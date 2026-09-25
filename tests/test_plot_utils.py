# -*- coding: utf-8 -*-
"""Помощники отрисовки: приставки единиц и диапазон напряжений."""

import numpy as np
import pytest

from mesa_diode.simulator.plot_utils import (
    adaptive_point_count, adaptive_voltage_range, auto_scale, robust_value_limits,
)


def test_auto_scale_picks_matching_prefix():
    factor, unit = auto_scale(np.array([2.5e-6]), "А")
    assert unit == "мкА"
    assert 2.5e-6 * factor == pytest.approx(2.5)


def test_auto_scale_handles_all_nan():
    assert auto_scale(np.array([np.nan, np.nan]), "Ф")[1] == "пФ"


def test_adaptive_voltage_range_keeps_default_without_data():
    vmin, vmax = adaptive_voltage_range([])
    assert vmin < -1.0 and vmax > 0.6


def test_adaptive_voltage_range_covers_experimental_data():
    vmin, vmax = adaptive_voltage_range([np.array([-6.0, -3.0, 1.5])])
    assert vmin <= -6.0 and vmax >= 1.5


def test_adaptive_point_count_bounds():
    assert adaptive_point_count(-6.0, 1.5) > adaptive_point_count(-1.0, 0.6)
    assert adaptive_point_count(0.0, 0.001) >= 121
    assert adaptive_point_count(-1000.0, 1000.0) <= 400


def test_robust_limits_follow_experiment_not_model_tail():
    lo, hi = robust_value_limits([np.array([-1e-6, 1e-3])], np.array([-1.0, 1e25]))
    assert hi < 1.0
