"""Проверки модели на синтетических данных.

Реальные измерения в тестах не используются: фикстура посчитана самой
моделью с заранее известными параметрами, поэтому подгонка обязана их вернуть.
"""

from pathlib import Path

import numpy as np
import pytest

from mesa_diode.diode import (
    DIFFUSION_IDEALITY,
    RECOMBINATION_IDEALITY,
    ideality_factor,
    mesa_area_cm2,
    optical_window_area_cm2,
    ring_contact_area_cm2,
    shockley_current,
    thermal_voltage,
    two_diode_current,
)
from mesa_diode.fit import fit_two_diode
from mesa_diode.loaders import load_columns

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_iv.csv"

# Параметры, которыми посчитана фикстура
TRUE_DIFFUSION_A = 2.5e-12
TRUE_RECOMBINATION_A = 4.0e-8
TRUE_SHUNT_OHM = 5.0e5
TEMPERATURE_K = 300.0


@pytest.fixture
def synthetic_iv():
    return load_columns(FIXTURE, "voltage_V", "current_A")


def test_thermal_voltage_at_room_temperature():
    assert thermal_voltage(300.0) == pytest.approx(0.02585, rel=1e-3)


def test_shockley_current_vanishes_at_zero_bias():
    assert shockley_current(0.0, 1e-12, 1.0) == pytest.approx(0.0, abs=1e-30)


def test_shockley_current_saturates_in_reverse():
    current = shockley_current(-1.0, 1e-12, 1.0)
    assert current == pytest.approx(-1e-12, rel=1e-6)


@pytest.mark.parametrize("ideality", [DIFFUSION_IDEALITY, 1.5, RECOMBINATION_IDEALITY])
def test_ideality_factor_recovers_exponential_slope(ideality):
    """Чистая экспонента без члена «−1» — наклон обязан дать n точно."""
    voltage = np.linspace(0.10, 0.30, 64)
    current = 1e-10 * np.exp(voltage / (ideality * thermal_voltage()))
    assert ideality_factor(voltage, current) == pytest.approx(ideality, rel=1e-9)


@pytest.mark.parametrize("ideality", [DIFFUSION_IDEALITY, RECOMBINATION_IDEALITY])
def test_ideality_of_shockley_branch_well_above_thermal_voltage(ideality):
    """На реальной ветви Шокли n восстанавливается при U >> n·kT/q."""
    window = (10 * ideality * thermal_voltage(), 20 * ideality * thermal_voltage())
    voltage = np.linspace(*window, 64)
    current = shockley_current(voltage, 1e-10, ideality)
    assert ideality_factor(voltage, current, window) == pytest.approx(ideality, rel=1e-3)


def test_mixed_branch_ideality_lies_between_one_and_two():
    voltage = np.linspace(0.10, 0.30, 64)
    current = two_diode_current(voltage, TRUE_DIFFUSION_A, TRUE_RECOMBINATION_A)
    assert 1.0 < ideality_factor(voltage, current) < 2.0


def test_fit_recovers_known_parameters(synthetic_iv):
    voltage, current = synthetic_iv
    result = fit_two_diode(
        voltage, current, temperature_K=TEMPERATURE_K, with_shunt=True
    )

    assert result.diffusion_saturation_A == pytest.approx(TRUE_DIFFUSION_A, rel=1e-3)
    assert result.recombination_saturation_A == pytest.approx(
        TRUE_RECOMBINATION_A, rel=1e-3
    )
    assert result.shunt_resistance_ohm == pytest.approx(TRUE_SHUNT_OHM, rel=1e-2)
    assert result.rms_log_residual < 1e-6


def test_fit_rejects_insufficient_forward_points():
    with pytest.raises(ValueError, match="минимум 3 точки"):
        fit_two_diode([-0.2, -0.1, 0.0], [-1e-9, -1e-9, 0.0])


def test_geometry_areas_from_meta():
    meta = {
        "geometry": {
            "mesa_diameter_um": 200.0,
            "top_contact": {"outer_diameter_um": 200.0, "inner_diameter_um": 100.0},
        }
    }

    assert mesa_area_cm2(meta) == pytest.approx(np.pi * 0.01**2)
    assert optical_window_area_cm2(meta) == pytest.approx(np.pi * 0.005**2)
    # Кольцо = внешний круг минус внутренний
    assert ring_contact_area_cm2(meta) == pytest.approx(
        mesa_area_cm2(meta) - optical_window_area_cm2(meta)
    )
