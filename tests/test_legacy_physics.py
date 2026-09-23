"""Проверки физического ядра GUI-симулятора на синтетических параметрах.

Симулятор — прямая модель (параметры -> кривые), поэтому тесты проверяют
внутреннюю согласованность формул, а не сравнение с измерениями.
"""

import numpy as np
import pytest

from mesa_diode.simulator.legacy_physics import (
    MesaParams, adaptive_point_count, adaptive_voltage_range, auto_scale,
    capacitance, current_density_from_area, depletion_width,
    estimate_grading_m, mobility, ni_of_T, shockley_current_density,
    solve_iv, MATERIAL_J0_A_CM2, MU_SI_N, NI_SI_300, EG_SI0,
)


@pytest.fixture
def params():
    return MesaParams(
        D_um=50.0, d_um=30.0, h_um=2.0, ND_si=1e17, NA_ge=5e17,
        T_K=300.0, Rs_ohm=10.0, Rsh_ohm=1e6, n2=2.0,
    )


def test_ni_of_T_matches_reference_at_300K():
    assert ni_of_T(NI_SI_300, EG_SI0, 300.0) == pytest.approx(NI_SI_300, rel=1e-9)


def test_ni_increases_with_temperature():
    assert ni_of_T(NI_SI_300, EG_SI0, 350.0) > ni_of_T(NI_SI_300, EG_SI0, 300.0)


def test_mobility_stays_within_model_bounds():
    for N in (1e14, 1e16, 1e18, 1e20):
        mu = mobility(N, MU_SI_N)
        assert MU_SI_N["mu_min"] <= mu <= MU_SI_N["mu_max"]


def test_mobility_decreases_with_doping():
    assert mobility(1e14, MU_SI_N) > mobility(1e19, MU_SI_N)


def test_mesa_area_from_diameter(params):
    expected = np.pi * (50.0 * 1e-4 / 2.0) ** 2
    assert params.A == pytest.approx(expected)


def test_built_in_potential_is_positive(params):
    assert params.Vbi > 0


def test_depletion_width_shrinks_toward_forward_bias(params):
    w_reverse = depletion_width(-1.0, params)
    w_zero = depletion_width(0.0, params)
    assert w_reverse > w_zero > 0


def test_capacitance_increases_toward_forward_bias(params):
    V = np.array([-1.0, -0.5, 0.0])
    C = capacitance(V, params)
    assert np.all(np.diff(C) > 0)


def test_capacitance_undefined_near_built_in_potential(params):
    C = capacitance(np.array([params.Vbi]), params)
    assert np.isnan(C[0])


def test_solve_iv_forward_current_exceeds_reverse(params):
    V = np.array([-0.5, 0.0, 0.3])
    I = solve_iv(V, params)
    assert I[2] > I[1] > I[0]


def test_solve_iv_zero_bias_gives_near_zero_current(params):
    I = solve_iv(np.array([0.0]), params)
    assert I[0] == pytest.approx(0.0, abs=1e-15)


def test_estimate_grading_m_needs_enough_points(params):
    V = np.array([-0.1, -0.2])
    C = capacitance(V, params)
    m, slope = estimate_grading_m(V, C, params.Vbi)
    assert m is None and slope is None


def test_estimate_grading_m_recovers_sharp_junction(params):
    V = np.linspace(-1.4, -0.1, 40)
    C = capacitance(V, params)
    m, slope = estimate_grading_m(V, C, params.Vbi)
    # Модель использует приближение резкого перехода (m -> 0)
    assert m == pytest.approx(0.0, abs=0.05)


def test_auto_scale_picks_matching_prefix():
    factor, unit = auto_scale(np.array([2.5e-6]), "А")
    assert unit == "мкА"
    assert 2.5e-6 * factor == pytest.approx(2.5)


def test_auto_scale_handles_all_nan():
    factor, unit = auto_scale(np.array([np.nan, np.nan]), "Ф")
    assert unit == "пФ"


# ---------------- уравнение Шокли для справочной кривой J-V ----------------

def test_shockley_current_density_vanishes_at_zero_bias():
    assert shockley_current_density(0.0, 1e-6) == pytest.approx(0.0, abs=1e-30)


def test_shockley_current_density_saturates_in_reverse():
    J = shockley_current_density(-1.0, 1e-6)
    assert J == pytest.approx(-1e-6, rel=1e-6)


def test_shockley_current_density_ge_exceeds_si_at_same_bias():
    # У Ge на много порядков выше характерный ток насыщения (см. physics.py)
    V = 0.3
    J_ge = shockley_current_density(V, MATERIAL_J0_A_CM2["Ge"])
    J_si = shockley_current_density(V, MATERIAL_J0_A_CM2["Si"])
    assert J_ge > J_si


def test_material_j0_has_si_and_ge():
    assert set(MATERIAL_J0_A_CM2) == {"Si", "Ge"}
    assert all(value > 0 for value in MATERIAL_J0_A_CM2.values())


# ---------------- плотность тока из площади мезы (J = I / S) ----------------

def test_current_density_from_area_matches_manual_division():
    current = np.array([1e-6, 2e-6, -1e-6])
    area = 2e-5
    J = current_density_from_area(current, area)
    np.testing.assert_allclose(J, current / area)


# ---------------- адаптивный диапазон напряжений под эксперимент ----------------

def test_adaptive_voltage_range_keeps_default_when_no_data():
    vmin, vmax = adaptive_voltage_range([])
    assert vmin < -1.0  # запас с краю
    assert vmax > 0.6


def test_adaptive_voltage_range_extends_to_cover_experimental_data():
    voltages = [np.array([-6.0, -3.0, 1.5])]
    vmin, vmax = adaptive_voltage_range(voltages)
    assert vmin <= -6.0
    assert vmax >= 1.5


def test_adaptive_voltage_range_ignores_narrower_experimental_data():
    voltages = [np.array([-0.5, 0.2])]
    vmin, vmax = adaptive_voltage_range(voltages)
    assert vmin <= -1.0
    assert vmax >= 0.6


def test_adaptive_point_count_grows_with_range():
    narrow = adaptive_point_count(-1.0, 0.6)
    wide = adaptive_point_count(-6.0, 1.5)
    assert wide > narrow


def test_adaptive_point_count_respects_bounds():
    assert adaptive_point_count(0.0, 0.001) >= 121
    assert adaptive_point_count(-1000.0, 1000.0) <= 400
