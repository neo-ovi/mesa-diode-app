"""Проверки физического ядра GUI-симулятора на синтетических параметрах.

Симулятор — прямая модель (параметры -> кривые), поэтому тесты проверяют
внутреннюю согласованность формул, а не сравнение с измерениями.
"""

import numpy as np
import pytest

from mesa_diode.simulator.physics import (
    MesaParams, auto_scale, capacitance, depletion_width,
    estimate_grading_m, mobility, ni_of_T, solve_iv, MU_SI_N, NI_SI_300, EG_SI0,
)


@pytest.fixture
def params():
    return MesaParams(
        D_um=50.0, h_um=2.0, ND_si=1e17, NA_ge=5e17,
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
