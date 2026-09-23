# -*- coding: utf-8 -*-
"""Тесты физического ядра по ТЗ §9 на синтетических и материальных данных.

Проверки, где нужны параметры реального образца, лежат вне этого
репозитория и выполняются tests/test_reference_sets.py через MESA_DATA_DIR.
"""

import numpy as np
import pytest

from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.materials import GE

T300 = 300.0


def rel(value, expected, tol):
    return value == pytest.approx(expected, rel=tol)


# ------------------------------------------------------------------ §2 --

def test_band_gap_ge_300k():
    assert rel(ph.band_gap(T300), 0.6634, 5e-3)


def test_effective_dos_ge_300k():
    Nc, Nv = ph.effective_dos(T300)
    assert rel(Nc, 1.029e19, 5e-3)
    assert rel(Nv, 4.99e18, 5e-3)


def test_intrinsic_concentration_ge_300k():
    assert rel(ph.intrinsic_concentration(T300), 1.918e13, 5e-3)


def test_equilibrium_carriers_mass_action_and_neutrality():
    ni = ph.intrinsic_concentration(T300)
    for N in (1e12, 1e14, 1e17):
        M, m = ph.equilibrium_carriers(N, ni)
        assert rel(M * m, ni * ni, 1e-12)
        assert rel(M - m, N, 1e-9)


def test_equilibrium_carriers_reduce_to_doping_when_n_much_greater_than_ni():
    ni = ph.intrinsic_concentration(T300)
    M, m = ph.equilibrium_carriers(1e18, ni)
    assert rel(M, 1e18, 1e-9)
    assert rel(m, ni * ni / 1e18, 1e-9)


@pytest.mark.parametrize("n, eta", [(1e18, -2.30), (1e19, 0.31)])
def test_reduced_fermi_level_electrons(n, eta):
    Nc, _ = ph.effective_dos(T300)
    assert ph.reduced_fermi_level(n, Nc) == pytest.approx(eta, abs=0.02)


def test_reduced_fermi_level_inverts_carriers_from_eta():
    Nc, _ = ph.effective_dos(T300)
    for eta in (-8.0, -3.0, 0.0, 3.0):
        assert ph.reduced_fermi_level(ph.carriers_from_eta(eta, Nc), Nc) == pytest.approx(eta, abs=1e-6)


@pytest.mark.parametrize("carrier, eta, expected", [
    ("electrons", -3.0, 5.0e17), ("electrons", -2.0, 1.3e18), ("electrons", 0.0, 7.9e18),
    ("holes", -3.0, 2.4e17), ("holes", -2.0, 6.5e17), ("holes", 0.0, 3.8e18),
])
def test_degeneracy_thresholds_table(carrier, eta, expected):
    table = ph.degeneracy_thresholds(T300)
    assert rel(table[carrier][eta], expected, 3e-2)


def test_boltzmann_ratio_limits():
    assert ph.boltzmann_ratio(-15.0) == pytest.approx(1.0, abs=1e-6)
    assert ph.boltzmann_ratio(0.0) < 0.8


def test_degeneracy_status_thresholds():
    assert ph.degeneracy_status(-5.0) == "невырожден"
    assert ph.degeneracy_status(-3.0) == "начало вырождения"
    assert ph.degeneracy_status(0.31) == "вырожден"


def test_resistivity_intrinsic_and_maximum():
    rho0 = ph.resistivity(0.0, T300)
    assert rel(rho0, 56.1, 5e-3)
    grid = np.logspace(11, 15, 400)
    rho = np.array([ph.resistivity(N, T300) for N in grid])
    assert rel(rho.max(), 59.5, 5e-3)
    assert 5e12 < grid[rho.argmax()] < 2e13


@pytest.mark.parametrize("rho", [0.1, 3.0, 20.0])
def test_acceptor_from_resistivity_inverts_resistivity(rho):
    NA, warnings = ph.acceptor_from_resistivity(rho, T300)
    assert warnings == []
    assert rel(ph.resistivity(NA, T300), rho, 1e-6)


def test_acceptor_from_resistivity_two_solutions_takes_larger():
    NA, warnings = ph.acceptor_from_resistivity(58.0, T300)
    assert len(warnings) == 1
    assert NA > 1e13
    assert rel(ph.resistivity(NA, T300), 58.0, 1e-6)


def test_acceptor_from_resistivity_above_maximum_has_no_solution():
    NA, warnings = ph.acceptor_from_resistivity(70.0, T300)
    assert np.isnan(NA) and warnings


def test_diffusion_coefficient_and_length():
    D = ph.diffusion_coefficient(GE.mu_n_max, T300)
    assert rel(D, ph.thermal_voltage(T300) * 3900.0, 1e-12)
    assert rel(ph.diffusion_length(D, 1e-6), np.sqrt(D * 1e-6), 1e-12)


# ------------------------------------------------------------------ §3 --
# Синтетическая структура для проверок свойств модели. Эталоны ТЗ §9
# (наборы Э1, Э0) — в tests/test_reference_sets.py через MESA_DATA_DIR.

UM = 1e-4


def synth(**overrides):
    params = dict(D=400 * UM, d_epi=3.2 * UM, h=3.2 * UM, d_n=0.4 * UM, d_sub=300 * UM,
                  ND_plus=5e17, N_i=5e15, rho_sub=5.0, scenario=ph.SCENARIO_A,
                  N_dis=0.0, Rs=10.0, Rsh=1e6, vbi_method=ph.VBI_BOLTZMANN)
    params.update(overrides)
    return ph.Structure(**params)




def test_inv_c2_slope_gives_n_eff_and_cutoff():
    s = synth()
    V = np.linspace(-1.0, 0.0, 41)
    N, V0 = ph.fit_inv_c2(V, ph.capacitance(s, V), s.area, s.eps)
    assert rel(N, s.N_eff, 1e-9)
    assert rel(V0, ph.c2_cutoff(s), 1e-9)
    assert rel(ph.c2_cutoff(s), s.Vbi - 2 * s.Vt, 1e-12)


def test_capacitance_is_eps_area_over_width_and_nan_near_vbi():
    s = synth()
    assert rel(ph.capacitance(s, -0.5), s.eps * s.area / ph.depletion_width(s, -0.5), 1e-12)
    assert np.isnan(ph.capacitance(s, s.Vbi - 2.9 * s.Vt))


def test_depletion_edges_charge_neutrality():
    s = synth()
    xn, xp = ph.depletion_edges(s, -1.0)
    assert rel(xn * s.n_side.N, xp * s.p_side.N, 1e-12)
    assert rel(xn + xp, ph.depletion_width(s, -1.0), 1e-12)


def test_vbi_degenerate_matches_boltzmann_for_nondegenerate_layers():
    s = synth(ND_plus=1e16, N_i=1e15)
    assert s.eta[0] < -3 and s.eta[1] < -3
    v31 = ph.built_in_potential(s, ph.VBI_BOLTZMANN)
    v31a = ph.built_in_potential(s, ph.VBI_DEGENERATE)
    assert v31a == pytest.approx(v31, rel=2e-3)


def test_scenario_mapping():
    a = synth()
    assert (a.n_side.layer, a.p_side.layer) == ("n+", "i")
    assert a.z_j == pytest.approx(0.4 * UM)
    assert rel(a.p_side.thickness, 2.8 * UM, 1e-12)
    b = a.with_scenario(ph.SCENARIO_B)
    assert (b.n_side.layer, b.p_side.layer) == ("i", "sub")
    assert b.z_j == pytest.approx(3.2 * UM)
    assert b.n_side.boundary == ph.REFLECT
    assert rel(ph.resistivity(b.p_side.N, 300.0), 5.0, 1e-6)
