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
