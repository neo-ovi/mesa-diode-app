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


# ------------------------------------------------------ §4, §5, §5А, §6.1 --

def test_long_base_equals_sze_eq_43():
    s = synth(bc_A_n=ph.LONG, bc_A_p=ph.LONG)
    (Dp, Lp), (Dn, Ln) = ph.minority_transport(s)
    p_n0, n_p0 = s.minority
    sze_43 = ph.Q * Dp * p_n0 / Lp + ph.Q * Dn * n_p0 / Ln
    assert rel(float(ph.saturation_current_density(s, -1.0)), sze_43, 1e-6)


def test_thick_base_sink_tends_to_long_base():
    thick = synth(d_epi=0.4 * UM + 5000 * UM)
    long_base = synth(d_epi=0.4 * UM + 5000 * UM, bc_A_p=ph.LONG)  # толстая только p-сторона
    assert rel(float(thick.area * ph.saturation_current_density(thick, 0.0)),
               float(long_base.area * ph.saturation_current_density(long_base, 0.0)), 1e-3)


def test_short_base_limits():
    u = np.array([1e-3])
    assert rel(ph.boundary_factor(u, ph.SINK)[0], 1.0 / np.tanh(1e-3), 1e-9)
    assert rel(ph.boundary_factor(u, ph.REFLECT)[0], np.tanh(1e-3), 1e-9)
    assert ph.boundary_factor(np.array([1e-8]), ph.SINK)[0] == pytest.approx(1e8)
    assert ph.boundary_factor(np.array([25.0]), ph.REFLECT)[0] == 1.0


def test_reflect_short_base_gives_q_n_w_over_tau():
    s = synth(bc_A_n=ph.REFLECT, bc_A_p=ph.REFLECT)
    (Dp, Lp), (Dn, Ln) = ph.minority_transport(s)
    _, electrons = ph.saturation_current_density_parts(s, -1.0)
    _, wp, _ = ph.neutral_widths(s, -1.0)
    tau_n = ph.minority_lifetime(s.p_side, s.N_dis)
    assert rel(float(electrons), ph.Q * s.minority[1] * float(wp) / tau_n, 2e-3)


def test_gr_current_vanishes_for_infinite_scr_lifetime():
    s = synth(tau0_bg=1e30)
    assert abs(float(ph.gr_current(s, -1.0))) < 1e-25


def test_sum_of_components_without_parasitics():
    s = synth(Rs=0.0, Rsh=float("inf"), I_L=0.0)
    V = np.linspace(-1.0, 0.3, 27)
    r = ph.solve_iv(s, V)
    assert np.allclose(r.I, ph.diffusion_current(s, V) + ph.gr_current(s, V), rtol=1e-12)


def test_zero_bias_zero_current_and_monotonic():
    s = synth(I_L=1e-6)
    V = np.linspace(-2.0, 0.4, 97)
    r = ph.solve_iv(s, V)
    assert r.warnings == []
    assert abs(r.I[np.argmin(np.abs(V))]) < 1e-15
    assert np.all(np.diff(r.I) > 0)


def test_series_resistance_voltage_drop_is_consistent():
    s = synth()
    r = ph.solve_iv(s, np.array([0.2]))
    assert rel(r.I[0], float(ph.junction_current(s, 0.2 - r.I[0] * s.Rs)), 1e-9)


def test_lifetimes_with_dislocations():
    s = synth(N_dis=1e6)
    assert rel(ph.minority_lifetime(s.p_side, s.N_dis),
               1.0 / (1.0 / 1e-6 + s.sigma_R_epi * 1e6), 1e-12)
    assert rel(ph.dislocation_lifetime(3.5e-3, 1e6), 1.0 / 3.5e3, 1e-12)
    assert ph.dislocation_lifetime(3.5e-3, 0.0) == float("inf")


def test_scr_sigma_follows_layer_with_larger_part_of_scr():
    a = synth()                                   # N_D⁺ ≫ N_i: ОПЗ в i-слое (p)
    assert ph.scr_sigma_R(a) == a.p_side.sigma_R
    b = synth(sigma_R_sub=1.0).with_scenario(ph.SCENARIO_B)  # N_i ≫ N_sub: ОПЗ в подложке
    assert ph.scr_sigma_R(b) == 1.0


def test_sns_factor_is_continuous_and_bounded():
    s = synth()
    V = np.linspace(-0.5, s.Vbi, 200)
    F = ph.sns_factor(s, V)
    assert np.all(F <= 1.0 + 1e-12) and np.all(F > 0)
    assert np.all(F[V <= 0] == 1.0)
    v3 = 3 * s.Vt
    assert ph.sns_factor(s, v3 - 1e-9) == pytest.approx(float(ph.sns_factor(s, v3)), rel=1e-6)


def test_edge_area_zero_when_scr_inside_mesa():
    assert float(ph.edge_area(synth(), -1.0)) == 0.0


# ------------------------------------------------- Справочник (§6.2, §6.4, §7.2) --

from mesa_diode.simulator import reference as ref



def test_boundary_recommendations_rule():
    b = synth(ND_plus=1e19, N_i=1e16).with_scenario(ph.SCENARIO_B)
    rec = ph.boundary_recommendations(b)
    assert rec["n"][1] == ph.RECOMMEND_REFLECT      # n⁺ в ≥10 раз сильнее
    assert rec["p"][1] == ph.RECOMMEND_SINK         # металл
    assert ph.boundary_recommendations(synth(N_i=1e15, rho_sub=0.05))["p"][1] == ph.RECOMMEND_REFLECT
    assert ph.boundary_recommendations(synth(N_i=1e16, rho_sub=50.0))["p"][1] == ph.RECOMMEND_SINK
    assert ph.boundary_recommendations(synth(N_i=1e15, rho_sub=10.0))["p"][1] == ph.RECOMMEND_INTERMEDIATE


def test_isolation_levels_scenario_b():
    b = synth(N_i=1e16, rho_sub=10.0).with_scenario(ph.SCENARIO_B)
    xn, xp = (float(x) for x in ph.depletion_edges(b, 0.0))
    from dataclasses import replace
    assert ph.isolation_status(replace(b, h=b.d_epi - 2 * xn), 0.0)[0] == ph.NOT_ISOLATED
    assert ph.isolation_status(replace(b, h=b.d_epi), 0.0)[0] == ph.EDGE
    assert ph.isolation_status(replace(b, h=b.d_epi + 2 * xp), 0.0)[0] == ph.ISOLATED


def test_reference_table_builds_and_flags_degenerate_layer():
    data = ref.reference_table(synth(ND_plus=1e19))
    titles = [g.title for g in data.groups]
    assert titles[0] == "Материал" and any(t.startswith("Переход") for t in titles)
    assert any("вырожд" in w for w in data.warnings)


def test_reference_table_flags_empirical_n2_and_sigma_extrapolation():
    data = ref.reference_table(synth(n2=1.8, N_dis=1e3))
    assert any("n₂" in w for w in data.warnings)
    assert any("экстраполяция" in w for w in data.warnings)


def test_reference_table_implant_group_from_metadata():
    meta = {"implant": {"dose_cm2": 2e14, "energy_keV": 30, "anneal": "—"}}
    data = ref.reference_table(synth(), metadata=meta)
    group = next(g for g in data.groups if g.title.startswith("Имплантация"))
    limit = next(r for r in group.rows if r.label.startswith("N_D ≤"))
    assert rel(limit.value, 2e14 / (0.4 * UM), 1e-12)
