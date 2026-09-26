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
    assert b.n_side.boundary == ph.REFLECT and b.n_side.neighbour.layer == "n+"   # умолчание ТЗ
    assert a.p_side.neighbour.layer == "sub"
    ui = presets.to_structure({"i_type": presets.I_TYPE_N})                      # умолчание окна
    assert ui.n_side.boundary == ph.LAYER
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
    limit = next(r for r in group.rows if r.label.startswith("N_{D} ≤"))
    assert rel(limit.value, 2e14 / (0.4 * UM), 1e-12)


# ----------------------------------------------- §6.3 и §6.6 --

def test_ideality_on_synthetic_data():
    Vt = ph.thermal_voltage(T300)
    V = np.linspace(0.08, 0.2, 40)
    I = 1e-9 * np.expm1(V / (1.4 * Vt))
    result = ph.ideality_from_data(V, I, T300)
    assert result.n == pytest.approx(1.400, abs=1e-3)


def test_ideality_corrects_series_resistance():
    Vt = ph.thermal_voltage(T300)
    Vj = np.linspace(0.08, 0.3, 60)
    I = 1e-9 * np.expm1(Vj / (1.4 * Vt))
    Rs = 50.0
    result = ph.ideality_from_data(Vj + I * Rs, I, T300, Rs=Rs)
    assert result.n == pytest.approx(1.400, abs=1e-3)


def test_ideality_window_stops_where_local_n_rises():
    Vt = ph.thermal_voltage(T300)
    Vj = np.linspace(0.0, 0.4, 161)
    I = 1e-9 * np.expm1(Vj / Vt)
    V = Vj + I * 2000.0            # изгиб от R_s, не учтённого в анализе
    result = ph.ideality_from_data(V, I, T300)
    assert result.V1 == pytest.approx(3 * Vt)
    assert result.V2 < V.max()
    # окно по правилу ТЗ допускает рост локального n до 20 % — отсюда смещение оценки
    assert result.n == pytest.approx(1.0, abs=0.05)


def test_manual_window_is_respected():
    Vt = ph.thermal_voltage(T300)
    V = np.linspace(0.08, 0.2, 40)
    result = ph.ideality_from_data(V, 1e-9 * np.expm1(V / (1.4 * Vt)), T300, window=(0.1, 0.15))
    assert (result.V1, result.V2) == (0.1, 0.15)


def test_effective_ideality_equal_components():
    assert ph.effective_ideality(1.0, 1.0) == pytest.approx(1.333, abs=5e-3)


def test_gr_share_limits():
    assert ph.gr_share_from_ideality(1.0) == 0.0
    assert ph.gr_share_from_ideality(2.0) == pytest.approx(1.0)


def test_model_ideality_is_one_for_pure_diffusion():
    s = synth(tau0_bg=1e30, Rs=0.0, Rsh=float("inf"), bc_A_n=ph.LONG, bc_A_p=ph.LONG)
    V = np.linspace(0.0, 0.2, 81)
    result = ph.ideality_from_data(V, ph.solve_iv(s, V).I, T300)
    assert result.n == pytest.approx(1.0, abs=0.01)


def test_compare_scenarios_recognises_source_scenario():
    s = synth(N_i=1e16, rho_sub=10.0)
    truth = s.with_scenario(ph.SCENARIO_A)
    V = np.linspace(-1.0, -0.1, 19)
    C = ph.capacitance(truth, V)
    I = ph.solve_iv(truth, V).I
    result = ph.compare_scenarios(s, exp_iv=(V, I), exp_cv=(V, C))
    assert result["A"]["delta_C"] < 1e-9 and result["A"]["delta_I"] < 1e-9
    assert result["B"]["delta_C"] > 0.1
    assert result["N_exp"] == pytest.approx(result["A"]["N_eff"], rel=5e-3)


# ------------------------------------------------------------ §5.4 наборы --

from mesa_diode.simulator import presets


def test_preset_roundtrip_is_lossless(tmp_path):
    original = presets.default_preset()
    original.params["N_i"] = 3.3e15
    original.params["i_type"] = presets.I_TYPE_BOTH
    original.metadata = {"implant": {"dose_cm2": 2e14}, "note": "проверка"}
    original.files = {"iv": ["iv/a.csv"], "cv": []}
    original.notes = {"6.3": "текст"}
    saved = presets.save_preset(original, tmp_path / "set.json")
    loaded = presets.load_preset(saved.path)
    assert loaded.to_json() == original.to_json()
    assert loaded.resolved_files("iv") == [tmp_path / "iv/a.csv"]


def test_preset_rejects_foreign_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        presets.load_preset(path)


def test_default_preset_builds_structure_in_calc_units():
    s = presets.to_structure(presets.DEFAULT_PARAMS)
    assert s.scenario == ph.SCENARIO_B
    assert s.D == pytest.approx(500e-4)
    assert presets.to_structure({"i_type": presets.I_TYPE_P}).scenario == ph.SCENARIO_A


def test_startup_without_data_dir_is_neutral(monkeypatch):
    monkeypatch.delenv("MESA_DATA_DIR", raising=False)
    assert presets.reference_preset() is None
    assert presets.startup_preset().params == presets.DEFAULT_PARAMS


def test_auto_window_keeps_at_least_three_points_on_sparse_data():
    Vt = ph.thermal_voltage(T300)
    Vj = np.linspace(0.02, 0.36, 14)
    I = 1e-9 * np.expm1(Vj / Vt)
    result = ph.ideality_from_data(Vj + I * 3e5, I, T300)  # сильный изгиб от R_s
    assert result is not None
    assert ((result.V_local >= result.V1) & (result.V_local <= result.V2)).sum() >= 3


# ------------------------------------- идеальность: реальные условия данных --

def _dense_iv(n=1.5, I0=2e-6, Rs=120.0, noise=0.0, seed=1, vmax=1.5):
    """Плотная ВАХ диода с заметным R_s (синтетика, не данные образца)."""
    from scipy.optimize import brentq
    Vt = ph.thermal_voltage(T300)

    def current(V):
        return brentq(lambda I: I - I0 * np.expm1(np.clip((V - I * Rs) / (n * Vt), -700, 700)), -1, 1)

    V = np.linspace(-1.0, vmax, 2500)
    I = np.array([current(v) for v in V])
    if noise:
        I = I * (1 + noise * np.random.default_rng(seed).standard_normal(I.size))
    return V, I


def test_ideality_dense_noisy_data_with_correct_rs():
    V, I = _dense_iv(noise=0.01)
    notes = []
    result = ph.ideality_from_data(V, I, T300, Rs=120.0, diagnostics=notes)
    assert result.n == pytest.approx(1.5, abs=0.02)
    assert notes == []


def test_ideality_reports_overestimated_rs():
    V, I = _dense_iv()
    notes = []
    result = ph.ideality_from_data(V, I, T300, Rs=500.0, diagnostics=notes)
    assert any("завышено" in note for note in notes)
    assert result is None or result.n < 1.5


def test_ideality_reports_reason_when_no_forward_points():
    notes = []
    assert ph.ideality_from_data(np.array([-1.0, -0.5]), np.array([-1e-6, -5e-7]), T300,
                                 diagnostics=notes) is None
    assert notes and "прямой ветви" in notes[0]


def test_ideality_ignores_repeated_voltages():
    V, I = _dense_iv(Rs=0.0, vmax=0.4)
    V2 = np.concatenate([V, V[::7]])
    I2 = np.concatenate([I, I[::7]])
    result = ph.ideality_from_data(V2, I2, T300)
    assert result.n == pytest.approx(1.5, abs=0.01)


# ------------------------------------------ эмпирическая модель и оценка τ₀ --

def test_empirical_model_follows_6_3():
    s = synth(model=ph.MODEL_EMPIRICAL, J0_emp=1e-5, n_emp=1.4, Rs=0.0, Rsh=float("inf"))
    V = np.linspace(-0.5, 0.3, 17)
    r = ph.solve_iv(s, V)
    expected = s.area * 1e-5 * np.expm1(V / (1.4 * s.Vt))
    assert np.allclose(r.I, expected, rtol=1e-12)
    assert set(r.parts) == {"emp", "sh", "L"}


def test_empirical_model_ideality_is_recovered():
    s = synth(model=ph.MODEL_EMPIRICAL, J0_emp=1e-5, n_emp=1.3, Rs=0.0, Rsh=float("inf"))
    V = np.linspace(0.0, 0.3, 121)
    assert ph.ideality_from_data(V, ph.solve_iv(s, V).I, T300).n == pytest.approx(1.3, abs=1e-3)


def test_tau0_estimate_recovers_model_lifetime():
    truth = synth(tau0_bg=3e-8, N_dis=1e5, Rsh=1e7)
    V = np.linspace(-2.0, 0.0, 81)
    I = ph.solve_iv(truth, V).I
    guess = synth(tau0_bg=1e-5, N_dis=1e5, Rsh=1e7)   # τ₀ неизвестно
    est = ph.estimate_tau0_from_reverse(guess, V, I, -1.0)
    assert est.tau0 == pytest.approx(ph.scr_lifetime(truth), rel=1e-3)
    assert est.tau0_bg == pytest.approx(3e-8, rel=1e-3)


def test_tau0_estimate_reports_when_current_is_shunt_only():
    s = synth(Rsh=1e3)
    V = np.linspace(-2.0, 0.0, 21)
    notes = []
    assert ph.estimate_tau0_from_reverse(s, V, V / 1e4, -1.0, notes) is None
    assert notes


def test_modes_select_model_and_editable_fields():
    basic = presets.to_structure({**presets.DEFAULT_PARAMS, "mode": presets.MODE_BASIC})
    assert basic.model == ph.MODEL_EMPIRICAL
    for mode in (presets.MODE_EXTENDED, presets.MODE_FIT):
        assert presets.to_structure({"mode": mode}).model == ph.MODEL_PHYSICAL
    basic_keys = presets.editable_keys(presets.MODE_BASIC)
    extended = presets.editable_keys(presets.MODE_EXTENDED)
    fit = presets.editable_keys(presets.MODE_FIT)
    # n и J₀ (6.3) — только базовый режим; расширенный — всё измеряемое; подгонка — плюс неизмеряемое
    assert basic_keys - presets.MODEL_KEYS < extended < fit
    assert not presets.MODEL_KEYS & (extended | fit)
    assert {"d_epi_um", "d_n_um", "d_sub_um", "rho_sub", "N_dis", "implant_dose"} <= extended
    assert {"mu_n", "tau0_bg", "sigma_R_epi", "n2"}.isdisjoint(extended)
    # эквивалентная схема (R_s, I_mod, R_sh, I_L, m) — во всех режимах
    assert presets.CIRCUIT_KEYS <= basic_keys and presets.CIRCUIT_KEYS <= extended
    assert fit | presets.MODEL_KEYS == {s.key for s in presets.NUMERIC_PARAMS}
    # поля другой модели тока базового режима не нужны
    two = presets.editable_keys(presets.MODE_BASIC, ph.MODEL_TWO_DIODE)
    assert {"J01_2d", "J02_2d"} <= two and not presets.EMPIRICAL_KEYS & two
    s2 = presets.to_structure({"mode": presets.MODE_BASIC, "basic_model": ph.MODEL_TWO_DIODE})
    assert s2.model == ph.MODEL_TWO_DIODE


def test_empty_field_outside_mode_uses_default():
    s = presets.to_structure({**presets.DEFAULT_PARAMS, "mu_n": None, "mode": presets.MODE_EXTENDED})
    assert s.mu_n == presets.DEFAULT_PARAMS["mu_n"]


def test_autofill_fills_only_empty_non_stored_fields():
    values = {**presets.DEFAULT_PARAMS, "N_i": None, "Rs": None, "afm_rms_nm": None}
    filled = presets.autofill(values, presets.editable_keys(presets.MODE_EXTENDED))
    assert set(filled) == {"N_i", "Rs"}
    assert filled["N_i"] == (presets.DEFAULT_PARAMS["N_i"], presets.AUTO_DEFAULT)


def test_autofill_estimates_nd_plus_from_dose_and_takes_ideality():
    values = {**presets.DEFAULT_PARAMS, "ND_plus": None, "implant_dose": 1e15, "d_n_um": 0.2,
              "n_emp": None, "J0_emp": None}
    filled = presets.autofill(values, presets.editable_keys(presets.MODE_EXTENDED))
    assert filled["ND_plus"][0] == pytest.approx(1e15 / 0.2e-4)
    assert filled["ND_plus"][1] == presets.AUTO_FROM_DOSE
    ideality = ph.IdealityResult(n=1.42, dn=0.0, I0=2e-9, V1=0.1, V2=0.2,
                                 V_local=np.array([]), n_local=np.array([]))
    filled = presets.autofill(values, presets.editable_keys(presets.MODE_BASIC), ideality)
    area = np.pi * (values["D_um"] * 1e-4) ** 2 / 4
    assert filled["n_emp"] == (1.42, presets.AUTO_FROM_IV)
    assert filled["J0_emp"][0] == pytest.approx(2e-9 / area, rel=1e-2)


def test_preset_without_mode_opens_in_fit_mode(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"format": "mesa-diode-preset/1", "params": {"N_i": 1e15}}', encoding="utf-8")
    preset = presets.load_preset(path)
    assert preset.params["mode"] == presets.MODE_FIT
    assert presets.to_structure(preset.params).model == ph.MODEL_PHYSICAL


def test_series_resistance_limit_is_close_to_true_rs():
    for rs in (30.0, 120.0):
        s = presets.to_structure({**presets.DEFAULT_PARAMS, "mode": presets.MODE_FIT, "Rs": rs,
                                  "tau0_bg": 3e-8})
        V = np.linspace(-3, 1.5, 300)
        limit = ph.series_resistance_limit(V, ph.solve_iv(s, V).I)
        # dV/dI верха прямой ветви = R_s + nkT/(qI): чуть больше истинного R_s
        assert rs < limit < 1.1 * rs


def test_ideality_failure_text_links_rs_and_gives_solution():
    from mesa_diode.simulator import hints
    short, text = hints.ideality_failure_text("a.csv", ["x"], 3000.0, 123.0, found=False)
    assert short == "R_s завышено"
    assert "V_j = V − I·R_s" in text and "R_s < 123 Ом" in text and "3000" in text
    short, text = hints.ideality_failure_text("a.csv", ["окно мало"], 10.0, 123.0, found=False)
    assert "окно мало" in text and "V₁ … V₂" in text
    assert hints.ideality_failure_text("a.csv", [], 10.0, 123.0, found=True) == ("", "")


# ------------------------------------- 3.3: граница «соседний слой» (4.3а) --

def test_finite_velocity_boundary_limits():
    u = np.array([1e-3, 0.5, 3.0])
    assert np.allclose(ph.boundary_factor(u, ph.SINK, r=1e12), 1.0 / np.tanh(u), rtol=1e-6)
    assert np.allclose(ph.boundary_factor(u, ph.LAYER, r=0.0), np.tanh(u), rtol=1e-12)
    assert ph.boundary_factor(np.array([1e-9]), ph.LAYER, r=7.0)[0] == pytest.approx(7.0, rel=1e-6)


def test_layer_boundary_between_sink_and_reflect():
    base = {"mode": presets.MODE_EXTENDED, "i_type": presets.I_TYPE_N, "N_i": 3e16, "ND_plus": 1e19}
    parts = {bc: ph.saturation_current_density_parts(presets.to_structure({**base, "bc_B_n": bc}), 0.0)[0]
             for bc in (ph.SINK, ph.REFLECT, ph.LAYER)}
    assert parts[ph.REFLECT] < parts[ph.LAYER] < parts[ph.SINK]
    # более слабо легированный n⁺ «отражает» хуже → ток дырок больше
    weak = ph.saturation_current_density_parts(presets.to_structure({**base, "ND_plus": 1e17}), 0.0)[0]
    assert weak > parts[ph.LAYER]


def test_punch_through_current_stays_finite():
    """Смыкание ОПЗ с границей i/подложка (сценарий A, N_i = 10¹⁴): ток — поток в подложку, не ∝ 1/w."""
    s = presets.to_structure({"mode": presets.MODE_EXTENDED, "i_type": presets.I_TYPE_P, "N_i": 1e14,
                              "ND_plus": 1e17})
    assert ph.neutral_widths(s, -3.0)[2]
    assert abs(ph.solve_iv(s, np.array([-3.0])).I[0]) < 1e-3


def test_iv_depends_on_weakly_doped_side():
    """Физическая модель: ток задаёт слабо легированная сторона (§6.8 методички)."""
    V = np.array([-1.0, 0.15])

    def current(**changes):
        params = {"mode": presets.MODE_EXTENDED, "i_type": presets.I_TYPE_P, "rho_sub": 40.0, **changes}
        return ph.solve_iv(presets.to_structure(params), V).I

    assert abs(current(N_i=2.9e14)[0]) > 5 * abs(current(N_i=2.9e17)[0])      # A: N_i важна
    assert current(ND_plus=1e17)[0] != current(ND_plus=1e20)[0]
    empirical = {"mode": presets.MODE_BASIC}
    same = [ph.solve_iv(presets.to_structure({**empirical, "N_i": n}), V).I for n in (1e14, 1e17)]
    assert np.allclose(same[0], same[1])                                       # базовая: по построению


# ------------------------------------------ 3.5: подвижность μ(N, T) (2.9) --

def test_mobility_doping_and_temperature():
    mu_n, mu_p = ph.mobility(1e13)
    assert (mu_n, mu_p) == (GE.mu_n_max, GE.mu_p_max)                      # g = 1 при малых N
    assert ph.mobility(1e18)[0] == pytest.approx(0.404 * GE.mu_n_max)       # [Зи, рис. 18]
    assert ph.mobility(1e18)[1] == pytest.approx(0.180 * GE.mu_p_max)
    assert ph.lattice_mobility(330.0)[0] == pytest.approx(GE.mu_n_max * 1.1 ** -1.66)
    # кривая Ирвина: ρ монотонно падает с N, p-Ge при 10¹⁶ ≈ 0.45 Ом·см
    Ns = np.logspace(15, 20, 30)
    rho = [ph.resistivity(N, 300.0) for N in Ns]
    assert all(a > b for a, b in zip(rho, rho[1:]))
    assert ph.resistivity(1e16, 300.0) == pytest.approx(0.447, rel=0.02)
    assert ph.resistivity(1e16, 300.0, kind="n") < ph.resistivity(1e16, 300.0)
