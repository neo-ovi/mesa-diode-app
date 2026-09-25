# -*- coding: utf-8 -*-
"""Автоподгонка ВАХ (§6.7): восстановление известных параметров на синтетике
и выбор механизмов по критерию (6.11). Реальная ВАХ — только с MESA_DATA_DIR."""

import numpy as np
import pytest

from mesa_diode.simulator import fitting, presets
from mesa_diode.simulator import physics as ph


def _empirical(**changes):
    params = {**presets.DEFAULT_PARAMS, "mode": presets.MODE_BASIC, "D_um": 500.0,
              "J0_emp": 1.5e-2, "n_emp": 1.3, "Rs": 130.0, "Rsh": 1.1e4, "I_L": 1.5e-5,
              "m_leak": 2.3, "I_mod": 0.08, **changes}
    return presets.to_structure(params)


def _synthetic(s, noise=0.0, quantum=0.0):
    V = np.linspace(-6.0, 1.5, 600)
    I = ph.solve_iv(s, V).I
    if noise:
        I = I * (1 + noise * np.random.default_rng(3).standard_normal(V.size))
    if quantum:
        I = np.round(I / quantum) * quantum
    return V, I


def test_series_resistance_modulation():
    s = _empirical()
    assert ph.series_resistance(s, 0.0) == pytest.approx(130.0)
    assert ph.series_resistance(s, 0.08) == pytest.approx(65.0)
    assert ph.series_resistance(presets.to_structure({"Rs": 50.0}), 1.0) == pytest.approx(50.0)


def test_solver_satisfies_circuit_equation_with_modulation():
    s = _empirical()
    V = np.linspace(-6, 1.5, 200)
    r = ph.solve_iv(s, V)
    assert np.all(np.isfinite(r.I))
    assert np.allclose(r.Vd + r.I * ph.series_resistance(s, r.I), V, atol=1e-9)
    assert np.allclose(ph.junction_current(s, r.Vd), r.I, rtol=1e-9, atol=1e-15)


def test_empirical_fit_recovers_parameters_and_mechanisms():
    s = _empirical()
    V, I = _synthetic(s, noise=0.002, quantum=1e-7)
    start = replace_params(s, Rs=100.0, Rsh=1e5, I_L=0.0, I_mod=float("inf"))
    r = fitting.fit_iv(start, V, I, fitting.EMPIRICAL, target=0.0)      # только BIC
    assert set(r.terms) == {fitting.TERM_LEAK, fitting.TERM_MOD}
    assert r.error < 0.01
    for key, true in (("n_emp", 1.3), ("Rs", 130.0), ("m_leak", 2.3), ("J0_emp", 1.5e-2)):
        assert r.params[key] == pytest.approx(true, rel=0.05), key


def test_fit_rejects_mechanisms_absent_in_data():
    s = _empirical(I_L=0.0, I_mod=float("inf"))
    V, I = _synthetic(s, noise=0.002)
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL)
    assert r.terms == ()
    assert r.error < 0.01
    assert any("выбрана" in n for n in r.notes)


def test_physical_fit_recovers_lifetime():
    params = {**presets.DEFAULT_PARAMS, "mode": presets.MODE_FIT, "tau0_bg": 3e-8, "Rs": 50.0,
              "Rsh": 1e5}
    s = presets.to_structure(params)
    V, I = _synthetic(s, noise=0.002)
    start = replace_params(s, tau0_bg=1e-6, tau_n_bg=1e-5, tau_p_bg=1e-5, Rs=20.0)
    r = fitting.fit_iv(start, V, I, fitting.PHYSICAL)
    assert r.error < 0.02
    assert r.params["tau0_bg"] == pytest.approx(3e-8, rel=0.15)


def test_reverse_exponent_and_error_metric():
    V = np.linspace(-6, -0.5, 50)
    assert fitting.reverse_exponent(V, -1e-5 * np.abs(V) ** 2.2) == pytest.approx(2.2, rel=1e-6)
    assert fitting.relative_error(np.array([1.1, 2.0]), np.array([1.0, 2.0]), 1e-3) == pytest.approx(
        np.sqrt(0.01 / 2))


def replace_params(s, **changes):
    from dataclasses import replace
    return replace(s, **changes)


def test_reference_iv_is_described_within_one_percent():
    """ВАХ опорного образца — из его набора в каталоге данных (files.iv)."""
    ref = presets.reference_preset()
    files = [p for p in ref.resolved_files("iv") if p.is_file()] if ref else []
    if not files:
        pytest.skip("ВАХ опорного образца недоступна: задайте MESA_DATA_DIR")
    from mesa_diode.simulator.io import load_xy_file
    V, I = load_xy_file(files[0])
    s = presets.to_structure({**ref.params, "mode": presets.MODE_BASIC})
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL)
    assert fitting.TERM_LEAK in r.terms and r.error <= fitting.TARGET_ERROR
    strict = fitting.fit_iv(s, V, I, fitting.EMPIRICAL, target=0.0)
    assert strict.error < 0.01 and set(strict.terms) == {fitting.TERM_LEAK, fitting.TERM_MOD}


def test_locked_parameters_keep_entered_values():
    """Зафиксированные поля не меняются, остальное подбирается."""
    s = _empirical()
    V, I = _synthetic(s, noise=0.002, quantum=1e-7)
    start = replace_params(s, Rs=130.0, n_emp=1.5, I_L=0.0, I_mod=float("inf"))
    r = fitting.fit_iv(start, V, I, fitting.EMPIRICAL, locked={"Rs", "tau_n_bg"}, target=0.0)
    assert r.params["Rs"] == 130.0 and "Rs" in r.locked
    assert r.params["n_emp"] == pytest.approx(1.3, rel=0.05)
    assert any("Зафиксированы" in n for n in r.notes)


def test_locked_mechanism_switch():
    """I_L = 0 с галочкой выключает утечку, ненулевой I_mod с галочкой — включает модуляцию."""
    s = _empirical()
    V, I = _synthetic(s, noise=0.002, quantum=1e-7)
    off = fitting.fit_iv(replace_params(s, I_L=0.0), V, I, fitting.EMPIRICAL, locked={"I_L"})
    assert all(fitting.TERM_LEAK not in c.terms for c in off.candidates)
    on = fitting.fit_iv(s, V, I, fitting.EMPIRICAL, locked={"I_mod"})
    assert all(fitting.TERM_MOD in c.terms for c in on.candidates)
    assert on.params["I_mod"] == 0.08


def test_everything_locked_only_evaluates():
    s = _empirical()
    V, I = _synthetic(s)
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL, locked=fitting.fittable_fields(fitting.EMPIRICAL))
    assert r.error < 1e-6 and len(r.candidates) == 1


def test_fittable_fields():
    assert fitting.fittable_fields(fitting.EMPIRICAL) == {"J0_emp", "n_emp", "Rs", "Rsh", "I_L",
                                                          "m_leak", "I_mod"}
    phys = fitting.fittable_fields(fitting.PHYSICAL)
    assert {"tau0_bg", "tau_n_bg", "tau_p_bg"} <= phys and "n_emp" not in phys
    assert "rho_sub" in presets.editable_keys(presets.MODE_BASIC)


def test_simplest_adequate_variant_is_chosen():
    """Физичность прежде точности: из вариантов с δ ≤ 5 % — самый простой."""
    s = _empirical()
    V, I = _synthetic(s, noise=0.002, quantum=1e-7)
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL)
    chosen = next(c for c in r.candidates if c.accepted)
    assert chosen.error <= fitting.TARGET_ERROR
    assert all(len(c.terms) >= len(chosen.terms) for c in r.candidates if c.error <= fitting.TARGET_ERROR)
    assert any("самый простой" in n for n in r.notes)


def test_selection_falls_back_to_bic():
    cands = [fitting.Candidate((), {}, 0.30, 100.0), fitting.Candidate(("leak",), {}, 0.20, 50.0),
             fitting.Candidate(("mod",), {}, 0.25, 95.0)]
    assert fitting.select_variant(cands).terms == ("leak",)
    cands[0].error = 0.04
    assert fitting.select_variant(cands).terms == ()


def test_ideality_bound_signals_other_mechanism():
    """n > 2 в данных: подгонка упирается в границу и объясняет это."""
    s = _empirical(n_emp=2.0, I_L=0.0, I_mod=float("inf"), Rs=1.0)
    V = np.linspace(-2, 0.8, 300)
    I = s.area * s.J0_emp * np.expm1(V / (3.0 * ph.thermal_voltage(300.0))) + V / s.Rsh
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL)
    assert "n_emp" in r.at_bounds
    assert any("n вне [1, 2]" in n for n in r.notes)


def test_diode_resistance_is_not_mistaken_for_modulation():
    """Спад dV/dI самого диода (n·kT/qI) не выдаётся за модуляцию R_s."""
    s = _empirical(I_L=0.0, I_mod=float("inf"), Rs=50.0)
    V, I = _synthetic(s, noise=0.001)
    r = fitting.fit_iv(s, V, I, fitting.EMPIRICAL)
    assert not any("падает с ростом тока" in n for n in r.notes)
