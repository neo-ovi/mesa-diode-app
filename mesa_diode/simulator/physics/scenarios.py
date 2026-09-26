# -*- coding: utf-8 -*-
"""§6.6. Сравнение сценариев A и B: (6.7), (6.8); методичка, п. 10.4."""

import numpy as np

from mesa_diode.simulator.physics.checks import isolation_status, low_injection_voltage
from mesa_diode.simulator.physics.circuit import solve_iv
from mesa_diode.simulator.physics.constants import SCENARIO_B, SCENARIOS
from mesa_diode.simulator.physics.electrostatics import (
    c2_cutoff, capacitance, depletion_width, fit_inv_c2,
)
from mesa_diode.simulator.physics.ideality import ideality_from_data


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def current_mismatch(s, V_exp, I_exp):
    """(6.7) δ_I = √(Σ(lg|I_mod| − lg|I_exp|)²/N) по точкам |V| > kT/q;
    для сценария B — только обратная ветвь. Метрика ТЗ."""
    V_exp = np.asarray(V_exp, dtype=float)
    I_exp = np.asarray(I_exp, dtype=float)
    mask = (np.abs(V_exp) > s.Vt) & (I_exp != 0) & np.isfinite(I_exp)
    if s.scenario == SCENARIO_B:
        mask &= V_exp < 0
    if not mask.any():
        return float("nan")
    I_mod = solve_iv(s, V_exp[mask]).I
    ok = I_mod != 0
    return _rms(np.log10(np.abs(I_mod[ok])) - np.log10(np.abs(I_exp[mask][ok])))


def capacitance_mismatch(s, V_exp, C_exp):
    """(6.8) δ_C = √(Σ((C_mod − C_exp)/C_exp)²/N). Метрика ТЗ."""
    V_exp = np.asarray(V_exp, dtype=float)
    C_exp = np.asarray(C_exp, dtype=float)
    mask = np.isfinite(C_exp) & (C_exp > 0)
    if not mask.any():
        return float("nan")
    C_mod = capacitance(s, V_exp[mask])
    return _rms((C_mod - C_exp[mask]) / C_exp[mask])


def scenario_summary(s, exp_iv=None, exp_cv=None, window=None):
    """Строка таблицы §6.6 для одного сценария: V_bi, W(0), C(0), C(−1 В),
    N_eff, отсечка 1/C², I(−1 В), n_мод, V_LI, δ_I, δ_C, изоляция (1.2)."""
    V_fwd = np.linspace(0.0, max(0.5, s.Vbi), 121)
    iv = solve_iv(s, V_fwd)
    ideality = ideality_from_data(V_fwd, iv.I, s.T, s.Rs, window, I_mod=s.I_mod)
    return {
        "scenario": s.scenario,
        "Vbi": s.Vbi,
        "W0": float(depletion_width(s, 0.0)),
        "C0": float(capacitance(s, 0.0)),
        "Cm1": float(capacitance(s, -1.0)),
        "N_eff": s.N_eff,
        "cutoff": c2_cutoff(s),
        "I_m1": float(solve_iv(s, np.array([-1.0])).I[0]),
        "n_mod": ideality.n if ideality else float("nan"),
        "V_LI": low_injection_voltage(s),
        "delta_I": current_mismatch(s, *exp_iv) if exp_iv is not None else float("nan"),
        "delta_C": capacitance_mismatch(s, *exp_cv) if exp_cv is not None else float("nan"),
        "isolation": isolation_status(s, 0.0)[1],
    }


def compare_scenarios(s, exp_iv=None, exp_cv=None, c2_window=None, window=None):
    """Режим «оба» (§6.6): расчёт A и B с общими параметрами, без подгонки.

    exp_iv = (V, I), exp_cv = (V, C) — эксперимент. Главный критерий —
    N из наклона экспериментальной 1/C² (3.5) против N_eff сценариев;
    второй — абсолютная ёмкость. Возвращает {"A": {...}, "B": {...}, "N_exp": ...}."""
    result = {sc: scenario_summary(s.with_scenario(sc), exp_iv, exp_cv, window)
              for sc in SCENARIOS}
    N_exp = float("nan")
    if exp_cv is not None:
        V, C = (np.asarray(a, dtype=float) for a in exp_cv)
        win = c2_window if c2_window is not None else (V.min(), 0.0)
        N_exp, _ = fit_inv_c2(V, C, s.area, s.eps, win)
    result["N_exp"] = N_exp
    return result
