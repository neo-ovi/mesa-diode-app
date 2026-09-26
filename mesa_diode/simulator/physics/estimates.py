# -*- coding: utf-8 -*-
"""Оценки по эксперименту: τ₀ по обратной ветви ВАХ (обращение (5.4))."""

from dataclasses import dataclass, replace

import numpy as np

from mesa_diode.simulator.materials import Q
from mesa_diode.simulator.physics.carriers import _expm1
from mesa_diode.simulator.physics.circuit import shunt_current
from mesa_diode.simulator.physics.constants import MODEL_PHYSICAL
from mesa_diode.simulator.physics.diffusion import diffusion_current
from mesa_diode.simulator.physics.electrostatics import depletion_width
from mesa_diode.simulator.physics.lifetimes import scr_sigma_R
from mesa_diode.simulator.physics.scr import gr_area, sns_factor


@dataclass
class Tau0Estimate:
    V: float
    I_gr: float          # ток ОПЗ, оставшийся после вычета шунта и диффузии, А
    tau0: float          # τ₀ по обращению (5.4), с
    tau0_bg: float       # τ₀^bg по (5.10): 1/τ₀^bg = 1/τ₀ − σ_R·N_dis (nan, если < 0)


def estimate_tau0_from_reverse(s, V_exp, I_exp, V_at=-1.0, diagnostics=None):
    """Оценка τ₀ по обратной ветви: обращение (5.4) при V = V_at.

    Из измеренного |I(V_at)| вычитаются шунт V/R_sh (6.1) и модельный
    диффузионный ток (4.4); остаток считается током ОПЗ (5.4), откуда
    τ₀ = q·n_i·W(V)·A·|e^{qV/n₂kT} − 1| / (2·|I_gr|). Это оценка: она верна,
    если обратный ток определяется генерацией в ОПЗ, а не утечкой по
    поверхности или пробоем. τ₀^bg — из (5.10) с дислокационным вкладом.
    Возвращает Tau0Estimate или None (причина — в diagnostics)."""
    notes = diagnostics if diagnostics is not None else []
    V_exp = np.asarray(V_exp, dtype=float)
    I_exp = np.asarray(I_exp, dtype=float)
    order = np.argsort(V_exp)
    V_exp, I_exp = V_exp[order], I_exp[order]
    if V_at >= 0 or not V_exp[0] <= V_at <= V_exp[-1]:
        notes.append(f"нет экспериментальных точек при V = {V_at:g} В на обратной ветви")
        return None
    I_meas = abs(float(np.interp(V_at, V_exp, I_exp)))
    physical = replace(s, model=MODEL_PHYSICAL)
    I_rest = I_meas - abs(float(shunt_current(physical, V_at))) - abs(float(diffusion_current(physical, V_at)))
    if I_rest <= 0:
        notes.append("обратный ток не превышает сумму шунта и диффузионного тока модели — "
                     "τ₀ по нему не оценить")
        return None
    W = float(depletion_width(physical, V_at))
    factor = abs(float(_expm1(V_at / (s.n2 * s.Vt))))
    area = float(gr_area(physical, V_at))
    if s.sns_refinement:
        factor *= float(sns_factor(physical, V_at))
    tau0 = Q * s.ni * W * area * factor / (2.0 * I_rest)
    inv_bg = 1.0 / tau0 - scr_sigma_R(physical) * s.N_dis
    tau0_bg = 1.0 / inv_bg if inv_bg > 0 else float("nan")
    if not np.isfinite(tau0_bg):
        notes.append("дислокационный вклад σ_R·N_dis один даёт больше тока ОПЗ, чем измерено")
    return Tau0Estimate(V=V_at, I_gr=I_rest, tau0=tau0, tau0_bg=tau0_bg)
