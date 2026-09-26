# -*- coding: utf-8 -*-
"""§5. Генерация–рекомбинация в ОПЗ: (5.1)–(5.6); методичка, гл. 7."""

import numpy as np

from mesa_diode.simulator.materials import Q
from mesa_diode.simulator.physics.carriers import _expm1
from mesa_diode.simulator.physics.electrostatics import depletion_width, edge_area
from mesa_diode.simulator.physics.lifetimes import scr_lifetime


def sns_factor(s, V):
    """(5.5) Опция «уточнение SNS»: F = min{1, π(kT/q)/(V_bi − V)} при V ≥ 3kT/q —
    [СНШ57, с. 1231, ур. (15), (16)]; на 0 < V < 3kT/q — линейная
    интерполяция от 1 (интерполяция); при V ≤ 0 F = 1."""
    V = np.asarray(V, dtype=float)

    def F(v):
        gap = s.Vbi - v
        return np.where(gap > np.pi * s.Vt, np.pi * s.Vt / np.where(gap > 0, gap, 1.0), 1.0)

    v3 = 3.0 * s.Vt
    F3 = F(v3)
    return np.where(V >= v3, F(V), np.where(V > 0, 1.0 + (F3 - 1.0) * V / v3, 1.0))


def gr_area(s, V):
    """Площадь для тока ОПЗ: A или A + ΔA (1.3) при включённой опции."""
    return s.area + edge_area(s, V) if s.edge_area else s.area * np.ones_like(np.asarray(V, dtype=float))


def gr_current(s, V):
    """(5.4) I_gr = A·q·n_i·W(V)/(2τ₀)·(e^{qV/n₂kT} − 1), А.

    Вывод: скорость рекомбинации (5.1) U = (pn − n_i²)/[τ_p0(n + n₁) + τ_n0(p + p₁)]
    [СНШ57, с. 1230, ур. (6)]; [Зи, с. 98, ур. (51)]; её максимум при n = p
    (5.3) U_max = n_i/(2τ₀)·(e^{qV/2kT} − 1) [СНШ57, с. 1231, ур. (13)].

    При n₂ = 2 при обратном смещении это [СНШ57, с. 1230, ур. (11)] =
    [Зи, с. 97, ур. (48)], при прямом — [Зи, с. 99, ур. (54)] (верхняя
    оценка). n₂ ≠ 2 — эмпирика (5.6) [Ман20, с. 45–46]; τ₀ по (5.10)."""
    V = np.asarray(V, dtype=float)
    J = Q * s.ni * depletion_width(s, V) / (2.0 * scr_lifetime(s)) * _expm1(V / (s.n2 * s.Vt))
    if s.sns_refinement:
        J = J * sns_factor(s, V)
    return gr_area(s, V) * J
