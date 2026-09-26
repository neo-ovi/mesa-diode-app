# -*- coding: utf-8 -*-
"""§4. Диффузионный ток и граничные условия базы: (4.1)–(4.4), (4.3а).

Методичка, гл. 6 (п. 6.4, 6.4а — граница «соседний слой»)."""

import numpy as np

from mesa_diode.simulator.materials import Q
from mesa_diode.simulator.physics.carriers import (
    _expm1, diffusion_coefficient, diffusion_length, equilibrium_carriers,
)
from mesa_diode.simulator.physics.constants import LAYER, LONG, REFLECT, SINK
from mesa_diode.simulator.physics.electrostatics import depletion_edges
from mesa_diode.simulator.physics.lifetimes import minority_lifetime


W_MIN = 1e-7      # см (1 нм): минимальная толщина нейтральной базы при смыкании
U_SMALL = 1e-6    # при u < 10⁻⁶: coth u = 1/u, tanh u = u
U_LARGE = 20.0    # при u > 20: f = 1


def boundary_factor(u, boundary, r=float("inf")):
    """f(w/L) в (4.3): 1 — длинная база [Зи, ур. (40)]; coth — «сток»,
    Δn(w) = 0 [Ш49, с. 470, ур. (5.5)]; tanh — «отражение», dΔn/dx(w) = 0
    [Ш49, с. 470, ур. (5.6) при p₁ = p₂].

    r = S·L/D — «сток» с конечной скоростью отвода носителей S на границе,
    −D·dΔn/dx = S·Δn (4.3а): f = (r + tanh u)/(r·tanh u + 1). При r → ∞ это
    coth u, при r = 0 — tanh u; при u → 0 f → r, и ток ограничен потоком
    q·n₀·S, а не растёт как 1/w (смыкание ОПЗ с границей слоя)."""
    u = np.asarray(u, dtype=float)
    if boundary == LONG:
        return np.ones_like(u)
    small = u < U_SMALL
    safe = np.where(small, U_SMALL, u)
    if boundary in (SINK, LAYER) and np.isfinite(r):
        t = np.where(small, u, np.tanh(safe))
        f = (r + t) / (r * t + 1.0)
    elif boundary in (SINK, LAYER):
        f = np.where(small, 1.0 / np.maximum(u, 1e-300), 1.0 / np.tanh(safe))
    elif boundary == REFLECT:
        f = np.where(small, u, np.tanh(safe))
    else:
        raise ValueError(f"неизвестное граничное условие: {boundary!r}")
    return np.where(u > U_LARGE, 1.0, f)


def neutral_widths(s, V):
    """Толщины нейтральных баз w_n = d_n-стороны − x_n(V), w_p = d_p-стороны − x_p(V), см.
    Возвращает (w_n, w_p, смыкание) — w ограничены снизу W_MIN."""
    xn, xp = depletion_edges(s, V)
    wn = s.n_side.thickness - xn
    wp = s.p_side.thickness - xp
    punch = bool(np.any(wn <= 0) or np.any(wp <= 0))
    return np.maximum(wn, W_MIN), np.maximum(wp, W_MIN), punch


def minority_transport(s):
    """D и L неосновных: дырки в n-области, электроны в p-области (2.5), (2.6), τ по (5.9).
    Возвращает ((D_p, L_p), (D_n, L_n))."""
    Dp = diffusion_coefficient(s.n_side.mu_minority, s.T)
    Dn = diffusion_coefficient(s.p_side.mu_minority, s.T)
    Lp = diffusion_length(Dp, minority_lifetime(s.n_side, s.N_dis))
    Ln = diffusion_length(Dn, minority_lifetime(s.p_side, s.N_dis))
    return (Dp, Lp), (Dn, Ln)


def boundary_velocity(s, side, minority0, vth):
    """Скорость отвода неосновных носителей S на дальней границе базы, см/с (4.3а).

    «Сток» — идеальный, S = ∞ (как в эталонах ТЗ §9).
    «Соседний слой» того же типа (изотипная граница i/n⁺, i/подложка):
    квазиуровень Ферми неосновных непрерывен через границу — то же условие,
    что (4.1) на краю ОПЗ [Зи, с. 92, ур. (28)], — поэтому Δn₂/n₂₀ = Δn₁/n₁₀,
    и поток в соседний слой по (4.3) даёт S = (D₂/L₂)·(n₂₀/n₁₀)·f₂(w₂/L₂)
    (вывод); f₂ — сток к контакту за соседним слоем. Сильнее легированный
    сосед (n₂₀ ≪ n₁₀) почти «отражает», слабее — почти «сток». Последовательно
    с тепловым ограничением (оценка): 1/S = 1/S₂ + 1/v_th. Без соседа — ∞."""
    if side.boundary == LAYER and side.neighbour is not None:
        nb = side.neighbour
        D2 = diffusion_coefficient(nb.mu_minority, s.T)
        L2 = diffusion_length(D2, minority_lifetime(nb, s.N_dis))
        minority2 = equilibrium_carriers(nb.N, s.ni)[1]
        S = D2 / L2 * minority2 / minority0 * float(boundary_factor(nb.thickness / L2, SINK))
        return 1.0 / (1.0 / max(S, 1e-300) + 1.0 / vth)
    return float("inf")


def saturation_current_density_parts(s, V):
    """(4.3) Слагаемые J_s(V): дырки в n-области и электроны в p-области, А/см².

    Вывод: граничное условие (4.1) n_p(−x_p) = n_p0·e^{qV/kT} [Зи, с. 92, ур. (28),
    (31)–(33)]; [Ш49, с. 458, ур. (4.4)]; уравнение диффузии (4.2)
    d²Δn/dx′² − Δn/L² = 0 [Зи, с. 94, ур. (39)]; [Ш49, с. 470, ур. (5.4)].
    J_s = qD_p·p_n0/L_p·f(w_n/L_p) + qD_n·n_p0/L_n·f(w_p/L_n) —
    [Зи, с. 94, ур. (44), (45)]; [Ш49, с. 460, ур. (4.13)]."""
    (Dp, Lp), (Dn, Ln) = minority_transport(s)
    p_n0, n_p0 = s.minority
    wn, wp, _ = neutral_widths(s, V)
    r_p = boundary_velocity(s, s.n_side, p_n0, s.material.vth_p) * Lp / Dp
    r_n = boundary_velocity(s, s.p_side, n_p0, s.material.vth_n) * Ln / Dn
    holes = Q * Dp * p_n0 / Lp * boundary_factor(wn / Lp, s.n_side.boundary, r_p)
    electrons = Q * Dn * n_p0 / Ln * boundary_factor(wp / Ln, s.p_side.boundary, r_n)
    return holes, electrons


def saturation_current_density(s, V):
    """(4.3) J_s(V), А/см²."""
    holes, electrons = saturation_current_density_parts(s, V)
    return holes + electrons


def diffusion_current(s, V):
    """(4.4) I_diff = A·J_s(V)·(e^{qV/kT} − 1), А."""
    V = np.asarray(V, dtype=float)
    return s.area * saturation_current_density(s, V) * _expm1(V / s.Vt)
