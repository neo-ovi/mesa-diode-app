# -*- coding: utf-8 -*-
"""§1, §3. Геометрия мезы, V_bi, ОПЗ, ёмкость, изоляция перехода травлением.

Формулы (1.1)–(1.3), (3.1)–(3.5); методичка, гл. 2, 4, 5."""

import numpy as np

from mesa_diode.simulator.materials import Q
from mesa_diode.simulator.physics.carriers import band_gap
from mesa_diode.simulator.physics.constants import VBI_BOLTZMANN, VBI_DEGENERATE


def built_in_potential(s, method=VBI_DEGENERATE):
    """Встроенный потенциал, В.

    (3.1)  V_bi = (kT/q)·ln(n_n0·p_p0/n_i²) — [Зи, с. 82, ур. (7), (7а)], n_n0, p_p0 по (2.4).
    (3.1а) qV_bi = E_g + kT(η_n + η_p) — [Зи, с. 82, ур. (7), первое равенство],
           η по (2.7); при η < −3 совпадает с (3.1).
    """
    if method == VBI_BOLTZMANN:
        n_n0, p_p0 = s.majority
        return s.Vt * np.log(n_n0 * p_p0 / s.ni ** 2)
    eta_n, eta_p = s.eta
    return band_gap(s.T, s.material) + s.Vt * (eta_n + eta_p)


def _depletion_argument(s, V):
    """V_bi − V − 2kT/q, ограниченное снизу kT/q (условие применимости §3)."""
    return np.maximum(s.Vbi - np.asarray(V, dtype=float) - 2.0 * s.Vt, s.Vt)


def depletion_width(s, V):
    """(3.2) W(V) = √[(2ε/q)·(N_A + N_D)/(N_A·N_D)·(V_bi − V − 2kT/q)], см —
    [Зи, с. 84, ур. (15), (16); с. 86, ур. (18)]; N — ионизованная примесь."""
    NA, ND = s.p_side.N, s.n_side.N
    return np.sqrt(2.0 * s.eps / Q * (NA + ND) / (NA * ND) * _depletion_argument(s, V))


def depletion_edges(s, V):
    """(3.2) x_p = W·N_D/(N_A + N_D), x_n = W·N_A/(N_A + N_D) — [Зи, с. 82, ур. (9)].
    Возвращает (x_n, x_p), см."""
    NA, ND = s.p_side.N, s.n_side.N
    W = depletion_width(s, V)
    return W * NA / (NA + ND), W * ND / (NA + ND)


def capacitance(s, V):
    """(3.4) C = εA/W, Ф — [Зи, с. 86, ур. (18)]. NaN при V > V_bi − 3kT/q."""
    V = np.asarray(V, dtype=float)
    C = s.eps * s.area / depletion_width(s, V)
    return np.where(V > s.Vbi - 3.0 * s.Vt, np.nan, C)


def inv_c2(s, V):
    """(3.5) 1/C² = 2(V_bi − 2kT/q − V)/(qεN_eff·A²), Ф⁻² —
    [Зи, с. 87, ур. (18а), (18б); с. 88, ур. (18в)]."""
    return 1.0 / capacitance(s, V) ** 2


def c2_cutoff(s):
    """Отсечка 1/C² по оси V: V_bi − 2kT/q, В (3.5)."""
    return s.Vbi - 2.0 * s.Vt


def fit_inv_c2(V, C, area, eps, window=None):
    """Прямая 1/C² = a + b·V на участке window = (V_min, V_max).

    Из наклона по (3.5): N = −2/(q·ε·A²·b); отсечка V₀ = −a/b.
    Возвращает (N, V₀) или (nan, nan), если точек меньше трёх.
    """
    V = np.asarray(V, dtype=float)
    C = np.asarray(C, dtype=float)
    mask = np.isfinite(C) & (C > 0)
    if window is not None:
        mask &= (V >= window[0]) & (V <= window[1])
    if mask.sum() < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(V[mask], 1.0 / C[mask] ** 2, 1)
    return -2.0 / (Q * eps * area ** 2 * b), -a / b


def estimate_grading_m(V, C, vbi_prime):
    """Показатель резкости по наклону ln C от ln(V_bi′ − V), V_bi′ = V_bi − 2kT/q.

    C ∝ (V_bi′ − V)^{−1/2} — резкий переход; C ∝ (V_bi − V)^{−1/3} —
    линейный [Зи, с. 89, ур. (23)]; общий показатель −1/(m + 2) — интерполяция.
    Возвращает (m, наклон) или (None, None) при нехватке точек (< 5, V < −0.05 В).
    """
    V = np.asarray(V, dtype=float)
    C = np.asarray(C, dtype=float)
    mask = np.isfinite(C) & (C > 0) & (V < -0.05) & ((vbi_prime - V) > 1e-6)
    if mask.sum() < 5:
        return None, None
    slope, _ = np.polyfit(np.log(vbi_prime - V[mask]), np.log(C[mask]), 1)
    if slope >= 0:
        return None, slope
    return -1.0 / slope - 2.0, slope


def edge_area(s, V):
    """(1.3) Добавочная площадь ОПЗ ΔA ≈ πD·max(0, x_p(V) − (h − z_j)), см² —
    оценка ТЗ (геометрическая). Для сценария B (z_j = d_epi) это
    πD·max(0, x_p − (h − d_epi)); в сценарии A ОПЗ, как правило, внутри мезы."""
    _, xp = depletion_edges(s, V)
    return np.pi * s.D * np.maximum(0.0, xp - (s.h - s.z_j))


def side_wall_area(s):
    """Площадь боковой стенки S_side = πDh, см² (справочно, для бэклога Б-3)."""
    return np.pi * s.D * s.h
