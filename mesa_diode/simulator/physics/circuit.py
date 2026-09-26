# -*- coding: utf-8 -*-
"""§6. Полный ток (6.1) с эквивалентной схемой и решатель; R_s(I) (6.9).

Ток перехода берётся у модели тока из реестра simulator/models.py
(методичка, п. 13.6); схема и решатель общие для всех моделей."""

from dataclasses import dataclass

import numpy as np


def shunt_current(s, V):
    """V_d/R_sh, А — [Зи, с. 97, п. 1]."""
    V = np.asarray(V, dtype=float)
    return V / s.Rsh if np.isfinite(s.Rsh) else np.zeros_like(V)


def leak_current(s, V):
    """I_L·sign(V_d)·|V_d/1 В|^m, А — мягкая обратная ВАХ [Кур74, с. 167–168]; эмпирика."""
    V = np.asarray(V, dtype=float)
    return s.I_L * np.sign(V) * np.abs(V) ** s.m_leak


def components(s, Vd):
    """Компоненты тока (6.1) при напряжении на переходе V_d, А.

    Ток перехода — по модели s.model из реестра models (физическая: I_diff,
    I_gr; эмпирическая (6.3) [Ман20, с. 45, ур. (2)]: I_emp; двухдиодная
    (6.3а) [Зи, с. 99, ур. (55)]: I_01, I_02); плюс общая для всех моделей
    эквивалентная схема: шунт I_sh и нелинейная утечка I_L."""
    from mesa_diode.simulator import models    # реестр импортирует physics

    Vd = np.asarray(Vd, dtype=float)
    parts = dict(models.get(s.model).parts(s, Vd))
    parts["sh"] = shunt_current(s, Vd)
    parts["L"] = leak_current(s, Vd)
    return parts


def junction_current(s, Vd):
    """Сумма компонент (6.1) при напряжении на переходе V_d, А."""
    return sum(components(s, Vd).values())


@dataclass
class IVResult:
    V: np.ndarray
    I: np.ndarray
    Vd: np.ndarray
    parts: dict
    warnings: list


BISECTION_STEPS = 52   # 2⁻⁵² от |V| — предел точности double


def series_resistance(s, I):
    """(6.9) R_s(I) = R_s/(1 + |I|/I_mod), Ом — эмпирика: модуляция проводимости
    высокоомной базы при высоком уровне инжекции [Зи, с. 97, п. 4]; I_mod = ∞ —
    R_s постоянно [Зи, с. 97, п. 5]."""
    I = np.asarray(I, dtype=float)
    if not np.isfinite(s.I_mod):
        return np.full_like(I, s.Rs)
    return s.Rs / (1.0 + np.abs(I) / s.I_mod)


def solve_iv(s, V):
    """(6.1) I = I_diff(V_d) + I_gr(V_d) + V_d/R_sh + I_L·sign(V_d)|V_d/1 В|^m,
    V_d = V − I·R_s(I) по (6.9).

    Уравнение решается относительно V_d: h(V_d) = V_d + F(V_d)·R_s(F) − V
    монотонно растёт, поэтому корень единственный и лежит между 0 и V;
    он находится бисекцией сразу для всех точек (52 шага). Если сумма
    компонент не конечна — NaN и предупреждение. R_s — [Зи, с. 97, п. 5;
    рис. 21, с. 99]."""
    V = np.asarray(V, dtype=float)
    if s.Rs == 0:
        Vd = V.copy()
    else:
        lo = np.minimum(V, 0.0)
        hi = np.maximum(V, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                F = junction_current(s, mid)
                h = mid + F * series_resistance(s, F) - V
                above = ~(h <= 0)          # NaN и +∞ считаются «выше корня»
                hi = np.where(above, mid, hi)
                lo = np.where(above, lo, mid)
        Vd = 0.5 * (lo + hi)
    with np.errstate(over="ignore", invalid="ignore"):
        I = np.asarray(junction_current(s, Vd), dtype=float)
    bad = ~np.isfinite(I)
    I = np.where(bad, np.nan, I)
    warnings = []
    if bad.any():
        warnings.append(f"Решатель (6.1) не сошёлся в {int(bad.sum())} точках — там NaN.")
    return IVResult(V=V, I=I, Vd=Vd, parts=components(s, Vd), warnings=warnings)
