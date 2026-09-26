# -*- coding: utf-8 -*-
"""§6.3. Коэффициент идеальности из эксперимента: (6.3)–(6.6); методичка, гл. 8."""

from dataclasses import dataclass

import numpy as np

from mesa_diode.simulator.physics.carriers import _expm1, thermal_voltage


N_RISE = 0.20   # правило ТЗ: V₂ — локальный n превысил минимум на [V₁; V] более чем на 20 %


@dataclass
class IdealityResult:
    n: float
    dn: float
    I0: float
    V1: float
    V2: float
    V_local: np.ndarray    # напряжение на переходе для локального n(V)
    n_local: np.ndarray


def local_ideality(V, I, T):
    """(6.4)/(6.5) n = [(kT/q)·Δln I/ΔV]⁻¹ центральными разностями —
    [Ман20, с. 46, ур. (4), (5)].

    Разность берётся по точкам, отстоящим не меньше чем на kT/q (на плотных
    данных соседние точки дали бы производную шума); на редких данных — по
    соседним точкам. V должно быть отсортировано по возрастанию."""
    V = np.asarray(V, dtype=float)
    lnI = np.log(np.asarray(I, dtype=float))
    Vt = thermal_voltage(T)
    if V.size < 3:
        return 1.0 / (Vt * np.gradient(lnI, V))
    step = float(np.median(np.diff(V)))
    k = max(1, int(np.ceil(Vt / step / 2.0))) if step > 0 else 1
    idx = np.arange(V.size)
    lo = np.clip(idx - k, 0, V.size - 1)
    hi = np.clip(idx + k, 0, V.size - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (V[hi] - V[lo]) / (Vt * (lnI[hi] - lnI[lo]))


def series_resistance_limit(V, I):
    """Наибольшее R_s, при котором напряжение на переходе V_j = V − I·R_s ещё
    растёт вместе с V на верху прямой ветви, Ом (или nan).

    dV_j/dV > 0 ⇔ dV/dI > R_s: R_s не может превышать дифференциальное
    сопротивление диода. Берётся наклон dV/dI прямой по последним 20 % точек
    прямой ветви (не меньше пяти) — там же, где ideality_from_data ищет перегиб.
    Это проверка согласованности данных, а не формула модели."""
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = np.isfinite(V) & np.isfinite(I) & (V > 0) & (I > 0)
    V, I = V[mask], I[mask]
    V, first = np.unique(V, return_index=True)
    I = I[first]
    tail = max(5, V.size // 5)
    if V.size < 5 or np.ptp(I[-tail:]) <= 0:
        return float("nan")
    slope = np.polyfit(I[-tail:], V[-tail:], 1)[0]
    return float(slope) if slope > 0 else float("nan")


N_RISE_RUN = 3        # рост должен держаться ≥ 3 точек подряд (одиночный шум не закрывает окно)
N_FIT_BOUNDS = (0.3, 20.0)


def ideality_from_data(V, I, T=300.0, Rs=0.0, window=None, diagnostics=None, I_mod=float("inf")):
    """Коэффициент идеальности по прямой ветви (алгоритм §6.3).

    1) V > 0, I > 0; при R_s > 0 V заменяется на V − I·R_s;
    2) локальный n(V) по (6.5);
    3) окно: V₁ = 3kT/q, V₂ — первая точка, где n превышает минимум на
       [V₁; V] более чем на 20 % (правило ТЗ); window = (V₁, V₂) задаёт окно вручную;
    4) n ± δn — подгонка (6.3) I = I₀[exp(qV/nkT) − 1] по ln I (curve_fit)
       [Ман20, с. 45, ур. (2)].

    Реализация: после поправки на R_s остаётся только участок, где V − I·R_s > 0
    и растёт вместе с V (иначе R_s завышено); рост n на 20 % засчитывается,
    если держится не менее трёх точек подряд; окно — не менее трёх точек.
    Возвращает IdealityResult или None; причина неудачи и замечания
    добавляются в список diagnostics, если он передан."""
    from scipy.optimize import curve_fit

    notes = diagnostics if diagnostics is not None else []

    def fail(reason):
        notes.append(reason)
        return None

    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = np.isfinite(V) & np.isfinite(I) & (V > 0) & (I > 0)
    V, I = V[mask], I[mask]
    V, first = np.unique(V, return_index=True)   # сортировка и удаление повторов V
    I = I[first]
    if V.size < 3:
        return fail("в прямой ветви меньше трёх точек с V > 0 и I > 0")

    Vt = thermal_voltage(T)
    # R_s(I) по (6.9); при I_mod = ∞ — постоянное R_s
    Rs_I = Rs / (1.0 + np.abs(I) / I_mod) if np.isfinite(I_mod) else Rs
    Vj = V - I * Rs_I if Rs > 0 else V.copy()
    if Rs > 0:
        # Перегиб: V − I·R_s устойчиво убывает на последних 20 % точек (наклон
        # прямой по ним < 0) — R_s завышено; шум отдельных точек так не срабатывает.
        tail = max(5, Vj.size // 5)
        if Vj.size >= 5 and np.polyfit(V[-tail:], Vj[-tail:], 1)[0] < 0:
            peak = int(np.argmax(Vj))
            notes.append(f"R_s = {Rs:g} Ом, вероятно, завышено: при V > {V[peak]:.3g} В "
                         "напряжение на переходе V − I·R_s убывает; эти точки не используются.")
            V, I, Vj = V[:peak + 1], I[:peak + 1], Vj[:peak + 1]
        # мелкое дрожание от шума тока: оставляем только точки, где V − I·R_s растёт
        rising = Vj >= np.maximum.accumulate(Vj)
        V, I, Vj = V[rising], I[rising], Vj[rising]
        Vj, first = np.unique(Vj, return_index=True)
        V, I = V[first], I[first]
        positive = Vj > 0
        V, I, Vj = V[positive], I[positive], Vj[positive]
        if Vj.size < 3:
            return fail(f"после поправки V − I·R_s (R_s = {Rs:g} Ом) осталось меньше трёх "
                        "точек: уменьшите R_s или задайте окно V₁…V₂ вручную")
    n_loc = local_ideality(Vj, I, T)

    if window is None:
        V1 = 3.0 * Vt
        V2 = Vj[-1]
        running = np.inf
        run = 0
        for v, n in zip(Vj, n_loc):
            if v < V1 or not np.isfinite(n) or n <= 0:
                continue
            if n > (1.0 + N_RISE) * running:
                run += 1
                if run == 1:
                    first_high = v
                if run >= N_RISE_RUN:
                    V2 = first_high
                    break
            else:
                run = 0
                running = min(running, n)
        # Подгонка двух параметров требует не менее трёх точек: на редких
        # данных окно по правилу 20 % расширяется до третьей точки от V₁.
        above = Vj[Vj >= V1]
        if above.size >= 3 and ((Vj >= V1) & (Vj <= V2)).sum() < 3:
            V2 = above[2]
    else:
        V1, V2 = window
    sel = (Vj >= V1) & (Vj <= V2)
    if sel.sum() < 3:
        return fail(f"в окне V₁…V₂ = {V1:.3g}…{V2:.3g} В меньше трёх точек прямой ветви")
    x, y = Vj[sel], np.log(I[sel])

    def model(v, lnI0, n):
        return lnI0 + np.log(np.expm1(v / (n * Vt)))

    slope, intercept = np.polyfit(x, y, 1)
    n0 = 1.0 / (Vt * slope) if slope > 0 else 1.5
    n0 = min(max(n0, N_FIT_BOUNDS[0] * 1.01), N_FIT_BOUNDS[1] * 0.99)
    try:
        popt, pcov = curve_fit(model, x, y, p0=(intercept, n0),
                               bounds=([-200.0, N_FIT_BOUNDS[0]], [50.0, N_FIT_BOUNDS[1]]))
    except (RuntimeError, ValueError):
        return fail("подгонка (6.3) не сошлась в выбранном окне")
    n = float(popt[1])
    if not N_FIT_BOUNDS[0] * 1.001 < n < N_FIT_BOUNDS[1] * 0.999:
        return fail(f"подгонка (6.3) упёрлась в границу n = {n:.3g}: окно "
                    f"{V1:.3g}…{V2:.3g} В не подходит — проверьте R_s или задайте окно вручную")
    if n < 1.0:
        notes.append(f"n = {n:.3g} < 1 физически невозможно для диода — вероятно, завышено R_s "
                     "или неудачно окно V₁…V₂.")
    dn = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else float("nan")
    return IdealityResult(n=n, dn=dn, I0=float(np.exp(popt[0])),
                          V1=float(V1), V2=float(V2), V_local=Vj, n_local=n_loc)


def empirical_current(V, I0, n, T=300.0):
    """(6.3) I = I₀[exp(qV/nkT) − 1] — [Ман20, с. 45, ур. (2)]; эмпирика."""
    return I0 * _expm1(np.asarray(V, dtype=float) / (n * thermal_voltage(T)))


def effective_ideality(I_diff, I_gr):
    """(6.6) n_eff = (I_diff + I_gr)/(I_diff + I_gr/2) — подстановка (4.4) и (5.4)
    в (6.4) при V ≫ kT/q."""
    return (I_diff + I_gr) / (I_diff + I_gr / 2.0)


def gr_share_from_ideality(n):
    """Доля тока ОПЗ из (6.6): 2(1 − 1/n)."""
    return 2.0 * (1.0 - 1.0 / n)
