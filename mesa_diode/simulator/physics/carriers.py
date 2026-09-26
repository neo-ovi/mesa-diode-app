# -*- coding: utf-8 -*-
"""§2. Материал: статистика носителей, подвижность, удельное сопротивление.

Формулы (2.1)–(2.9); методичка, гл. 3, п. 10.1, п. 10.1а."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from mesa_diode.simulator.materials import GE, K_B, Q

SQRT_PI = np.sqrt(np.pi)

# Пороги вырождения по η (2.7): «несколько kT» [Зи, с. 22] в численной трактовке ТЗ.
ETA_ONSET = -3.0
ETA_DEGENERATE = 0.0

EXP_LIMIT = 700.0  # защита exp от переполнения


def _expm1(x):
    return np.expm1(np.clip(x, -EXP_LIMIT, EXP_LIMIT))


def thermal_voltage(T):
    """V_t = kT/q, В."""
    return K_B * T / Q



def band_gap(T, mat=GE):
    """(2.1) E_g(T) = E_g(0) − αT²/(T + β), эВ — [Зи, с. 20, табл. на рис. 8]."""
    return mat.Eg0_eV - mat.alpha_eV_K * T ** 2 / (T + mat.beta_K)


def effective_dos(T, mat=GE):
    """(2.2) N_C = N_C0·T^{3/2}, N_V = N_V0·T^{3/2}, см⁻³ — [Ioffe-Ge-b]."""
    t32 = T ** 1.5
    return mat.Nc0 * t32, mat.Nv0 * t32


def intrinsic_concentration(T, mat=GE):
    """(2.3) n_i = √(N_C·N_V)·exp(−E_g/2kT), см⁻³ — [Зи, с. 24, ур. (19), (19а)]."""
    Nc, Nv = effective_dos(T, mat)
    return np.sqrt(Nc * Nv) * np.exp(-band_gap(T, mat) / (2.0 * thermal_voltage(T)))


def equilibrium_carriers(N, ni):
    """(2.4) Равновесные концентрации слоя с легированием N (полная ионизация):
    основные M = N/2 + √(N²/4 + n_i²), неосновные m = n_i²/M.
    Из np = n_i² [Зи, с. 24, ур. (19)] и электронейтральности.
    Возвращает (M, m)."""
    majority = N / 2.0 + np.sqrt(N * N / 4.0 + ni * ni)
    return majority, ni * ni / majority


def diffusion_coefficient(mu, T):
    """(2.5) D = (kT/q)·μ, см²/с — [Зи, с. 36, ур. (44)]."""
    return thermal_voltage(T) * mu


def diffusion_length(D, tau):
    """(2.6) L = √(Dτ), см — [Зи, с. 94, ур. (41)]."""
    return np.sqrt(D * tau)


def fermi_integral_half(eta):
    """F_{1/2}(η) = ∫₀^∞ x^{1/2} dx / (1 + e^{x−η}) — [Зи, с. 22, ур. (11); с. 23, рис. 10]."""
    upper = max(eta, 0.0) + 60.0
    # epsabs=0: при сильно отрицательном η значения ~e^η малы, абсолютный
    # допуск quad по умолчанию дал бы относительную ошибку ~10⁻⁵.
    value, _ = quad(lambda x: np.sqrt(x) * expit(eta - x), 0.0, upper,
                    limit=200, epsabs=0.0, epsrel=1e-10)
    return value


def carriers_from_eta(eta, N_band):
    """(2.7) n = N_C·(2/√π)·F_{1/2}(η) — [Зи, с. 22, ур. (11), (14)]."""
    return N_band * 2.0 / SQRT_PI * fermi_integral_half(eta)


def reduced_fermi_level(n, N_band):
    """(2.7) η по концентрации основных носителей: обращение
    n = N·(2/√π)·F_{1/2}(η) (quad + brentq). Для электронов η = (E_F − E_C)/kT,
    для дырок η = (E_V − E_F)/kT."""
    target = n / N_band
    lower = np.log(target)  # статистика Больцмана завышает n, поэтому η ≥ ln(n/N)
    upper = max(lower, 0.0) + 2.0 + (0.75 * SQRT_PI * target) ** (2.0 / 3.0)
    return brentq(lambda eta: 2.0 / SQRT_PI * fermi_integral_half(eta) - target,
                  lower, upper, xtol=1e-10)


def boltzmann_ratio(eta):
    """(2.7) Ошибка Больцмана: (2/√π)·F_{1/2}(η)·e^{−η} — отношение точной
    концентрации к больцмановской при том же η (1 — нет ошибки)."""
    return 2.0 / SQRT_PI * fermi_integral_half(eta) * np.exp(-eta)


def degeneracy_status(eta):
    """Статус вырождения по η: ≥ 0 — вырожден, ≥ −3 — начало вырождения."""
    if eta >= ETA_DEGENERATE:
        return "вырожден"
    if eta >= ETA_ONSET:
        return "начало вырождения"
    return "невырожден"


def degeneracy_thresholds(T, mat=GE, etas=(-3.0, -2.0, 0.0)):
    """Концентрации электронов и дырок при заданных η (таблица порогов (2.7))."""
    Nc, Nv = effective_dos(T, mat)
    return {
        "electrons": {eta: carriers_from_eta(eta, Nc) for eta in etas},
        "holes": {eta: carriers_from_eta(eta, Nv) for eta in etas},
    }


def resistivity(NA, T, mat=GE):
    """ρ = 1/[q(μ_n·n₀ + μ_p·p₀)], Ом·см — [Зи, с. 36, ур. (47)];
    n₀, p₀ по (2.4), μ — чистого материала (§4)."""
    p0, n0 = equilibrium_carriers(NA, intrinsic_concentration(T, mat))
    return 1.0 / (Q * (mat.mu_n_max * n0 + mat.mu_p_max * p0))


def resistivity_max(T, mat=GE):
    """ρ_max = 1/(2q·n_i·√(μ_n·μ_p)), Ом·см — наибольшее ρ материала при T.

    Следует из (2.8) при n₀p₀ = n_i² (2.4): проводимость q(μ_n·n₀ + μ_p·n_i²/n₀)
    минимальна при n₀ = n_i·√(μ_p/μ_n) [Зи, с. 36, ур. (47)]. Больше ρ_max не
    бывает: собственные носители проводят ток при любом легировании."""
    return 1.0 / (2.0 * Q * intrinsic_concentration(T, mat) * np.sqrt(mat.mu_n_max * mat.mu_p_max))


def acceptor_from_resistivity(rho, T, mat=GE):
    """(2.8) N_A подложки по ρ_sub: решение ρ(N_A) совместно с (2.4).

    T — температура, при которой измерено ρ (не температура образца: N_A от
    T не зависит, а ρ зависит сильно). ρ(N_A) немонотонна (максимум около
    собственной концентрации): при ρ(0) < ρ ≤ ρ_max решений два — берётся
    большее, с предупреждением. Возвращает (N_A, [предупреждения]); при
    ρ > ρ_max — (nan, [...]).
    """
    warnings = []
    peak = minimize_scalar(lambda lg: -resistivity(10.0 ** lg, T, mat),
                           bounds=(8.0, 18.0), method="bounded", options={"xatol": 1e-6})
    lg_peak = peak.x
    rho_max = resistivity(10.0 ** lg_peak, T, mat)
    rho_zero = resistivity(0.0, T, mat)
    if rho > rho_max:
        ni = intrinsic_concentration(T, mat)
        return float("nan"), [
            f"ρ_{{sub}} = {rho:g} Ом·см больше максимально возможного для {mat.name} при "
            f"T = {T:g} К ({rho_max:.1f} Ом·см): такой подложки не бывает, N_{{A}} найти нельзя. "
            f"Причина: даже в чистом (собственном) {mat.name} ток переносят собственные носители "
            f"n_{{i}} = {ni:.2g} см⁻³, поэтому ρ ≤ ρ_max = 1/(2q·n_{{i}}·√(μ_{{n}}μ_{{p}})) (2.8). "
            "n_{i} быстро растёт с температурой, и ρ_max падает: около 60 Ом·см при 300 К и "
            "13 Ом·см при 330 К. Что сделать: ρ из паспорта пластины измерено при комнатной "
            "температуре — укажите её в поле «T изм. ρ» (не температуру образца); если ρ "
            "измерено при этой температуре — проверьте значение и единицы."]
    if rho > rho_zero:
        warnings.append(
            f"ρ_{{sub}} = {rho:g} Ом·см > ρ(0) = {rho_zero:.1f} Ом·см: два решения "
            "(2.8), взято большее N_A.")
    lg = brentq(lambda x: resistivity(10.0 ** x, T, mat) - rho, lg_peak, 22.0, xtol=1e-12)
    return 10.0 ** lg, warnings
