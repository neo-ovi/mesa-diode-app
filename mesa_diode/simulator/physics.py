# -*- coding: utf-8 -*-
"""Физическое ядро симулятора мезадиода (Ge, гомопереход).

Номера формул (раздел.номер) совпадают с вкладкой «Модель» окна
«Формулы и параметры». Источники указаны библиографически в docstring
каждой функции; коды источников — см. реестр в simulator/formulas.py.

Единицы в расчёте: см, см⁻³, с, В, А, Ф, К; E_g — эВ. Знаки: V > 0 —
прямое смещение, I > 0 — прямой ток [Ш49, с. 460].
"""

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from mesa_diode.simulator.materials import EPS0, GE, K_B, Q

SQRT_PI = np.sqrt(np.pi)

# Пороги вырождения по η (2.7): «несколько kT» [Зи, с. 22] в численной трактовке ТЗ.
ETA_ONSET = -3.0
ETA_DEGENERATE = 0.0


def thermal_voltage(T):
    """V_t = kT/q, В."""
    return K_B * T / Q


# ------------------------------------------------------------ §2. Материал --

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


def acceptor_from_resistivity(rho, T, mat=GE):
    """(2.8) N_A подложки по ρ_sub: решение ρ(N_A) совместно с (2.4).

    ρ(N_A) немонотонна (максимум около собственной концентрации): при
    ρ(0) < ρ ≤ ρ_max решений два — берётся большее, с предупреждением.
    Возвращает (N_A, [предупреждения]); при ρ > ρ_max — (nan, [...]).
    """
    warnings = []
    peak = minimize_scalar(lambda lg: -resistivity(10.0 ** lg, T, mat),
                           bounds=(8.0, 18.0), method="bounded", options={"xatol": 1e-6})
    lg_peak = peak.x
    rho_max = resistivity(10.0 ** lg_peak, T, mat)
    rho_zero = resistivity(0.0, T, mat)
    if rho > rho_max:
        return float("nan"), [
            f"ρ_sub = {rho:g} Ом·см больше максимально возможного для {mat.name} "
            f"({rho_max:.1f} Ом·см при T = {T:g} К): решения нет."]
    if rho > rho_zero:
        warnings.append(
            f"ρ_sub = {rho:g} Ом·см > ρ(0) = {rho_zero:.1f} Ом·см: два решения "
            "(2.8), взято большее N_A.")
    lg = brentq(lambda x: resistivity(10.0 ** x, T, mat) - rho, lg_peak, 22.0, xtol=1e-12)
    return 10.0 ** lg, warnings


# ------------------------------------------ §5. Структура и два сценария --

SCENARIO_A = "A"   # i-слой p: переход n⁺/i на глубине d_n
SCENARIO_B = "B"   # i-слой n (по умолчанию): переход i/подложка на дне мезы
SCENARIOS = (SCENARIO_A, SCENARIO_B)

SINK, REFLECT, LONG = "sink", "reflect", "long"
BOUNDARY_LABELS = {SINK: "сток", REFLECT: "отражение", LONG: "длинная база"}

VBI_DEGENERATE = "degenerate"   # (3.1а), используется в расчёте всегда
VBI_BOLTZMANN = "boltzmann"     # (3.1), для сравнения и эталонов §9


@dataclass(frozen=True)
class Side:
    """Нейтральная область одной стороны перехода."""

    kind: str            # "n" или "p"
    layer: str           # "n+", "i" или "sub"
    N: float             # концентрация ионизованной примеси, см⁻³
    thickness: float     # толщина слоя, см
    mu_minority: float   # подвижность неосновных носителей, см²/(В·с)
    tau_bg: float        # фоновое время жизни неосновных, с
    sigma_R: float       # рекомбинационная эффективность дислокации, см²/с
    boundary: str        # граничное условие на дальней границе


@dataclass(frozen=True)
class Structure:
    """Параметры структуры (§5.3) в единицах расчёта: см, см⁻³, с, Ом, А, К.

    Сценарий A: i-слой p-типа, переход n⁺/i (z_j = d_n).
    Сценарий B: i-слой n-типа, переход i/подложка (z_j = d_epi).
    Граничные условия дальних границ: bc_A_n (верхний контакт),
    bc_A_p (граница i/подложка), bc_B_n (граница n/n⁺), bc_B_p (тыльный контакт).
    """

    D: float
    d_epi: float
    h: float
    d_n: float
    d_sub: float
    ND_plus: float
    N_i: float
    rho_sub: float
    T: float = 300.0
    scenario: str = SCENARIO_B
    mu_n: float = GE.mu_n_max        # электроны — неосновные в p-области
    mu_p_i: float = GE.mu_p_max      # дырки в i-слое
    mu_p_nplus: float = GE.mu_p_max  # дырки в n⁺ (значение задаёт набор образца)
    tau_n_bg: float = 1e-6
    tau_p_bg: float = 1e-6
    tau0_bg: float = 1e-7
    N_dis: float = 0.0
    sigma_R_epi: float = 3.5e-3      # i-слой и n⁺
    sigma_R_sub: float = 5.5e-4      # подложка
    n2: float = 2.0
    Rs: float = 0.0
    Rsh: float = float("inf")
    I_L: float = 0.0
    m_leak: float = 3.0
    bc_A_n: str = SINK
    bc_A_p: str = SINK
    bc_B_n: str = REFLECT
    bc_B_p: str = SINK
    sns_refinement: bool = False
    edge_area: bool = False
    vbi_method: str = VBI_DEGENERATE
    material: object = GE

    def with_scenario(self, scenario):
        return replace(self, scenario=scenario)

    # --- материал при температуре T ---
    @cached_property
    def Vt(self):
        return thermal_voltage(self.T)

    @cached_property
    def ni(self):
        return intrinsic_concentration(self.T, self.material)

    @cached_property
    def eps(self):
        """ε = ε_r·ε₀, Ф/см [Ioffe-SiGe]; без усреднения по материалам."""
        return self.material.eps_r * EPS0

    @cached_property
    def area(self):
        """(1.1) A = πD²/4, см²."""
        return np.pi * self.D ** 2 / 4.0

    @cached_property
    def d_i(self):
        """Толщина нелегированного слоя d_i = d_epi − d_n, см."""
        return self.d_epi - self.d_n

    @cached_property
    def substrate(self):
        """N_A подложки по ρ_sub (2.8): (N_A, [предупреждения])."""
        return acceptor_from_resistivity(self.rho_sub, self.T, self.material)

    # --- стороны перехода по сценарию (§5.2) ---
    @cached_property
    def n_side(self):
        if self.scenario == SCENARIO_A:
            return Side("n", "n+", self.ND_plus, self.d_n, self.mu_p_nplus,
                        self.tau_p_bg, self.sigma_R_epi, self.bc_A_n)
        return Side("n", "i", self.N_i, self.d_i, self.mu_p_i,
                    self.tau_p_bg, self.sigma_R_epi, self.bc_B_n)

    @cached_property
    def p_side(self):
        if self.scenario == SCENARIO_A:
            return Side("p", "i", self.N_i, self.d_i, self.mu_n,
                        self.tau_n_bg, self.sigma_R_epi, self.bc_A_p)
        return Side("p", "sub", self.substrate[0], self.d_sub, self.mu_n,
                    self.tau_n_bg, self.sigma_R_sub, self.bc_B_p)

    @cached_property
    def z_j(self):
        """Глубина p-n перехода от поверхности мезы, см."""
        return self.d_n if self.scenario == SCENARIO_A else self.d_epi

    @cached_property
    def majority(self):
        """Равновесные основные носители сторон (2.4): (n_n0, p_p0)."""
        return (equilibrium_carriers(self.n_side.N, self.ni)[0],
                equilibrium_carriers(self.p_side.N, self.ni)[0])

    @cached_property
    def minority(self):
        """Равновесные неосновные носители (2.4): (p_n0, n_p0)."""
        return (equilibrium_carriers(self.n_side.N, self.ni)[1],
                equilibrium_carriers(self.p_side.N, self.ni)[1])

    @cached_property
    def eta(self):
        """(2.7) η основных носителей: (η_n n-стороны, η_p p-стороны)."""
        Nc, Nv = effective_dos(self.T, self.material)
        n_n0, p_p0 = self.majority
        return reduced_fermi_level(n_n0, Nc), reduced_fermi_level(p_p0, Nv)

    @cached_property
    def Vbi(self):
        return built_in_potential(self, self.vbi_method)

    @cached_property
    def N_eff(self):
        """N_eff = N_A·N_D/(N_A + N_D), см⁻³ (3.5)."""
        NA, ND = self.p_side.N, self.n_side.N
        return NA * ND / (NA + ND)


# ------------------------------------------------- §3. Электростатика --

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
