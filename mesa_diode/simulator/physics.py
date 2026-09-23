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


# ------------------------------------------- §1. Геометрия и изоляция --

def edge_area(s, V):
    """(1.3) Добавочная площадь ОПЗ ΔA ≈ πD·max(0, x_p(V) − (h − z_j)), см² —
    оценка ТЗ (геометрическая). Для сценария B (z_j = d_epi) это
    πD·max(0, x_p − (h − d_epi)); в сценарии A ОПЗ, как правило, внутри мезы."""
    _, xp = depletion_edges(s, V)
    return np.pi * s.D * np.maximum(0.0, xp - (s.h - s.z_j))


def side_wall_area(s):
    """Площадь боковой стенки S_side = πDh, см² (справочно, для бэклога Б-3)."""
    return np.pi * s.D * s.h


# ------------------------------------------ §5А. Дислокации и время жизни --

def dislocation_lifetime(sigma_R, N_dis):
    """τ_dis = 1/(σ_R·N_dis), с — из [KKA56, с. 1289, ур. (3)]: λ = σ_R·N_D·ΔP."""
    return float("inf") if N_dis <= 0 else 1.0 / (sigma_R * N_dis)


def minority_lifetime(side, N_dis):
    """(5.9) 1/τ = 1/τ^bg + σ_R·N_dis: каналы рекомбинации складываются;
    второе слагаемое — [KKA56, ур. (3)]."""
    return 1.0 / (1.0 / side.tau_bg + side.sigma_R * N_dis)


def scr_sigma_R(s):
    """σ_R слоя, в котором лежит бо́льшая часть ОПЗ (x_p > x_n ⇔ N_D > N_A)."""
    return s.p_side.sigma_R if s.n_side.N > s.p_side.N else s.n_side.sigma_R


def scr_lifetime(s):
    """(5.10) 1/τ₀ = 1/τ₀^bg + σ_R·N_dis — допущение модели: перенос
    [KKA56, ур. (3)] (объёмное время жизни) на ОПЗ."""
    return 1.0 / (1.0 / s.tau0_bg + scr_sigma_R(s) * s.N_dis)


# --------------------------------------------------- §4. Диффузионный ток --

W_MIN = 1e-7      # см (1 нм): минимальная толщина нейтральной базы при смыкании
U_SMALL = 1e-6    # при u < 10⁻⁶: coth u = 1/u, tanh u = u
U_LARGE = 20.0    # при u > 20: f = 1


def boundary_factor(u, boundary):
    """f(w/L) в (4.3): 1 — длинная база [Зи, ур. (40)]; coth — «сток»,
    Δn(w) = 0 [Ш49, с. 470, ур. (5.5)]; tanh — «отражение», dΔn/dx(w) = 0
    [Ш49, с. 470, ур. (5.6) при p₁ = p₂]."""
    u = np.asarray(u, dtype=float)
    if boundary == LONG:
        return np.ones_like(u)
    small = u < U_SMALL
    safe = np.where(small, U_SMALL, u)
    if boundary == SINK:
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


def saturation_current_density_parts(s, V):
    """(4.3) Слагаемые J_s(V): дырки в n-области и электроны в p-области, А/см².
    J_s = qD_p·p_n0/L_p·f(w_n/L_p) + qD_n·n_p0/L_n·f(w_p/L_n) —
    [Зи, с. 94, ур. (44), (45)]; [Ш49, с. 460, ур. (4.13)]."""
    (Dp, Lp), (Dn, Ln) = minority_transport(s)
    p_n0, n_p0 = s.minority
    wn, wp, _ = neutral_widths(s, V)
    holes = Q * Dp * p_n0 / Lp * boundary_factor(wn / Lp, s.n_side.boundary)
    electrons = Q * Dn * n_p0 / Ln * boundary_factor(wp / Ln, s.p_side.boundary)
    return holes, electrons


def saturation_current_density(s, V):
    """(4.3) J_s(V), А/см²."""
    holes, electrons = saturation_current_density_parts(s, V)
    return holes + electrons


EXP_LIMIT = 700.0  # защита exp от переполнения


def _expm1(x):
    return np.expm1(np.clip(x, -EXP_LIMIT, EXP_LIMIT))


def diffusion_current(s, V):
    """(4.4) I_diff = A·J_s(V)·(e^{qV/kT} − 1), А."""
    V = np.asarray(V, dtype=float)
    return s.area * saturation_current_density(s, V) * _expm1(V / s.Vt)


# ------------------------------------- §5. Генерация–рекомбинация в ОПЗ --

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

    При n₂ = 2 при обратном смещении это [СНШ57, с. 1230, ур. (11)] =
    [Зи, с. 97, ур. (48)], при прямом — [Зи, с. 99, ур. (54)] (верхняя
    оценка). n₂ ≠ 2 — эмпирика (5.6) [Ман20, с. 45–46]; τ₀ по (5.10)."""
    V = np.asarray(V, dtype=float)
    J = Q * s.ni * depletion_width(s, V) / (2.0 * scr_lifetime(s)) * _expm1(V / (s.n2 * s.Vt))
    if s.sns_refinement:
        J = J * sns_factor(s, V)
    return gr_area(s, V) * J


# -------------------------------------------------- §6. Полный ток (6.1) --

def shunt_current(s, V):
    """V_d/R_sh, А — [Зи, с. 97, п. 1]."""
    V = np.asarray(V, dtype=float)
    return V / s.Rsh if np.isfinite(s.Rsh) else np.zeros_like(V)


def leak_current(s, V):
    """I_L·sign(V_d)·|V_d/1 В|^m, А — мягкая обратная ВАХ [Кур74, с. 167–168]; эмпирика."""
    V = np.asarray(V, dtype=float)
    return s.I_L * np.sign(V) * np.abs(V) ** s.m_leak


def components(s, Vd):
    """Компоненты тока (6.1) при напряжении на переходе V_d: I_diff, I_gr, I_sh, I_L, А."""
    return {
        "diff": diffusion_current(s, Vd),
        "gr": gr_current(s, Vd),
        "sh": shunt_current(s, Vd),
        "L": leak_current(s, Vd),
    }


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


MAX_NEWTON = 100


def _solve_point(s, V, I_guess):
    """Ньютон по I для I = F(V − I·R_s); шаг ограничен так, что |ΔI·R_s| ≤ 2kT/q."""
    if s.Rs == 0:
        return float(junction_current(s, V)), True
    I = I_guess
    max_dv = 2.0 * s.Vt
    for _ in range(MAX_NEWTON):
        Vd = V - I * s.Rs
        F = float(junction_current(s, Vd))
        h = 1e-6
        dF = float(junction_current(s, Vd + h) - junction_current(s, Vd - h)) / (2 * h)
        g = I - F
        step = -g / (1.0 + s.Rs * dF)
        if abs(step) * s.Rs > max_dv:
            step = np.sign(step) * max_dv / s.Rs
        I += step
        if abs(step) <= 1e-10 * abs(I) + 1e-18:
            return I, True
    return float("nan"), False


def solve_iv(s, V):
    """(6.1) I = I_diff(V_d) + I_gr(V_d) + V_d/R_sh + I_L·sign(V_d)|V_d/1 В|^m, V_d = V − I·R_s.

    Ньютон по I с ограничением шага; развёртка от V = 0 к краям (решение
    соседней точки — начальное приближение); не более 100 итераций; при
    несходимости — NaN и предупреждение. R_s — [Зи, с. 97, п. 5; рис. 21, с. 99]."""
    V = np.asarray(V, dtype=float)
    I = np.full_like(V, np.nan)
    order = np.argsort(np.abs(V))
    failed = []
    # два прохода от V ≈ 0: вверх и вниз, чтобы начальное приближение было соседним
    for sign in (1, -1):
        guess = 0.0
        for idx in sorted((i for i in order if np.sign(V[i]) in (sign, 0)),
                          key=lambda i: abs(V[i])):
            value, ok = _solve_point(s, float(V[idx]), guess)
            I[idx] = value
            if ok:
                guess = value
            else:
                failed.append(float(V[idx]))
    Vd = V - np.nan_to_num(I) * s.Rs
    warnings = []
    if failed:
        warnings.append(f"Решатель (6.1) не сошёлся в {len(failed)} точках — там NaN.")
    return IVResult(V=V, I=I, Vd=Vd, parts=components(s, Vd), warnings=warnings)


# ------------------------------------------ Справочник: проверки §1, §6 --

def low_injection_voltage(s):
    """(6.4) V_LI = (kT/q)·ln(0.1·M_B/m_B0), В: M_B, m_B0 — равновесные основные
    и неосновные носители более слабой стороны (2.4). Условие —
    [Зи, с. 94, перед ур. (38)]; выше V_LI модель неприменима [Зи, с. 97, п. 4]."""
    weak = s.n_side if s.n_side.N < s.p_side.N else s.p_side
    M, m = equilibrium_carriers(weak.N, s.ni)
    return s.Vt * np.log(0.1 * M / m)


ISOLATED, EDGE, NOT_ISOLATED = "ok", "warn", "error"


def isolation_status(s, V):
    """(1.2) Изоляция перехода травлением мезы при смещении V.
    Возвращает (уровень, текст): уровень ok / warn / error."""
    if s.scenario == SCENARIO_A:
        if s.h > s.d_n:
            return ISOLATED, "переход изолирован (h > d_n)"
        return NOT_ISOLATED, "не изолирован: h ≤ d_n"
    xn, xp = depletion_edges(s, V)
    xn, xp = float(xn), float(xp)
    if s.h < s.d_epi - xn:
        return NOT_ISOLATED, "не изолирован: n-слой соединяет мезу с полем"
    if s.h < s.d_epi + xp:
        return EDGE, "ОПЗ выходит на поверхность поля у подножия мезы"
    return ISOLATED, "ОПЗ внутри мезы"


RECOMMEND_REFLECT = "отражение"
RECOMMEND_SINK = "сток"
RECOMMEND_INTERMEDIATE = "промежуточный"


def _recommend(N_layer, N_neighbour):
    """Правило ТЗ (6.2): соседний слой того же типа легирован в ≥ 10 раз
    сильнее → отражение; в ≥ 10 раз слабее или металл → сток; иначе промежуточный."""
    if N_neighbour is None or N_neighbour <= N_layer / 10.0:
        return RECOMMEND_SINK
    if N_neighbour >= 10.0 * N_layer:
        return RECOMMEND_REFLECT
    return RECOMMEND_INTERMEDIATE


def boundary_recommendations(s):
    """(6.2) Рекомендации для дальних границ n- и p-стороны текущего сценария.
    Возвращает {"n": (граница, рекомендация), "p": (...)}; None — металл."""
    if s.scenario == SCENARIO_A:
        return {"n": ("верхний контакт", _recommend(s.ND_plus, None)),
                "p": ("граница i/подложка", _recommend(s.N_i, s.substrate[0]))}
    return {"n": ("граница n/n⁺", _recommend(s.N_i, s.ND_plus)),
            "p": ("тыльный контакт", _recommend(s.substrate[0], None))}


def drude_mean_free_path(vth, mu, m_rel):
    """λ = v_th·μ·m_cc/q, см — оценка (соотношение Друде); источника со
    страницей в проекте нет, только для сравнения с L."""
    from mesa_diode.simulator.materials import M0
    return (vth * 1e-2) * (mu * 1e-4) * (m_rel * M0) / Q * 1e2
