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


# ------------------------------------------ §5. Структура и два сценария --

SCENARIO_A = "A"   # i-слой p: переход n⁺/i на глубине d_n
SCENARIO_B = "B"   # i-слой n (по умолчанию): переход i/подложка на дне мезы
SCENARIOS = (SCENARIO_A, SCENARIO_B)

SINK, REFLECT, LONG, LAYER = "sink", "reflect", "long", "layer"
BOUNDARY_LABELS = {SINK: "сток", REFLECT: "отражение", LONG: "длинная база", LAYER: "соседний слой"}

MODEL_PHYSICAL = "physical"     # физическая модель §2–§6
MODEL_EMPIRICAL = "empirical"   # эмпирическая модель (6.3): J₀ и n задаются (базовый режим)
MODEL_TWO_DIODE = "two_diode"   # двухдиодная модель (6.3а): J₀₁ (n = 1) и J₀₂ (n = 2)
# Формулы тока перехода каждой модели — в реестре simulator/models.py.

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
    neighbour: object = None   # слой того же типа за дальней границей (Side) или None — контакт


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
    T_rho: float = 300.0             # температура, при которой измерено ρ_sub, К
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
    I_mod: float = float("inf")       # ток модуляции R_s (6.9), А; inf — R_s постоянно
    Rsh: float = float("inf")
    I_L: float = 0.0
    m_leak: float = 3.0
    bc_A_n: str = SINK
    bc_A_p: str = SINK                # умолчания ТЗ (эталоны §9); в окне по умолчанию LAYER
    bc_B_n: str = REFLECT
    bc_B_p: str = SINK
    sns_refinement: bool = False
    edge_area: bool = False
    vbi_method: str = VBI_DEGENERATE
    model: str = MODEL_PHYSICAL
    J0_emp: float = 1e-6              # А/см², эмпирическая модель (6.3)
    n_emp: float = 1.5                # идеальность эмпирической модели (6.3)
    J01_2d: float = 1e-7              # А/см², двухдиодная модель (6.3а): диффузия, n = 1
    J02_2d: float = 1e-5              # А/см², двухдиодная модель (6.3а): ОПЗ, n = 2
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
        """N_A подложки по ρ_sub (2.8) при температуре измерения ρ T_ρ:
        (N_A, [предупреждения]). N_A от температуры не зависит (полная ионизация)."""
        return acceptor_from_resistivity(self.rho_sub, self.T_rho, self.material)

    # --- стороны перехода по сценарию (§5.2) ---
    @cached_property
    def n_side(self):
        if self.scenario == SCENARIO_A:
            return Side("n", "n+", self.ND_plus, self.d_n, self.mu_p_nplus,
                        self.tau_p_bg, self.sigma_R_epi, self.bc_A_n)
        nplus = Side("n", "n+", self.ND_plus, self.d_n, self.mu_p_nplus,
                     self.tau_p_bg, self.sigma_R_epi, SINK)          # за ним верхний контакт
        return Side("n", "i", self.N_i, self.d_i, self.mu_p_i,
                    self.tau_p_bg, self.sigma_R_epi, self.bc_B_n, nplus)

    @cached_property
    def p_side(self):
        if self.scenario == SCENARIO_A:
            sub = Side("p", "sub", self.substrate[0], self.d_sub, self.mu_n,
                       self.tau_n_bg, self.sigma_R_sub, SINK)        # за ней тыльный контакт
            return Side("p", "i", self.N_i, self.d_i, self.mu_n,
                        self.tau_n_bg, self.sigma_R_epi, self.bc_A_p, sub)
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


BISECTION_STEPS = 64   # 2⁻⁶⁴ от |V|: точность V_d лучше 10⁻¹⁸ В


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
    он находится бисекцией сразу для всех точек (64 шага). Если сумма
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


# ------------------------------------------ Справочник: проверки §1, §6 --

def low_injection_voltage(s):
    """§6.4. Граница низкой инжекции V_LI = (kT/q)·ln(0.1·M_B/m_B0), В: M_B, m_B0 — равновесные основные
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


# ------------------------------------- §6.3. Идеальность из эксперимента --

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


# ------------------------------------------- §6.6. Сравнение сценариев --

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


# ------------------------------------- оценка τ₀ по обратной ветви ВАХ --

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
