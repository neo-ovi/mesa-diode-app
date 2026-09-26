# -*- coding: utf-8 -*-
"""§5. Структура образца: слои, стороны перехода, два сценария.

Методичка, гл. 2 (слои, п. 2.6 — обогащённый слой подложки), п. 10.4."""

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from mesa_diode.simulator.materials import EPS0, GE
from mesa_diode.simulator.physics.carriers import (
    acceptor_from_resistivity, effective_dos, equilibrium_carriers, intrinsic_concentration,
    reduced_fermi_level, thermal_voltage,
)
from mesa_diode.simulator.physics.constants import (
    MODEL_PHYSICAL, REFLECT, SCENARIO_A, SCENARIO_B, SINK, VBI_DEGENERATE,
)
from mesa_diode.simulator.physics.electrostatics import built_in_potential


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
