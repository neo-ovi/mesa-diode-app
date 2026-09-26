# -*- coding: utf-8 -*-
"""Материальные и физические постоянные симулятора (ТЗ §4).

Каждое значение сопровождается библиографической ссылкой в поле ``source``.
Новые константы сюда добавляются только вместе с источником.
"""

from dataclasses import dataclass, field

# Физические постоянные [CODATA 2018 / SI 2019]; q и k точные по определению.
Q = 1.602176634e-19        # Кл
K_B = 1.380649e-23         # Дж/К
EPS0 = 8.8541878128e-14    # Ф/см
M0 = 9.1093837e-31         # кг

CODATA = "CODATA 2018 / SI 2019"
ZI = "Зи С. Физика полупроводниковых приборов. Кн. 1. М.: Мир, 1984"
IOFFE_GE_B = "Ioffe NSM Archive, Ge: band structure"
IOFFE_GE_E = "Ioffe NSM Archive, Ge: electrical properties"
IOFFE_SI_B = "Ioffe NSM Archive, Si: band structure"
IOFFE_SI_E = "Ioffe NSM Archive, Si: electrical properties"
IOFFE_SIGE = "Ioffe NSM Archive, SiGe: basic parameters"
KKA56 = ("Kurtz A.D., Kulin S.A., Averbach B.L. // Phys. Rev. 1956. "
         "Vol. 101, № 4. P. 1285–1291")


@dataclass(frozen=True)
class Material:
    """Параметры полупроводника. E_g(T) = E_g(0) − αT²/(T + β) (2.1),
    N_C,V = N_C0,V0·T^{3/2} (2.2)."""

    name: str
    Eg0_eV: float
    alpha_eV_K: float
    beta_K: float
    Nc0: float                 # см⁻³·К^{−3/2}
    Nv0: float                 # см⁻³·К^{−3/2}
    eps_r: float
    mu_n_max: float            # см²/(В·с), чистый материал (верхний предел)
    mu_p_max: float
    vth_n: float               # см/с
    vth_p: float
    m_cc: float | None = None  # в единицах m0, только для Справочника
    a_angstrom: float | None = None
    mu_T_exp_n: float = 0.0    # μ_L ∝ T^{−α}: показатели α_n, α_p (2.9а)
    mu_T_exp_p: float = 0.0
    mu_doping: tuple = ()      # (lg N, g_n, g_p): g = μ(N)/μ(N → 0) (2.9)
    source: dict = field(default_factory=dict)


# Зависимость подвижности от концентрации примеси в Ge при 300 K: g = μ(N)/μ(N → 0)
# для электронов и дырок, оцифровка [Зи, с. 34, рис. 18] с шагом полдекады
# (точность отсчёта ≈ 5 %). Абсолютный предел μ(N → 0) берётся из mu_n_max,
# mu_p_max [Ioffe-Ge-e]; методичка, п. 3.4.
GE_MOBILITY_DOPING = (
    (14.0, 1.000, 1.000), (14.5, 0.968, 0.995), (15.0, 0.930, 0.965),
    (15.5, 0.872, 0.873), (16.0, 0.800, 0.735), (16.5, 0.710, 0.585),
    (17.0, 0.627, 0.415), (17.5, 0.528, 0.298), (18.0, 0.404, 0.180),
    (18.5, 0.294, 0.106), (19.0, 0.186, 0.056),
)

GE = Material(
    name="Ge",
    Eg0_eV=0.7437, alpha_eV_K=4.774e-4, beta_K=235.0,
    Nc0=1.98e15, Nv0=9.6e14,
    eps_r=16.2,
    mu_n_max=3900.0, mu_p_max=1900.0,
    vth_n=3.1e7, vth_p=1.9e7,
    m_cc=0.12, a_angstrom=5.658,
    mu_T_exp_n=1.66, mu_T_exp_p=2.33,
    mu_doping=GE_MOBILITY_DOPING,
    source={
        "Eg": f"{ZI}, с. 20, табл. на рис. 8",
        "Nc_Nv": IOFFE_GE_B,
        "eps_r": IOFFE_SIGE,
        "mu": IOFFE_GE_E,
        "vth": IOFFE_GE_E,
        "m_cc": IOFFE_GE_B,
        "a": IOFFE_SIGE,
        "mu_T": f"{ZI}, с. 35 (μ_n ∝ T^−1.66, μ_p ∝ T^−2.33 для Ge)",
        "mu_doping": f"{ZI}, с. 34, рис. 18 (оцифровка; те же данные — кривые Ирвина, с. 39, рис. 22)",
    },
)

# Заводится для бэклога Б-1 (подложка Si) и в расчёте не используется.
SI = Material(
    name="Si",
    Eg0_eV=1.170, alpha_eV_K=4.73e-4, beta_K=636.0,
    Nc0=6.2e15, Nv0=3.5e15,
    eps_r=11.7,
    mu_n_max=1400.0, mu_p_max=450.0,
    vth_n=2.3e7, vth_p=1.65e7,
    source={
        "Eg": f"{ZI}, с. 20, табл. на рис. 8",
        "Nc_Nv": IOFFE_SI_B,
        "eps_r": IOFFE_SIGE,
        "mu": IOFFE_SI_E,
        "vth": IOFFE_SI_E,
    },
)

# Рекомбинационная эффективность дислокации σ_R, см²/с [KKA56, с. 1289].
# Данные — объёмные кристаллы при плотности дислокаций 10⁵–10⁷ см⁻²;
# для эпитаксии и меньших плотностей — экстраполяция (§5А).
SIGMA_R = {
    "Ge высокоомный (30–40 Ом·см)": 5.5e-4,
    "Ge низкоомный (3–5 Ом·см)": 3.5e-3,
    "Si высокоомный": 1.65e-2,
}
SIGMA_R_SOURCE = f"{KKA56}, с. 1289"
SIGMA_R_VALID_RANGE = (1e5, 1e7)  # см⁻², диапазон N_dis в измерениях KKA56
