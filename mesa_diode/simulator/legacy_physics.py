# -*- coding: utf-8 -*-
"""Физическое ядро симулятора мезоструктуры Ge/Si.

Прямая модель: параметры структуры задаются пользователем (не из измерений),
и по ним рассчитываются ВАХ и ВФХ. Это отличает симулятор от остальной части
пакета `mesa_diode`, которая, наоборот, подгоняет параметры под уже измеренную
ВАХ образца (см. `mesa_diode.fit`).

Без изменений перенесено из первой версии симулятора (mesa_simulator.py).
"""

import numpy as np

Q = 1.602176634e-19      # Кл, заряд электрона
K = 1.380649e-23         # Дж/К, постоянная Больцмана
EPS0 = 8.8541878128e-14  # Ф/см, электрическая постоянная

# Материальные константы (фиксированы; не выводятся как отдельные поля ввода,
# т.к. запрошен МИНИМАЛЬНЫЙ набор геометрических/электрофизических параметров
# мезы, а не полная база данных материалов).
# Если нужно поменять материал/справочные значения — меняйте только здесь,
# формулы ниже их не дублируют.
EPS_R_SI = 11.7
EPS_R_GE = 16.2
EG_SI0 = 1.12    # эВ, 300 К
EG_GE0 = 0.66    # эВ, 300 К
NI_SI_300 = 1.5e10    # см^-3
NI_GE_300 = 2.4e13    # см^-3

# Эмпирическая модель подвижности Кофи-Томаса: mu(N) = mu_min + (mu_max-mu_min)/(1+(N/Nref)^alpha)
MU_SI_N = dict(mu_max=1417.0, mu_min=60.0, Nref=9.68e16, alpha=0.68)
MU_GE_P = dict(mu_max=1900.0, mu_min=50.0, Nref=1.0e17, alpha=0.6)

TAU_MINORITY = 1e-7   # с, время жизни неосновных носителей (диффузионный ток)
TAU0_SCR = 1e-8       # с, время жизни в ОПЗ (рекомбинационный ток)

# Характерные (табличные) плотности тока насыщения p-n перехода при 300 К —
# только для справочного графика "идеальный диод по уравнению Шокли"
# (см. shockley_current_density ниже). НЕ используются в двухдиодной
# геометрической модели мезы выше (там ток насыщения p.J0 считается из
# реальных параметров структуры). Порядок величины: у Ge собственная
# концентрация носителей на несколько порядков больше, чем у Si (см.
# NI_GE_300 против NI_SI_300), поэтому и характерный ток насыщения на
# несколько порядков выше. Значения ориентировочные (типичный порядок
# величины из учебной литературы) — в интерфейсе их можно заменить своими.
MATERIAL_J0_A_CM2 = {
    "Si": 1e-12,
    "Ge": 1e-6,
}


def mobility(N, prm):
    """Формула Кофи-Томаса — см. simulator/formulas.py, раздел 7."""
    return prm['mu_min'] + (prm['mu_max'] - prm['mu_min']) / (
        1.0 + (N / prm['Nref']) ** prm['alpha'])


def ni_of_T(ni_300, Eg_eV, T):
    """Температурная зависимость собственной концентрации носителей."""
    k_eV = 8.617333e-5
    return ni_300 * (T / 300.0) ** 1.5 * np.exp(
        -Eg_eV / (2 * k_eV) * (1.0 / T - 1.0 / 300.0))


class MesaParams:
    """Собирает первичные параметры и рассчитывает производные величины.
    Порядок вычислений соответствует разделам 1-6 окна «Формулы и параметры»."""

    def __init__(self, D_um, d_um, h_um, ND_si, NA_ge, T_K, Rs_ohm, Rsh_ohm, n2):
        self.D_um, self.d_um, self.h_um = D_um, d_um, h_um
        self.ND, self.NA = ND_si, NA_ge
        self.T = T_K
        self.Rs, self.Rsh, self.n2 = Rs_ohm, Rsh_ohm, n2

        D_cm = D_um * 1e-4
        self.A = np.pi * (D_cm / 2.0) ** 2          # см^2, площадь мезы (раздел 1)

        self.Vt = K * self.T / Q                     # тепловой потенциал, В

        self.ni_si = ni_of_T(NI_SI_300, EG_SI0, self.T)
        self.ni_ge = ni_of_T(NI_GE_300, EG_GE0, self.T)
        self.ni_eff = np.sqrt(self.ni_si * self.ni_ge)

        self.mu_n_si = mobility(self.ND, MU_SI_N)
        self.mu_p_ge = mobility(self.NA, MU_GE_P)
        self.Dn_si = self.Vt * self.mu_n_si
        self.Dp_ge = self.Vt * self.mu_p_ge
        self.Ln_si = np.sqrt(self.Dn_si * TAU_MINORITY)
        self.Lp_ge = np.sqrt(self.Dp_ge * TAU_MINORITY)

        self.eps = EPS0 * (EPS_R_SI + EPS_R_GE) / 2.0
        self.Neff = 1.0 / (1.0 / self.ND + 1.0 / self.NA)
        self.Vbi = self.Vt * np.log((self.ND * self.NA) / (self.ni_eff ** 2))

        pn0 = self.ni_ge ** 2 / self.NA
        np0 = self.ni_si ** 2 / self.ND
        self.J0 = Q * (self.Dp_ge * pn0 / self.Lp_ge + self.Dn_si * np0 / self.Ln_si)  # раздел 2


def depletion_width(V, p):
    """Ширина ОПЗ, см. раздел 5."""
    dV = np.maximum(p.Vbi - V, 1e-4)
    return np.sqrt(2 * p.eps * dV / (Q * p.Neff))


def capacitance(V, p):
    """Барьерная ёмкость C(V), см. раздел 6. Возвращает NaN вблизи V=Vbi
    (сингулярность резкого перехода) и правее — модель там не определена."""
    Vlim = 0.95 * p.Vbi
    Vc = np.minimum(V, Vlim)
    C = p.eps * p.A / depletion_width(Vc, p)
    return np.where(V < Vlim, C, np.nan)


def estimate_grading_m(V, C, Vbi):
    """
    Оценка коэффициента градиента перехода m по наклону зависимости
    ln(C) от ln(Vbi - V) (см. раздел 6 окна формул):
        C ~ (Vbi - V)^(-1/(m+2))  =>  наклон = -1/(m+2)
    m -> 0 : резкий (ступенчатый) переход;  m -> 1 : линейно-плавный переход.
    Используются только точки при V < -0.05 В (подальше от Vbi, устойчивее оценка).
    Возвращает (m, наклон) или (None, None), если точек недостаточно.
    """
    mask = np.isfinite(C) & (C > 0) & (V < -0.05) & ((Vbi - V) > 1e-6)
    if np.sum(mask) < 5:
        return None, None
    x = np.log(Vbi - V[mask])
    y = np.log(C[mask])
    slope, _intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None, slope
    m = (-1.0 / slope) - 2.0
    return m, slope


def _Idiode(Vd, p, W0):
    Jrec0 = Q * p.ni_eff * W0 / (2 * TAU0_SCR)
    x1 = np.clip(Vd / p.Vt, -50, 80)
    x2 = np.clip(Vd / (p.n2 * p.Vt), -50, 80)
    return p.A * (p.J0 * np.expm1(x1) + Jrec0 * np.expm1(x2))


def _dIdiode(Vd, p, W0):
    Jrec0 = Q * p.ni_eff * W0 / (2 * TAU0_SCR)
    x1 = np.clip(Vd / p.Vt, -50, 80)
    x2 = np.clip(Vd / (p.n2 * p.Vt), -50, 80)
    return p.A * (p.J0 * np.exp(x1) / p.Vt + Jrec0 * np.exp(x2) / (p.n2 * p.Vt))


def solve_iv(V_array, p):
    """Численное решение неявного уравнения (раздел 4):
       I = I_диод(V - I*Rs) + (V - I*Rs)/Rsh   (метод Ньютона по каждой точке V)."""
    W0 = depletion_width(0.0, p)
    I_out = np.zeros_like(V_array, dtype=float)
    I_guess = 0.0
    for i, V in enumerate(V_array):
        for _ in range(60):
            Vd = V - I_guess * p.Rs
            f = I_guess - _Idiode(Vd, p, W0) - Vd / p.Rsh
            dVd_dI = -p.Rs
            df = 1.0 - _dIdiode(Vd, p, W0) * dVd_dI - dVd_dI / p.Rsh
            if abs(df) < 1e-30:
                break
            I_new = I_guess - f / df
            if abs(I_new - I_guess) > 1.0:
                I_new = I_guess + np.sign(I_new - I_guess) * 1.0
            if abs(I_new - I_guess) < 1e-16 * max(1.0, abs(I_new)):
                I_guess = I_new
                break
            I_guess = I_new
        I_out[i] = I_guess
    return I_out


def auto_scale(array, base_unit):
    """Подбор удобной приставки (базовая -> м -> мк -> н -> п) для отображения.
    base_unit — строка базовой единицы ('А' для тока, 'Ф' для ёмкости)."""
    finite = array[np.isfinite(array)]
    maxval = np.max(np.abs(finite)) if finite.size else 0.0
    for factor, prefix in ((1, ""), (1e3, "м"), (1e6, "мк"),
                            (1e9, "н"), (1e12, "п")):
        if maxval * factor >= 1.0 or prefix == "п":
            return factor, prefix + base_unit
    return 1e12, "п" + base_unit


def shockley_current_density(V, J0_A_cm2, T_K=300.0, n=1.0):
    """Идеальное уравнение Шокли для плотности тока: J = J0*(exp(qV/(n*kT)) - 1).

    Не связано с двухдиодной геометрической моделью мезы выше — это
    справочная кривая по характерной плотности тока насыщения материала
    (см. MATERIAL_J0_A_CM2), чтобы сравнить ВАХ мезы с «типичным» переходом
    Si или Ge.
    """
    Vt = K * T_K / Q
    x = np.clip(np.asarray(V, dtype=float) / (n * Vt), -50, 80)
    return J0_A_cm2 * np.expm1(x)


def current_density_from_area(current_A, area_cm2):
    """Плотность тока J = I / S по геометрической площади мезы S = area_cm2."""
    return np.asarray(current_A, dtype=float) / area_cm2


def robust_value_limits(primary_arrays, fallback_array, margin_fraction=0.2):
    """Границы оси Y по «содержательным» данным, а не по хвосту модели.

    Экспоненциальные модели (Шокли, двухдиодная) при широком диапазоне
    напряжений могут давать физически нереалистичные значения на краях
    (реальный диод либо ограничен Rs, либо электрически пробивается задолго
    до таких V) — если строить ось Y по ним, содержательная часть графика
    (там, где есть эксперимент) сжимается в незаметную линию у нуля.

    Если ``primary_arrays`` непусты (обычно — экспериментальные значения),
    границы считаются по ним с запасом margin_fraction, а модельная кривая
    может выходить за пределы видимой области — это ожидаемо. Если
    ``primary_arrays`` пуст (данных не загружено), используются границы
    ``fallback_array`` (обычно — сама модельная кривая) целиком.
    """
    values = [np.asarray(a) for a in primary_arrays if np.asarray(a).size]
    if values:
        combined = np.concatenate(values)
        vmin, vmax = float(np.min(combined)), float(np.max(combined))
    else:
        fb = np.asarray(fallback_array)
        finite = fb[np.isfinite(fb)]
        vmin, vmax = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)

    span = vmax - vmin
    margin = span * margin_fraction if span > 0 else max(abs(vmin), abs(vmax), 1.0) * margin_fraction
    return vmin - margin, vmax + margin


def adaptive_voltage_range(voltage_arrays, default_min=-1.0, default_max=0.6,
                            margin_fraction=0.05):
    """Диапазон напряжений для расчёта модели: покрывает значения по
    умолчанию и все переданные экспериментальные массивы (с запасом по
    краям), чтобы модельная кривая не обрывалась раньше точек эксперимента.
    """
    vmin, vmax = default_min, default_max
    for arr in voltage_arrays:
        arr = np.asarray(arr)
        if arr.size:
            vmin = min(vmin, float(np.min(arr)))
            vmax = max(vmax, float(np.max(arr)))

    span = vmax - vmin
    margin = span * margin_fraction if span > 0 else 0.05
    return vmin - margin, vmax + margin


def adaptive_point_count(v_min, v_max, base_count=121, resolution_V=0.01, max_count=400):
    """Число точек расчёта ВАХ: гуще на широком диапазоне напряжений, но не
    в ущерб отклику интерфейса (solve_iv — метод Ньютона, до 60 итераций на
    точку)."""
    span = max(v_max - v_min, 0.0)
    return int(np.clip(np.ceil(span / resolution_V), base_count, max_count))
