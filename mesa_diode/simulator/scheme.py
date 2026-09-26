# -*- coding: utf-8 -*-
"""Правая панель окна: эквивалентная схема модели тока, основная формула и
расчёт одной точки вручную на текущих числах (методичка, п. 1А.20).

Здесь — содержание панели: чистые функции elements() и formula_lines(), их
проверяют тесты без окна; рисует scheme_panel.SchemePanel. Для модели тока, которой
здесь нет (своя модель реестра models), рисуется один блок «ток перехода»
с её описанием summary."""

import math
from dataclasses import dataclass

import numpy as np

from mesa_diode.simulator import hints, models, presets
from mesa_diode.simulator import physics as ph

SAMPLE_VD = 0.20          # напряжение на диоде для расчёта точки вручную, В
HIGH_RSH = 1e11           # шунт больше этого считается выключенным
LAYER_NAMES = {"n+": "n⁺-слой", "i": "i-слой", "surf": "обогащённый слой подложки", "sub": "подложка"}


@dataclass(frozen=True)
class Element:
    key: str             # имя элемента (для подсказки и тестов)
    symbol: str          # diode | source | resistor | varres | box
    label: str           # подпись на схеме
    tip: str             # подсказка при наведении: что это и почему здесь
    active: bool = True  # выключенный элемент — пунктиром


def _g(x):
    """Число для текста: 3 значащие цифры, ∞ для бесконечности."""
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "∞"
    return f"{x:.3g}"


# --------------------------------------------------------------- схема --

def elements(s, mode):
    """Элементы схемы: (параллельные ветви между шинами, последовательный R_s)."""
    basic = mode == presets.MODE_BASIC
    model = models.get(s.model) if s.model in models.MODELS else None
    branches = []
    if s.model == ph.MODEL_EMPIRICAL:
        branches.append(Element(
            "diode", "diode", "D: J₀, n",
            f"Диод — сам p-n переход. Ток I_д = A·J₀·(e^{{V_д/(n·kT/q)}} − 1): J₀ = {_g(s.J0_emp)} А/см² "
            f"— «пропускная способность» при малом напряжении, n = {_g(s.n_emp)} — механизм "
            "(n = 1 — диффузия, n = 2 — рекомбинация в ОПЗ, между ними — смесь). Структуру слоёв эта "
            "формула не знает: n и J₀ находят по ВАХ. " + hints.reference("n_emp")))
    elif s.model == ph.MODEL_TWO_DIODE:
        branches += [
            Element("d1", "diode", "D₁: J₀₁, n=1",
                    f"Диффузионный диод, n = 1: A·J₀₁·(e^{{V_д/(kT/q)}} − 1), J₀₁ = {_g(s.J01_2d)} А/см². "
                    "Преобладает при большом прямом смещении. " + hints.reference("J01_2d")),
            Element("d2", "diode", "D₂: J₀₂, n=2",
                    f"Рекомбинационный диод, n = 2: A·J₀₂·(e^{{V_д/(2kT/q)}} − 1), J₀₂ = {_g(s.J02_2d)} А/см². "
                    "Ток рекомбинации в ОПЗ, преобладает при малом смещении. " + hints.reference("J02_2d"))]
    elif s.model == ph.MODEL_PHYSICAL:
        branches += [
            Element("diff", "source", "I_diff",
                    "Диффузионный ток: неосновные носители диффундируют через нейтральные слои, n = 1. "
                    "Считается из структуры: концентрации, толщины слоёв, времена жизни, граничные условия "
                    "(4.3), (4.4). " + hints.reference("sensitivity")),
            Element("gr", "source", "I_gr",
                    f"Ток генерации–рекомбинации в ОПЗ, n ≈ 2: пропорционален ширине ОПЗ W и 1/τ₀ "
                    f"(τ₀^bg = {_g(s.tau0_bg)} с). Главный вклад обратного тока и малых прямых. "
                    + hints.reference("tau0_bg"))]
    else:
        summary = model.summary if model else "ток перехода"
        branches.append(Element("junction", "box", "I_перехода", f"Ток перехода модели «{s.model}»: {summary}"))
    shunt_on = np.isfinite(s.Rsh) and s.Rsh < HIGH_RSH
    branches.append(Element(
        "shunt", "resistor", "R_sh",
        f"Шунт R_sh = {_g(s.Rsh)} Ом — утечка в обход перехода (боковая стенка, поверхность, дефекты). "
        "Даёт прямую линию на обратной ветви и при малых напряжениях. Не считается из ρ слоёв: "
        "это путь мимо перехода. " + hints.reference("Rsh") + ("" if shunt_on else " Сейчас выключен (R_sh "
                                                                  "очень большое)."), shunt_on))
    if not basic:
        branches.append(Element(
            "leak", "source", "I_L|V|^m",
            f"Нелинейная утечка I_L·|V_д|^m (I_L = {_g(s.I_L)} А, m = {_g(s.m_leak)}): «мягкая» обратная "
            "ветвь, растущая быстрее прямой линии шунта. Эмпирика: подгонка подключает её, только если "
            "без неё ошибка больше 5 %. " + hints.reference("I_L") + ("" if s.I_L > 0 else
                                                                  " Сейчас выключена (I_L = 0)."),
            s.I_L > 0))
    modulated = not basic and np.isfinite(s.I_mod)
    series = Element(
        "rs", "varres" if modulated else "resistor", "R_s(I)" if modulated else "R_s",
        f"Последовательное сопротивление R_s = {_g(s.Rs)} Ом: контакты, подложка, растекание тока. На нём "
        "падает часть напряжения: V = V_д + I·R_s, поэтому верх прямой ветви выпрямляется. "
        + ("Здесь R_s уменьшается с током (6.9): R_s/(1 + |I|/I_mod), "
           f"I_mod = {_g(s.I_mod)} А. " if modulated else "")
        + "Оценка по геометрии — Справочник. " + hints.reference("Rs"))
    return branches, series


# -------------------------------------------------------------- формулы --

def _cv_lines(s):
    Vt = s.Vt
    NA, ND = s.p_side.N, s.n_side.N
    C0 = float(ph.capacitance(s, 0.0))
    vbi = ("V_bi = (kT/q)·ln(N_A·N_D/n_i²)" if s.vbi_method == ph.VBI_BOLTZMANN
           else "V_bi — по уровням Ферми с учётом вырождения (3.1а)")
    return [
        "ВФХ (вкладка «ВФХ и 1/C²»):",
        "  C = A·√(q·ε·N_эф / (2·(V_bi − V − 2kT/q)))",
        "  N_эф = N_A·N_D/(N_A + N_D);  " + vbi,
        f"  p-сторона — {LAYER_NAMES[s.p_side.layer]}, N_A = {_g(NA)} см⁻³;",
        f"  n-сторона — {LAYER_NAMES[s.n_side.layer]}, N_D = {_g(ND)} см⁻³",
        f"  n_i = {_g(s.ni)} см⁻³, V_bi = {s.Vbi:.3f} В, N_эф = {_g(s.N_eff)} см⁻³",
        f"  C(0) = {_g(C0 * 1e12)} пФ   (kT/q = {1e3 * Vt:.2f} мВ)",
    ]


def formula_lines(s, mode):
    """Строки панели: заголовок, основная формула, точка вручную, ВФХ."""
    basic = mode == presets.MODE_BASIC
    Vd = SAMPLE_VD
    A, Vt = s.area, s.Vt
    I = float(ph.junction_current(s, Vd))
    Rs_I = float(ph.series_resistance(s, I))
    V = Vd + I * Rs_I
    title = {ph.MODEL_EMPIRICAL: "Простой диод (базовая модель)" if basic else "Однодиодная модель (6.3)",
             ph.MODEL_TWO_DIODE: "Двухдиодная модель (6.3а)",
             ph.MODEL_PHYSICAL: "Физическая модель (гл. 4–7)"}.get(s.model, f"Модель «{s.model}»")
    lines = [title]
    circuit = " + V_д/R_sh" + ("" if basic else " + I_L·|V_д|^m")
    series = "V = V_д + I·R_s" if basic else "V = V_д + I·R_s/(1 + |I|/I_mod)"
    if s.model == ph.MODEL_EMPIRICAL:
        lines += [f"  I = A·J₀·(e^(V_д/(n·kT/q)) − 1){circuit}", f"  {series}"]
    elif s.model == ph.MODEL_TWO_DIODE:
        lines += [f"  I = A·[J₀₁·(e^(V_д/(kT/q)) − 1) + J₀₂·(e^(V_д/(2kT/q)) − 1)]{circuit}", f"  {series}"]
    elif s.model == ph.MODEL_PHYSICAL:
        lines += [f"  I = I_diff(V_д) + I_gr(V_д){circuit}",
                  "  I_diff = A·J_s·(e^(V_д/(kT/q)) − 1),  J_s — из слоёв (4.3)",
                  "  I_gr = A·q·n_i·W/(2τ₀)·(e^(V_д/(n₂kT/q)) − 1)  (5.4)", f"  {series}"]
    else:
        lines += [f"  I = I_перехода(V_д){circuit}", f"  {series}"]
    lines += ["  V_д — напряжение на самом переходе; A = πD²/4", "",
              f"Точка вручную при V_д = {Vd:.2f} В:",
              f"  A = π·({1e4 * s.D:g} мкм)²/4 = {_g(A)} см²,  kT/q = {1e3 * Vt:.2f} мВ (T = {s.T:g} К)"]
    if s.model == ph.MODEL_EMPIRICAL:
        Id = A * s.J0_emp * math.expm1(Vd / (s.n_emp * Vt))
        lines.append(f"  I_д = {_g(A)}·{_g(s.J0_emp)}·(e^({Vd:.2f}/({_g(s.n_emp)}·{Vt:.4f})) − 1) = {_g(Id)} А")
    elif s.model == ph.MODEL_TWO_DIODE:
        I1 = A * s.J01_2d * math.expm1(Vd / Vt)
        I2 = A * s.J02_2d * math.expm1(Vd / (2 * Vt))
        lines.append(f"  I₀₁ = {_g(I1)} А,  I₀₂ = {_g(I2)} А")
    elif s.model == ph.MODEL_PHYSICAL:
        Js = float(ph.saturation_current_density(s, Vd))
        lines.append(f"  J_s = {_g(Js)} А/см²: I_diff = {_g(float(ph.diffusion_current(s, Vd)))} А, "
                     f"I_gr = {_g(float(ph.gr_current(s, Vd)))} А")
    lines.append(f"  I_sh = V_д/R_sh = {_g(float(ph.shunt_current(s, Vd)))} А")
    if not basic and s.I_L > 0:
        lines.append(f"  I_L·|V_д|^m = {_g(float(ph.leak_current(s, Vd)))} А")
    lines += [f"  I = {_g(I)} А;  V = {Vd:.2f} + {_g(I)}·{_g(Rs_I)} = {V:.4g} В", ""]
    return lines + _cv_lines(s)


def model_summary(s, mode):
    """Одна-две фразы: как считает модель и что делать дальше."""
    if mode == presets.MODE_BASIC:
        return ("Базовая модель — простой диод: диод, шунт R_sh и последовательное R_s. Точку можно "
                "посчитать вручную (ниже). Погрешность большая — это намеренно: модель показывает, что "
                "это за диод. Точнее — режим «Расширенная».")
    model = models.get(s.model) if s.model in models.MODELS else None
    text = model.summary if model else ""
    if mode == presets.MODE_EXTENDED:
        text += " Модель меняется списком «Модель тока» сверху."
    return text
