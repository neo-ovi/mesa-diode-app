# -*- coding: utf-8 -*-
"""Правая панель окна: эквивалентная схема модели тока, её формулы и расчёт
одной точки вручную на текущих числах (методичка, п. 1А.20).

Здесь — содержание панели: чистые функции elements() (схема) и panel_blocks()
(формулы с номерами, пояснением «где» и сноской; расчёт точки «формула →
числа → результат»; ВФХ); их проверяют тесты без окна. formula_lines() — та же
панель обычным текстом. Рисует scheme_panel.SchemePanel, набор формул —
typeset.py. Для модели тока, которой здесь нет (своя модель реестра models),
рисуется один блок «ток перехода» с её описанием summary."""

import math
import re
from dataclasses import dataclass, replace

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
    tex: tuple = ()      # подпись формулами (mathtext), по строке на элемент кортежа — так она на схеме


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
            "формула не знает: n и J₀ находят по ВАХ. " + hints.reference("n_emp"), tex=(r"$D$", r"$J_{0},\ n$")))
    elif s.model == ph.MODEL_TWO_DIODE:
        branches += [
            Element("d1", "diode", "D₁: J₀₁, n=1",
                    f"Диффузионный диод, n = 1: A·J₀₁·(e^{{V_д/(kT/q)}} − 1), J₀₁ = {_g(s.J01_2d)} А/см². "
                    "Преобладает при большом прямом смещении. " + hints.reference("J01_2d"),
                    tex=(r"$D_{1}$", r"$J_{01},\ n=1$")),
            Element("d2", "diode", "D₂: J₀₂, n=2",
                    f"Рекомбинационный диод, n = 2: A·J₀₂·(e^{{V_д/(2kT/q)}} − 1), J₀₂ = {_g(s.J02_2d)} А/см². "
                    "Ток рекомбинации в ОПЗ, преобладает при малом смещении. " + hints.reference("J02_2d"),
                    tex=(r"$D_{2}$", r"$J_{02},\ n=2$"))]
    elif s.model == ph.MODEL_PHYSICAL:
        branches += [
            Element("diff", "source", "I_diff",
                    "Диффузионный ток: неосновные носители диффундируют через нейтральные слои, n = 1. "
                    "Считается из структуры: концентрации, толщины слоёв, времена жизни, граничные условия "
                    "(4.3), (4.4). " + hints.reference("sensitivity"), tex=(r"$I_{\mathrm{diff}}$",)),
            Element("gr", "source", "I_gr",
                    f"Ток генерации–рекомбинации в ОПЗ, n ≈ 2: пропорционален ширине ОПЗ W и 1/τ₀ "
                    f"(τ₀^bg = {_g(s.tau0_bg)} с). Главный вклад обратного тока и малых прямых. "
                    + hints.reference("tau0_bg"), tex=(r"$I_{\mathrm{gr}}$",))]
    else:
        summary = model.summary if model else "ток перехода"
        branches.append(Element("junction", "box", "I_перехода", f"Ток перехода модели «{s.model}»: {summary}",
                                tex=(r"$I_{\mathrm{перехода}}$",)))
    shunt_on = np.isfinite(s.Rsh) and s.Rsh < HIGH_RSH
    branches.append(Element(
        "shunt", "resistor", "R_sh",
        f"Шунт R_sh = {_g(s.Rsh)} Ом — утечка в обход перехода (боковая стенка, поверхность, дефекты). "
        "Даёт прямую линию на обратной ветви и при малых напряжениях. Не считается из ρ слоёв: "
        "это путь мимо перехода. " + hints.reference("Rsh") + ("" if shunt_on else " Сейчас выключен (R_sh "
                                                                  "очень большое)."), shunt_on,
        tex=(r"$R_{\mathrm{sh}}$",)))
    if not basic:
        branches.append(Element(
            "leak", "source", "I_L|V|^m",
            f"Нелинейная утечка I_L·|V_д|^m (I_L = {_g(s.I_L)} А, m = {_g(s.m_leak)}): «мягкая» обратная "
            "ветвь, растущая быстрее прямой линии шунта. Эмпирика: подгонка подключает её, только если "
            "без неё ошибка больше 5 %. " + hints.reference("I_L") + ("" if s.I_L > 0 else
                                                                  " Сейчас выключена (I_L = 0)."),
            s.I_L > 0, tex=(r"$I_{L}\,|V|^{m}$",)))
    modulated = not basic and np.isfinite(s.I_mod)
    series = Element(
        "rs", "varres" if modulated else "resistor", "R_s(I)" if modulated else "R_s",
        f"Последовательное сопротивление R_s = {_g(s.Rs)} Ом: контакты, подложка, растекание тока. На нём "
        "падает часть напряжения: V = V_д + I·R_s, поэтому верх прямой ветви выпрямляется. "
        + ("Здесь R_s уменьшается с током (6.9): R_s/(1 + |I|/I_mod), "
           f"I_mod = {_g(s.I_mod)} А. " if modulated else "")
        + "Оценка по геометрии — Справочник. " + hints.reference("Rs"),
        tex=(r"$R_{\mathrm{s}}(I)$" if modulated else r"$R_{\mathrm{s}}$",))
    return branches, series


# ------------------------------------------------------- формулы панели --
# Блоки панели. Формулы — mathtext ($...$): кириллица только в \mathrm{…};
# тексты — с разметкой индексов _{…}/^{…}. У каждого блока есть text —
# та же строка обычным текстом (тесты, копирование расчёта).

@dataclass(frozen=True)
class Heading:
    text: str


@dataclass(frozen=True)
class Eq:
    """Выключная формула: строка или система строк, номер, «где», сноска."""
    tex: tuple
    number: str
    text: str
    where: tuple = ()      # (символ mathtext, пояснение) — символ поясняется один раз на панели
    note: str = ""         # источник и пункт методички


@dataclass(frozen=True)
class Step:
    """Строка расчёта вручную: формула → числа → результат."""
    label: str             # «по (6.3)» — откуда формула
    tex: str               # подстановка чисел
    result: str            # «= число единица» (mathtext) или ""
    text: str


@dataclass(frozen=True)
class Remark:
    text: str


def tex_num(x, digits=3):
    """Число для формулы: 0.2, 25.9, 3.41·10⁻⁸; ∞ для бесконечности."""
    if x is None or not np.isfinite(x):
        return r"\infty"
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    if -3 < exponent < digits:
        return f"{x:.{digits}g}"
    mantissa = f"{x / 10 ** exponent:.{digits}g}"
    if mantissa.startswith("10"):            # округление 9.996 → 10
        mantissa, exponent = mantissa.replace("10", "1", 1), exponent + 1
    return r"%s\cdot10^{%d}" % (mantissa, exponent)


def markup_num(x, digits=3):
    """Число для текста с разметкой индексов: 1.92·10^{13}."""
    tex = tex_num(x, digits)
    return tex.replace(r"\cdot", "·").replace(r"\infty", "∞")


def _u(unit):
    """Единица в формуле: прямым шрифтом через пробел (ISO 80000-1)."""
    return r"\ \mathrm{%s}" % unit


# Символы, общие для формул модели: (символ mathtext, пояснение).
W_A = (r"$A$", "площадь мезы, A = πD^{2}/4 (1.1), см^{2}")
W_VD = (r"$V_{\mathrm{д}}$", "напряжение на самом p-n переходе, В")
W_KT = (r"$kT/q$", "тепловое напряжение: 25.85 мВ при 300 К")
W_I = (r"$I$", "ток через диод — по оси графика, А")
W_V = (r"$V$", "напряжение на выводах диода — по оси графика, В")
W_RSH = (r"$R_{\mathrm{sh}}$", "шунт: утечка в обход перехода (боковая стенка, поверхность), Ом")
W_RS = (r"$R_{\mathrm{s}}$", "последовательное сопротивление: контакты, подложка, растекание тока, Ом")

TITLES = {ph.MODEL_EMPIRICAL: "Однодиодная модель (6.3)", ph.MODEL_TWO_DIODE: "Двухдиодная модель (6.3а)",
          ph.MODEL_PHYSICAL: "Физическая модель (гл. 6, 7)"}


_TEX_WORDS = ((r"\varepsilon", "ε"), (r"\tau", "τ"), (r"\eta", "η"), (r"\pi", "π"), (r"\ ", " "),
              (r"\,", " "))


def plain_tex(tex):
    """Символ mathtext обычным текстом: $V_{\\mathrm{д}}$ → V_д (для текстовой версии панели)."""
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", tex.strip("$"))
    for command, char in _TEX_WORDS:
        text = text.replace(command, char)
    return re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", text)


_SUPERSCRIPT = str.maketrans("0123456789−-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺")


def plain_markup(text):
    """Текст с разметкой индексов обычным текстом: см^{−3} → см⁻³, V_{bi} → V_bi."""
    def sup(match):
        body = match.group(1)
        return body.translate(_SUPERSCRIPT) if all(c in "0123456789−-+" for c in body) else "^" + body
    text = re.sub(r"\^\{([^{}]*)\}", sup, text)
    return re.sub(r"_\{([^{}]*)\}", r"_\1", text)


def panel_title(s, mode):
    if mode == presets.MODE_BASIC and s.model == ph.MODEL_EMPIRICAL:
        return "Простой диод (базовая модель)"
    return TITLES.get(s.model, f"Модель «{s.model}»")


def _junction_equations(s):
    """Формулы тока самого перехода I_д(V_д) для модели s.model."""
    if s.model == ph.MODEL_EMPIRICAL:
        return [Eq((r"$I_{\mathrm{д}}=AJ_{0}\left[\exp\left(\dfrac{V_{\mathrm{д}}}{n\,kT/q}\right)-1\right]$",),
                   "(6.3)", "I_д = A·J₀·[exp(V_д/(n·kT/q)) − 1]",
                   where=(W_A, (r"$J_{0}$", "плотность тока насыщения, А/см^{2}"),
                          W_VD, (r"$n$", "коэффициент идеальности: 1 — диффузия, 2 — рекомбинация в ОПЗ, "
                                         "между ними — смесь механизмов"), W_KT),
                   note="При n = 1 — формула Шокли [Ш49, с. 460, ур. (4.13)]; n — [Ман20, с. 45, ур. (2)]. "
                        "Методичка: п. 8.1, п. 1А.20.")]
    if s.model == ph.MODEL_TWO_DIODE:
        return [Eq((r"$I_{\mathrm{д}}=I_{01}+I_{02}$",
                    r"$I_{01}=AJ_{01}\left[\exp\left(\dfrac{V_{\mathrm{д}}}{kT/q}\right)-1\right]$",
                    r"$I_{02}=AJ_{02}\left[\exp\left(\dfrac{V_{\mathrm{д}}}{2kT/q}\right)-1\right]$"),
                   "(6.3а)", "I_д = A·J₀₁·[exp(V_д/(kT/q)) − 1] + A·J₀₂·[exp(V_д/(2kT/q)) − 1]",
                   where=(W_A, (r"$J_{01}$", "плотность тока насыщения диффузионного диода (n = 1), А/см^{2}"),
                          W_VD, W_KT,
                          (r"$J_{02}$", "то же для рекомбинации в ОПЗ (n = 2), А/см^{2}")),
                   note="[Зи, с. 99, ур. (55)]. Методичка: п. 8.4.")]
    if s.model == ph.MODEL_PHYSICAL:
        return [Eq((r"$I_{\mathrm{diff}}=AJ_{s}\left[\exp\left(\dfrac{V_{\mathrm{д}}}{kT/q}\right)-1\right]$",),
                   "(4.4)", "I_diff = A·J_s·[exp(V_д/(kT/q)) − 1]",
                   where=(W_A, (r"$J_{s}$", "плотность тока насыщения из параметров слоёв (4.3): концентрации "
                                            "неосновных носителей, D, L, толщины баз, граничные условия"),
                          W_VD, W_KT),
                   note="Диффузия неосновных носителей, n = 1 [Ш49, с. 460, ур. (4.13)]; [Зи, с. 94, ур. (44), "
                        "(45)]. Методичка: гл. 6."),
                Eq((r"$I_{\mathrm{gr}}=A\,\dfrac{q\,n_{i}W}{2\tau_{0}}\left[\exp\left(\dfrac{V_{\mathrm{д}}}"
                    r"{n_{2}kT/q}\right)-1\right]$",),
                   "(5.4)", "I_gr = A·q·n_i·W/(2τ₀)·[exp(V_д/(n₂·kT/q)) − 1]",
                   where=((r"$q$", "заряд электрона, Кл"), (r"$n_{i}$", "собственная концентрация Ge при T (2.3), "
                                                                       "см^{−3}"),
                          (r"$W$", "ширина ОПЗ при V_{д} (3.2), см"),
                          (r"$\tau_{0}$", "время жизни в ОПЗ (5.10), с"),
                          (r"$n_{2}$", "показатель тока ОПЗ, по теории 2 (5.6)")),
                   note="Генерация–рекомбинация через дефекты в ОПЗ [СНШ57, с. 1230, ур. (11)]; [Зи, с. 97–99, "
                        "ур. (48), (54)]. Методичка: гл. 7.")]
    model = models.get(s.model) if s.model in models.MODELS else None
    return [Remark(f"Ток перехода I_{{д}}(V_{{д}}) — модель «{s.model}»: " + (model.summary if model else ""))]


def _circuit_equation(s, basic):
    """Эквивалентная схема (6.1): как ток перехода складывается с утечками и R_s."""
    junction = "I_{\\mathrm{diff}}+I_{\\mathrm{gr}}" if s.model == ph.MODEL_PHYSICAL else "I_{\\mathrm{д}}"
    plain = "I_diff + I_gr" if s.model == ph.MODEL_PHYSICAL else "I_д"
    if basic:
        return Eq((rf"$I={junction}+\dfrac{{V_{{\mathrm{{д}}}}}}{{R_{{\mathrm{{sh}}}}}}$",
                   r"$V=V_{\mathrm{д}}+IR_{\mathrm{s}}$"),
                  "(6.1)", f"I = {plain} + V_д/R_sh;  V = V_д + I·R_s",
                  where=(W_I, W_RSH, W_V, W_RS),
                  note="Шунт — [Зи, с. 97, п. 1], R_{s} — [Зи, с. 97, п. 5]. Решатель подбирает V_{д} так, чтобы "
                       "выполнялись оба равенства. Методичка: гл. 9, пп. 9.1, 9.3.")
    return Eq((rf"$I={junction}+\dfrac{{V_{{\mathrm{{д}}}}}}{{R_{{\mathrm{{sh}}}}}}"
               r"+I_{L}\,\mathrm{sign}\,V_{\mathrm{д}}\left|\dfrac{V_{\mathrm{д}}}{1\ \mathrm{В}}\right|^{m}$",
               r"$V=V_{\mathrm{д}}+IR_{\mathrm{s}}(I),\qquad R_{\mathrm{s}}(I)=\dfrac{R_{\mathrm{s}}}"
               r"{1+|I|/I_{\mathrm{mod}}}$"),
              "(6.1), (6.9)", f"I = {plain} + V_д/R_sh + I_L·sign(V_д)·|V_д/1 В|^m;  V = V_д + I·R_s(I), "
                              "R_s(I) = R_s/(1 + |I|/I_mod)",
              where=(W_I, W_RSH, (r"$I_{L},\ m$", "нелинейная утечка и её показатель (эмпирика); I_{L} = 0 — "
                                                  "утечки нет"),
                     W_V, W_RS, (r"$I_{\mathrm{mod}}$", "ток, при котором R_{s} вдвое меньше (модуляция при высоком "
                                                       "уровне инжекции); ∞ — R_{s} постоянно")),
              note="Шунт — [Зи, с. 97, п. 1]; утечка — [Кур74, с. 167–168], [MG40]; R_{s} — [Зи, с. 97, п. 5], "
                   "модуляция — [Зи, с. 97, п. 4]. Методичка: гл. 9, пп. 9.1–9.4.")


def _step(label, tex, result, text):
    return Step(label, f"${tex}$", f"${result}$" if result else "", text)


def _hand_steps(s, basic, Vd):
    """Расчёт одной точки: те же числа, что у решателя (6.1)."""
    A, Vt = s.area, s.Vt
    I = float(ph.junction_current(s, Vd))
    Rs_I = float(ph.series_resistance(s, I))
    V = Vd + I * Rs_I
    D_cm = s.D
    steps = [
        _step("по (1.1)", r"A=\dfrac{\pi D^{2}}{4}=\dfrac{\pi\cdot(%s\ \mathrm{см})^{2}}{4}" % tex_num(D_cm, 4),
              r"=%s%s^{2}" % (tex_num(A), _u("см")),
              f"A = π·({_g(D_cm)} см)²/4 = {_g(A)} см²"),
        _step(f"при T = {s.T:g} К", r"kT/q=%s%s" % (tex_num(1e3 * Vt, 4), _u("мВ")), "",
              f"kT/q = {1e3 * Vt:.2f} мВ (T = {s.T:g} К)"),
    ]
    vd, vt = f"{Vd:.2f}", tex_num(Vt, 4)
    names = []
    if s.model == ph.MODEL_EMPIRICAL:
        Id = A * s.J0_emp * math.expm1(Vd / (s.n_emp * Vt))
        steps.append(_step("по (6.3)", r"I_{\mathrm{д}}=%s\cdot%s\cdot\left[\exp\left(\dfrac{%s}{%s\cdot%s}\right)"
                                       r"-1\right]" % (tex_num(A), tex_num(s.J0_emp), vd, tex_num(s.n_emp), vt),
                           r"=%s%s" % (tex_num(Id), _u("А")),
                           f"I_д = {_g(A)}·{_g(s.J0_emp)}·[exp({Vd:.2f}/({_g(s.n_emp)}·{Vt:.4f})) − 1] = {_g(Id)} А"))
        names.append(("I_{\\mathrm{д}}", "I_д"))
    elif s.model == ph.MODEL_TWO_DIODE:
        I1 = A * s.J01_2d * math.expm1(Vd / Vt)
        I2 = A * s.J02_2d * math.expm1(Vd / (2 * Vt))
        steps += [
            _step("по (6.3а)", r"I_{01}=%s\cdot%s\cdot\left[\exp\left(\dfrac{%s}{%s}\right)-1\right]"
                  % (tex_num(A), tex_num(s.J01_2d), vd, vt), r"=%s%s" % (tex_num(I1), _u("А")),
                  f"I₀₁ = {_g(A)}·{_g(s.J01_2d)}·[exp({Vd:.2f}/{Vt:.4f}) − 1] = {_g(I1)} А"),
            _step("по (6.3а)", r"I_{02}=%s\cdot%s\cdot\left[\exp\left(\dfrac{%s}{2\cdot%s}\right)-1\right]"
                  % (tex_num(A), tex_num(s.J02_2d), vd, vt), r"=%s%s" % (tex_num(I2), _u("А")),
                  f"I₀₂ = {_g(A)}·{_g(s.J02_2d)}·[exp({Vd:.2f}/(2·{Vt:.4f})) − 1] = {_g(I2)} А")]
        names += [("I_{01}", "I₀₁"), ("I_{02}", "I₀₂")]
    elif s.model == ph.MODEL_PHYSICAL:
        Js = float(ph.saturation_current_density(s, Vd))
        Idiff = float(ph.diffusion_current(s, Vd))
        Igr = float(ph.gr_current(s, Vd))
        W = float(ph.depletion_width(s, Vd))
        tau0 = ph.scr_lifetime(s)
        steps += [
            _step("по (4.3) — из параметров слоёв", r"J_{s}=%s%s^{2}" % (tex_num(Js), _u("А/см")), "",
                  f"J_s = {_g(Js)} А/см²"),
            _step("по (4.4)", r"I_{\mathrm{diff}}=%s\cdot%s\cdot\left[\exp\left(\dfrac{%s}{%s}\right)-1\right]"
                  % (tex_num(A), tex_num(Js), vd, vt), r"=%s%s" % (tex_num(Idiff), _u("А")),
                  f"I_diff = {_g(A)}·{_g(Js)}·[exp({Vd:.2f}/{Vt:.4f}) − 1] = {_g(Idiff)} А")]
        if s.sns_refinement or s.edge_area:
            steps.append(_step("по (5.4) с опциями «уточнение SNS» / «краевая площадь»",
                               r"I_{\mathrm{gr}}=%s%s" % (tex_num(Igr), _u("А")), "", f"I_gr = {_g(Igr)} А"))
        else:
            steps.append(_step(
                "по (5.4)", r"I_{\mathrm{gr}}=%s\cdot\dfrac{1.602\cdot10^{-19}\cdot%s\cdot%s}{2\cdot%s}\cdot"
                            r"\left[\exp\left(\dfrac{%s}{%s\cdot%s}\right)-1\right]"
                % (tex_num(A), tex_num(s.ni), tex_num(W), tex_num(tau0), vd, tex_num(s.n2), vt),
                r"=%s%s" % (tex_num(Igr), _u("А")),
                f"I_gr = {_g(A)}·q·{_g(s.ni)}·{_g(W)}/(2·{_g(tau0)})·[exp({Vd:.2f}/({_g(s.n2)}·{Vt:.4f})) − 1] "
                f"= {_g(Igr)} А"))
        names += [("I_{\\mathrm{diff}}", "I_diff"), ("I_{\\mathrm{gr}}", "I_gr")]
    else:
        Ij = I - float(ph.shunt_current(s, Vd)) - float(ph.leak_current(s, Vd))
        steps.append(_step("модель реестра", r"I_{\mathrm{д}}=%s%s" % (tex_num(Ij), _u("А")), "",
                           f"I_д = {_g(Ij)} А"))
        names.append(("I_{\\mathrm{д}}", "I_д"))
    Ish = float(ph.shunt_current(s, Vd))
    steps.append(_step("шунт", r"I_{\mathrm{sh}}=\dfrac{V_{\mathrm{д}}}{R_{\mathrm{sh}}}=\dfrac{%s}{%s}" % (
        vd, tex_num(s.Rsh)), r"=%s%s" % (tex_num(Ish), _u("А")), f"I_sh = V_д/R_sh = {_g(Ish)} А"))
    names.append(("I_{\\mathrm{sh}}", "I_sh"))
    if not basic and s.I_L > 0:
        IL = float(ph.leak_current(s, Vd))
        steps.append(_step("утечка", r"I_{L}\left|\dfrac{V_{\mathrm{д}}}{1\ \mathrm{В}}\right|^{m}=%s\cdot%s^{%s}"
                           % (tex_num(s.I_L), vd, tex_num(s.m_leak)), r"=%s%s" % (tex_num(IL), _u("А")),
                           f"I_L·|V_д|^m = {_g(IL)} А"))
        names.append(("I_{\\mathrm{ут}}", "I_ут"))
    total = "+".join(t for t, _p in names)
    steps.append(_step("сумма ветвей", r"I=%s" % total, r"=%s%s" % (tex_num(I), _u("А")),
                       f"I = {' + '.join(p for _t, p in names)} = {I:.3g} А"))
    rs_text = "R_s(I)" if not basic and np.isfinite(s.I_mod) else "R_s"
    rs_tex = r"R_{\mathrm{s}}(I)" if rs_text == "R_s(I)" else r"R_{\mathrm{s}}"
    steps.append(_step("по (6.1)", r"V=V_{\mathrm{д}}+I%s=%s+%s\cdot%s" % (rs_tex, vd, tex_num(I), tex_num(Rs_I)),
                       r"=%s%s" % (tex_num(V, 4), _u("В")),
                       f"V = V_д + I·{rs_text} = {Vd:.2f} + {I:.3g}·{_g(Rs_I)} = {V:.4g} В"))
    return steps


def _cv_blocks(s, mode):
    """ВФХ: формулы ёмкости и V_bi, стороны перехода и числа при V = 0."""
    Vt = s.Vt
    NA, ND = s.p_side.N, s.n_side.N
    n_n0, p_p0 = s.majority
    C0 = float(ph.capacitance(s, 0.0))
    boltzmann = s.vbi_method == ph.VBI_BOLTZMANN
    blocks = [Heading("ВФХ — вкладка «ВФХ и 1/C²»"),
              Eq((r"$C=\dfrac{\varepsilon A}{W}=A\sqrt{\dfrac{q\,\varepsilon N_{\mathrm{эф}}}"
                  r"{2\left(V_{\mathrm{bi}}-V-2kT/q\right)}}$",
                  r"$N_{\mathrm{эф}}=\dfrac{N_{A}N_{D}}{N_{A}+N_{D}}$"),
                 "(3.2)–(3.5)", "C = εA/W = A·√(q·ε·N_эф/(2·(V_bi − V − 2kT/q)));  N_эф = N_A·N_D/(N_A + N_D)",
                 where=((r"$\varepsilon$", "диэлектрическая проницаемость Ge, ε_{r}ε_{0}"),
                        (r"$W$", "ширина ОПЗ (3.2), см"), (r"$q$", "заряд электрона, Кл"),
                        (r"$N_{\mathrm{эф}}$", "приведённая концентрация перехода, см^{−3}"),
                        (r"$V_{\mathrm{bi}}$", "контактная разность потенциалов, В"), W_V,
                        (r"$N_{A},\ N_{D}$", "концентрации примеси p- и n-стороны перехода, см^{−3}")),
                 note="Приближение обеднения, резкий переход [Зи, с. 82–88, ур. (9)–(18в)]. Методичка: гл. 5, п. 4.4.")]
    if boltzmann:
        blocks.append(Eq((r"$V_{\mathrm{bi}}=\dfrac{kT}{q}\ln\dfrac{n_{n0}\,p_{p0}}{n_{i}^{2}}$",), "(3.1)",
                         "V_bi = (kT/q)·ln(n_n0·p_p0/n_i²)",
                         where=((r"$n_{n0},\ p_{p0}$", "равновесные концентрации основных носителей сторон (2.4); "
                                                       "при N ≫ n_{i} равны N_{D} и N_{A}"),
                                (r"$n_{i}$", "собственная концентрация Ge при T (2.3), см^{−3}")),
                         note="Статистика Больцмана [Зи, с. 82, ур. (7), (7а)]. Методичка: пп. 4.1, 4.2."))
    else:
        blocks.append(Eq((r"$qV_{\mathrm{bi}}=E_{g}+kT\,(\eta_{n}+\eta_{p})$",), "(3.1а)",
                         "qV_bi = E_g + kT·(η_n + η_p)",
                         where=((r"$E_{g}$", "ширина запрещённой зоны Ge при T (2.1), эВ"),
                                (r"$\eta_{n},\ \eta_{p}$", "приведённые уровни Ферми n- и p-стороны (2.7): точная "
                                                          "статистика, учитывает вырождение")),
                         note="[Зи, с. 82, ур. (7)]; при η < −3 совпадает с (3.1). Методичка: пп. 4.1–4.3."))
    blocks += [
        Remark(f"p-сторона — {LAYER_NAMES[s.p_side.layer]}, N_{{A}} = {markup_num(NA)} см^{{−3}}; "
               f"n-сторона — {LAYER_NAMES[s.n_side.layer]}, N_{{D}} = {markup_num(ND)} см^{{−3}}; "
               f"n_{{i}} = {markup_num(s.ni)} см^{{−3}}."),
        _step("по (3.5)", r"N_{\mathrm{эф}}=\dfrac{%s\cdot%s}{%s+%s}" % (tex_num(NA), tex_num(ND), tex_num(NA),
                                                                       tex_num(ND)),
              r"=%s%s^{-3}" % (tex_num(s.N_eff), _u("см")), f"N_эф = {_g(s.N_eff)} см⁻³"),
    ]
    if boltzmann:
        blocks.append(_step("по (3.1)", r"V_{\mathrm{bi}}=%s\cdot\ln\dfrac{%s\cdot%s}{(%s)^{2}}" % (
            tex_num(Vt, 4), tex_num(n_n0), tex_num(p_p0), tex_num(s.ni)), r"=%s%s" % (tex_num(s.Vbi), _u("В")),
            f"V_bi = {s.Vbi:.3f} В"))
    else:
        eta_n, eta_p = s.eta
        Eg = float(ph.band_gap(s.T, s.material))
        blocks.append(_step("по (3.1а); E_{g} в эВ, поэтому E_{g}/q — в вольтах",
                            r"V_{\mathrm{bi}}=\dfrac{E_{g}}{q}+\dfrac{kT}{q}(\eta_{n}+\eta_{p})=%s+%s\cdot(%s%s)" % (
            tex_num(Eg), tex_num(Vt, 4), tex_num(eta_n), ("+" if eta_p >= 0 else "") + tex_num(eta_p)),
            r"=%s%s" % (tex_num(s.Vbi), _u("В")), f"V_bi = {s.Vbi:.3f} В"))
    blocks.append(_step("при V = 0", r"C(0)=%s%s" % (tex_num(C0 * 1e12), _u("пФ")), "",
                        f"C(0) = {_g(C0 * 1e12)} пФ   (kT/q = {1e3 * Vt:.2f} мВ)"))
    return blocks


def panel_blocks(s, mode):
    """Содержание панели под схемой: формулы модели, расчёт точки, ВФХ."""
    basic = mode == presets.MODE_BASIC
    blocks = [Heading("Формулы модели")]
    blocks += _junction_equations(s)
    blocks.append(_circuit_equation(s, basic))
    blocks.append(Heading(f"Расчёт точки вручную (V_{{д}} = {SAMPLE_VD:.2f} В)"))
    blocks += _hand_steps(s, basic, SAMPLE_VD)
    blocks.append(Remark("Те же числа даёт решатель программы: точка (V, I) лежит на кривой модели."))
    return _explain_once(blocks + _cv_blocks(s, mode))


def _explain_once(blocks):
    """Символ поясняется под первой формулой, где он появился (как в учебнике)."""
    seen, result = set(), []
    for block in blocks:
        if isinstance(block, Eq):
            where = tuple(row for row in block.where if row[0] not in seen)
            seen.update(row[0] for row in where)
            block = replace(block, where=where)
        result.append(block)
    return result


def formula_lines(s, mode):
    """Панель обычным текстом: заголовок модели, формулы, расчёт точки, ВФХ."""
    lines = [panel_title(s, mode)]
    for block in panel_blocks(s, mode):
        if isinstance(block, Heading):
            lines += ["", block.text + ":"]
        elif isinstance(block, Eq):
            lines.append(f"  {block.text}   {block.number}")
            for index, (symbol, description) in enumerate(block.where):
                lines.append(f"    {'где ' if index == 0 else '    '}{plain_tex(symbol)} — "
                             f"{plain_markup(description)}")
        elif isinstance(block, Step):
            lines.append(f"  {block.text}")
        elif isinstance(block, Remark):
            lines.append(f"  {plain_markup(block.text)}")
    return lines


def model_summary(s, mode):
    """Одна-две фразы: как считает модель и что делать дальше."""
    if mode == presets.MODE_BASIC:
        return ("Базовая модель — простой диод: диод, шунт R_{sh} и последовательное R_{s}. Точку можно "
                "посчитать вручную (ниже). Погрешность большая — это намеренно: модель показывает, что "
                "это за диод. Точнее — режим «Расширенная».")
    model = models.get(s.model) if s.model in models.MODELS else None
    text = model.summary if model else ""
    if mode == presets.MODE_EXTENDED:
        text += " Модель меняется списком «Модель тока» сверху."
    return text
