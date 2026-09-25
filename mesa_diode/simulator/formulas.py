# -*- coding: utf-8 -*-
"""Окно «Формулы и параметры» (ТЗ §6, §7): вкладки «Модель» и «Справочник».

Содержание вкладки «Модель» — реестр SECTIONS: разделы §0–§7, каждый —
список блоков (текст, формула с источником или пометкой, плашка,
«→ График N», таблица параметров, «При текущих параметрах»). ID формул
совпадают с docstring функций simulator/physics.py (это проверяет тест).

Два вида строк — не смешивать:
  * mathtext ($...$) — только в формулах и символах таблиц; разрешены
    \\frac \\sqrt \\exp \\ln \\coth \\tanh \\sinh \\cosh \\min \\max \\int \\infty
    \\left \\right \\mathrm \\varepsilon \\mathcal{E} \\eta \\sigma \\lambda \\Delta
    \\ll \\gg \\approx \\cdot, индексы и степени; запрещены \\text, \\operatorname,
    \\begin и кириллица внутри $…$;
  * обычный текст — с разметкой индексов _{...} и ^{...}; символы $ и \\
    в нём не используются.

Пояснения с числами конкретного образца в код не пишутся: они приходят
из набора образца (поле notes, блок Note) — см. simulator/presets.py.
"""

import base64
import io
import re
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mesa_diode.simulator import physics as ph
from mesa_diode.simulator import presets
from mesa_diode.simulator.materials import GE, SIGMA_R, SIGMA_R_SOURCE

UM = 1e-4

# ------------------------------------------------------------ блоки разделов --

ASSUMPTION = "допущение"
EMPIRICAL = "эмпирика"
INTERPOLATION = "интерполяция"
ESTIMATE = "оценка"
TZ_RULE = "правило ТЗ"
TZ_METRIC = "метрика ТЗ"
IMPORTANT = "важно"

BADGE_COLORS = {
    ASSUMPTION: "#fff3b0",
    IMPORTANT: "#fff3b0",
    EMPIRICAL: "#ffd8a8",
    INTERPOLATION: "#e6e6e6",
    ESTIMATE: "#e6e6e6",
    TZ_RULE: "#e6e6e6",
    TZ_METRIC: "#e6e6e6",
}


@dataclass(frozen=True)
class Text:
    text: str
    style: str = "normal"   # normal / small / muted


@dataclass(frozen=True)
class Formula:
    tex: str
    source: str = ""
    badge: str = ""
    fid: str = ""           # ID вида "(2.1)"; у итогов — рамка
    result: bool = False
    note: str = ""          # текст под формулой


@dataclass(frozen=True)
class Plaque:
    kind: str
    text: str


@dataclass(frozen=True)
class Graph:
    text: str


@dataclass(frozen=True)
class Params:
    rows: tuple             # (символ mathtext, описание, единицы)


@dataclass(frozen=True)
class Current:
    key: str                # ключ функции в CURRENT_VALUES


@dataclass(frozen=True)
class Note:
    key: str                # ключ в preset.notes


@dataclass(frozen=True)
class Section:
    sid: str
    title: str
    why: str
    blocks: tuple = field(default_factory=tuple)


def _t(text, style="normal"):
    return Text(text, style)


SECTIONS = [
    Section("§0", "Введение",
            "Программа рассчитывает темновую ВАХ и ВФХ мезадиода по параметрам структуры "
            "и сравнивает их с экспериментом.",
            (
                _t("Ток складывается из:\n"
                   "  • диффузии неосновных носителей (идеальный диод Шокли, n = 1);\n"
                   "  • генерации–рекомбинации через дефекты в области пространственного заряда "
                   "(модель Са–Нойса–Шокли, n = 2);\n"
                   "  • утечек (R_{sh}, нелинейная утечка).\n"
                   "К ним добавляется падение напряжения на R_{s}."),
                _t("Время жизни в ОПЗ связано с дефектами, в том числе с плотностью дислокаций, "
                   "поэтому ВАХ — мост к структурным измерениям."),
                _t("Тип проводимости нелегированного слоя можно проверить, сравнивая модельную "
                   "ВФХ с экспериментом."),
                _t("Коэффициент идеальности программа определяет из эксперимента сама."),
                _t("Подробные выкладки — в Методичке (меню «Справка»)."),
                _t("Соглашения: V > 0 — прямое смещение (плюс на p-стороне), I > 0 — прямой ток "
                   "[Ш49, с. 460]. У Зи формулы записаны как V_{bi} ± V с «+» для обратного "
                   "смещения; в программе везде V_{bi} − V с V со знаком. Единицы расчёта: см, "
                   "см⁻³, с, В, А, Ф, К; E_{g} — в эВ. V_{t} = kT/q.", "muted"),
            )),

    Section("§1", "Геометрия и изоляция перехода",
            "Площадь мезы A задаёт масштаб всех токов и ёмкости. Высота мезы h (глубина "
            "травления) определяет, изолирован ли переход.",
            (
                Formula(r"$A=\frac{\pi D^{2}}{4}$", "площадь круга диаметром D", fid="(1.1)", result=True),
                _t("(1.2) Изоляция перехода (только Справочник; объяснение — Методичка §2.4):\n"
                   "  • сценарий A: изолирован, если h > d_{n};\n"
                   "  • сценарий B (z_{j} = d_{epi}): h < d_{epi} − x_{n}(V) — «не изолирован: n-слой "
                   "соединяет мезу с полем» (красное); d_{epi} − x_{n} ≤ h < d_{epi} + x_{p}(V) — «ОПЗ "
                   "выходит на поверхность поля у подножия мезы» (жёлтое); h ≥ d_{epi} + x_{p}(V) — "
                   "«ОПЗ внутри мезы»."),
                Formula(r"$\Delta A\approx\pi D\,\max\left(0,\ x_{p}(V)-(h-z_{j})\right)$", badge=ESTIMATE,
                        fid="(1.3)", result=True,
                        note="Добавочная площадь ОПЗ, оценка ТЗ (геометрическая); для сценария B "
                             "z_{j} = d_{epi}. Выводится в Справочнике. При включённой опции ток ОПЗ (5.4) "
                             "считается по A + ΔA, диффузионный — по A."),
                Formula(r"$S_{side}=\pi D h$", "площадь боковой стенки цилиндра",
                        note="Справочно: площадь боковой стенки; нужна для бэклога Б-3."),
                Graph("→ Графики 1–3: площадь A (1.1) входит во все токи и в ёмкость; (1.2) и (1.3) — "
                      "во вкладке «Справочник»."),
                Params((
                    (r"$D$", "диаметр мезы (внешний диаметр кольцевого контакта)", "мкм"),
                    (r"$h$", "высота мезы (глубина травления)", "мкм"),
                    (r"$d_{epi},\ d_{n}$", "толщина эпитаксии и n⁺-слоя", "мкм"),
                    (r"$z_{j}$", "глубина перехода: d_{n} (A) или d_{epi} (B)", "мкм"),
                    (r"$x_{n},\ x_{p}$", "края ОПЗ в n- и p-области (3.2)", "мкм"),
                )),
                Current("geometry"),
            )),

    Section("§2", "Материал и равновесие",
            "Ширина запрещённой зоны и эффективные плотности состояний дают собственную "
            "концентрацию n_{i}(T), а из неё и легирования — равновесные концентрации "
            "носителей в каждом слое.",
            (
                Formula(r"$E_{g}(T)=E_{g}(0)-\frac{\alpha T^{2}}{T+\beta}$",
                        "[Зи, с. 20, табл. на рис. 8]", fid="(2.1)"),
                Formula(r"$N_{C}=N_{C0}T^{3/2},\qquad N_{V}=N_{V0}T^{3/2}$", "[Ioffe-Ge-b]", fid="(2.2)"),
                Formula(r"$n_{i}=\sqrt{N_{C}N_{V}}\exp\left(-\frac{E_{g}}{2kT}\right)$",
                        "[Зи, с. 24, ур. (19), (19а)]", fid="(2.3)"),
                Formula(r"$p_{0}=\frac{N_{A}}{2}+\sqrt{\frac{N_{A}^{2}}{4}+n_{i}^{2}},\qquad n_{0}=\frac{n_{i}^{2}}{p_{0}}$",
                        "[Зи, с. 24, ур. (19)] и электронейтральность (полная ионизация)",
                        fid="(2.4)", result=True,
                        note="Для p-слоя; n-слой — симметрично. При N ≫ n_{i}: p_{0} = N_{A}, "
                             "n_{0} = n_{i}²/N_{A}. Для почти собственного слоя обязательна полная форма."),
                Formula(r"$D=\frac{kT}{q}\mu$", "[Зи, с. 36, ур. (44)]; для вырожденного случая — ур. (43а)",
                        fid="(2.5)"),
                Formula(r"$L=\sqrt{D\tau}$", "[Зи, с. 94, ур. (41)]", fid="(2.6)"),
                Formula(r"$n=N_{C}\frac{2}{\sqrt{\pi}}F_{1/2}(\eta_{n}),\qquad \eta_{n}=\frac{E_{F}-E_{C}}{kT}$",
                        "[Зи, с. 22, ур. (11), (13)]", fid="(2.7)"),
                Formula(r"$p=N_{V}\frac{2}{\sqrt{\pi}}F_{1/2}(\eta_{p}),\qquad \eta_{p}=\frac{E_{V}-E_{F}}{kT}$",
                        "[Зи, с. 22, ур. (14)]"),
                Formula(r"$F_{1/2}(\eta)=\int_{0}^{\infty}\frac{x^{1/2}\,dx}{1+e^{x-\eta}}$",
                        "[Зи, с. 22, ур. (11); с. 23, рис. 10]",
                        note="η находится для равновесной концентрации основных носителей слоя (2.4) "
                             "численно (quad + brentq)."),
                Formula(r"$\frac{2}{\sqrt{\pi}}F_{1/2}(\eta)\,e^{-\eta}$", "ошибка Больцмана: отношение точной "
                        "концентрации к больцмановской при том же η, из (2.7)"),
                Plaque(TZ_RULE, "Предупреждения: η ≥ −3 — «начало вырождения, ошибка X %»; η ≥ 0 — "
                                "«вырожден». Порог 3kT — численная трактовка «несколько kT» [Зи, с. 22]. "
                                "Таблица порогов при текущей T — во вкладке «Справочник»."),
                Formula(r"$\rho=\frac{1}{q(\mu_{n}n_{0}+\mu_{p}p_{0})}$", "[Зи, с. 36, ур. (47)]",
                        fid="(2.8)", result=True,
                        note="N_{A} подложки по ρ_{sub}: решение совместно с (2.4); μ — чистого Ge (§4 ТЗ). "
                             "ρ(N_{A}) немонотонна: максимум ≈ 59.5 Ом·см при N_{A} ≈ 10^{13} см⁻³, "
                             "ρ(0) = 56.1 Ом·см; при ρ > 56.1 решений два — предупреждение, берётся большее."),
                Formula(r"$R_{sub}=\frac{\rho_{sub}d_{sub}}{A}$", "[Зи, с. 36, ур. (45)–(47)]", badge=ESTIMATE,
                        note="Справочно: верхняя оценка сопротивления подложки под мезой."),
                Formula(r"$\lambda=\frac{v_{th}\,\mu\,m_{cc}}{q}$", badge=ESTIMATE,
                        note="Справочно: длина свободного пробега по соотношению Друде; источника со "
                             "страницей в проекте нет — только для сравнения с L."),
                Graph("→ n_{i}, равновесные концентрации, D и L входят во все формулы §3–§6; "
                      "вычисленные значения — во вкладке «Справочник»."),
                Params((
                    (r"$E_{g}(0),\ \alpha,\ \beta$", f"{GE.Eg0_eV} эВ; {GE.alpha_eV_K:g} эВ/К; {GE.beta_K:g} К "
                                                     f"[{GE.source['Eg']}]", ""),
                    (r"$N_{C0},\ N_{V0}$", f"{GE.Nc0:.3g}; {GE.Nv0:.3g} [{GE.source['Nc_Nv']}]", "см⁻³·К^{−3/2}"),
                    (r"$\varepsilon_{r}$", f"{GE.eps_r:g} [{GE.source['eps_r']}]", "—"),
                    (r"$\mu_{n},\ \mu_{p}$", f"{GE.mu_n_max:g}; {GE.mu_p_max:g} — чистый Ge, верхние пределы "
                                             f"[{GE.source['mu']}]", "см²/(В·с)"),
                    (r"$v_{th,n},\ v_{th,p}$", f"{GE.vth_n:.2g}; {GE.vth_p:.2g} [{GE.source['vth']}]", "см/с"),
                    (r"$m_{cc}$", f"{GE.m_cc:g}·m_{{0}} [{GE.source['m_cc']}], только Справочник", "—"),
                )),
                Current("material"),
            )),

    Section("§3", "Электростатика → График 2",
            "V_{bi} — разность положений уровня Ферми в n- и p-областях; в невырожденном Ge "
            "V_{bi} < E_{g}/q ≈ 0.66 В. Подробно — Методичка, гл. 4.",
            (
                Formula(r"$V_{bi}=\frac{kT}{q}\ln\frac{n_{n0}\,p_{p0}}{n_{i}^{2}}$", "[Зи, с. 82, ур. (7), (7а)]",
                        fid="(3.1)",
                        note="n_{n0}, p_{p0} по (2.4). При N ≫ n_{i}: (kT/q)·ln(N_{A}N_{D}/n_{i}²)."),
                Formula(r"$qV_{bi}=E_{g}+kT(\eta_{n}+\eta_{p})$", "[Зи, с. 82, ур. (7), первое равенство]",
                        fid="(3.1а)", result=True,
                        note="η из (2.7). Используется в расчёте всегда; при η < −3 совпадает с (3.1)."),
                _t("Вывод W: 1) уравнение Пуассона в приближении обеднения [Зи, с. 82, ур. (10а), (10б)]; "
                   "2) N_{A}x_{p} = N_{D}x_{n} [ур. (9)]; 3) поле [с. 83, ур. (11), (12)]; "
                   "4) V_{bi} = ℰ_{m}W/2 [ур. (14)]; 5) W [с. 84, ур. (15)], поправка −2kT/q [ур. (16)], "
                   "смещение [с. 86, ур. (18)]."),
                Formula(r"$W(V)=\sqrt{\frac{2\varepsilon}{q}\frac{N_{A}+N_{D}}{N_{A}N_{D}}\left(V_{bi}-V-\frac{2kT}{q}\right)}$",
                        "[Зи, с. 84, ур. (15), (16); с. 86, ур. (18)]", fid="(3.2)", result=True,
                        note="N_{A} и N_{D} — концентрации ионизованной примеси."),
                Formula(r"$x_{p}=W\frac{N_{D}}{N_{A}+N_{D}},\qquad x_{n}=W\frac{N_{A}}{N_{A}+N_{D}}$",
                        "[Зи, с. 82, ур. (9)]"),
                Formula(r"$C=\frac{\varepsilon A}{W}$", "[Зи, с. 86, ур. (18)]", fid="(3.4)", result=True),
                Formula(r"$\frac{1}{C^{2}}=\frac{2(V_{bi}-2kT/q-V)}{q\varepsilon N_{eff}A^{2}},\qquad N_{eff}=\frac{N_{A}N_{D}}{N_{A}+N_{D}}$",
                        "[Зи, с. 87, ур. (18а), (18б); с. 88, ур. (18в)]", fid="(3.5)", result=True,
                        note="Отсечка = V_{bi} − 2kT/q. Вкладка «1/C²» Графика 2: эксперимент и модель, "
                             "прямые на выбранном участке, N из наклона, отсечки."),
                Formula(r"$C\propto(V_{bi}'-V)^{-1/2}$", "резкий переход, по (3.4) и (3.2); V_{bi}′ = V_{bi} − 2kT/q"),
                Formula(r"$C\propto(V_{bi}-V)^{-1/3}$", "линейный переход [Зи, с. 89, ур. (23)]"),
                Formula(r"$C\propto(V_{bi}-V)^{-1/(m+2)}$", badge=INTERPOLATION,
                        note="Общий показатель резкости m: m = 0 — резкий, m = 1 — линейный."),
                _t("Применимость: аргумент корня ≥ kT/q; C = NaN при V > V_{bi} − 3kT/q; смыкание, если "
                   "x ≥ толщины слоя; при N_{A} < 10·n_{i} — предупреждение «приближение обеднения грубое "
                   "(почти собственный слой)»."),
                Graph("(3.2)–(3.5) → График 2 (ВФХ и 1/C²)."),
                Params((
                    (r"$\varepsilon$", "ε_{r}·ε_{0} для Ge (без усреднения)", "Ф/см"),
                    (r"$N_{A},\ N_{D}$", "ионизованная примесь p- и n-стороны перехода по сценарию", "см⁻³"),
                    (r"$V_{bi}$", "встроенный потенциал", "В"),
                    (r"$W,\ x_{n},\ x_{p}$", "ширина ОПЗ и её края", "мкм"),
                    (r"$N_{eff}$", "приведённая концентрация перехода", "см⁻³"),
                )),
                Current("electrostatics"),
            )),

    Section("§4", "Диффузионный ток → Графики 1 и 3",
            "Диффузия неосновных носителей через нейтральные области — идеальный диод Шокли (n = 1).",
            (
                Plaque(ASSUMPTION, "Допущения Шокли [Зи, с. 91, §2.4.1]: резкие границы ОПЗ; статистика "
                                   "Больцмана; низкий уровень инжекции; в ОПЗ нет генерации и рекомбинации "
                                   "(их даёт §5)."),
                _t("Вывод (электроны в p-области; x′ отсчитывается от края ОПЗ): 1) равновесие (2.4);"),
                Formula(r"$n_{p}(-x_{p})=n_{p0}e^{qV/kT}$",
                        "[Зи, с. 92, ур. (28), (31)–(33)]; [Ш49, с. 458, ур. (4.4)]", fid="(4.1)",
                        note="2) граничное условие на краю ОПЗ;"),
                Formula(r"$\frac{d^{2}\Delta n}{dx'^{2}}-\frac{\Delta n}{L_{n}^{2}}=0$",
                        "[Зи, с. 94, ур. (39)]; [Ш49, с. 470, ур. (5.4)]", fid="(4.2)",
                        note="3) уравнение диффузии; 4) решения для базы толщиной w:"),
                Formula(r"$\Delta n=\Delta n(0)e^{-x'/L}$", "длинная база [Зи, ур. (40)]"),
                Formula(r"$\Delta n=\Delta n(0)\frac{\sinh((w-x')/L)}{\sinh(w/L)}$",
                        "«сток», Δn(w) = 0 [Ш49, с. 470, ур. (5.5)]"),
                Formula(r"$\Delta n=\Delta n(0)\frac{\cosh((w-x')/L)}{\cosh(w/L)}$",
                        "«отражение», dΔn/dx′(w) = 0 [Ш49, с. 470, ур. (5.6) при p_{1} = p_{2}]"),
                Formula(r"$J=qD\left|\frac{d\Delta n}{dx'}\right|_{0}=\frac{qD\Delta n(0)}{L}f\left(\frac{w}{L}\right),\quad f=1,\ \coth,\ \tanh$",
                        "дифференцирование решений шага 4",
                        note="5) ток на краю ОПЗ; 6) сумма сторон [Зи, с. 94, ур. (44), (45)]; "
                             "[Ш49, с. 460, ур. (4.13)]."),
                Formula(r"$J_{s}(V)=\frac{qD_{p}p_{n0}}{L_{p}}f\left(\frac{w_{n}}{L_{p}}\right)+\frac{qD_{n}n_{p0}}{L_{n}}f\left(\frac{w_{p}}{L_{n}}\right)$",
                        "[Зи, с. 94, ур. (44), (45)]; [Ш49, с. 460, ур. (4.13)]", fid="(4.3)", result=True,
                        note="w_{n} = (толщина n-стороны) − x_{n}(V); w_{p} = (толщина p-стороны) − x_{p}(V); "
                             "неосновные n_{p0}, p_{n0} — по (2.4); τ — по (5.9)."),
                Formula(r"$I_{diff}=A\,J_{s}(V)\left(e^{qV/kT}-1\right)$",
                        "[Зи, с. 94, ур. (45)]; [Ш49, с. 460, ур. (4.13)]", fid="(4.4)", result=True),
                _t("Пределы: w ≫ L → f = 1; w ≪ L: «сток» даёт qDn_{p0}/w, «отражение» даёт q·n_{p0}·w/τ. "
                   "Реализация: coth = 1/tanh; при u < 10^{−6} coth = 1/u, tanh = u; при u > 20 f = 1; "
                   "w ≤ 0 → «смыкание», w_{min} = 1 нм."),
                Graph("(4.4) → кривая “I_{diff}” на Графике 1; J = I/A — на Графике 3."),
                Params((
                    (r"$D_{n},\ D_{p}$", "коэффициенты диффузии неосновных (2.5)", "см²/с"),
                    (r"$L_{n},\ L_{p}$", "диффузионные длины (2.6) с τ по (5.9)", "мкм"),
                    (r"$w_{n},\ w_{p}$", "толщины нейтральных баз", "мкм"),
                    (r"$f$", "граничное условие дальней границы: сток / отражение / длинная база", "—"),
                )),
                Current("diffusion"),
            )),

    Section("§5", "Генерация–рекомбинация в ОПЗ → Графики 1 и 3",
            "Генерация–рекомбинация через дефекты в области пространственного заряда "
            "(модель Са–Нойса–Шокли, n = 2).",
            (
                Formula(r"$U=\frac{pn-n_{i}^{2}}{\tau_{p0}(n+n_{1})+\tau_{n0}(p+p_{1})}$",
                        "[СНШ57, с. 1230, ур. (6)]; [Зи, с. 98, ур. (51)]", fid="(5.1)"),
                Formula(r"$E_{t}=E_{i},\qquad \tau_{n0}=\tau_{p0}=\tau_{0}=\frac{1}{\sigma v_{th}N_{t}}$",
                        "[СНШ57, с. 1230]; [Зи, с. 98, ур. (52)]"),
                Formula(r"$pn=n_{i}^{2}e^{qV/kT}$", "[Зи, с. 92, ур. (28), (31)]"),
                Formula(r"$U_{max}=\frac{n_{i}}{2\tau_{0}}\left(e^{qV/2kT}-1\right)$",
                        "[СНШ57, с. 1231, ур. (13)]", fid="(5.3)",
                        note="Максимум при n = p. Пределы: [СНШ57, ур. (14)], [Зи, ур. (53)] при прямом "
                             "смещении; [СНШ57, ур. (10)] при обратном."),
                Formula(r"$J_{gr}=\frac{q\,n_{i}\,W(V)}{2\tau_{0}}\left(e^{qV/2kT}-1\right)$",
                        "при обратном смещении = [СНШ57, с. 1230, ур. (11)] = [Зи, с. 97, ур. (48)] (τ_{e} = 2τ_{0}); "
                        "при прямом = [Зи, с. 99, ур. (54)], верхняя оценка [Зи, с. 99, сноска]",
                        fid="(5.4)", result=True),
                Formula(r"$F=\min\left\{1,\ \frac{\pi(kT/q)}{V_{bi}-V}\right\}$",
                        "[СНШ57, с. 1231, ур. (15), (16), множитель π/2]", fid="(5.5)",
                        note="Опция «уточнение SNS» для V ≥ 3kT/q."),
                Plaque(INTERPOLATION, "На 0 < V < 3kT/q множитель F (5.5) линейно интерполируется от 1."),
                Formula(r"$J_{gr}\propto e^{qV/n_{2}kT},\qquad n_{2}\neq 2$", "[Ман20, с. 45–46]",
                        badge=EMPIRICAL, fid="(5.6)", note="По умолчанию n_{2} = 2."),
                Formula(r"$\sigma v_{th}N_{t}=\frac{1}{\tau_{0}}$", "из [Зи, с. 98, ур. (52)]",
                        note="Выводится во вкладке «Справочник»."),
                Graph("(5.4) → кривая “I_{gr}” на Графике 1; J = I/A — на Графике 3."),
                Params((
                    (r"$n_{i}$", "собственная концентрация (2.3)", "см⁻³"),
                    (r"$W(V)$", "ширина ОПЗ (3.2)", "мкм"),
                    (r"$\tau_{0}$", "время жизни в ОПЗ (5.10)", "с"),
                    (r"$n_{2}$", "показатель тока ОПЗ (теория: 2)", "—"),
                )),
                Current("scr"),
            )),

    Section("§5А", "Дислокации и время жизни",
            "Дислокации — центры рекомбинации. Скорость рекомбинации пропорциональна избытку "
            "неосновных носителей и плотности дислокаций; коэффициент σ_{R} — рекомбинационная "
            "эффективность единицы длины дислокационной линии [KKA56, с. 1289, ур. (3)].",
            (
                Formula(r"$\lambda=\frac{\Delta P}{\tau}=\sigma_{R}N_{D}\Delta P\quad\Rightarrow\quad\sigma_{R}=\frac{1}{\tau N_{D}}$",
                        "[KKA56, с. 1289, ур. (3)]",
                        note="Здесь N_{D} — плотность дислокаций в обозначениях [KKA56]."),
                Formula(r"$\frac{1}{\tau_{n,p}}=\frac{1}{\tau_{n,p}^{bg}}+\sigma_{R}N_{dis}$",
                        "второе слагаемое — [KKA56, ур. (3)]", fid="(5.9)", result=True,
                        note="Независимые каналы рекомбинации складываются."),
                Formula(r"$\frac{1}{\tau_{0}}=\frac{1}{\tau_{0}^{bg}}+\sigma_{R}N_{dis}$", badge=ASSUMPTION,
                        fid="(5.10)", result=True,
                        note="Допущение модели: перенос на ОПЗ (в [KKA56] измерялось объёмное время жизни). "
                             "σ_{R} берётся для слоя, в котором лежит бо́льшая часть ОПЗ."),
                Params(tuple((r"$\sigma_{R}$", f"{name}: {value:g} [{SIGMA_R_SOURCE}]", "см²/с")
                             for name, value in SIGMA_R.items())),
                Plaque(ESTIMATE, "Данные σ_{R} — объёмные кристаллы при плотности дислокаций 10^{5}–10^{7} см⁻²; "
                                 "для эпитаксии и N < 10^{5} — экстраполяция."),
                Formula(r"$\tau_{dis}=\frac{1}{\sigma_{R}N_{dis}}$", "из [KKA56, с. 1289, ур. (3)]",
                        note="Справочник: τ_{dis} по слоям и доля дислокаций в 1/τ и 1/τ_{0}."),
                Plaque(IMPORTANT, "Если дислокационный вклад в 1/τ_{0} мал, наблюдаемый ток определяют "
                                  "другие центры (точечные дефекты, остатки имплантационных повреждений, "
                                  "поверхность); их описывает τ_{0}^{bg}."),
                Note("5A"),
                _t("Корреляция плотности прорастающих дислокаций и времени жизни в плёнках Ge/GeSn "
                   "наблюдалась во многих работах; непреднамеренное легирование плёнок Ge обычно "
                   "~10^{16} см⁻³, n- и p-типа [Giunto24, с. 041333-28, -34].", "small"),
                Graph("→ τ по (5.9) входит в (4.3), τ_{0} по (5.10) — в (5.4): Графики 1 и 3."),
                Current("dislocations"),
            )),

    Section("§6", "Полный ток, идеальность, сравнение сценариев → Графики 1–3",
            "Полный ток складывает компоненты §4, §5 и утечки с учётом R_{s}; коэффициент "
            "идеальности определяется из эксперимента, режим «оба» сравнивает сценарии.",
            (
                _t("6.1 Эквивалентная схема", "bold"),
                Formula(r"$I=I_{diff}(V_{d})+I_{gr}(V_{d})+\frac{V_{d}}{R_{sh}}+I_{L}\,\mathrm{sign}(V_{d})\left|\frac{V_{d}}{1\,\mathrm{V}}\right|^{m},\quad V_{d}=V-IR_{s}$",
                        "R_{s}: [Зи, с. 97, п. 5; рис. 21, с. 99], контакты — [Кур74, с. 238]; "
                        "R_{sh}: [Зи, с. 97, п. 1]; [Кур74, с. 254–255]", fid="(6.1)", result=True,
                        note="Решение — Ньютон по I с ограничением шага; развёртка от V = 0 к краям; "
                             "не более 100 итераций; при несходимости NaN и предупреждение."),
                Plaque(EMPIRICAL, "Последнее слагаемое — «мягкая» обратная ВАХ I ∝ U^{m}, 3 < m < 7 "
                                  "(микромостики вдоль дислокаций) [Кур74, с. 167–168]; по умолчанию выключено."),
                _t("6.2 Рекомендация граничного условия (Справочник)", "bold"),
                Plaque(TZ_RULE, "Соседний слой того же типа легирован в ≥ 10 раз сильнее → «отражение»; "
                                "в ≥ 10 раз слабее или металлический контакт → «сток»; иначе → «промежуточный»."),
                _t("6.3 Коэффициент идеальности из эксперимента", "bold"),
                Formula(r"$I=I_{0}\left[\exp\left(\frac{qV}{nkT}\right)-1\right]$", "[Ман20, с. 45, ур. (2)]",
                        fid="(6.3)", result=True),
                Formula(r"$n=\left[\frac{kT}{q}\frac{d\ln I}{dV}\right]^{-1}$", "[Ман20, с. 46, ур. (4)]", fid="(6.4)"),
                Formula(r"$n=\left[\frac{kT}{q}\frac{\Delta\ln I}{\Delta V}\right]^{-1}$", "[Ман20, с. 46, ур. (5)]",
                        fid="(6.5)", note="То же, что qΔV/(kT·Δln I)."),
                _t("Алгоритм (при загрузке ВАХ и при каждом «Рассчитать»): 1) прямая ветвь, V > 0, I > 0; "
                   "при R_{s} > 0 V заменяется на V − I·R_{s}; 2) локальный n(V) по (6.5) центральными "
                   "разностями — на правой оси Графика 3; 3) окно V_{1} = 3kT/q … V_{2}; 4) n_{эксп} ± δn — "
                   "подгонка (6.3) по (I_{0}, n) по ln I; 5) то же для модели → n_{мод}; 6) прямая (6.3) на "
                   "Графике 3, эмпирическая кривая (6.3) — серым пунктиром на Графике 1."),
                Plaque(TZ_RULE, "V_{2} — первая точка, где локальный n превышает минимум на [V_{1}; V] более "
                                "чем на 20 %. V_{1} и V_{2} можно править; окно содержит не менее трёх точек."),
                Plaque(IMPORTANT, "Что такое n. При n = 1 ток растёт в 10 раз на каждые 59.5 мВ [Зи, с. 95], "
                                  "при n = 2 — на каждые 119 мВ.\nПочему измеряется промежуточное значение. "
                                  "Диффузионный ток имеет n = 1, ток ОПЗ — n = 2. Прибор измеряет их сумму, "
                                  "наклон которой лежит между ними:"),
                Formula(r"$n_{eff}=\frac{I_{diff}+I_{gr}}{I_{diff}+I_{gr}/2}$",
                        "подстановка (4.4) и (5.4) в (6.4) при V ≫ kT/q", fid="(6.6)", result=True,
                        note="Откуда доля тока ОПЗ равна 2(1 − 1/n)."),
                Plaque(IMPORTANT, "Поэтому n_{эксп} не подставляется в физическую модель. Модель сама даёт "
                                  "n_{мод}; совпадение n_{мод} и n_{эксп} — критерий правильного подбора τ_{0}, τ, "
                                  "R_{s}. Кнопка «n_{2} ← n_{эксп}» — только для ручных экспериментов, с пометкой "
                                  "«эмпирика»."),
                Note("6.3"),
                _t("6.4 Граница низкой инжекции (Справочник)", "bold"),
                Formula(r"$V_{LI}=\frac{kT}{q}\ln\frac{0.1\,M_{B}}{m_{B0}}$",
                        "условие — [Зи, с. 94, перед ур. (38)]; выше V_{LI} модель неприменима [Зи, с. 97, п. 4]",
                        note="M_{B}, m_{B0} — равновесные концентрации основных и неосновных носителей (2.4) "
                             "более слабой стороны."),
                _t("При почти собственной подложке в сценарии B прямая ветвь модели неприменима почти сразу "
                   "(высокий уровень инжекции в подложке); для этого сценария сравнивайте обратную ветвь и ВФХ."),
                _t("6.5 График 3: |J| = |I|/A (модель, компоненты, эксперимент J(S)) в полулогарифмическом "
                   "масштабе; штриховая линия J_{s}(0); прямая n_{эксп}.", "bold"),
                _t("6.6 Сравнение сценариев (режим «оба», без подгонки)", "bold"),
                _t("Расчёт A и B с общими параметрами. На Графиках 1–3 и на «1/C²»: A — синие, B — красные, "
                   "эксперимент — точки."),
                Formula(r"$\delta_{I}=\sqrt{\frac{1}{N}\sum\left(\log_{10}|I_{mod}|-\log_{10}|I_{exp}|\right)^{2}}$",
                        badge=TZ_METRIC, fid="(6.7)",
                        note="По точкам с |V| > kT/q; для сценария B — только обратная ветвь."),
                Formula(r"$\delta_{C}=\sqrt{\frac{1}{N}\sum\left(\frac{C_{mod}-C_{exp}}{C_{exp}}\right)^{2}}$",
                        badge=TZ_METRIC, fid="(6.8)"),
                _t("Главный критерий: N из наклона экспериментальной 1/C² по (3.5) против N_{eff} сценариев; "
                   "второй — абсолютная ёмкость. Таблица по сценариям — во вкладке «Справочник»."),
                Note("6.6"),
                _t("Сценарий, у которого совпадает N из наклона 1/C² и меньше δ_{C}, — кандидат на тип "
                   "i-слоя. Окончательно тип подтверждают знак эффекта Холла или ECV."),
                Graph("→ Графики 1–3: сумма (6.1) — сплошная, компоненты — пунктир; эмпирика (6.3) — серым "
                      "на Графике 1; прямая (6.3) и локальный n(V) — на Графике 3."),
                Params((
                    (r"$R_{s},\ R_{sh}$", "последовательное сопротивление и шунт", "Ом"),
                    (r"$I_{L},\ m$", "нелинейная утечка и её показатель (эмпирика)", "А, —"),
                    (r"$n_{exp},\ n_{mod}$",
                     "идеальность эксперимента и модели (§6.3)", "—"),
                    (r"$V_{1},\ V_{2}$", "окно подгонки идеальности", "В"),
                )),
                Current("total"),
            )),

    Section("§7", "Связь с дефектами",
            "Время жизни в ОПЗ связывает ВАХ с плотностью дефектов.",
            (
                Formula(r"$\tau_{0}=\frac{1}{\sigma v_{th}N_{t}}$", "[Зи, с. 98]",
                        note="Больше дефектов → меньше τ_{0} → больше I_{gr} → n ближе к 2."),
                _t("Дислокации вносят вклад по (5.9)–(5.10) и дают каналы утечки (R_{sh}, плавный пробой) "
                   "[Кур74, с. 167–168]."),
                _t("Сопоставление методов контроля дефектов — бэклог Б-8."),
                Current("defects"),
            )),
]

SOURCES = [
    ("Зи", "Зи С. Физика полупроводниковых приборов. Кн. 1. — М.: Мир, 1984. — 456 с."),
    ("Ш49", "Shockley W. The Theory of p-n Junctions in Semiconductors and p-n Junction Transistors // "
            "Bell Syst. Tech. J. 1949. Vol. 28. P. 435–489."),
    ("СНШ57", "Sah C.-T., Noyce R.N., Shockley W. Carrier Generation and Recombination in P-N Junctions "
              "and P-N Junction Characteristics // Proc. IRE. 1957. Vol. 45. P. 1228–1243. "
              "DOI: 10.1109/JRPROC.1957.278528"),
    ("KKA56", "Kurtz A.D., Kulin S.A., Averbach B.L. Effect of Dislocations on the Minority Carrier "
              "Lifetime in Semiconductors // Phys. Rev. 1956. Vol. 101, № 4. P. 1285–1291. "
              "DOI: 10.1103/PhysRev.101.1285"),
    ("Giunto24", "Giunto A., Fontcuberta i Morral A. Defects in Ge and GeSn and their impact on "
                 "optoelectronic properties // Appl. Phys. Rev. 2024. Vol. 11. 041333. DOI: 10.1063/5.0218623"),
    ("Ман20", "Маняхин Ф.И., Ваттана А.Б., Мокрецова Л.О. Применение механизма рекомбинации "
              "Шокли–Нойса–Саа для модели ВАХ светодиодных структур с квантовыми ямами // "
              "Светотехника. 2020. № 4."),
    ("Кур74", "Курносов А.И., Юдин В.В. Технология производства полупроводниковых приборов. — "
              "М.: Высшая школа, 1974. — 400 с."),
    ("Ioffe-Ge-b, Ioffe-Ge-e", "Ioffe NSM Archive, Ge: band structure / electrical properties — "
                               "https://www.ioffe.ru/SVA/NSM/Semicond/Ge/bandstr.html; …/Ge/electric.html"),
    ("Ioffe-Si-b, Ioffe-Si-e", "Ioffe NSM Archive, Si (для бэклога Б-1) — "
                               "https://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html; …/Si/electric.html"),
    ("Ioffe-SiGe", "Ioffe NSM Archive, SiGe basic parameters — "
                   "https://www.ioffe.ru/SVA/NSM/Semicond/SiGe/basic.html"),
    ("CODATA", "CODATA 2018 / SI 2019: q и k точные по определению."),
]

# Где используется каждый параметр основного окна (итоговая таблица).
PARAMETER_USE = {
    "D_um": "(1.1): площадь A во всех токах и ёмкости",
    "D_inner_um": "только схема мезы",
    "d_epi_um": "z_{j} и d_{i} (§5.2 ТЗ), изоляция (1.2)",
    "h_um": "изоляция (1.2), ΔA (1.3)",
    "d_n_um": "(4.3), z_{j}",
    "ND_plus": "(2.4), (3.1а)",
    "N_i": "(2.4), (3.x), (4.3)",
    "rho_sub": "(2.8) → N_{A} подложки",
    "d_sub_um": "(4.3), R_{sub}",
    "T": "все формулы",
    "mu_n": "(2.5): электроны в p-области",
    "mu_p_i": "(2.5): дырки в i-слое (сценарий B)",
    "mu_p_nplus": "(2.5): дырки в n⁺ (сценарий A)",
    "tau_n_bg": "(5.9)",
    "tau_p_bg": "(5.9)",
    "tau0_bg": "(5.10)",
    "N_dis": "(5.9), (5.10)",
    "sigma_R_epi": "(5.9), (5.10): i-слой и n⁺",
    "sigma_R_sub": "(5.9), (5.10): подложка",
    "n2": "(5.6)",
    "Rs": "(6.1)",
    "Rsh": "(6.1)",
    "I_L": "(6.1), эмпирика",
    "m_leak": "(6.1), эмпирика",
}


def formula_ids():
    """Все ID формул во вкладке «Модель» (для проверки совпадения с docstring)."""
    ids = []
    for section in SECTIONS:
        for block in section.blocks:
            if isinstance(block, Formula) and block.fid:
                ids.append(block.fid)
            if isinstance(block, Text):
                ids += re.findall(r"^\((\d\.\d+а?)\)", block.text)
    return ids


# ------------------------------------------------ «При текущих параметрах» --

def _structure(app):
    if app is None or not getattr(app, "structures", None):
        return None
    return app.structures[list(app.structures)[-1]]


def _num(value, unit="", digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}g} {unit}".strip()


def _current_geometry(app, s):
    level, text = ph.isolation_status(s, 0.0)
    return [("A", _num(s.area, "см²")), ("S_{side}", _num(ph.side_wall_area(s), "см²")),
            ("ΔA(−1 В)", _num(float(ph.edge_area(s, -1.0)), "см²")), ("изоляция при 0 В", text)]


def _current_material(app, s):
    Nc, Nv = ph.effective_dos(s.T, s.material)
    return [("E_{g}", _num(ph.band_gap(s.T, s.material), "эВ")), ("n_{i}", _num(s.ni, "см⁻³")),
            ("N_{C} / N_{V}", f"{Nc:.3e} / {Nv:.3e} см⁻³"), ("N_{A} подложки (2.8)", _num(s.substrate[0], "см⁻³")),
            ("η основных n / p", f"{s.eta[0]:.2f} / {s.eta[1]:.2f}")]


def _current_electrostatics(app, s):
    return [("V_{bi} (3.1а)", _num(s.Vbi, "В")), ("W(0)", _num(float(ph.depletion_width(s, 0.0)) / UM, "мкм")),
            ("C(0)", _num(float(ph.capacitance(s, 0.0)) * 1e12, "пФ")), ("N_{eff}", _num(s.N_eff, "см⁻³")),
            ("отсечка 1/C²", _num(ph.c2_cutoff(s), "В"))]


def _current_diffusion(app, s):
    holes, electrons = ph.saturation_current_density_parts(s, 0.0)
    (_, Lp), (_, Ln) = ph.minority_transport(s)
    return [("J_{s}(0): дырки / электроны", f"{float(holes):.3e} / {float(electrons):.3e} А/см²"),
            ("L_{p} / L_{n}", f"{Lp / UM:.3g} / {Ln / UM:.3g} мкм"),
            ("I_{diff}(−1 В)", _num(float(ph.diffusion_current(s, -1.0)), "А"))]


def _current_scr(app, s):
    tau0 = ph.scr_lifetime(s)
    return [("τ_{0} (5.10)", _num(tau0, "с")), ("1/τ_{0} = σv_{th}N_{t}", _num(1 / tau0, "с⁻¹")),
            ("I_{gr}(−1 В)", _num(float(ph.gr_current(s, -1.0)), "А"))]


def _current_dislocations(app, s):
    return [("τ_{dis}: i-слой и n⁺", _num(ph.dislocation_lifetime(s.sigma_R_epi, s.N_dis), "с")),
            ("τ_{dis}: подложка", _num(ph.dislocation_lifetime(s.sigma_R_sub, s.N_dis), "с")),
            ("τ n-стороны / p-стороны (5.9)",
             f"{ph.minority_lifetime(s.n_side, s.N_dis):.3g} / {ph.minority_lifetime(s.p_side, s.N_dis):.3g} с"),
            ("τ_{0} (5.10)", _num(ph.scr_lifetime(s), "с"))]


def _current_total(app, s):
    rows = [("I(−1 В) (6.1)", _num(float(ph.solve_iv(s, np.array([-1.0])).I[0]), "А")),
            ("V_{LI}", _num(ph.low_injection_voltage(s), "В"))]
    for name, attr in (("n_{эксп}", "ideality"), ("n_{мод}", "model_ideality")):
        result = getattr(app, attr, None)
        rows.append((name, f"{result.n:.3f} ± {result.dn:.3f}" if result else "—"))
        if result:
            rows.append((f"доля тока ОПЗ по (6.6), {name}",
                         f"{ph.gr_share_from_ideality(result.n) * 100:.0f} %"))
    return rows


def _current_defects(app, s):
    return [("σv_{th}N_{t} = 1/τ_{0}", _num(1 / ph.scr_lifetime(s), "с⁻¹"))]


CURRENT_VALUES = {
    "geometry": _current_geometry,
    "material": _current_material,
    "electrostatics": _current_electrostatics,
    "diffusion": _current_diffusion,
    "scr": _current_scr,
    "dislocations": _current_dislocations,
    "total": _current_total,
    "defects": _current_defects,
}


def current_text(key, app):
    s = _structure(app)
    if s is None:
        return "нет расчёта — нажмите «Рассчитать» в основном окне"
    rows = CURRENT_VALUES[key](app, s)
    return f"сценарий {s.scenario}:  " + ";   ".join(f"{label} = {value}" for label, value in rows)


# --------------------------------------------------------------- вёрстка --

TEXT_FONT = ("Segoe UI", 10)
SMALL_FONT = ("Segoe UI", 8)
BOLD_FONT = ("Segoe UI", 10, "bold")
TITLE_FONT = ("Segoe UI", 13, "bold")
SOURCE_FONT = ("Segoe UI", 8)

_INDEX_MARKUP = re.compile(r"([_^])\{([^{}]*)\}")


def split_index_markup(text):
    """Разбивает обычный текст на куски [(фрагмент, None|"sub"|"sup"), ...].
    Понимает только _{...} и ^{...}; одиночные подчёркивания (solve_iv) — как есть."""
    parts, pos = [], 0
    for m in _INDEX_MARKUP.finditer(text):
        if m.start() > pos:
            parts.append((text[pos:m.start()], None))
        parts.append((m.group(2), "sub" if m.group(1) == "_" else "sup"))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], None))
    return parts


def _background_hex(widget):
    color = ttk.Style(widget).lookup("TFrame", "background") or "#f0f0f0"
    r, g, b = widget.winfo_rgb(color)
    return "#%02x%02x%02x" % (r >> 8, g >> 8, b >> 8)


def _set_rich_text(widget, text):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for chunk, tag in split_index_markup(text):
        widget.insert("end", chunk, tag or ())
    widget.configure(state="disabled")


def _rich_text(parent, text, bg, font=TEXT_FONT, foreground="#000000"):
    """Нередактируемый текст с переносом по ширине и настоящими индексами;
    высота подстраивается под число строк после переноса."""
    base = tkfont.Font(parent, font=font)
    small = tkfont.Font(parent, font=font)
    size = abs(base.actual("size"))
    small.configure(size=max(round(size * 0.75), 6))
    # width=1: ширину задаёт pack(fill="x"), а не число символов по умолчанию.
    offset = max(size // 3, 2)
    # pady = сдвиг индекса: место под верхний/нижний индекс первой и последней строки.
    widget = tk.Text(parent, wrap="word", width=1, height=1, borderwidth=0, highlightthickness=0,
                     padx=0, pady=offset, background=bg, foreground=foreground, font=base,
                     cursor="arrow", takefocus=0, spacing2=offset)
    widget.tag_configure("sub", offset=-offset, font=small)
    widget.tag_configure("sup", offset=offset, font=small)
    _set_rich_text(widget, text)
    widget._fonts = (base, small)
    def _fit_height(_event=None):
        # -update: Tk считает переносы строк лениво, без него число строк ещё неизвестно.
        lines = int(widget.tk.call(widget._w, "count", "-update", "-displaylines", "1.0", "end"))
        lines = max(lines, 1)
        if int(widget.cget("height")) != lines:
            widget.configure(height=lines)

    widget.bind("<Configure>", _fit_height)
    widget.fit_height = _fit_height
    return widget


RENDER_DPI = 150


def _render_math(parent, tex, bg, fontsize=15, color="#000000"):
    """mathtext-строка → картинка PNG в Tk-подписи, обрезанная по тексту.
    PNG вместо отдельного холста matplotlib на каждую формулу: окно из
    сотни формул открывается в разы быстрее."""
    fig = Figure(dpi=RENDER_DPI)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(bg)
    fig.text(0, 0, tex, fontsize=fontsize, va="bottom", ha="left", color=color)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=RENDER_DPI, bbox_inches="tight", pad_inches=0.03,
                facecolor=bg)
    image = tk.PhotoImage(master=parent, data=base64.b64encode(buffer.getvalue()))
    label = tk.Label(parent, image=image, background=bg, borderwidth=0, highlightthickness=0)
    label.image = image  # ссылка, иначе картинку удалит сборщик мусора
    return label


class _ScrollTab:
    """Прокручиваемая вкладка: canvas + внутренняя рамка body."""

    def __init__(self, notebook, title, bg):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=title)
        self.canvas = tk.Canvas(self.frame, highlightthickness=0, background=bg)
        scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.body = ttk.Frame(self.canvas)
        body_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(body_id, width=e.width))

    def scroll_to(self, widget):
        self.canvas.update_idletasks()
        total = max(self.body.winfo_height(), 1)
        self.canvas.yview_moveto(widget.winfo_y() / total)


class FormulasWindow:
    """Окно «Формулы и параметры»: вкладки «Модель» и «Справочник»."""

    PAD = 14

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app)
        self.win.title("Формулы и параметры")
        self.win.geometry("1100x900")
        self.win.minsize(760, 500)
        self.bg = _background_hex(self.win)
        self.notebook = ttk.Notebook(self.win)
        self.notebook.pack(fill="both", expand=True)
        self.model_tab = _ScrollTab(self.notebook, "Модель", self.bg)
        self.reference_tab = _ScrollTab(self.notebook, "Справочник", self.bg)
        self.current_widgets = []
        self.note_widgets = []
        self._build_model_tab()
        self._build_reference_tab()

        self.win.bind_all("<MouseWheel>", self._on_wheel)
        self.win.bind_all("<Button-4>", lambda _e: self._active().canvas.yview_scroll(-3, "units"))
        self.win.bind_all("<Button-5>", lambda _e: self._active().canvas.yview_scroll(3, "units"))
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Button(self.win, text="Закрыть", command=self.close).pack(side="bottom", pady=6)

    def _active(self):
        index = self.notebook.index(self.notebook.select())
        return self.model_tab if index == 0 else self.reference_tab

    def _on_wheel(self, event):
        self._active().canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def close(self):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.win.unbind_all(sequence)
        self.win.destroy()
        if self.app is not None and getattr(self.app, "formulas_window", None) is self:
            self.app.formulas_window = None

    # ------------------------------------------------------ вкладка «Модель»
    def _build_model_tab(self):
        body = self.model_tab.body
        pad = self.PAD
        toc = ttk.Frame(body)
        toc.pack(fill="x", padx=pad, pady=(pad, 4))
        ttk.Label(toc, text="Оглавление:", font=BOLD_FONT).pack(side="left")
        frames = {}
        for section in SECTIONS:
            frame = ttk.Frame(body)
            frame.pack(fill="x", padx=pad, pady=(pad, 4))
            frames[section.sid] = frame
            self._build_section(frame, section)
            link = ttk.Label(toc, text=section.sid, foreground="#1a4fa0", cursor="hand2", font=TEXT_FONT)
            link.pack(side="left", padx=6)
            link.bind("<Button-1>", lambda _e, f=frame: self.model_tab.scroll_to(f))
        self._build_parameters(body)
        self._build_sources(body)

    def _build_section(self, frame, section):
        _rich_text(frame, f"{section.sid}. {section.title}", self.bg, font=TITLE_FONT).pack(fill="x")
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 6))
        _rich_text(frame, "Зачем: " + section.why, self.bg, foreground="#1d3557").pack(fill="x", pady=(0, 6))
        for block in section.blocks:
            self._build_block(frame, block)

    def _build_block(self, frame, block):
        bg = self.bg
        if isinstance(block, Text):
            font = {"small": SMALL_FONT, "muted": SMALL_FONT, "bold": BOLD_FONT}.get(block.style, TEXT_FONT)
            color = "#555555" if block.style in ("small", "muted") else "#000000"
            _rich_text(frame, block.text, bg, font=font, foreground=color).pack(fill="x", pady=(2, 4))
        elif isinstance(block, Formula):
            holder = ttk.Frame(frame)
            holder.pack(fill="x", pady=(2, 2))
            row = tk.Frame(holder, background=bg, highlightthickness=1 if block.result else 0,
                           highlightbackground="#1a4fa0")
            row.pack(anchor="w", padx=(20, 0))
            if block.fid:
                tk.Label(row, text=block.fid, background=bg, font=BOLD_FONT,
                         foreground="#1a4fa0").pack(side="left", padx=(6, 8))
            _render_math(row, block.tex, bg, fontsize=15).pack(side="left", padx=(0, 6))
            if block.badge:
                tk.Label(row, text=block.badge, background=BADGE_COLORS[block.badge],
                         font=SMALL_FONT, padx=4).pack(side="left", padx=4)
            if block.source:
                _rich_text(holder, "Источник: " + block.source, bg, font=SOURCE_FONT,
                           foreground="#777777").pack(fill="x", padx=(24, 0))
            if block.note:
                _rich_text(holder, block.note, bg).pack(fill="x", padx=(24, 0), pady=(1, 0))
        elif isinstance(block, Plaque):
            color = BADGE_COLORS[block.kind]
            box = tk.Frame(frame, background=color, padx=8, pady=4)
            box.pack(fill="x", pady=4, padx=(20, 0))
            tk.Label(box, text=block.kind, background=color, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            _rich_text(box, block.text, color).pack(fill="x")
        elif isinstance(block, Graph):
            _rich_text(frame, block.text, bg, font=BOLD_FONT, foreground="#1a6b2f").pack(fill="x", pady=(4, 4))
        elif isinstance(block, Params):
            box = ttk.Frame(frame)
            box.pack(fill="x", padx=(20, 0), pady=(2, 4))
            for symbol, description, unit in block.rows:
                row = ttk.Frame(box)
                row.pack(fill="x", anchor="w")
                _render_math(row, symbol, bg, fontsize=11, color="#1a4fa0").pack(side="left", anchor="n")
                text = "— " + description + (f", {unit}" if unit else "")
                _rich_text(row, text, bg).pack(side="left", fill="x", expand=True, padx=(6, 0), pady=(3, 0))
        elif isinstance(block, Current):
            box = tk.Frame(frame, background="#eef4fb", padx=8, pady=4)
            box.pack(fill="x", pady=4, padx=(20, 0))
            tk.Label(box, text="При текущих параметрах", background="#eef4fb",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            widget = _rich_text(box, current_text(block.key, self.app), "#eef4fb")
            widget.pack(fill="x")
            self.current_widgets.append((block.key, widget))
        elif isinstance(block, Note):
            widget = _rich_text(frame, self._note_text(block.key), bg, font=TEXT_FONT, foreground="#7a4b00")
            widget.pack(fill="x", padx=(20, 0), pady=(0, 4))
            self.note_widgets.append((block.key, widget))

    def _note_text(self, key):
        preset = getattr(self.app, "preset", None)
        note = preset.notes.get(key) if preset else None
        return f"Набор «{preset.name}»: {note}" if note else ""

    def _build_parameters(self, body):
        frame = ttk.Frame(body)
        frame.pack(fill="x", padx=self.PAD, pady=(self.PAD, 4))
        _rich_text(frame, "Параметры моделирования (основное окно)", self.bg, font=TITLE_FONT).pack(fill="x")
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 6))
        for spec in presets.NUMERIC_PARAMS:
            text = f"{spec.label}, {spec.unit}. Используется в: {PARAMETER_USE[spec.key]}."
            _rich_text(frame, "• " + text, self.bg).pack(fill="x", pady=1)

    def _build_sources(self, body):
        frame = ttk.Frame(body)
        frame.pack(fill="x", padx=self.PAD, pady=(self.PAD, self.PAD))
        _rich_text(frame, "Источники", self.bg, font=TITLE_FONT).pack(fill="x")
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 6))
        for code, reference in SOURCES:
            _rich_text(frame, f"[{code}] {reference}", self.bg).pack(fill="x", pady=1)

    # --------------------------------------------------- вкладка «Справочник»
    def _build_reference_tab(self):
        for child in self.reference_tab.body.winfo_children():
            child.destroy()
        body = self.reference_tab.body
        pad = self.PAD
        data = getattr(self.app, "reference_data", None)
        s = _structure(self.app)
        if data is None or s is None:
            _rich_text(body, "Нет расчёта — нажмите «Рассчитать» в основном окне.", self.bg).pack(
                fill="x", padx=pad, pady=pad)
            return
        head = f"Сценарий {s.scenario}; набор «{self.app.preset.name}». Обновляется при «Рассчитать»."
        _rich_text(body, head, self.bg, foreground="#555555").pack(fill="x", padx=pad, pady=(pad, 4))
        if data.warnings:
            box = tk.Frame(body, background=BADGE_COLORS[IMPORTANT], padx=8, pady=4)
            box.pack(fill="x", padx=pad, pady=4)
            for warning in data.warnings:
                _rich_text(box, "⚠ " + warning, BADGE_COLORS[IMPORTANT]).pack(fill="x")
        for group in data.groups:
            self._table(body, group.title, [(r.label, _format_value(r.value), r.unit, r.note) for r in group.rows])
        comparison = getattr(self.app, "comparison", None)
        if comparison:
            self._comparison_table(body, comparison)

    def _table(self, body, title, rows):
        frame = ttk.Frame(body)
        frame.pack(fill="x", padx=self.PAD, pady=(8, 2))
        _rich_text(frame, title, self.bg, font=BOLD_FONT).pack(fill="x")
        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=3)
        grid.columnconfigure(1, weight=2)
        grid.columnconfigure(2, weight=3)
        for index, (label, value, unit, note) in enumerate(rows):
            cells = (label, f"{value} {unit}".strip(), note)
            for column, text in enumerate(cells):
                widget = _rich_text(grid, text, self.bg, font=SMALL_FONT if column == 2 else TEXT_FONT,
                                    foreground="#666666" if column == 2 else "#000000")
                widget.grid(row=index, column=column, sticky="new", padx=(0, 8))

    def _comparison_table(self, body, comparison):
        rows = []
        labels = [("Vbi", "V_{bi}", "В"), ("W0", "W(0)", "мкм"), ("C0", "C(0)", "пФ"), ("Cm1", "C(−1 В)", "пФ"),
                  ("N_eff", "N_{eff}", "см⁻³"), ("cutoff", "отсечка 1/C²", "В"), ("I_m1", "I(−1 В)", "А"),
                  ("n_mod", "n_{мод}", ""), ("V_LI", "V_{LI}", "В"), ("delta_I", "δ_{I} (6.7)", ""),
                  ("delta_C", "δ_{C} (6.8)", ""), ("isolation", "изоляция (1.2)", "")]
        scale = {"W0": 1 / UM, "C0": 1e12, "Cm1": 1e12}
        for key, label, unit in labels:
            values = []
            for sc in ph.SCENARIOS:
                value = comparison[sc][key]
                values.append(_format_value(value * scale.get(key, 1) if isinstance(value, float) else value))
            rows.append((label, f"A: {values[0]}   |   B: {values[1]}", unit, ""))
        rows.append(("N из наклона эксп. 1/C² (3.5)", _format_value(comparison["N_exp"]), "см⁻³",
                     "главный критерий — сравнить с N_{eff} сценариев"))
        self._table(body, "Сравнение сценариев (режим «оба», §6.6)", rows)
        note = self._note_text("6.6")
        if note:
            _rich_text(body, note, self.bg, foreground="#7a4b00").pack(fill="x", padx=self.PAD, pady=4)

    # --------------------------------------------------------------- обновление
    def refresh(self):
        for key, widget in self.current_widgets:
            _set_rich_text(widget, current_text(key, self.app))
            widget.fit_height()
        for key, widget in self.note_widgets:
            _set_rich_text(widget, self._note_text(key))
            widget.fit_height()
        self._build_reference_tab()


def _format_value(value):
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—" if np.isnan(value) else "∞"
        return f"{value:.4g}"
    return str(value)


def open_formulas_window(app):
    """Открывает (или поднимает уже открытое) окно «Формулы и параметры»."""
    existing = getattr(app, "formulas_window", None)
    if existing is not None:
        existing.win.lift()
        return existing
    window = FormulasWindow(app)
    if app is not None:
        app.formulas_window = window
    return window
