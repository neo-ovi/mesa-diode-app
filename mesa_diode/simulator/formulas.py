# -*- coding: utf-8 -*-
"""Окно «Формулы и параметры» — содержание и вёрстка справочного материала.

ЧТОБЫ ИЗМЕНИТЬ СОДЕРЖАНИЕ: правьте только структуры FORMULA_SECTIONS и
PARAMETERS_TABLE ниже. Формулы задаются строками mathtext — это встроенный
в matplotlib LaTeX-подобный движок разметки формул, работающий без
установленного LaTeX; поддерживает дроби (\\frac), степени/индексы (^, _),
корни (\\sqrt), автоматически масштabируемые скобки (\\left...\\right) и
греческие буквы. Каждая такая строка рендерится как отдельная картинка
ровно по размеру текста — отсюда настоящие дроби с чертой и степени, а не
текстовые заменители вида "V/Vt" или "N^2".

Каждый раздел — словарь с ключами:
  title    — заголовок раздела (с номером, см. существующие для стиля);
  formulas — список mathtext-строк (r"$...$"), выводятся друг под другом;
  legend   — список пар (mathtext-строка символа, расшифровка на русском) —
             выводится под формулами как список "обозначение — расшифровка";
  principle — короткий текст: как ведёт себя формула, в т.ч. в граничных
              случаях. Не обязателен.

Нумерация разделов (§N в текстах ссылок ниже) соответствует порядку
элементов списка — если меняете порядок, поправьте и перекрёстные ссылки
в соседних разделах и в PARAMETERS_TABLE.

Окно строится функцией open_formulas_window(parent) — вызывается из
MesaApp.open_formulas_window (см. simulator/app.py). Сама эта функция
ничего не решает по содержанию — только вёрстка.
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

INTRO_TEXT = (
    "Ниже — все формулы, по которым физическое ядро симулятора "
    "(simulator/physics.py) считает ВАХ, ВФХ и справочную кривую J–V, "
    "и расшифровка всех переменных. Номер раздела (§N) совпадает с "
    "порядком блоков на этой странице — на него ссылаются соседние "
    "разделы и итоговая таблица параметров в самом низу."
)

FORMULA_SECTIONS = [
    {
        "title": "§1. Геометрия мезы",
        "formulas": [
            r"$A = \pi\left(\frac{D}{2}\right)^{2}$",
        ],
        "legend": [
            (r"$D$", "диаметр мезы, мкм (в формулу подставляется в см: "
                      "$D_{\\mathrm{см}} = D_{\\mathrm{мкм}}\\times 10^{-4}$)"),
            (r"$d$", "диаметр окна кольцевого контакта на поверхности мезы, "
                      "мкм — в формулу площади не входит, только на схеме"),
            (r"$h$", "глубина (высота) мезы, мкм — тоже не входит в A"),
            (r"$A$", "площадь мезы, см² — торцевая площадь, через которую "
                      "течёт ток; входит во все формулы тока §3–§5 и в "
                      "барьерную ёмкость §7"),
        ],
        "principle": (
            "Площадь круга растёт квадратично с диаметром — при увеличении "
            "D вдвое A увеличивается вчетверо, и во столько же раз "
            "масштabируются все токи и ёмкость модели (они везде входят "
            "как A·J, A·C_уд.). Параметры d и h на электрический расчёт "
            "не влияют — они нужны только для схемы мезы."
        ),
    },
    {
        "title": "§2. Температурная зависимость собственной концентрации носителей",
        "formulas": [
            r"$n_i(T) = n_i(300\,\mathrm{K})\left(\frac{T}{300}\right)^{3/2}"
            r"\exp\left[-\frac{E_g}{2k}\left(\frac{1}{T}-\frac{1}{300}\right)\right]$",
        ],
        "legend": [
            (r"$n_i(300\,\mathrm{K})$", "собственная концентрация носителей "
                                          "при 300 К, см⁻³ — табличная "
                                          "константа отдельно для Si и Ge"),
            (r"$E_g$", "ширина запрещённой зоны при 300 К, эВ — табличная "
                        "константа отдельно для Si и Ge"),
            (r"$k$", "постоянная Больцмана в эВ/К (физическая константа)"),
            (r"$T$", "температура, К — вводится пользователем"),
        ],
        "principle": (
            "Формула применяется отдельно к Si и Ge (свои n_i(300 К) и E_g), "
            "давая $n_{i,\\mathrm{Si}}(T)$ и $n_{i,\\mathrm{Ge}}(T)$, которые "
            "используются во всех формулах §3–§6. При T → 300 К выражение "
            "точно даёт табличное значение; с ростом T концентрация растёт "
            "быстро (степенной множитель и экспонента действуют в одну "
            "сторону) — типичный полупроводник даёт рост n_i на порядок "
            "величины при нагреве на 10–15 °C."
        ),
    },
    {
        "title": "§3. Диффузионный ток (уравнение Шокли для гетероперехода)",
        "formulas": [
            r"$p_{n0} = \frac{n_{i,\mathrm{Ge}}^{2}}{N_A}\,,\qquad"
            r"n_{p0} = \frac{n_{i,\mathrm{Si}}^{2}}{N_D}$",
            r"$D_{n,\mathrm{Si}} = \frac{kT}{q}\,\mu_{n,\mathrm{Si}}\,,\qquad"
            r"D_{p,\mathrm{Ge}} = \frac{kT}{q}\,\mu_{p,\mathrm{Ge}}$",
            r"$L_{n,\mathrm{Si}} = \sqrt{D_{n,\mathrm{Si}}\,\tau}\,,\qquad"
            r"L_{p,\mathrm{Ge}} = \sqrt{D_{p,\mathrm{Ge}}\,\tau}$",
            r"$J_0 = q\left(\frac{D_{p,\mathrm{Ge}}\,p_{n0}}{L_{p,\mathrm{Ge}}}"
            r"+ \frac{D_{n,\mathrm{Si}}\,n_{p0}}{L_{n,\mathrm{Si}}}\right)$",
            r"$I_{\mathrm{diff}}(V) = A\,J_0\left[\exp\left(\frac{V}{V_T}\right)-1\right]$",
        ],
        "legend": [
            (r"$q$", "элементарный заряд электрона, Кл (физическая константа)"),
            (r"$V_T = kT/q$", "тепловой потенциал, В"),
            (r"$N_D$", "концентрация доноров в Si (подложка), см⁻³ — "
                        "вводится пользователем"),
            (r"$N_A$", "концентрация акцепторов в Ge (меза), см⁻³ — "
                        "вводится пользователем"),
            (r"$p_{n0}$", "равновесная концентрация неосновных дырок в Ge, см⁻³"),
            (r"$n_{p0}$", "равновесная концентрация неосновных электронов в "
                           "Si, см⁻³"),
            (r"$\mu_{n,\mathrm{Si}}, \mu_{p,\mathrm{Ge}}$",
             "подвижности электронов (Si) и дырок (Ge), см²/(В·с) — модель "
             "подвижности §8"),
            (r"$D_{n,\mathrm{Si}}, D_{p,\mathrm{Ge}}$",
             "коэффициенты диффузии (соотношение Эйнштейна), см²/с"),
            (r"$\tau$", "время жизни неосновных носителей, с (константа)"),
            (r"$L_{n,\mathrm{Si}}, L_{p,\mathrm{Ge}}$", "диффузионные длины, см"),
            (r"$J_0$", "плотность тока насыщения диффузионной компоненты, "
                        "А/см²"),
            (r"$I_{\mathrm{diff}}(V)$", "диффузионная компонента полного "
                                          "тока мезы, А"),
        ],
        "principle": (
            "Классическое диодное уравнение Шокли, применённое к "
            "гетеропереходу Ge–Si: прямой ток растёт экспоненциально с V, "
            "в обратном смещении (V ≪ −V_T) ток насыщается на уровне "
            "−A·J₀. Поскольку n_{i,Ge} на несколько порядков больше "
            "n_{i,Si}, вклад инжекции дырок в Ge обычно доминирует над "
            "инжекцией электронов в Si."
        ),
    },
    {
        "title": "§4. Рекомбинационный ток в ОПЗ (модель Sah-Noyce-Shockley)",
        "formulas": [
            r"$n_{i,\mathrm{eff}} = \sqrt{n_{i,\mathrm{Si}}\,n_{i,\mathrm{Ge}}}$",
            r"$J_{\mathrm{rec,0}} = \frac{q\,n_{i,\mathrm{eff}}\,W(0)}{2\,\tau_0}$",
            r"$I_{\mathrm{rec}}(V) = A\,J_{\mathrm{rec,0}}"
            r"\left[\exp\left(\frac{V}{n_2 V_T}\right)-1\right]$",
        ],
        "legend": [
            (r"$n_{i,\mathrm{eff}}$", "эффективная собственная концентрация "
                                       "гетероперехода — упрощение для "
                                       "оценки рекомбинации, см⁻³"),
            (r"$W(0)$", "ширина ОПЗ при V=0, см — §6"),
            (r"$\tau_0$", "время жизни носителей в ОПЗ, с (константа)"),
            (r"$n_2$", "коэффициент неидеальности рекомбинационного тока, "
                        "безразмерный — вводится пользователем, обычно 1…2"),
            (r"$J_{\mathrm{rec,0}}$", "плотность тока насыщения "
                                        "рекомбинационной компоненты, А/см²"),
            (r"$I_{\mathrm{rec}}(V)$", "рекомбинационная компонента полного "
                                         "тока мезы, А"),
        ],
        "principle": (
            "Описывает генерацию-рекомбинацию через ловушки в ОПЗ; вклад "
            "существен при малых прямых смещениях. Характерный наклон "
            "ln I(V) для этой компоненты равен 1/(n₂V_T) — при n₂=2 вдвое "
            "положе, чем у чисто диффузионного тока §3 (наклон 1/V_T); по "
            "этому различию на практике разделяют вклад двух механизмов."
        ),
    },
    {
        "title": "§5. Полный ток мезы с учётом $R_s$ и $R_{sh}$",
        "formulas": [
            r"$I = I_{\mathrm{diff}}(V-I R_s) + I_{\mathrm{rec}}(V-I R_s)"
            r"+ \frac{V-I R_s}{R_{sh}}$",
        ],
        "legend": [
            (r"$R_s$", "последовательное сопротивление, Ом — вводится "
                        "пользователем"),
            (r"$R_{sh}$", "шунтирующее (параллельное) сопротивление, Ом — "
                           "вводится пользователем"),
            (r"$I$", "полный ток мезы при заданном V, А — неизвестная, "
                      "уравнение относительно неё неявное"),
        ],
        "principle": (
            "I входит в правую часть через падение напряжения на R_s, "
            "поэтому уравнение решается численно (метод Ньютона, "
            "physics.solve_iv, ≤60 итераций на точку). При R_s → 0 и "
            "R_{sh} → ∞ уравнение вырождается в простую сумму §3+§4. "
            "R_s сглаживает (ограничивает) рост тока при больших прямых "
            "смещениях, R_{sh} определяет наклон ВАХ и ток утечки вблизи "
            "V=0 и при малых обратных смещениях."
        ),
    },
    {
        "title": "§6. Встроенный потенциал и ширина ОПЗ",
        "formulas": [
            r"$\varepsilon = \varepsilon_0\,\frac{\varepsilon_{r,\mathrm{Si}}"
            r"+\varepsilon_{r,\mathrm{Ge}}}{2}$",
            r"$V_{bi} = V_T \ln\left(\frac{N_D N_A}{n_{i,\mathrm{eff}}^{2}}\right)$",
            r"$\frac{1}{N_{\mathrm{eff}}} = \frac{1}{N_D}+\frac{1}{N_A}$",
            r"$W(V) = \sqrt{\frac{2\varepsilon\,(V_{bi}-V)}{q\,N_{\mathrm{eff}}}}$",
        ],
        "legend": [
            (r"$\varepsilon_0$", "электрическая постоянная, Ф/см "
                                   "(физическая константа)"),
            (r"$\varepsilon_{r,\mathrm{Si}}, \varepsilon_{r,\mathrm{Ge}}$",
             "относительные диэлектрические проницаемости Si и Ge "
             "(табличные константы)"),
            (r"$\varepsilon$", "диэлектрическая проницаемость гетероперехода "
                                 "в модели — среднее Si и Ge, Ф/см"),
            (r"$V_{bi}$", "встроенный потенциал перехода, В"),
            (r"$N_{\mathrm{eff}}$", "эффективная (приведённая) концентрация "
                                      "легирования перехода, см⁻³"),
            (r"$W(V)$", "ширина ОПЗ при смещении V, см"),
        ],
        "principle": (
            "При росте обратного смещения (V < 0, |V| растёт) подкоренное "
            "выражение (V_{bi}-V) увеличивается — ширина ОПЗ W растёт, как "
            "и положено p-n переходу. При V → V_{bi}⁻ (глубокое прямое "
            "смещение) W → 0 — приближение резкого перехода вблизи этой "
            "точки перестаёт быть физичным (см. ограничение в §7)."
        ),
    },
    {
        "title": "§7. Барьерная ёмкость (ВФХ) и оценка резкости перехода",
        "formulas": [
            r"$C(V) = \frac{\varepsilon A}{W(V)}$",
            r"$C \propto (V_{bi}-V)^{-\frac{1}{m+2}}$",
            r"$m = -\frac{1}{k}-2\,,\qquad k=\frac{d\ln C}{d\ln (V_{bi}-V)}$",
        ],
        "legend": [
            (r"$C(V)$", "барьерная (переходная) ёмкость мезы, Ф"),
            (r"$m$", "коэффициент градиента перехода, безразмерный — "
                      "m=0 резкий (ступенчатый) переход, m=1 "
                      "линейно-плавный (градиентный)"),
            (r"$k$", "наклон линейной аппроксимации ln C от ln(V_{bi}-V) по "
                      "точкам — оценивается функцией estimate_grading_m"),
        ],
        "principle": (
            "Общий степенной закон C-V связывает наклон lg C(lg(V_{bi}-V)) "
            "с типом перехода: k=-1/2 для резкого (m=0), k=-1/3 для "
            "линейно-градиентного (m=1). Приложение оценивает m "
            "автоматически по точкам с V < −0.05 В (подальше от V_{bi}, "
            "где приближение резкого перехода менее устойчиво) — оценка "
            "выводится под графиком ВФХ, если точек достаточно."
        ),
    },
    {
        "title": "§8. Подвижность носителей (эмпирическая модель Кофи-Томаса)",
        "formulas": [
            r"$\mu(N) = \mu_{\min} + \frac{\mu_{\max}-\mu_{\min}}"
            r"{1+\left(N/N_{\mathrm{ref}}\right)^{\alpha}}$",
        ],
        "legend": [
            (r"$\mu_{\min}, \mu_{\max}$", "подвижность при предельно "
                                            "высоком/низком легировании, "
                                            "см²/(В·с) — табличные "
                                            "константы отдельно для "
                                            "электронов в Si и дырок в Ge"),
            (r"$N_{\mathrm{ref}}$", "характерная концентрация, при которой "
                                      "подвижность падает вдвое от "
                                      "максимума, см⁻³"),
            (r"$\alpha$", "показатель степени эмпирической модели, "
                           "безразмерный"),
            (r"$N$", "концентрация легирующей примеси ($N_D$ для Si, $N_A$ "
                      "для Ge), см⁻³"),
        ],
        "principle": (
            "При N ≪ N_{ref} рассеяние на примесях мало — μ → μ_{max}; "
            "при N ≫ N_{ref} подвижность падает до μ_{min} из-за "
            "усиленного рассеяния носителей на ионах примеси. Результат "
            "формулы используется в §3 (D_n, D_p через соотношение "
            "Эйнштейна)."
        ),
    },
    {
        "title": "§9. Справочная кривая J–V (идеальный диод Шокли для материала)",
        "formulas": [
            r"$J(V) = J_0\left[\exp\left(\frac{qV}{kT}\right)-1\right]$",
            r"$J_{(S)} = \frac{I_{\mathrm{exp}}}{S}$",
        ],
        "legend": [
            (r"$J_0$", "характерное (табличное) значение плотности тока "
                        "насыщения для Si или Ge — переключатель Si/Ge "
                        "подставляет его в редактируемое поле; не "
                        "совпадает с $J_0$ модели мезы §3 (там оно "
                        "вычисляется из параметров структуры, а не берётся "
                        "из справочника)"),
            (r"$J(V)$", "плотность тока идеального диода Шокли для "
                         "выбранного материала, А/см² — кривая на третьем "
                         "графике"),
            (r"$I_{\mathrm{exp}}$", "экспериментальный ток из загруженного "
                                      "файла ВАХ, А"),
            (r"$S$", "площадь мезы (совпадает с $A$ §1), см²"),
            (r"$J_{(S)}$", "плотность тока, посчитанная прямо из "
                            "эксперимента и геометрии мезы — точки "
                            "тоггла «J(S)»"),
        ],
        "principle": (
            "Это отдельная справочная модель, не связанная с двухдиодной "
            "геометрической моделью мезы §3–§6: J₀ здесь — табличное "
            "значение для объёмного p-n перехода нужного материала, а не "
            "результат расчёта по параметрам структуры. Она нужна, чтобы "
            "визуально сравнить смоделированную/измеренную мезу с "
            "«типичным» переходом Si или Ge. J_{(S)} — независимый способ "
            "получить плотность тока прямо из измерения, без какой-либо "
            "модели, кроме геометрической площади."
        ),
    },
]


PARAMETERS_TITLE = "Параметры моделирования"

PARAMETERS_TABLE = [
    (r"$D$", "диаметр мезы", "мкм",
     "геометрия мезы §1 — определяет площадь A, от которой пропорционально "
     "зависят все токовые компоненты §3–§5 и барьерная ёмкость §7"),
    (r"$d$", "диаметр окна кольцевого контакта на поверхности мезы", "мкм",
     "только схема мезы (simulator/diagram.py) — в расчёт ВАХ/ВФХ не входит"),
    (r"$h$", "глубина (высота) мезы", "мкм",
     "только схема мезы — в текущей модели ВАХ/ВФХ не учитывается "
     "(приближение резкого перехода без боковой поверхности)"),
    (r"$N_D$", "концентрация доноров в Si (подложка)", "см⁻³",
     "диффузионный ток §3 ($n_{p0}$, $D_n$, $L_n$), встроенный потенциал §6, "
     "барьерная ёмкость §7"),
    (r"$N_A$", "концентрация акцепторов в Ge (меза)", "см⁻³",
     "диффузионный ток §3 ($p_{n0}$, $D_p$, $L_p$), встроенный потенциал §6, "
     "барьерная ёмкость §7"),
    (r"$T$", "температура", "К",
     "тепловой потенциал $V_T=kT/q$ и температурная зависимость $n_i(T)$ §2 "
     "— входит во все формулы §3–§7"),
    (r"$R_s$", "последовательное сопротивление мезы", "Ом",
     "полная модель тока мезы §5 — ограничивает рост тока при больших "
     "прямых смещениях"),
    (r"$R_{sh}$", "шунтирующее (параллельное) сопротивление", "Ом",
     "полная модель тока мезы §5 — определяет ток утечки и наклон ВАХ "
     "вблизи V=0"),
    (r"$n_2$", "коэффициент неидеальности рекомбинационного тока",
     "безразмерный", "рекомбинационный ток в ОПЗ §4"),
    (r"$\mathrm{Si\,/\,Ge}$", "переключатель материала для справочной "
     "кривой J–V", "—",
     "выбирает табличное значение $J_0$ §9 (не влияет на модель мезы §2–§6)"),
    (r"$J_0$", "характерная плотность тока насыщения выбранного материала "
     "(редактируемое поле)", "А/см²",
     "справочная кривая идеального диода Шокли §9 на третьем графике"),
]


# ------------------------------------------------------------------ рендер --

def _render_math(parent, tex, fontsize=15, pad_px=4, color="#000000"):
    """Рендерит одну mathtext-строку в компактную Tk-канву, обрезанную по
    фактическому размеру текста (без рамки и лишних полей вокруг)."""
    fig = Figure(dpi=150)
    fig.patch.set_facecolor("#ffffff")
    txt = fig.text(0, 0, tex, fontsize=fontsize, va="bottom", ha="left", color=color)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    bbox = txt.get_window_extent(renderer=canvas.get_renderer())
    w_px = max(bbox.width + 2 * pad_px, 1.0)
    h_px = max(bbox.height + 2 * pad_px, 1.0)
    dpi = fig.dpi
    fig.set_size_inches(w_px / dpi, h_px / dpi)
    txt.set_position((pad_px / w_px, pad_px / h_px))
    canvas.draw()

    widget = canvas.get_tk_widget()
    widget.configure(width=int(w_px), height=int(h_px), highlightthickness=0)
    return widget


def _build_legend_row(parent, symbol_tex, description, symbol_fontsize=13):
    row = ttk.Frame(parent)
    row.pack(fill="x", anchor="w", pady=1)
    sym_widget = _render_math(row, symbol_tex, fontsize=symbol_fontsize,
                               color="#1a4fa0")
    sym_widget.pack(side="left", anchor="n")
    ttk.Label(row, text="— " + description, wraplength=760,
              justify="left").pack(side="left", padx=(6, 0), fill="x", anchor="n")
    return row


def _build_section(parent, section, pad):
    frame = ttk.Frame(parent)
    frame.pack(fill="x", padx=pad, pady=(pad, 4))

    ttk.Label(frame, text=section["title"],
              font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 8))

    for tex in section["formulas"]:
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(0, 6))
        _render_math(row, tex, fontsize=16).pack(side="left", padx=(20, 0))

    for symbol_tex, description in section.get("legend", []):
        _build_legend_row(frame, symbol_tex, description)

    principle = section.get("principle")
    if principle:
        ttk.Label(frame, text=principle, wraplength=880, justify="left",
                  foreground="#333333",
                  font=("Segoe UI", 9, "italic")).pack(fill="x", pady=(6, 4),
                                                        anchor="w")


def _build_parameters_table(parent, pad):
    frame = ttk.Frame(parent)
    frame.pack(fill="x", padx=pad, pady=(pad, pad))

    ttk.Label(frame, text=PARAMETERS_TITLE,
              font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(2, 8))

    for symbol_tex, description, units, used_in in PARAMETERS_TABLE:
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=4, anchor="w")
        _render_math(row, symbol_tex, fontsize=14,
                     color="#1a4fa0").pack(side="left", anchor="n")
        text = "— {}, {}. Используется в: {}.".format(description, units, used_in)
        ttk.Label(row, text=text, wraplength=800, justify="left").pack(
            side="left", padx=(8, 0), fill="x", anchor="n")


def open_formulas_window(parent):
    """Открывает окно «Формулы и параметры модели» (Toplevel над ``parent``).

    Прокручиваемый список: вступление, разделы FORMULA_SECTIONS по порядку,
    затем таблица PARAMETERS_TABLE. Содержание и порядок правьте только в
    структурах выше — эта функция лишь строит из них виджеты.
    """
    win = tk.Toplevel(parent)
    win.title("Формулы и параметры модели")
    win.geometry("1000x850")
    win.minsize(700, 500)

    outer = ttk.Frame(win)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0, background="#ffffff")
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    vscroll.pack(side="right", fill="y")

    body = ttk.Frame(canvas)
    body_id = canvas.create_window((0, 0), window=body, anchor="nw")

    def _on_body_configure(_event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        canvas.itemconfig(body_id, width=event.width)

    body.bind("<Configure>", _on_body_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Колесо мыши работает только пока это окно открыто — обработчик
    # снимается в _on_close, чтобы не перехватывать прокрutку других окон
    # приложения после закрытия.
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_scroll_up(_event):
        canvas.yview_scroll(-3, "units")

    def _on_scroll_down(_event):
        canvas.yview_scroll(3, "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows / macOS
    canvas.bind_all("<Button-4>", _on_scroll_up)          # Linux
    canvas.bind_all("<Button-5>", _on_scroll_down)        # Linux

    PAD = 14
    ttk.Label(body, text=INTRO_TEXT, wraplength=920, justify="left",
              font=("Segoe UI", 10)).pack(fill="x", padx=PAD, pady=(PAD, 6),
                                          anchor="w")

    for section in FORMULA_SECTIONS:
        _build_section(body, section, pad=PAD)

    ttk.Separator(body, orient="horizontal").pack(fill="x", padx=PAD, pady=(6, 0))
    _build_parameters_table(body, pad=PAD)

    def _on_close():
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _on_close)
    ttk.Button(win, text="Закрыть", command=_on_close).pack(side="bottom", pady=8)
