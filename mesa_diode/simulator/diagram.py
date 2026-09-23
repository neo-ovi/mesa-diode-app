# -*- coding: utf-8 -*-
"""Схематическое изображение мезы для GUI-симулятора.

Стиль подписей — как на типовой технической схеме подключения образца:
элементы помечены короткими номерами (кружки 1-4), а расшифровка вынесена
в одну подпись под рисунком. Это осознанный выбор, а не просто вкус —
длинные надписи внутри самих фигур («нижний контакт (металл)», «верхний
контакт») не помещались в свои прямоугольники при типичных размерах окна
и наезжали на соседние элементы и на стрелки размеров.
"""

from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# Геометрия схемы в условных единицах холста (не мкм — см. draw_mesa_diagram)
# ----------------------------------------------------------------------
XLIM = (-2.0, 12.0)
YLIM = (-2.0, 9.5)
# ВАЖНО: увеличивать этот диапазон ради «больше места для текста» —
# контринтуитивно, но НЕПРАВИЛЬНЫЙ способ чинить наложения. Текст рисуется
# фиксированным размером в пунктах, а не в единицах холста; при aspect="equal"
# и ограниченной высоте подграфика (см. MesaApp._build_figure_area, схема
# делит окно с тремя другими графиками) увеличение YLIM снижает физический
# масштаб (единиц данных на дюйм становится больше), и тот же текст в pt
# занимает БОЛЬШУЮ, а не меньшую долю холста. Если элементы наезжают друг на
# друга при уменьшении подграфика — увеличивайте высоту, выделенную схеме
# в _build_figure_area (height_ratios), а не YLIM здесь.

SUBSTRATE = dict(x0=1.0, x1=11.0, y0=0.0, y1=3.2)
MESA = dict(x0=4.5, x1=7.5, y0=3.2, y1=6.0)
CONTACT_HEIGHT = 0.35
BACK_CONTACT = dict(x0=2.0, x1=10.0, y0=-0.35, y1=0.0)

# Окно кольцевого контакта рисуется пропорционально p.d_um/p.D_um (см.
# draw_mesa_diagram) — не в масштабе схемы целиком, но соотношение окно/меза
# соответствует введённым значениям. Клампы — чтобы вырожденные значения
# (d около 0 или d около D) не схлопывали рисунок в нечитаемую полоску.
MIN_GAP_RATIO = 0.05
MAX_GAP_RATIO = 0.9

CALLOUT_STYLE = dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="black")


def _callout(ax, x, y, number, ha="center"):
    """Кружок с номером элемента — как на технической схеме подключения."""
    ax.text(x, y, str(number), ha=ha, va="center", fontsize=9,
            fontweight="bold", bbox=CALLOUT_STYLE, zorder=5)


def _probe_lead(ax, start_xy, bend_xy, end_xy):
    """Ломаный проводник к внешнему пробнику (маленький кружок на конце)."""
    xs = [start_xy[0], bend_xy[0], end_xy[0]]
    ys = [start_xy[1], bend_xy[1], end_xy[1]]
    ax.plot(xs, ys, color="black", linewidth=1.2, zorder=4)
    ax.plot(end_xy[0], end_xy[1], marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor="black", zorder=5)


def _dimension(ax, x0, x1, y, label, tick_y0=None, tick_y1=None):
    """Горизонтальная стрелка размера с выносными линиями от объекта."""
    if tick_y0 is not None and tick_y1 is not None:
        for x in (x0, x1):
            ax.plot([x, x], [tick_y0, y], color="#888888",
                    linewidth=0.7, linestyle="--", zorder=1)
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2),
                zorder=3)
    ax.text((x0 + x1) / 2, y + 0.25, label, ha="center", va="bottom",
            fontsize=9, fontweight="bold")


def draw_mesa_diagram(ax, p):
    """Рисует схему мезы на переданных осях ``ax`` по параметрам ``p``.

    ``p`` — объект с атрибутами D_um (диаметр мезы), d_um (внутренний
    диаметр кольца), h_um и i_type (подпись типа i-слоя). Вызывается из
    MesaApp._plot_mesa после ``ax.clear()``.
    """
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Схема мезы (не в масштабе), {p.i_type}",
        fontsize=9)

    mesa_cx = (MESA["x0"] + MESA["x1"]) / 2

    # --- подложка p-Ge, N_A из ρ_sub ---
    ax.add_patch(Rectangle(
        (SUBSTRATE["x0"], SUBSTRATE["y0"]),
        SUBSTRATE["x1"] - SUBSTRATE["x0"], SUBSTRATE["y1"] - SUBSTRATE["y0"],
        facecolor="#cfe3f7", edgecolor="black", linewidth=1))
    ax.text(mesa_cx, (SUBSTRATE["y0"] + SUBSTRATE["y1"]) / 2,
            "подложка p-Ge, N_A", ha="center", va="center", fontsize=11)
    _callout(ax, SUBSTRATE["x1"] + 0.5, (SUBSTRATE["y0"] + SUBSTRATE["y1"]) / 2, 4)

    # --- меза: эпитаксия n⁺ / i (прямоугольник, без искусственного скоса) ---
    ax.add_patch(Rectangle(
        (MESA["x0"], MESA["y0"]), MESA["x1"] - MESA["x0"], MESA["y1"] - MESA["y0"],
        facecolor="#d9c2a3", edgecolor="black", linewidth=1))
    ax.text(mesa_cx, (MESA["y0"] + MESA["y1"]) / 2, "n⁺/i",
            ha="center", va="center", fontsize=10)
    _callout(ax, MESA["x1"] + 0.5, (MESA["y0"] + MESA["y1"]) / 2, 3)

    # --- верхний контакт: кольцо в разрезе — две половинки с окном между ними ---
    # Ширина окна пропорциональна d/D (клампы MIN/MAX_GAP_RATIO — см. выше).
    mesa_width = MESA["x1"] - MESA["x0"]
    gap_ratio = min(max(p.d_um / p.D_um, MIN_GAP_RATIO), MAX_GAP_RATIO)
    contact_gap = mesa_width * gap_ratio
    contact_half_width = (mesa_width - contact_gap) / 2

    left_block = (mesa_cx - contact_gap / 2 - contact_half_width,
                  mesa_cx - contact_gap / 2)
    right_block = (mesa_cx + contact_gap / 2,
                   mesa_cx + contact_gap / 2 + contact_half_width)
    contact_y0, contact_y1 = MESA["y1"], MESA["y1"] + CONTACT_HEIGHT
    for x0, x1 in (left_block, right_block):
        ax.add_patch(Rectangle((x0, contact_y0), x1 - x0, contact_y1 - contact_y0,
                                facecolor="#8a8a8a", edgecolor="black", linewidth=1))
    _callout(ax, right_block[1] + 0.5, (contact_y0 + contact_y1) / 2, 1)

    # --- тыльный контакт (штриховка — металлизация на подложке) ---
    ax.add_patch(Rectangle(
        (BACK_CONTACT["x0"], BACK_CONTACT["y0"]),
        BACK_CONTACT["x1"] - BACK_CONTACT["x0"], BACK_CONTACT["y1"] - BACK_CONTACT["y0"],
        facecolor="#bbbbbb", edgecolor="black", linewidth=1, hatch="//"))
    _callout(ax, BACK_CONTACT["x1"] + 0.5, (BACK_CONTACT["y0"] + BACK_CONTACT["y1"]) / 2, 2)

    # --- проводники к внешним пробникам, оба выведены влево (как на схеме подключения) ---
    _probe_lead(ax,
                start_xy=(mesa_cx, (contact_y0 + contact_y1) / 2),
                bend_xy=(-1.0, (contact_y0 + contact_y1) / 2),
                end_xy=(-1.0, 8.0))
    _probe_lead(ax,
                start_xy=(BACK_CONTACT["x0"], (BACK_CONTACT["y0"] + BACK_CONTACT["y1"]) / 2),
                bend_xy=(-1.0, (BACK_CONTACT["y0"] + BACK_CONTACT["y1"]) / 2),
                end_xy=(-1.0, -1.2))

    # --- размер D: ширина мезы, вынесена НАД контактом — там гарантированно пусто ---
    _dimension(ax, MESA["x0"], MESA["x1"], y=8.0, label=f"D = {p.D_um:g} мкм",
               tick_y0=contact_y1, tick_y1=8.0)

    # --- размер d: ширина окна кольца, отдельной строкой НИЖЕ стрелки D ---
    # (y=7.1 — между верхом контакта ~contact_y1 и строкой D на y=8.0, не
    # пересекается ни с той, ни с другой при типичных пропорциях мезы)
    _dimension(ax, left_block[1], right_block[0], y=7.1, label=f"d = {p.d_um:g} мкм",
               tick_y0=contact_y1, tick_y1=7.1)

    # --- размер h: высота мезы (от подложки до контакта), вынесен вправо от мезы ---
    h_x = MESA["x1"] + 2.0
    ax.plot([MESA["x1"], h_x], [MESA["y0"], MESA["y0"]],
            color="#888888", linewidth=0.7, linestyle="--")
    ax.plot([right_block[1], h_x], [contact_y1, contact_y1],
            color="#888888", linewidth=0.7, linestyle="--")
    ax.annotate("", xy=(h_x, contact_y1), xytext=(h_x, MESA["y0"]),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
    ax.text(h_x + 0.2, (MESA["y0"] + contact_y1) / 2, f"h = {p.h_um:g} мкм",
            ha="left", va="center", fontsize=9, fontweight="bold", rotation=90)

    # --- расшифровка номеров — одной строкой под схемой, как в оригинале ---
    ax.text((XLIM[0] + XLIM[1]) / 2 + 0.6, YLIM[0] + 0.3,
            "1 — кольцо   2 — тыльный контакт   "
            "3 — меза n⁺/i   4 — подложка",
            ha="center", va="bottom", fontsize=8, color="#333333")
