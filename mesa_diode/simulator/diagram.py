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

SUBSTRATE = dict(x0=1.0, x1=11.0, y0=0.0, y1=3.2)
MESA = dict(x0=4.5, x1=7.5, y0=3.2, y1=6.0)
CONTACT_GAP = 1.0                 # окно между половинками кольцевого контакта
CONTACT_HALF_WIDTH = 0.8          # ширина каждой половинки кольца
CONTACT_HEIGHT = 0.35
BACK_CONTACT = dict(x0=2.0, x1=10.0, y0=-0.35, y1=0.0)

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

    ``p`` — объект с атрибутами D_um, h_um, ND, NA, T (см. physics.MesaParams).
    Вызывается из MesaApp._plot_mesa после ``ax.clear()``.
    """
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Схема мезы (условно, не в масштабе): D — диаметр, h — глубина, "
        "N_D — доноры (подложка Ge(100), ГДГ-45), N_A — акцепторы (меза Ge)",
        fontsize=9)

    mesa_cx = (MESA["x0"] + MESA["x1"]) / 2

    # --- подложка ГДГ-45, Ge(100), N_D ---
    ax.add_patch(Rectangle(
        (SUBSTRATE["x0"], SUBSTRATE["y0"]),
        SUBSTRATE["x1"] - SUBSTRATE["x0"], SUBSTRATE["y1"] - SUBSTRATE["y0"],
        facecolor="#cfe3f7", edgecolor="black", linewidth=1))
    ax.text(mesa_cx, (SUBSTRATE["y0"] + SUBSTRATE["y1"]) / 2,
            "ГДГ-45, Ge(100), N_D", ha="center", va="center", fontsize=11)
    _callout(ax, SUBSTRATE["x1"] + 0.5, (SUBSTRATE["y0"] + SUBSTRATE["y1"]) / 2, 4)

    # --- меза Ge, N_A (прямоугольник, без искусственного скоса) ---
    ax.add_patch(Rectangle(
        (MESA["x0"], MESA["y0"]), MESA["x1"] - MESA["x0"], MESA["y1"] - MESA["y0"],
        facecolor="#d9c2a3", edgecolor="black", linewidth=1))
    ax.text(mesa_cx, (MESA["y0"] + MESA["y1"]) / 2, "Ge, N_A",
            ha="center", va="center", fontsize=11)
    _callout(ax, MESA["x1"] + 0.5, MESA["y1"] - 0.4, 3)

    # --- верхний контакт: кольцо в разрезе — две половинки с окном между ними ---
    left_block = (mesa_cx - CONTACT_GAP / 2 - CONTACT_HALF_WIDTH,
                  mesa_cx - CONTACT_GAP / 2)
    right_block = (mesa_cx + CONTACT_GAP / 2,
                   mesa_cx + CONTACT_GAP / 2 + CONTACT_HALF_WIDTH)
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

    # --- сводка параметров ---
    info = f"N_D = {p.ND:.2e} см⁻³\nN_A = {p.NA:.2e} см⁻³\nT = {p.T:g} К"
    ax.text(XLIM[1] - 0.2, YLIM[1] - 0.6, info, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999999"))

    # --- расшифровка номеров — одной строкой под схемой, как в оригинале ---
    ax.text((XLIM[0] + XLIM[1]) / 2, YLIM[0] + 0.3,
            "1 — контактное кольцо   2 — тыльный контакт (к подложке)   "
            "3 — меза-структура   4 — подложка",
            ha="center", va="bottom", fontsize=8, color="#333333")
