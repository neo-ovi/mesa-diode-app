# -*- coding: utf-8 -*-
"""Набор формул как в учебнике (typeset.py) — без окна Tk."""

import pytest

from mesa_diode.simulator import typeset


def test_descriptive_subscripts_are_upright():
    """ISO 80000-2: индекс из двух и более букв — описательный, прямым шрифтом."""
    assert typeset.upright(r"$R_{sh}+V_{bi}$") == r"$R_{\mathrm{sh}}+V_{\mathrm{bi}}$"
    assert typeset.upright(r"$\tau^{bg}$") == r"$\tau^{\mathrm{bg}}$"
    # однобуквенные индексы, показатели степени и уже прямые индексы не трогаются
    for tex in (r"$R_{s}$", r"$e^{qV/kT}$", r"$J_{01}$", r"$\tau_{n,p}$", r"$V_{\mathrm{д}}$"):
        assert typeset.upright(tex) == tex


def test_cyrillic_only_inside_mathrm():
    assert typeset.cyrillic_outside_mathrm(r"$V_{\mathrm{д}}=1\ \mathrm{В}$") == []
    assert typeset.cyrillic_outside_mathrm(r"$V_{д}$") == ["д"]


def test_raster_is_cached_and_has_ink():
    tex = r"$I=I_{0}\left[\exp\left(\dfrac{qV}{nkT}\right)-1\right]$"
    typeset.raster.cache_clear()
    alpha, depth = typeset.raster(tex, typeset.DISPLAY_PT, 96)
    assert alpha.ndim == 2 and alpha.max() > 200 and depth >= 0
    typeset.raster(tex, typeset.DISPLAY_PT, 96)
    assert typeset.raster.cache_info().hits == 1


def test_bigger_size_gives_bigger_formula():
    small = typeset.raster(r"$A=\dfrac{\pi D^{2}}{4}$", typeset.SYMBOL_PT, 96)[0]
    big = typeset.raster(r"$A=\dfrac{\pi D^{2}}{4}$", typeset.DISPLAY_PT, 96)[0]
    assert big.shape[0] > small.shape[0] and big.shape[1] > small.shape[1]


def test_fit_size_shrinks_wide_formula_but_not_below_minimum():
    tex = r"$I=I_{\mathrm{д}}+\dfrac{V_{\mathrm{д}}}{R_{\mathrm{sh}}}+I_{L}\left|V_{\mathrm{д}}\right|^{m}$"
    width = typeset.raster(tex, typeset.DISPLAY_PT, 96)[0].shape[1]
    assert typeset.fit_size(tex, typeset.DISPLAY_PT, 96, None) == typeset.DISPLAY_PT
    assert typeset.fit_size(tex, typeset.DISPLAY_PT, 96, width + 5) == typeset.DISPLAY_PT
    smaller = typeset.fit_size(tex, typeset.DISPLAY_PT, 96, width * 0.8)
    assert typeset.MIN_PT <= smaller < typeset.DISPLAY_PT
    assert typeset.fit_size(tex, typeset.DISPLAY_PT, 96, 10) == typeset.MIN_PT


@pytest.mark.parametrize("color, rgb", [("#ffffff", (255, 255, 255)), ("white", (255, 255, 255)),
                                        ("#1a4fa0", (26, 79, 160))])
def test_ppm_colors(color, rgb):
    data, width, height = typeset.ppm(r"$x$", 12, 96, fg="#000000", bg=color)
    header = b"P6 %d %d 255\n" % (width, height)
    assert data.startswith(header) and len(data) == len(header) + 3 * width * height
    assert tuple(data[len(header):len(header) + 3]) == rgb      # угол картинки — фон


def test_index_markup():
    assert typeset.split_index_markup("V_{bi} и 10^{−4}") == [
        ("V", None), ("bi", "sub"), (" и 10", None), ("−4", "sup")]
    assert typeset.split_index_markup("physics.solve_iv") == [("physics.solve_iv", None)]


def test_display_fractions_are_full_size():
    """В выключной формуле дроби — в полный размер (displaystyle), \\dfrac не трогается."""
    assert typeset.display(r"$\frac{kT}{q}\ln\frac{a}{b}$") == r"$\dfrac{kT}{q}\ln\dfrac{a}{b}$"
    assert typeset.display(r"$\dfrac{a}{b}$") == r"$\dfrac{a}{b}$"
    tex = r"$\frac{a}{b}$"
    assert typeset.raster(typeset.display(tex), 15, 96)[0].shape[0] > typeset.raster(tex, 15, 96)[0].shape[0]
