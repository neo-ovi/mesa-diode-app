# -*- coding: utf-8 -*-
"""Эталоны ТЗ §9, зависящие от параметров реального образца.

Сами значения хранятся вне публичного репозитория, в каталоге данных
(MESA_DATA_DIR/validation/section9.json). Без него тесты пропускаются.
"""

import json

import numpy as np
import pytest

from mesa_diode import config
from mesa_diode.simulator import physics as ph


def _load():
    try:
        path = config.data_dir() / "validation" / "section9.json"
    except RuntimeError:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


DATA = _load()
pytestmark = pytest.mark.skipif(DATA is None, reason="нет MESA_DATA_DIR/validation/section9.json")


def _cases(key):
    return DATA.get(key, []) if DATA else []


@pytest.mark.parametrize("case", _cases("acceptor_from_resistivity"))
def test_acceptor_from_resistivity(case):
    NA, _ = ph.acceptor_from_resistivity(case["rho"], case["T"])
    assert NA == pytest.approx(case["expected"], rel=case["rel"])


@pytest.mark.parametrize("case", _cases("equilibrium_carriers"))
def test_equilibrium_carriers(case):
    M, m = ph.equilibrium_carriers(case["N"], ph.intrinsic_concentration(case["T"]))
    assert M == pytest.approx(case["majority"], rel=case["rel"])
    assert m == pytest.approx(case["minority"], rel=case["rel"])


UM = 1e-4


def _quantities(s):
    """Величины §9, которые сравниваются с эталонами (ключи файла эталонов)."""
    V = np.linspace(-1.0, 0.0, 41)
    N_slope, _ = ph.fit_inv_c2(V, ph.capacitance(s, V), s.area, s.eps)
    return {
        "Vbi": lambda: s.Vbi,
        "Vbi_degenerate": lambda: ph.built_in_potential(s, ph.VBI_DEGENERATE),
        "W0_um": lambda: ph.depletion_width(s, 0.0) / UM,
        "Wm1_um": lambda: ph.depletion_width(s, -1.0) / UM,
        "C0_pF": lambda: ph.capacitance(s, 0.0) * 1e12,
        "Cm1_pF": lambda: ph.capacitance(s, -1.0) * 1e12,
        "cutoff": lambda: ph.c2_cutoff(s),
        "N_slope": lambda: N_slope,
    }


def _structure_cases():
    return [pytest.param(case, key, id=f"{case['name']}-{key}")
            for case in _cases("structures") for key in case["expected"]]


@pytest.mark.parametrize("case, key", _structure_cases())
def test_structure_reference_value(case, key):
    s = ph.Structure(**case["params"])
    expected, tol = case["expected"][key]
    assert float(_quantities(s)[key]()) == pytest.approx(expected, rel=tol)
