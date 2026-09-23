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
from mesa_diode.simulator import reference


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
        "Idiff_m1": lambda: ph.diffusion_current(s, -1.0),
        "Igr_m1": lambda: ph.gr_current(s, -1.0),
        "I_m1": lambda: ph.solve_iv(s, np.array([-1.0])).I[0],
        "I_p02": lambda: ph.solve_iv(s, np.array([0.2])).I[0],
        "dA_m1_cm2": lambda: ph.edge_area(s, -1.0),
        "VLI": lambda: ph.low_injection_voltage(s),
    }


def _structure_cases():
    return [pytest.param(case, key, id=f"{case['name']}-{key}")
            for case in _cases("structures") for key in case["expected"]]


@pytest.mark.parametrize("case, key", _structure_cases())
def test_structure_reference_value(case, key):
    s = ph.Structure(**case["params"])
    expected, tol, *abs_tol = case["expected"][key]
    value = float(_quantities(s)[key]())
    assert value == pytest.approx(expected, rel=tol, abs=abs_tol[0] if abs_tol else 1e-12)


@pytest.mark.parametrize("case", [c for c in _cases("structures") if "expected_warning" in c],
                         ids=lambda c: c["name"])
def test_structure_reference_warning(case):
    data = reference.reference_table(ph.Structure(**case["params"]))
    assert any(case["expected_warning"] in w for w in data.warnings)


@pytest.mark.parametrize("case", _cases("tau_dis"))
def test_dislocation_lifetime(case):
    tau = ph.dislocation_lifetime(case["sigma_R"], case["N_dis"])
    assert tau == pytest.approx(case["expected_s"], rel=case["rel"])
