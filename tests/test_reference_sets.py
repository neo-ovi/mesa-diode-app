# -*- coding: utf-8 -*-
"""Эталоны ТЗ §9, зависящие от параметров реального образца.

Сами значения хранятся вне публичного репозитория, в каталоге данных
(MESA_DATA_DIR/validation/section9.json). Без него тесты пропускаются.
"""

import json

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
