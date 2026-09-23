# -*- coding: utf-8 -*-
"""Шаг колеса мыши и тексты подсказок (без окна Tk)."""

import pytest

from mesa_diode.simulator import hints, presets
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.widgets import wheel_step


@pytest.mark.parametrize("value, direction, fine, expected", [
    (500.0, 1, False, 510.0),
    (500.0, -1, False, 490.0),
    (500.0, 1, True, 501.0),
    (1.36, 1, False, 1.46),
    (1.0, -1, False, 0.9),
    (0.3, 1, False, 0.31),
    (5e15, 1, False, 5.1e15),
    (1e-6, -1, False, 9e-7),
])
def test_wheel_step_is_tenth_of_leading_digit(value, direction, fine, expected):
    assert wheel_step(value, direction, fine) == pytest.approx(expected, rel=1e-12)


def test_wheel_step_never_reaches_zero():
    value = 1.0
    for _ in range(200):
        value = wheel_step(value, -1)
        assert value > 0


def test_wheel_step_from_zero():
    assert wheel_step(0.0, -1, key="N_dis") == 0.0
    assert wheel_step(0.0, 1, key="N_dis") == 1e3
    assert wheel_step(0.0, 1, key="I_L") == 1e-6
    assert wheel_step(0.0, 1, key="Rs") == 0.1


def test_every_curve_has_a_hint():
    pytest.importorskip("tkinter")
    from mesa_diode.simulator import app
    assert {key for curves in app.CURVES.values() for key, _ in curves} <= set(hints.CURVE_HINTS)


def test_every_parameter_mode_and_boundary_has_a_hint():
    assert set(hints.PARAM_HINTS) == {spec.key for spec in presets.NUMERIC_PARAMS}
    assert set(hints.MODE_HINTS) == set(presets.MODES)
    assert set(hints.BOUNDARY_HINTS) == set(ph.BOUNDARY_LABELS)


def test_hints_have_no_tex_and_plain_removes_markup():
    texts = [*hints.PARAM_HINTS.values(), *hints.CURVE_HINTS.values(),
             *hints.MODE_HINTS.values(), *hints.BOUNDARY_HINTS.values()]
    for text in texts:
        assert "$" not in text and "\\" not in text
        assert "_{" not in hints.plain(text) and "^{" not in hints.plain(text)
    assert hints.plain("V_{bi} и τ^{bg}") == "Vbi и τbg"


def test_rs_and_rsh_hints_explain_the_difference():
    assert "последовательно" in hints.PARAM_HINTS["Rs"]
    assert "параллельно" in hints.PARAM_HINTS["Rsh"]
