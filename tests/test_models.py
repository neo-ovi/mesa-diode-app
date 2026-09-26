# -*- coding: utf-8 -*-
"""Реестр моделей тока: встроенные модели и подключение своей без правок остального кода."""

from dataclasses import replace

import numpy as np
import pytest

from mesa_diode.simulator import fitting, models, presets
from mesa_diode.simulator import physics as ph


def test_builtin_models_registered():
    assert {ph.MODEL_EMPIRICAL, ph.MODEL_TWO_DIODE, ph.MODEL_PHYSICAL} <= set(models.MODELS)
    assert [m.key for m in models.basic_models()] == [ph.MODEL_EMPIRICAL, ph.MODEL_TWO_DIODE]
    for m in models.MODELS.values():
        s = presets.to_structure({"mode": presets.MODE_FIT})
        parts = m.parts(replace(s, model=m.key), np.array([-0.5, 0.2]))
        assert set(parts) == set(m.components) and set(m.components) <= set(models.COMPONENTS)
        assert set(m.fit_core) <= set(fitting.PARAMS)
    with pytest.raises(ValueError, match="неизвестная модель"):
        models.get("нет такой")


def test_two_diode_equals_sze_55_sum():
    s = presets.to_structure({"mode": presets.MODE_BASIC, "basic_model": ph.MODEL_TWO_DIODE,
                              "J01_2d": 1e-8, "J02_2d": 1e-6})
    V = np.array([0.3])
    parts = ph.components(s, V)
    Vt = ph.thermal_voltage(300.0)
    assert parts["d1"][0] == pytest.approx(s.area * 1e-8 * np.expm1(0.3 / Vt))
    assert parts["d2"][0] == pytest.approx(s.area * 1e-6 * np.expm1(0.3 / (2 * Vt)))


def test_two_diode_fit_recovers_parameters():
    true = presets.to_structure({"mode": presets.MODE_BASIC, "basic_model": ph.MODEL_TWO_DIODE,
                                 "J01_2d": 3e-8, "J02_2d": 2e-5, "Rs": 40.0, "Rsh": 2e5})
    V = np.linspace(-3, 1, 300)
    I = ph.solve_iv(true, V).I * (1 + 0.003 * np.random.default_rng(1).standard_normal(V.size))
    start = presets.to_structure({"mode": presets.MODE_BASIC, "basic_model": ph.MODEL_TWO_DIODE})
    r = fitting.fit_iv(start, V, I, fitting.TWO_DIODE)
    assert r.error < 0.01
    assert r.params["J01_2d"] == pytest.approx(3e-8, rel=0.1)
    assert r.params["J02_2d"] == pytest.approx(2e-5, rel=0.1)
    assert any("τ₀ ≈" in n for n in r.notes)


def test_custom_model_plugs_in(monkeypatch):
    """Своя упрощённая модель: только parts и регистрация — решатель и подгонка работают."""
    def parts(s, Vd):
        return {"emp": s.area * s.J0_emp * ph._expm1(np.asarray(Vd) / s.Vt)}   # n = 1, один параметр

    custom = models.CurrentModel(
        key="ideal", label="идеальный диод", formula="(тест)", summary="n = 1",
        fields=frozenset({"J0_emp"}), fit_core=("J0_emp",), components=("emp",), parts=parts,
        saturation=lambda s: (s.J0_emp, "J₀"), start=lambda s, V, I: {"J0_emp": 1e-6})
    monkeypatch.setitem(models.MODELS, "ideal", custom)
    s = replace(presets.to_structure({"mode": presets.MODE_BASIC}), model="ideal", J0_emp=2e-6,
                Rs=20.0, Rsh=1e6)
    V = np.linspace(-1, 0.6, 120)
    I = ph.solve_iv(s, V).I
    assert np.all(np.isfinite(I))
    r = fitting.fit_iv(replace(s, J0_emp=1e-4), V, I, "ideal")
    assert r.params["J0_emp"] == pytest.approx(2e-6, rel=0.02)
