# -*- coding: utf-8 -*-
"""Правая панель (схема и формула), обучение и объяснения закрытых полей — без окна Tk."""

import numpy as np
import pytest

from mesa_diode.simulator import hints, presets, scheme, tutorial
from mesa_diode.simulator import physics as ph


def _structure(mode, **changes):
    return presets.to_structure({**presets.DEFAULT_PARAMS, "mode": mode, **changes})


def test_basic_scheme_is_diode_shunt_and_series_resistor():
    s = _structure(presets.MODE_BASIC, I_L=1e-5, I_mod=0.1)
    branches, series = scheme.elements(s, presets.MODE_BASIC)
    assert [e.key for e in branches] == ["diode", "shunt"]
    assert series.symbol == "resistor"                     # в базовой модели R_s постоянно


@pytest.mark.parametrize("model, keys", [
    (ph.MODEL_EMPIRICAL, ["diode", "shunt", "leak"]),
    (ph.MODEL_TWO_DIODE, ["d1", "d2", "shunt", "leak"]),
    (ph.MODEL_PHYSICAL, ["diff", "gr", "shunt", "leak"]),
])
def test_extended_scheme_follows_current_model(model, keys):
    s = _structure(presets.MODE_EXTENDED, extended_model=model, I_L=0.0, I_mod=0.1)
    branches, series = scheme.elements(s, presets.MODE_EXTENDED)
    assert [e.key for e in branches] == keys
    assert not branches[-1].active and "выключена" in branches[-1].tip      # I_L = 0
    assert series.symbol == "varres"                                        # R_s(I) при конечном I_mod
    assert all(e.tip for e in branches + [series])


def test_unknown_model_is_drawn_as_one_block():
    from dataclasses import replace
    s = replace(_structure(presets.MODE_EXTENDED), model="своя")
    branches, _series = scheme.elements(s, presets.MODE_EXTENDED)
    assert branches[0].symbol == "box"


def test_hand_calculation_matches_solver():
    """Точка вручную в панели — те же числа, что считает решатель (6.1)."""
    s = _structure(presets.MODE_BASIC)
    lines = scheme.formula_lines(s, presets.MODE_BASIC)
    I = float(ph.junction_current(s, scheme.SAMPLE_VD))
    V = scheme.SAMPLE_VD + I * s.Rs
    assert any(f"I = {I:.3g} А" in line and f"{V:.4g} В" in line for line in lines)
    # и совпадает с решателем, решающим V → I
    assert float(ph.solve_iv(s, np.array([V])).I[0]) == pytest.approx(I, rel=1e-6)
    assert "V = V_д + I·R_s" in "\n".join(lines) and "I_L" not in lines[1]
    assert any("C(0)" in line for line in lines)


def test_panel_lines_for_every_model():
    for mode, model in ((presets.MODE_EXTENDED, ph.MODEL_TWO_DIODE), (presets.MODE_FIT, ph.MODEL_PHYSICAL)):
        s = _structure(mode, extended_model=model)
        text = "\n".join(scheme.formula_lines(s, mode))
        assert "Точка вручную" in text and "ВФХ" in text
        assert scheme.model_summary(s, mode)


def test_tutorial_steps_and_example():
    assert {step.target for step in tutorial.STEPS} <= set(tutorial.TARGET_KEYS)
    assert tutorial.STEPS[0].run == tutorial.PREPARE
    runs = [step.run for step in tutorial.STEPS if step.run]
    assert runs == [tutorial.PREPARE, tutorial.LOAD_EXAMPLE, tutorial.FIT]
    V, I = tutorial.example_iv()
    assert V.size == I.size and V.min() < 0 < V.max() and np.all(np.isfinite(I))
    assert np.array_equal(V, tutorial.example_iv()[0])          # воспроизводимо


def test_tutorial_autostart_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tutorial, "SETTINGS_PATH", tmp_path / "s" / "settings.json")
    monkeypatch.delenv(tutorial.DISABLE_ENV, raising=False)
    assert tutorial.should_autostart()
    tutorial.mark_done()
    assert not tutorial.should_autostart()
    monkeypatch.setattr(tutorial, "SETTINGS_PATH", tmp_path / "other.json")
    monkeypatch.setenv(tutorial.DISABLE_ENV, "1")
    assert not tutorial.should_autostart()


def test_closed_fields_explain_why_and_where():
    basic, ext, fit = presets.MODE_BASIC, presets.MODE_EXTENDED, presets.MODE_FIT
    assert hints.field_reason("D_um", basic, ph.MODEL_EMPIRICAL, ["B"]) == ""
    assert "«Расширенная»" in hints.field_reason("I_L", basic, ph.MODEL_EMPIRICAL, ["B"])
    assert "«Расширенная»" in hints.field_reason("d_epi_um", basic, ph.MODEL_EMPIRICAL, ["B"])
    reason = hints.field_reason("J01_2d", ext, ph.MODEL_PHYSICAL, ["B"])
    assert "двухдиодная" in reason and "Модель тока" in reason
    assert "«Подгонка»" in hints.field_reason("mu_n", ext, ph.MODEL_PHYSICAL, ["B"])
    assert "сценарий A" in hints.field_reason("mu_p_i", fit, ph.MODEL_PHYSICAL, ["A"],
                                              unused_by_scenario={"A": {"mu_p_i"}})
    # каждому закрытому полю — объяснение
    for mode in presets.MODES:
        model = presets.current_model({"mode": mode})
        editable = presets.editable_keys(mode, model)
        for spec in presets.NUMERIC_PARAMS:
            if spec.key not in editable:
                assert hints.field_reason(spec.key, mode, model, ["B"]), (mode, spec.key)


def test_curves_caption_names_missing_components():
    text = hints.curves_caption(ph.MODEL_EMPIRICAL, presets.MODE_BASIC, {"d1", "d2", "diff", "gr", "L"})
    assert "I_{01}" in text and "I_{diff}" in text and "двухдиодная" in text and "утечки" in text
