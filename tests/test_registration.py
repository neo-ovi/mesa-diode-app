# -*- coding: utf-8 -*-
"""Новое поле окна или параметр подгонки зарегистрированы во всех реестрах.

Список мест, где регистрируется поле, — методичка, п. 13.9; упавший тест
называет пропущенное место."""

from dataclasses import fields, replace

import pytest

from mesa_diode.simulator import fitting, hints, models, presets
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.physics import structure

FIELD_KEYS = {spec.key for spec in presets.NUMERIC_PARAMS}
STRUCTURE_FIELDS = {f.name for f in fields(ph.Structure)}


def test_every_field_is_registered():
    grouped = [key for _title, keys, _about in hints.FIELD_GROUPS for key in keys]
    assert sorted(grouped) == sorted(FIELD_KEYS), "hints.FIELD_GROUPS: каждое поле ровно в одном разделе"
    assert set(hints.PARAM_HINTS) == FIELD_KEYS, "hints.PARAM_HINTS"
    assert set(hints.METODICHKA_REFS) >= FIELD_KEYS, "hints.METODICHKA_REFS"
    assert set(presets.DEFAULT_PARAMS) >= FIELD_KEYS, "presets.DEFAULT_PARAMS"
    for spec in presets.NUMERIC_PARAMS:
        assert spec.field is None or spec.field in STRUCTURE_FIELDS, f"поле Structure для {spec.key}"


def test_every_field_is_registered_in_window_modules():
    pytest.importorskip("tkinter")
    from mesa_diode.simulator import app, formulas
    assert set(app.SHORT_LABELS) >= FIELD_KEYS, "app.SHORT_LABELS"
    assert set(formulas.PARAMETER_USE) == FIELD_KEYS, "formulas.PARAMETER_USE"


def test_every_fit_parameter_maps_to_structure_and_window():
    for key, p in fitting.PARAMS.items():
        field = p[0]
        assert field is None or field in STRUCTURE_FIELDS, f"FitParam({key}).field"
    for model in models.MODELS.values():
        assert set(model.fit_core) <= set(models.FIT_PARAMS), f"fit_core модели {model.key}"
    assert set(fitting.FIELD_TO_PARAM.values()) <= set(fitting.PARAMS)


@pytest.mark.parametrize("name", sorted(structure._NON_ELECTROSTATIC))
def test_cached_electrostatics_does_not_depend_on_fit_fields(name):
    """Structure.updated переносит кэш только для полей, от которых он не зависит."""
    s = presets.to_structure({**presets.DEFAULT_PARAMS, "mode": presets.MODE_FIT})
    value = getattr(s, name)
    changed = 0.37 * value if value and value != float("inf") else 1.23e-3
    fresh = replace(s, **{name: changed})
    kept = s.updated(**{name: changed})
    for prop in structure._ELECTROSTATIC_CACHE:
        assert repr(getattr(kept, prop)) == repr(getattr(fresh, prop)), (name, prop)
