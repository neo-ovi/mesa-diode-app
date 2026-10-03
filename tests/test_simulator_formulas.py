# -*- coding: utf-8 -*-
"""Проверки окна «Формулы и параметры» (ТЗ §7.1, приёмка §12).

GUI не поднимается: проверяются данные реестра SECTIONS. Если в окружении
нет Tk (Linux-CI без python3-tk), модуль импортируется с заглушкой.
"""

import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import tkinter  # noqa: F401
except ImportError:
    sys.modules.setdefault("tkinter", MagicMock())
    sys.modules.setdefault("tkinter.ttk", MagicMock())
    sys.modules.setdefault("tkinter.font", MagicMock())
    sys.modules.setdefault("matplotlib.backends.backend_tkagg", MagicMock())

from matplotlib.mathtext import MathTextParser

from mesa_diode.simulator import formulas as fm
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator import presets, typeset

MATH = MathTextParser("agg")
FORBIDDEN = (r"\text", r"\operatorname", r"\begin")
REF = re.compile(r"^(п\. \d+[АA]?\.\d+[а-я]?|гл\. \d+[АA]?)$")
# Формулы модели — в docstring physics.py, формулы автоподгонки (§6.7) — в fitting.py.
PHYSICS_SOURCE = "\n".join(
    [p.read_text(encoding="utf-8") for p in sorted(Path(ph.__file__).parent.glob("*.py"))]
    + [Path(importlib.import_module(f"mesa_diode.simulator.{m}").__file__).read_text(encoding="utf-8")
       for m in ("fitting", "defects", "models")])


def _blocks(kind):
    return [(s, b) for s in fm.SECTIONS for b in s.blocks if isinstance(b, kind)]


def _all_tex():
    tex = [b.tex for _s, b in _blocks(fm.Formula)]
    tex += [symbol for _s, b in _blocks(fm.Formula) for symbol, _text in b.where]
    tex += [row[0] for _s, b in _blocks(fm.Params) for row in b.rows]
    return tex


def _all_plain():
    plain = [s.title for s in fm.SECTIONS] + [s.why for s in fm.SECTIONS]
    plain += [p.title for p in fm.PARTS] + [p.lead for p in fm.PARTS]
    for _s, b in _blocks(fm.Text):
        plain.append(b.text)
    for _s, b in _blocks(fm.Formula):
        plain += [b.source, b.note, b.ref] + [text for _symbol, text in b.where]
    plain += [b.text for _s, b in _blocks(fm.Plaque)] + [b.text for _s, b in _blocks(fm.Graph)]
    for _s, b in _blocks(fm.Params):
        plain += [row[1] + row[2] for row in b.rows]
    plain += [ref for _code, ref in fm.SOURCES] + list(fm.PARAMETER_USE.values())
    return [p for p in plain if p]


@pytest.mark.parametrize("tex", _all_tex())
def test_formula_renders_with_mathtext(tex):
    MATH.parse(tex, dpi=100)


@pytest.mark.parametrize("tex", _all_tex())
def test_formula_uses_only_allowed_markup(tex):
    assert not any(cmd in tex for cmd in FORBIDDEN)
    assert not typeset.cyrillic_outside_mathrm(tex), tex      # кириллица — только в \mathrm{…}


def test_every_formula_has_source_or_badge():
    missing = [b.tex for _s, b in _blocks(fm.Formula) if not (b.source or b.badge)]
    assert missing == []


def test_badges_are_known():
    assert {b.badge for _s, b in _blocks(fm.Formula) if b.badge} <= set(fm.BADGE_COLORS)
    assert {b.kind for _s, b in _blocks(fm.Plaque)} <= set(fm.BADGE_COLORS)


@pytest.mark.parametrize("text", _all_plain())
def test_plain_text_has_no_unrendered_markup(text):
    assert "$" not in text and "\\" not in text
    rendered = "".join(chunk for chunk, _tag in fm.split_index_markup(text))
    assert "{" not in rendered and "}" not in rendered


def test_every_section_with_result_points_to_a_graph():
    for section in fm.SECTIONS:
        has_result = any(isinstance(b, fm.Formula) and b.result for b in section.blocks)
        if has_result:
            assert any(isinstance(b, fm.Graph) for b in section.blocks), section.sid


def test_every_section_has_why():
    assert all(s.why.strip() for s in fm.SECTIONS)
    assert [s.sid for s in fm.SECTIONS] == (
        [f"Б{i}" for i in range(8)] + [f"Р{i}" for i in range(9)] + [f"П{i}" for i in range(6)])


def test_parts_go_from_simple_to_complex():
    """Базовая → расширенная → подгонка; каждая следующая часть начинается с того,
    что было в предыдущей."""
    assert [p.letter for p in fm.PARTS] == ["Б", "Р", "П"]
    letters = [s.sid[0] for s in fm.SECTIONS]
    assert letters == sorted(letters, key="БРП".index)
    assert "В базовой модели мы использовали" in fm.PARTS[1].lead
    assert "В расширенной модели" in fm.PARTS[2].lead


def test_basic_part_holds_every_formula_of_the_basic_mode():
    """Часть Б — те формулы, по которым считает базовый режим: простой диод, схема, ВФХ, материал."""
    basic = {b.fid for s, b in _blocks(fm.Formula) if s.sid.startswith("Б") and b.fid}
    assert {"(6.3)", "(6.1)", "(1.1)", "(3.1)", "(3.2)", "(3.4)", "(3.5)", "(2.1)", "(2.2)", "(2.3)", "(2.4)",
            "(2.8)", "(2.9)", "(4.4)", "(6.4)", "(6.5)"} <= basic
    # сложное — не в базовой части
    assert not basic & {"(3.1а)", "(4.3)", "(4.3а)", "(5.4)", "(5.9)", "(5.10)", "(6.9)", "(6.11)"}


def test_every_numbered_formula_points_to_the_metodichka():
    missing = [b.fid for _s, b in _blocks(fm.Formula) if b.fid and not b.ref]
    assert missing == []
    for _s, b in _blocks(fm.Formula):
        for ref in filter(None, (r.strip() for r in b.ref.split(";"))):
            assert REF.match(ref), (b.fid, ref)


@pytest.mark.parametrize("fid", sorted(set(fm.formula_ids())))
def test_formula_id_matches_physics_docstring(fid):
    assert fid in PHYSICS_SOURCE, f"{fid} нет в docstring physics.py"


def test_formula_ids_are_unique():
    """Номер у формулы один; повтор (другая форма той же формулы) помечен repeat."""
    ids = [b.fid for _s, b in _blocks(fm.Formula) if b.fid and not b.repeat]
    assert len(ids) == len(set(ids))
    repeats = {b.fid for _s, b in _blocks(fm.Formula) if b.repeat}
    assert repeats <= set(ids)


def test_parameter_use_covers_main_window_parameters():
    assert set(fm.PARAMETER_USE) == {spec.key for spec in presets.NUMERIC_PARAMS}


def test_current_values_for_every_section():
    s = presets.to_structure(presets.DEFAULT_PARAMS)
    app = SimpleNamespace(structures={s.scenario: s}, ideality=None, model_ideality=None)
    keys = {b.key for _s, b in _blocks(fm.Current)}
    assert keys == set(fm.CURRENT_VALUES)
    for key in keys:
        text = fm.current_text(key, app)
        assert text.startswith("сценарий B") and "$" not in text


def test_current_values_without_calculation():
    assert "Рассчитать" in fm.current_text("geometry", None)


def test_split_index_markup_sub_and_sup():
    assert fm.split_index_markup("V_{bi} и 10^{−4}") == [
        ("V", None), ("bi", "sub"), (" и 10", None), ("−4", "sup"),
    ]


def test_split_index_markup_keeps_plain_underscores():
    text = "physics.solve_iv и estimate_grading_m"
    assert fm.split_index_markup(text) == [(text, None)]


def test_fit_guide_covers_modes_groups_boundaries_and_tables():
    guide = dict(fm.fit_guide())
    text = "\n".join(p for paragraphs in guide.values() for p in paragraphs)
    for spec in presets.NUMERIC_PARAMS:
        assert spec.label in text
    for label in ph.BOUNDARY_LABELS.values():
        assert label in text.lower()
    assert "3900" in text and "1900" in text and "Ioffe" in text
    assert "$" not in text and "\\" not in text
    assert fm.parameter_modes("D_um") == "Б, Р, П"
    assert fm.parameter_modes("d_epi_um") == "Р, П"
    assert fm.parameter_modes("mu_n") == "П"
    assert fm.parameter_modes("n_emp") == "Б, Р"          # простой диод и однодиодная модель
    assert fm.parameter_modes("J01_2d") == "Р" and fm.parameter_modes("I_L") == "Р, П"
