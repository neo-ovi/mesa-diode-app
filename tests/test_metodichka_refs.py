# -*- coding: utf-8 -*-
"""Ссылки подсказок «методичка, п. …» указывают на существующие пункты.

Методичка хранится в каталоге данных ($MESA_DATA_DIR/docs/METODICHKA.md);
без него тест пропускается."""

import re

import pytest

from mesa_diode import config
from mesa_diode.simulator import hints


def _metodichka_text():
    try:
        path = config.data_dir() / "docs" / "METODICHKA.md"
    except RuntimeError:
        path = None
    if path is None or not path.is_file():
        pytest.skip("методичка недоступна: задайте MESA_DATA_DIR")
    return path.read_text(encoding="utf-8")


def test_every_hint_reference_points_to_a_heading():
    text = _metodichka_text()
    missing = []
    for key, refs in hints.METODICHKA_REFS.items():
        for ref in (r.strip() for r in refs.split(";")):
            kind, number = ref.split(" ", 1)
            if kind == "п.":
                pattern = rf"^\*\*{re.escape(number)}[ .]"
            else:
                pattern = rf"^## Глава {re.escape(number)}\."
            if not re.search(pattern, text, flags=re.MULTILINE):
                missing.append(f"{key}: {ref}")
    assert not missing, "нет в методичке: " + ", ".join(missing)


def test_error_references_point_to_headings():
    text = _metodichka_text()
    refs = {ref for _pattern, ref in hints.ERROR_REFS} | {hints.DEFAULT_ERROR_REF}
    missing = []
    for ref in refs:
        kind, number = ref.split(" ", 1)
        pattern = (rf"^\*\*{re.escape(number)}[ .]" if kind == "п." else
                   rf"^## Глава {re.escape(number)}\.")
        if not re.search(pattern, text, flags=re.MULTILINE):
            missing.append(ref)
    assert not missing, "нет в методичке: " + ", ".join(missing)


def test_formula_references_point_to_headings():
    """Сноски «Методичка: п. …» под формулами вкладки «Модель» и правой панели."""
    import sys
    from unittest.mock import MagicMock
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.modules.setdefault("tkinter", MagicMock())
        sys.modules.setdefault("tkinter.ttk", MagicMock())
        sys.modules.setdefault("tkinter.font", MagicMock())
    from mesa_diode.simulator import formulas, presets, scheme

    text = _metodichka_text()
    refs = {r.strip() for s in formulas.SECTIONS for b in s.blocks
            if isinstance(b, formulas.Formula) for r in b.ref.split(";") if r.strip()}
    for mode in presets.MODES:
        s = presets.to_structure({**presets.DEFAULT_PARAMS, "mode": mode})
        for block in scheme.panel_blocks(s, mode):
            if isinstance(block, scheme.Eq):
                refs |= set(re.findall(r"(?:п\. \d+[АA]?\.\d+[а-я]?|гл\. \d+[АA]?)", block.note))
    missing = []
    for ref in refs:
        kind, number = ref.split(" ", 1)
        pattern = (rf"^\*\*{re.escape(number)}[ .]" if kind == "п." else
                   rf"^## Глава {re.escape(number)}\.")
        if not re.search(pattern, text, flags=re.MULTILINE):
            missing.append(ref)
    assert not missing, "нет в методичке: " + ", ".join(sorted(missing))
