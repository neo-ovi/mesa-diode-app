# -*- coding: utf-8 -*-
"""Проверки содержимого окна «Формулы и параметры» (simulator/formulas.py).

Тесты проверяют только данные (FORMULA_SECTIONS, PARAMETERS_TABLE) — сам
GUI не поднимается. simulator/formulas.py импортирует tkinter и
matplotlib.backends.backend_tkagg на уровне модуля (для встраивания
отрендеренных формул в окно); в окружениях без установленного Tk
(например, Linux-CI без python3-tk) настоящего tkinter может не быть —
тогда перед импортом модуля подставляется временная заглушка только для
этого случая, чтобы можно было проверить данные без реального GUI. Там,
где Tk есть (обычная машина разработчика), тесты используют настоящий
tkinter как обычно.
"""

import sys
from unittest.mock import MagicMock

import pytest

try:
    import tkinter  # noqa: F401
except ImportError:
    sys.modules.setdefault("tkinter", MagicMock())
    sys.modules.setdefault("tkinter.ttk", MagicMock())
    sys.modules.setdefault("matplotlib.backends.backend_tkagg", MagicMock())

from matplotlib.mathtext import MathTextParser

from mesa_diode.simulator import formulas

MATH_PARSER = MathTextParser("agg")


def _all_mathtext_strings():
    seen = []
    for section in formulas.FORMULA_SECTIONS:
        for tex in section["formulas"]:
            seen.append(tex)
        for symbol_tex, _desc in section.get("legend", []):
            seen.append(symbol_tex)
    for symbol_tex, _desc, _units, _used_in in formulas.PARAMETERS_TABLE:
        seen.append(symbol_tex)
    return seen


def test_formula_sections_have_required_fields():
    assert formulas.FORMULA_SECTIONS
    for section in formulas.FORMULA_SECTIONS:
        assert section["title"]
        assert section["formulas"], section["title"]


def test_section_titles_are_unique():
    titles = [s["title"] for s in formulas.FORMULA_SECTIONS]
    assert len(titles) == len(set(titles))


def test_every_legend_entry_is_a_symbol_description_pair():
    for section in formulas.FORMULA_SECTIONS:
        for entry in section.get("legend", []):
            assert len(entry) == 2
            symbol_tex, description = entry
            assert symbol_tex.strip()
            assert description.strip()


def test_parameters_table_rows_are_4_tuples():
    assert formulas.PARAMETERS_TABLE
    for row in formulas.PARAMETERS_TABLE:
        assert len(row) == 4
        symbol_tex, description, units, used_in = row
        assert symbol_tex.strip() and description.strip()
        assert units.strip() and used_in.strip()


def _all_plain_strings():
    plain = [formulas.INTRO_TEXT, formulas.PARAMETERS_TITLE]
    for section in formulas.FORMULA_SECTIONS:
        plain.append(section["title"])
        plain.extend(desc for _sym, desc in section.get("legend", []))
        if section.get("principle"):
            plain.append(section["principle"])
    for _sym, description, units, used_in in formulas.PARAMETERS_TABLE:
        plain.extend([description, units, used_in])
    return plain


@pytest.mark.parametrize("text", _all_plain_strings())
def test_plain_text_has_no_unrendered_markup(text):
    # Обычный текст выводится виджетом Tk, а не mathtext: $ и \ в нём
    # показались бы на экране как есть. После разбора индексов не должно
    # остаться и фигурных скобок.
    assert "$" not in text and "\\" not in text
    rendered = "".join(chunk for chunk, _tag in formulas.split_index_markup(text))
    assert "{" not in rendered and "}" not in rendered


def test_split_index_markup_sub_and_sup():
    assert formulas.split_index_markup("V_{bi} и 10^{−4}") == [
        ("V", None), ("bi", "sub"), (" и 10", None), ("−4", "sup"),
    ]


def test_split_index_markup_keeps_plain_underscores():
    text = "physics.solve_iv и estimate_grading_m"
    assert formulas.split_index_markup(text) == [(text, None)]


@pytest.mark.parametrize("tex", _all_mathtext_strings())
def test_mathtext_syntax_is_valid(tex):
    # Настоящая проверка синтаксиса формул движком matplotlib — упадёт,
    # если в mathtext-строке опечатка или неподдерживаемая команда.
    MATH_PARSER.parse(tex, dpi=100)
