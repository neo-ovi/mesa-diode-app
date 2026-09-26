# -*- coding: utf-8 -*-
"""Чтение и проверка файлов измерений (datafile)."""

import numpy as np
import pytest

from mesa_diode.simulator import datafile as df
from mesa_diode.simulator.io import load_xy_file


def write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def iv_lines(sep=";", decimal=".", n=41):
    V = np.linspace(-2, 1, n)
    I = 1e-9 * (np.exp(V / 0.05) - 1) + V / 1e6
    return V, I, [f"{v:.4f}{sep}{i:.6e}".replace(".", decimal) if decimal != "." else
                  f"{v:.4f}{sep}{i:.6e}" for v, i in zip(V, I)]


# ------------------------------------------------------------ примеры
@pytest.mark.parametrize("name, kind", [("iv_example.csv", df.KIND_IV),
                                        ("iv_example_decimal_comma.txt", df.KIND_IV),
                                        ("iv_example.xlsx", df.KIND_IV),
                                        ("cv_example.csv", df.KIND_CV)])
def test_examples_load_without_warnings(name, kind):
    report = df.analyze(df.EXAMPLES_DIR / name, kind)
    assert report.ok and not report.warnings, report.text()
    assert report.voltage.size > 50
    assert np.all(np.diff(report.voltage) >= 0)


def test_examples_agree_across_formats():
    """Один и тот же набор в CSV (А), TXT (мкА, запятая) и Excel (мА)."""
    base = df.analyze(df.EXAMPLES_DIR / "iv_example.csv")
    for name in ("iv_example_decimal_comma.txt", "iv_example.xlsx"):
        other = df.analyze(df.EXAMPLES_DIR / name)
        np.testing.assert_allclose(other.voltage, base.voltage)
        np.testing.assert_allclose(other.value, base.value, rtol=1e-3)


def test_example_files_listed_and_format_help_mentions_formats():
    assert {"iv_example.csv", "iv_example.xlsx", "cv_example.csv"} <= set(df.example_files())
    for fmt in (".csv", ".txt", ".xlsx", ".xls"):
        assert fmt in df.FORMAT_HELP


# ------------------------------------------------------------ текст
@pytest.mark.parametrize("sep, decimal", [(";", "."), (";", ","), ("\t", ","), (",", "."), (" ", ".")])
def test_text_separators_and_decimal_comma(tmp_path, sep, decimal):
    V, I, lines = iv_lines(sep, decimal)
    path = write(tmp_path, "iv.txt", "\n".join(lines))
    report = df.analyze(path)
    assert report.ok, report.text()
    np.testing.assert_allclose(report.value, I, rtol=1e-5)


def test_cp1251_header_and_units(tmp_path):
    V, I, lines = iv_lines()
    text = "Напряжение, В;Ток, мА\n" + "\n".join(f"{v};{1e3 * i}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text, encoding="cp1251"))
    assert report.ok
    np.testing.assert_allclose(report.value, I, rtol=1e-9)
    assert any("мА" in i.text for i in report.infos)


def test_comma_as_both_separators_is_error(tmp_path):
    # «-1,5,2,3» — это -1,5 и 2,3 или четыре числа? Неоднозначно.
    text = "\n".join(f"{int(v)},{k % 10},{k},{k % 7}" for k, v in enumerate(np.linspace(-1, 1, 30)))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert not report.ok
    assert "Запятая" in report.errors[0].text


def test_header_selects_columns_among_many(tmp_path):
    V, I, _ = iv_lines()
    text = "t;I (uA);T;V (mV)\n" + "\n".join(
        f"{k};{1e6 * i};300;{1e3 * v}" for k, (v, i) in enumerate(zip(V, I)))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert report.ok, report.text()
    np.testing.assert_allclose(report.voltage, V, rtol=1e-9)
    np.testing.assert_allclose(report.value, I, rtol=1e-9)


def test_many_columns_without_header_warns(tmp_path):
    V, I, _ = iv_lines()
    text = "\n".join(f"{v};{i};300" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert report.ok and any("первые два" in w.text for w in report.warnings)


# ------------------------------------------------------------ проверки ВАХ
def test_absolute_reverse_current_gets_sign(tmp_path):
    V, I, _ = iv_lines()
    text = "\n".join(f"{v};{abs(i)}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert report.ok
    np.testing.assert_allclose(report.value, I, rtol=1e-9)
    assert any("по модулю" in w.text for w in report.warnings)


def test_inverted_polarity_is_flipped(tmp_path):
    V, I, _ = iv_lines()
    text = "\n".join(f"{v};{-i}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    np.testing.assert_allclose(report.value, I, rtol=1e-9)
    assert any("Знак тока обратный" in w.text for w in report.warnings)


def test_swapped_columns_is_error(tmp_path):
    V, I, _ = iv_lines()
    text = "\n".join(f"{i};{v}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert not report.ok and "перепутаны" in report.errors[0].text


def test_only_forward_branch_and_few_points_warn(tmp_path):
    text = "\n".join(f"{v};{1e-9 * np.exp(v / 0.05)}" for v in np.linspace(0.1, 0.5, 8))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    texts = " ".join(w.text for w in report.warnings)
    assert report.ok and "обратной ветви" in texts and "Мало точек" in texts


def test_current_in_milliamps_without_header_warns(tmp_path):
    V, I, _ = iv_lines()
    text = "\n".join(f"{v};{1e6 * i}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert any("мА" in w.hint for w in report.warnings)


def test_current_density_is_error(tmp_path):
    V, I, _ = iv_lines()
    text = "V;J (A/cm2)\n" + "\n".join(f"{v};{i}" for v, i in zip(V, I))
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert not report.ok and "плотность" in report.errors[0].text


def test_repeated_voltages_and_bad_rows(tmp_path):
    V, I, lines = iv_lines()
    text = "\n".join(lines + ["ошибка прибора;;"] + lines[::-1])
    report = df.analyze(write(tmp_path, "iv.csv", text))
    texts = " ".join(w.text for w in report.warnings)
    assert report.ok and "повторяются" in texts and "не прочитаны" in texts


def test_no_rectification_warns(tmp_path):
    V = np.linspace(-1, 1, 41)
    text = "\n".join(f"{v};{v / 1e3 * (1 + (v < 0))}" for v in V)
    report = df.analyze(write(tmp_path, "iv.csv", text))
    assert any("выпрямления нет" in w.text for w in report.warnings)


# ------------------------------------------------------------ проверки ВФХ
def test_cv_units_and_nonpositive(tmp_path):
    V = np.linspace(-3, 0, 31)
    text = "V;C, пФ\n" + "\n".join(f"{v};{10 / np.sqrt(1 - v) if v < -1 else -1}" for v in V)
    report = df.analyze(write(tmp_path, "cv.csv", text), df.KIND_CV)
    assert report.ok and report.value.max() < 1e-10
    assert any("≤ 0" in w.text for w in report.warnings)


def test_cv_file_loaded_as_iv_warns(tmp_path):
    text = "V;C (pF)\n" + "\n".join(f"{v};{10 + v}" for v in np.linspace(-3, 0, 31))
    report = df.analyze(write(tmp_path, "cv.csv", text), df.KIND_IV)
    assert any("ёмкость" in w.text for w in report.warnings)


# ------------------------------------------------------------ формат файла
def test_unsupported_and_missing_files(tmp_path):
    assert not df.analyze(tmp_path / "нет.csv").ok
    assert "не поддерживается" in df.analyze(write(tmp_path, "a.ods", "x")).errors[0].text
    binary = tmp_path / "a.dat"
    binary.write_bytes(b"\x00\x01\x02" * 100)
    assert "двоичный" in df.analyze(binary).errors[0].text
    assert "ни одной строки" in df.analyze(write(tmp_path, "e.csv", "V;I\nтекст;текст\n")).errors[0].text


def test_xls_without_xlrd_gives_hint(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xlrd":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = tmp_path / "old.xls"
    path.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    report = df.analyze(path)
    assert not report.ok and ".xlsx" in report.errors[0].hint


def test_excel_picks_sheet_with_data(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    V, I, _ = iv_lines()
    book = openpyxl.Workbook()
    book.active.title = "Описание"
    book.active.append(["образец", "пример"])
    sheet = book.create_sheet("Данные")
    sheet.append(["V", "I, nA"])
    for v, i in zip(V, I):
        sheet.append([float(v), f"{1e9 * i:.6g}".replace(".", ",")])   # текст-число с запятой
    path = tmp_path / "iv.xlsx"
    book.save(path)
    report = df.analyze(path)
    assert report.ok and report.sheet == "Данные"
    np.testing.assert_allclose(report.value, I, rtol=1e-5)


def test_broken_excel_is_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"not a zip")
    report = df.analyze(path)
    assert not report.ok and "не открывается" in report.errors[0].text


def test_every_issue_refers_to_metodichka(tmp_path):
    report = df.analyze(write(tmp_path, "e.csv", "V;I\n"))
    assert all(i.ref for i in report.issues)
    assert "методичка" in report.text()


def test_load_xy_file_raises_value_error_with_hint(tmp_path):
    with pytest.raises(ValueError, match="Что сделать"):
        load_xy_file(write(tmp_path, "e.csv", "V;I\n"))
    V, I, lines = iv_lines()
    v, i = load_xy_file(write(tmp_path, "ok.csv", "\n".join(lines)))
    np.testing.assert_allclose(i, I, rtol=1e-5)


def test_column_recognition():
    assert df.column_quantity("Напряжение, В") == "voltage"
    assert df.column_quantity("I (мкА)") == "current"
    assert df.column_quantity("C [pF]") == "capacitance"
    assert df.column_scale("I (мкА)", "current") == pytest.approx(1e-6)
    assert df.column_scale("V, mV", "voltage") == pytest.approx(1e-3)
    assert df.column_scale("I", "current") is None
    assert df.parse_number("1,5e-3") == pytest.approx(1.5e-3)
    assert df.parse_number("−2") == -2
    assert df.parse_number("abc") is None
