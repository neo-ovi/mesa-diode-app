# -*- coding: utf-8 -*-
"""Поиск методички (ТЗ §8.1): без файла — None, по переменным — верный путь."""

import json

from mesa_diode.simulator import help as hp


def _docs(tmp_path, *names):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"x")
    return folder


def test_missing_file_returns_none(tmp_path):
    assert hp.find_metodichka("pdf", config_path=tmp_path / "none.json", environ={}) is None


def test_docs_environment_variable(tmp_path):
    folder = _docs(tmp_path, "METODICHKA.pdf", "METODICHKA.docx")
    env = {hp.DOCS_ENV: str(folder)}
    assert hp.find_metodichka("pdf", tmp_path / "c.json", env) == folder / "METODICHKA.pdf"
    assert hp.find_metodichka("docx", tmp_path / "c.json", env) == folder / "METODICHKA.docx"


def test_data_dir_docs_fallback(tmp_path):
    folder = _docs(tmp_path, "METODICHKA.pdf")
    env = {hp.config.ENV_VAR: str(tmp_path)}
    assert hp.find_metodichka("pdf", tmp_path / "c.json", env) == folder / "METODICHKA.pdf"
    assert hp.find_metodichka("docx", tmp_path / "c.json", env) is None


def test_config_has_priority_and_is_saved(tmp_path):
    folder = _docs(tmp_path, "METODICHKA.pdf", "METODICHKA.docx")
    other = tmp_path / "other"
    other.mkdir()
    (other / "METODICHKA.pdf").write_bytes(b"y")
    config_path = tmp_path / "cfg" / "config.json"
    hp.save_metodichka_path(other / "METODICHKA.pdf", config_path)
    assert json.loads(config_path.read_text(encoding="utf-8"))[hp.CONFIG_KEY].endswith("METODICHKA.pdf")
    env = {hp.DOCS_ENV: str(folder)}
    assert hp.find_metodichka("pdf", config_path, env) == other / "METODICHKA.pdf"
    # Word рядом с указанным PDF нет — берётся следующий источник
    assert hp.find_metodichka("docx", config_path, env) == folder / "METODICHKA.docx"


def test_broken_config_is_ignored(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{не json", encoding="utf-8")
    assert hp.find_metodichka("pdf", config_path, {}) is None
