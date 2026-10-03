# -*- coding: utf-8 -*-
import json

import pytest

from mesa_diode import config
from mesa_diode.simulator import desktop, presets, tutorial


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    """Настройки пользователя (~/.mesa_diode: папка с данными, масштаб окна) не влияют
    на тесты: каталог данных — только из MESA_DATA_DIR."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "user" / "config.json")
    monkeypatch.setattr(desktop, "SETTINGS_PATH", tmp_path / "user" / "settings.json")
    monkeypatch.setattr(tutorial, "SETTINGS_PATH", tmp_path / "user" / "settings.json")


@pytest.fixture
def validation_preset():
    """Набор образца из каталога данных, на котором проверяются эталоны §9 ТЗ
    ("metadata": {"validation": "section9"}); без каталога данных — пропуск.
    Опорный образец программы — модельная структура, а не этот набор."""
    for path in presets.data_presets():
        try:
            preset = presets.load_preset(path)
        except (ValueError, json.JSONDecodeError):
            continue
        if preset.metadata.get("validation") == "section9":
            return preset
    pytest.skip("в каталоге данных нет набора для проверки эталонов §9: задайте MESA_DATA_DIR")
