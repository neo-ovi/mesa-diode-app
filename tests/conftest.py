# -*- coding: utf-8 -*-
import json

import pytest

from mesa_diode.simulator import presets


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
