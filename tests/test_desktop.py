# -*- coding: utf-8 -*-
"""Окно на разных системах (desktop.py), папка с данными (config.py) и проверка
окружения --check — без окна Tk."""

import pytest

from mesa_diode import config
from mesa_diode.simulator import desktop
from mesa_diode.simulator import help as hp


def test_px_follows_scale(monkeypatch):
    monkeypatch.setattr(desktop, "SCALE", 1.0)
    assert desktop.px(380) == 380
    monkeypatch.setattr(desktop, "SCALE", 1.5)
    assert desktop.px(380) == 570


def test_parse_xft_dpi():
    assert desktop.parse_xft_dpi("Xft.antialias:\t1\nXft.dpi:\t120\n") == 120.0
    assert desktop.parse_xft_dpi("Xcursor.size: 24") is None
    assert desktop.parse_xft_dpi("Xft.dpi: abc") is None
    assert desktop.parse_xft_dpi("") is None


@pytest.mark.parametrize("environ, settings, dpi, expected", [
    ({}, {}, None, (1.0, "default")),
    ({}, {}, 144.0, (1.5, "xft")),                                   # «Масштаб интерфейса» 150 %
    ({}, {"ui_scale": 1.25}, 192.0, (1.25, "settings")),             # настройка главнее рабочего стола
    ({"MESA_UI_SCALE": "2"}, {"ui_scale": 1.25}, 96.0, (2.0, "env")),  # переменная главнее всего
    ({"MESA_UI_SCALE": "125%"}, {}, None, (1.25, "env")),
    ({"MESA_UI_SCALE": "1,5"}, {}, None, (1.5, "env")),
    ({"MESA_UI_SCALE": "auto"}, {}, 120.0, (1.25, "xft")),
    ({"MESA_UI_SCALE": "крупно"}, {}, None, (1.0, "default")),
    ({}, {}, 960.0, (3.0, "xft")),                                   # предел сверху
])
def test_choose_scale(environ, settings, dpi, expected):
    assert desktop.choose_scale(environ, settings, dpi) == expected


def test_desktop_dpi_is_read_only_when_needed():
    calls = []

    def reader():
        calls.append(1)
        return 120.0

    assert desktop.choose_scale({"MESA_UI_SCALE": "1"}, {}, reader) == (1.0, "env")
    assert not calls
    assert desktop.choose_scale({}, {}, reader) == (1.25, "xft")
    assert calls


def test_settings_round_trip(tmp_path):
    path = tmp_path / "s" / "settings.json"
    assert desktop.read_settings(path) == {}
    assert desktop.write_setting("ui_scale", 1.5, path)
    assert desktop.write_setting("tutorial_done", True, path)
    assert desktop.read_settings(path) == {"ui_scale": 1.5, "tutorial_done": True}
    desktop.write_setting("ui_scale", None, path)
    assert desktop.read_settings(path) == {"tutorial_done": True}
    path.write_text("[1, 2]", encoding="utf-8")
    assert desktop.read_settings(path) == {}


def test_file_types_case_insensitive_on_linux():
    types = [("Данные", "*.csv *.xlsx"), ("Все файлы", "*.*")]
    assert desktop.file_types(types, linux=True) == [("Данные", "*.csv *.xlsx *.CSV *.XLSX"),
                                                     ("Все файлы", "*")]
    assert desktop.file_types(types, linux=False) == types


def test_system_env_restores_library_path():
    """xdg-open из собранной программы — с исходным LD_LIBRARY_PATH, без переменных сборки."""
    frozen_env = {"LD_LIBRARY_PATH": "/tmp/_MEI1:/opt/lib", "LD_LIBRARY_PATH_ORIG": "/opt/lib",
                  "TCL_LIBRARY": "/tmp/_MEI1/_tcl_data", "_PYI_ARCHIVE_FILE": "/x", "HOME": "/home/u"}
    env = desktop.system_env(frozen_env, frozen=True, bundle="/tmp/_MEI1")
    assert env["LD_LIBRARY_PATH"] == "/opt/lib" and "LD_LIBRARY_PATH_ORIG" not in env
    assert "TCL_LIBRARY" not in env and "_PYI_ARCHIVE_FILE" not in env and env["HOME"] == "/home/u"
    env = desktop.system_env({"LD_LIBRARY_PATH": "/tmp/_MEI1"}, frozen=True, bundle="/tmp/_MEI1")
    assert "LD_LIBRARY_PATH" not in env                       # до запуска переменной не было
    assert desktop.system_env(frozen_env, frozen=False) == frozen_env     # из исходников — как есть


def test_keycode_event_only_for_non_latin_layouts():
    assert desktop.keycode_event("Cyrillic_em", 55) == "<<Paste>>"       # Ctrl+V на русской раскладке
    assert desktop.keycode_event("Cyrillic_es", 54) == "<<Copy>>"
    assert desktop.keycode_event("v", 55) is None             # латиница — стандартная привязка Tk
    assert desktop.keycode_event("Cyrillic_a", 10) is None    # не клавиша правки


def test_desktop_entry_quotes_exec_and_names_window_class(tmp_path):
    text = desktop.desktop_entry(["/home/u/Мои программы/MesaSimulator", "--x=100%"], tmp_path / "i.png")
    exec_line = next(line for line in text.splitlines() if line.startswith("Exec="))
    assert exec_line == 'Exec="/home/u/Мои программы/MesaSimulator" --x=100%%'
    assert f"StartupWMClass={desktop.WM_CLASS}" in text and "Terminal=false" in text
    assert desktop._exec_arg('a"b') == '"a\\\\"b"'     # \" — для Exec, затем \\ — общее правило строк


def test_menu_entry_paths_follow_xdg(tmp_path):
    entry, icon = desktop.menu_entry_paths({"XDG_DATA_HOME": str(tmp_path)})
    assert entry == tmp_path / "applications" / "mesa-simulator.desktop"
    assert icon.parent == tmp_path / "icons" / "hicolor" / "128x128" / "apps"


def test_removable_drive():
    assert desktop.on_removable_drive("/media/user/FLASH/MesaSimulator")
    assert not desktop.on_removable_drive("/home/u/MesaSimulator")


def test_icon_pixels():
    assert desktop._icon_pixel(0.5, 0.5) is None              # скруглённый угол — прозрачный
    assert desktop._icon_pixel(32, 32) == "#ffffff"           # символ диода
    assert desktop._icon_pixel(32, 5) == "#1f4fb2"            # фон


def test_data_dir_from_settings(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="Настройки → Папка с данными"):
        config.data_dir()
    folder = tmp_path / "data"
    (folder / "samples").mkdir(parents=True)
    config.save_data_dir(folder)
    assert config.data_dir() == folder
    assert config.data_dir_setting() == (folder, "settings")
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))         # MESA_DATA_DIR главнее выбора в программе
    assert config.data_dir_setting() == (tmp_path, "env")


def test_missing_saved_data_dir_is_explained(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    config.save_data_dir(tmp_path / "нет такой")
    with pytest.raises(RuntimeError, match="Укажите её заново"):
        config.data_dir()


def test_metodichka_found_in_saved_data_dir(tmp_path):
    cfg = tmp_path / "cfg.json"
    docs = tmp_path / "data" / "docs"
    docs.mkdir(parents=True)
    (docs / "METODICHKA.pdf").write_bytes(b"x")
    config.save_data_dir(tmp_path / "data", cfg)
    assert hp.find_metodichka("pdf", cfg, {}) == docs / "METODICHKA.pdf"
    assert hp.find_metodichka("docx", cfg, {}) is None


def test_selfcheck_reports_calculation(monkeypatch):
    from mesa_diode.simulator import selfcheck

    monkeypatch.setattr(selfcheck, "_check_tk", lambda out, problems: None)   # без окна
    lines = []
    assert selfcheck.run(lines.append) == 0
    text = "\n".join(lines)
    assert "Расчёт и подгонка" in text and "работают" in text and text.endswith("Всё работает.")
    assert "iv_example.xlsx" in text                         # Excel (openpyxl) читается
