# -*- mode: python ; coding: utf-8 -*-
# Спецификация PyInstaller: программа одним файлом — MesaSimulator.exe (Windows)
# или MesaSimulator (Linux).
#
# Запускать из КОРНЯ репозитория:
#   pyinstaller packaging/mesa_simulator.spec
#
# Готовый файл появится в dist/MesaSimulator(.exe). С MESA_ONEDIR=1 — папка
# dist/MesaSimulator/ (исполняемый файл и _internal): она быстрее запускается
# (ничего не распаковывает) и в архиве xz почти вдвое меньше одного файла.
# Подробности — в mesa_diode/simulator/README.md. Linux: собирать на самой
# старой системе, на которой программа должна работать (Ubuntu 22.04 → Linux
# Mint 21 и новее): packaging/build_linux.sh делает это в Docker.

import os
import re
import sys
from pathlib import Path

block_cipher = None
project_root = Path.cwd()
linux = sys.platform.startswith("linux")
onedir = os.environ.get("MESA_ONEDIR") == "1"

# Linux: библиотеки, которые программа берёт из системы, а не из сборки. Шрифты
# (fontconfig, freetype) должны совпадать с настройкой шрифтов системы, libX11
# и libxcb — с X-сервером; libstdc++ и libgcc_s системы не старше, чем в сборке
# (Ubuntu 22.04), а более новым системным библиотекам нужны именно они. Всё это
# есть в любой системе с рабочим столом. Tk, Tcl, libXft, libXss, libXrender
# остаются в сборке: Tk собран с Xft (сглаженные шрифты), а libXft и libXss есть
# не везде. Только точные имена системных библиотек: свои копии пакетов
# (pillow.libs/libxcb-ad31f5a3.so…) нужны. libBLT тоже остаётся: с ней
# слинкован _tkinter Ubuntu.
SYSTEM_LIBS = re.compile(r"(libX11|libX11-xcb|libxcb|libXau|libXdmcp|libfontconfig|libfreetype|libexpat|libz"
                         r"|libpng16|libbrotli(common|dec|enc)|libbsd|libmd|libuuid|libstdc\+\+|libgcc_s)"
                         r"\.so(\..*)?")

# Что программе не нужно (сборка меньше на 15–20 МБ): средства разработки и
# другие оконные библиотеки; PyYAML (нужен только show_config numpy и scipy);
# readline и OpenSSL (сеть и интерактивная консоль); форматы Pillow, которыми
# matplotlib не пользуется (AVIF, WebP, ICC-профили, ImageMath).
EXCLUDES = ["pytest", "_pytest", "pluggy", "iniconfig", "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6",
            "gi", "wx", "yaml", "_yaml", "readline", "ssl", "_ssl", "_hashlib",
            "PIL._avif", "PIL.AvifImagePlugin", "PIL._webp", "PIL.WebPImagePlugin",
            "PIL._imagingcms", "PIL.ImageCms", "PIL._imagingmath", "PIL.ImageMath"]

a = Analysis(
    [str(project_root / "scripts" / "run_simulator.py")],
    pathex=[str(project_root)],
    binaries=[],
    # примеры файлов данных (Данные → Формат файлов данных)
    datas=[(str(project_root / "mesa_diode" / "simulator" / "examples"), "mesa_diode/simulator/examples")],
    hiddenimports=[
        "matplotlib.backends.backend_tkagg",
        "openpyxl",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
if linux:
    a.binaries = [entry for entry in a.binaries if not SYSTEM_LIBS.fullmatch(Path(entry[0]).name)]
a.datas = [entry for entry in a.datas if "mpl-data/sample_data" not in entry[0].replace("\\", "/")]
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_options = dict(
    name="MesaSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not linux,          # UPX в Linux портит разделяемые библиотеки
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if onedir:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_options)
    coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=not linux, upx_exclude=[],
                   name="MesaSimulator")
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [], runtime_tmpdir=None, **exe_options)
