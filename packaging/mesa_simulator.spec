# -*- mode: python ; coding: utf-8 -*-
# Спецификация PyInstaller для сборки MesaSimulator.exe.
#
# Запускать из КОРНЯ репозитория:
#   pyinstaller packaging/mesa_simulator.spec
#
# Готовый файл появится в dist/MesaSimulator(.exe).
# Подробности и альтернативная однострочная команда — в
# mesa_diode/simulator/README.md.

import sys
from pathlib import Path

block_cipher = None
project_root = Path.cwd()

a = Analysis(
    [str(project_root / "scripts" / "run_simulator.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "matplotlib.backends.backend_tkagg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MesaSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
