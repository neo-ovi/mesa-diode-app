# -*- coding: utf-8 -*-
"""Проверка окружения: `MesaSimulator --check` (из исходников —
`python scripts/run_simulator.py --check`); методичка, п. 1А.22.

Печатает версию, систему, библиотеки, Tk (система шрифтов, выбранные шрифты,
масштаб), папку с данными и методичку; проверяет расчёт: ВАХ модельной
структуры, подгонку примера ВАХ обучения, чтение примеров файлов (CSV, Excel)
и рисование графика с кириллицей.
Окно не открывается; без графического сеанса пропускается только часть Tk.
Код выхода 0 — всё работает, 1 — есть ошибка (причина — в выводе)."""

import io
import os
import sys
import time

from mesa_diode import config
from mesa_diode.simulator import __version__, desktop, presets


def _line(out, label, value):
    out(f"{label:<24}{value}")


def _check_tk(out, problems):
    """Tk: версия, система окон и шрифтов, выбранные шрифты, масштаб, экран."""
    try:
        import tkinter as tk
    except ImportError as e:
        problems.append(f"нет tkinter: {e}")
        _line(out, "Tk", "НЕТ (в Linux: sudo apt install python3-tk)")
        return
    try:
        root = tk.Tk(className=desktop.WM_CLASS)
    except tk.TclError as e:
        _line(out, "Tk", f"{tk.TkVersion}; окно не открыть: {e}")
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            problems.append(f"Tk не открыл окно: {e}")
        return
    try:
        root.withdraw()
        try:
            fonts = root.tk.call("::tk::pkgconfig", "get", "fontsystem")
        except tk.TclError:
            fonts = "?"
        system = root.tk.call("tk", "windowingsystem")
        _line(out, "Tk", f"{root.tk.call('info', 'patchlevel')}; окна {system}, шрифты {fonts}")
        if system == "x11" and fonts != "xft":
            problems.append("Tk без Xft: шрифты будут без сглаживания")
        scale, source = desktop.setup(root)
        names = {"env": f"переменная {desktop.SCALE_ENV}", "settings": "настройка программы",
                 "xft": "Xft.dpi рабочего стола", "default": "по умолчанию"}
        _line(out, "Шрифты", f"{desktop.UI_FAMILY}; моноширинный — {desktop.MONO_FAMILY}")
        _line(out, "Масштаб", f"{round(100 * scale)} % ({names.get(source, source)})")
        _line(out, "Экран", f"{root.winfo_screenwidth()}×{root.winfo_screenheight()}, "
                            f"{root.winfo_fpixels('1i'):.0f} точек на дюйм")
    finally:
        root.destroy()


def _check_data(out):
    path, source = config.data_dir_setting()
    if path is None:
        _line(out, "Папка с данными", "не задана (работа на модельном образце)")
    else:
        where = f"переменная {config.ENV_VAR}" if source == "env" else "настройка программы"
        state = f"наборов: {len(presets.data_presets())}" if path.is_dir() else "НЕ НАЙДЕНА"
        _line(out, "Папка с данными", f"{path} ({where}); {state}")
    from mesa_diode.simulator import help as help_module
    found = help_module.find_metodichka("pdf")
    _line(out, "Методичка (PDF)", found or "не найдена (меню «Справка» спросит файл)")


def _check_calculation(out, problems):
    """ВАХ модельной структуры, подгонка примера обучения, график с кириллицей."""
    import numpy as np

    from mesa_diode.simulator import fitting, tutorial
    from mesa_diode.simulator import physics as ph

    try:
        start = time.perf_counter()
        params = {**presets.DEFAULT_PARAMS, "mode": presets.MODE_BASIC}
        s = presets.to_structure(params)
        V = np.linspace(-1.0, 0.6, 33)
        I = ph.solve_iv(s, V).I
        if not np.all(np.isfinite(I)):
            raise ValueError("ВАХ с нечисловыми значениями")
        V_ex, I_ex = tutorial.example_iv()
        result = fitting.fit_iv(s, V_ex, I_ex, model=presets.current_model(params),
                                terms=presets.MODE_TERMS[presets.MODE_BASIC])
        _line(out, "Расчёт и подгонка", f"работают: ошибка подгонки примера {100 * result.error:.1f} %, "
                                        f"{time.perf_counter() - start:.1f} с")
    except Exception as e:  # noqa: BLE001 — любая поломка расчёта — в отчёт
        problems.append(f"расчёт: {type(e).__name__}: {e}")
        _line(out, "Расчёт и подгонка", f"ОШИБКА: {type(e).__name__}: {e}")
    try:
        from mesa_diode.simulator import datafile

        names = []
        for name in ("iv_example.csv", "iv_example.xlsx"):
            report = datafile.analyze(datafile.EXAMPLES_DIR / name, datafile.KIND_IV)
            if not report.ok:
                raise ValueError(f"{name}: " + report.text(("error",)))
            names.append(f"{name} (точек: {len(report.voltage)})")
        _line(out, "Файлы измерений", "читаются: " + ", ".join(names))
    except Exception as e:  # noqa: BLE001
        problems.append(f"файлы: {type(e).__name__}: {e}")
        _line(out, "Файлы измерений", f"ОШИБКА: {type(e).__name__}: {e}")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(2, 1), dpi=50)
        FigureCanvasAgg(figure)
        figure.text(0.1, 0.5, "ВАХ: I, А; μ, τ₀")
        figure.savefig(io.BytesIO(), format="png")
        _line(out, "Графики", "matplotlib рисует (Agg)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"графики: {type(e).__name__}: {e}")
        _line(out, "Графики", f"ОШИБКА: {type(e).__name__}: {e}")


def run(out=print):
    """Печатает отчёт; возвращает 0, если всё работает, иначе 1."""
    import matplotlib
    import numpy
    import scipy

    problems = []
    _line(out, "Программа", f"{desktop.APP_NAME} {__version__}")
    _line(out, "Система", desktop.system_summary())
    _line(out, "Библиотеки", f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
                             f"matplotlib {matplotlib.__version__}")
    _check_tk(out, problems)
    _check_data(out)
    _check_calculation(out, problems)
    if problems:
        out("ЕСТЬ ПРОБЛЕМЫ:\n  " + "\n  ".join(problems))
        return 1
    out("Всё работает.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
