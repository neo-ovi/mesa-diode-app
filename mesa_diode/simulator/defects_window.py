# -*- coding: utf-8 -*-
"""Окно «Методы дефектов»: сопоставление травления (EPD), АСМ и XRD.

Расчёты — в defects.py; здесь только вёрстка. Окно открывается из меню
«Данные» и с панели инструментов; текущий образец берётся из полей
основного окна (N_dis — ямки травления, данные АСМ и XRD), другие образцы —
из наборов каталога данных или из таблицы (CSV, TXT, Excel).
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mesa_diode.simulator import datafile, defects, hints, presets
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.widgets import Tooltip

FONT = ("Segoe UI", 9)
CURRENT = "текущий образец"
EXAMPLE = datafile.EXAMPLES_DIR / "defects_example.csv"

INTRO = ("Сопоставление разрушающего метода — ямок травления (EPD) — с неразрушающими: полушириной кривой "
         "качания XRD и АСМ. XRD пересчитывается в плотность дислокаций по (7.1), (7.2) [KKA56]; по "
         "нескольким образцам строится калибровка EPD = c·Xᵏ, чтобы дальше оценивать образцы без травления. "
         + hints.reference("defects"))


class DefectsWindow(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Методы контроля дефектов: травление, АСМ, XRD")
        self.geometry("1250x820")
        self.others = []                     # образцы из наборов и таблиц
        self.x_key = tk.StringVar(value="xrd_density")
        self._build()
        self.refresh()

    # ---------------------------------------------------------- вёрстка
    def _build(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text=hints.plain(INTRO), wraplength=1200, justify="left", font=FONT).pack(anchor="w")

        bar = ttk.Frame(self, padding=(8, 0))
        bar.pack(fill=tk.X)
        for text, command, tip in (
                ("Добавить таблицу образцов...", self.add_table,
                 "CSV, TXT или Excel: столбцы «образец», «EPD», «АСМ», «RMS», «FWHM», «прибор», «естеств.»."),
                ("Добавить наборы каталога данных", self.add_presets,
                 "Все наборы образцов из MESA_DATA_DIR/samples."),
                ("Открыть пример таблицы", self.open_example, "Синтетическая таблица образцов для проверки."),
                ("Очистить список", self.clear, "Оставить только текущий образец."),
                ("Обновить", self.refresh, "Перечитать поля основного окна.")):
            button = ttk.Button(bar, text=text, command=command)
            button.pack(side=tk.LEFT, padx=(0, 4), pady=4)
            Tooltip(button, tip)
        ttk.Label(bar, text="  Калибровка EPD по:", font=FONT).pack(side=tk.LEFT)
        box = ttk.Combobox(bar, state="readonly", width=30,
                           values=[label for label, _g in defects.X_VALUES.values()])
        box.set(defects.X_VALUES[self.x_key.get()][0])
        box.pack(side=tk.LEFT, padx=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self._set_x(box.get()))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        ttk.Label(left, text="Текущий образец: методы", font=(FONT[0], 9, "bold")).pack(anchor="w")
        self.methods = ttk.Treeview(left, columns=("n", "ratio", "tau"), height=4)
        for col, text, width in (("#0", "метод", 230), ("n", "плотность, см⁻²", 130),
                                 ("ratio", "к EPD", 90), ("tau", "τ_dis (5.10), с", 120)):
            self.methods.heading(col, text=text)
            self.methods.column(col, width=width, anchor="w" if col == "#0" else "center")
        self.methods.pack(fill=tk.X)

        ttk.Label(left, text="Образцы", font=(FONT[0], 9, "bold")).pack(anchor="w", pady=(6, 0))
        cols = ("epd", "afm", "rms", "fwhm", "nx", "r")
        self.table = ttk.Treeview(left, columns=cols, height=8)
        for col, text, width in (("#0", "образец", 150), ("epd", "EPD", 80), ("afm", "АСМ", 80),
                                 ("rms", "RMS, нм", 65), ("fwhm", "Δω₁/₂, ″", 65), ("nx", "N_XRD", 85),
                                 ("r", "EPD/N_XRD", 80)):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor="w" if col == "#0" else "center")
        self.table.pack(fill=tk.X)

        ttk.Label(left, text="Выводы", font=(FONT[0], 9, "bold")).pack(anchor="w", pady=(6, 0))
        self.text = tk.Text(left, wrap="word", height=14, font=FONT, relief="flat",
                            background=ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0")
        self.text.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5.2, 5.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------------------------------------------------------- данные
    def _current(self):
        try:
            params = self.app._read_params()
        except ValueError:
            params = dict(self.app.preset.params)
        return defects.sample_from_params(CURRENT, params)

    def _sigma_R(self):
        structures = getattr(self.app, "structures", None)
        if structures:
            return ph.scr_sigma_R(next(iter(structures.values())))
        return presets.to_structure(presets.DEFAULT_PARAMS).sigma_R_sub

    def _set_x(self, label):
        self.x_key.set(next(k for k, (lab, _g) in defects.X_VALUES.items() if lab == label))
        self.refresh()

    def add_table(self, path=None):
        path = path or filedialog.askopenfilename(parent=self, title="Таблица образцов",
                                                  filetypes=datafile.FILE_TYPES)
        if not path:
            return
        try:
            self.others += defects.samples_from_table(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Таблица образцов", hints.with_error_reference(
                f"{Path(path).name}: {e}"), parent=self)
            return
        self.refresh()

    def open_example(self):
        self.add_table(EXAMPLE)

    def add_presets(self):
        found = 0
        current = getattr(self.app.preset, "path", None)
        names = {s.name for s in self.others}
        for path in presets.data_presets():
            try:
                preset = presets.load_preset(path)
            except (OSError, ValueError):
                continue
            if (current is not None and Path(path) == Path(current)) or preset.name in names:
                continue                     # текущий образец уже в списке
            sample = defects.sample_from_params(preset.name, preset.params)
            if sample.densities():
                self.others.append(sample)
                found += 1
        if not found:
            messagebox.showinfo("Наборы", "В каталоге данных нет других наборов с данными EPD, АСМ или "
                                          "XRD (текущий образец уже в списке; или MESA_DATA_DIR не задан).",
                                parent=self)
        self.refresh()

    def clear(self):
        self.others = []
        self.refresh()

    # ---------------------------------------------------------- вывод
    def refresh(self):
        current = self._current()
        samples = [current] + self.others
        sigma_R = self._sigma_R()
        self._fill_methods(current, sigma_R)
        self._fill_table(samples)
        calib = defects.calibrate(samples, self.x_key.get())
        self._write_notes(current, samples, calib, sigma_R)
        self._plot(samples, calib)

    def _fill_methods(self, sample, sigma_R):
        self.methods.delete(*self.methods.get_children())
        d = sample.densities()
        tau = defects.lifetime_by_method(sample, sigma_R)
        epd = d.get(defects.METHOD_EPD)
        for method, label in defects.METHOD_LABELS.items():
            if method not in d:
                continue
            ratio = f"{d[method] / epd:.3g}" if epd else "—"
            self.methods.insert("", "end", text=label,
                                values=(f"{d[method]:.3g}", ratio, f"{tau[method]:.3g}"))

    def _fill_table(self, samples):
        self.table.delete(*self.table.get_children())

        def fmt(v, spec=".3g"):
            return "—" if v in (None, 0) or v != v else format(v, spec)

        for s in samples:
            nx = s.densities().get(defects.METHOD_XRD)
            ratio = s.epd / nx if s.epd and nx else None
            self.table.insert("", "end", text=s.name, values=(
                fmt(s.epd), fmt(s.afm_density), fmt(s.afm_rms), fmt(s.xrd_fwhm, "g"), fmt(nx), fmt(ratio)))

    def _write_notes(self, current, samples, calib, sigma_R):
        lines = defects.compare(current)["notes"]
        tau = defects.lifetime_by_method(current, sigma_R)
        if tau:
            worst = min(tau.values())
            model_tau = None
            if getattr(self.app, "structures", None):
                model_tau = ph.scr_lifetime(next(iter(self.app.structures.values())))
            text = (f"Время жизни, ограниченное дислокациями (5.10), при σ_R = {sigma_R:.2g} см²/с: "
                    f"не меньше {worst:.3g} с при любой из оценок плотности.")
            if model_tau:
                text += (f" В модели τ₀ = {model_tau:.3g} с: " + (
                    "дислокации на ток не влияют при любой оценке — расхождение методов для ВАХ не важно."
                    if worst > 100 * model_tau else
                    "дислокации заметно влияют на ток — выбор метода плотности важен для ВАХ."))
            lines.append(text)
        label = defects.X_VALUES[self.x_key.get()][0]
        if calib is None:
            lines.append(f"Калибровка по «{label}»: нет образцов, где есть и EPD, и эта величина.")
        elif calib.n == 1:
            lines.append(f"Калибровка по «{label}»: один образец — только множитель EPD = {calib.ratio:.3g}·X. "
                         "Для степенной зависимости нужны хотя бы три образца с разной плотностью дефектов.")
        else:
            text = (f"Калибровка по «{label}» ({calib.n} образцов): EPD = 10^{calib.intercept:.2f}·X^{calib.slope:.2f}")
            if np.isfinite(calib.slope_err):
                text += f" (k = {calib.slope:.2f} ± {calib.slope_err:.2f})"
            if np.isfinite(calib.r2):
                text += f", R² = {calib.r2:.2f}"
            text += "."
            if self.x_key.get() == "xrd_density" and np.isfinite(calib.slope):
                text += (" По [KKA56] ожидается k ≈ 1 и EPD/N_XRD от 1/3 до 1." if abs(calib.slope - 1) < 0.3
                         else " k заметно отличается от 1: связь не пропорциональна — проверьте поправки XRD и "
                              "одинаковость травления.")
            lines.append(text)
        if calib is not None and defects.usable_for_calibration(current, self.x_key.get()):
            x = defects.X_VALUES[self.x_key.get()][1](current)
            if x:
                lines.append(f"Оценка по калибровке для текущего образца: EPD ≈ {calib.predict(x):.3g} см⁻² "
                             "(без травления).")
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join("• " + hints.plain(line) for line in lines) or "Нет данных.")
        self.text.configure(state="disabled")

    def _plot(self, samples, calib):
        ax = self.ax
        ax.clear()
        label, getter = defects.X_VALUES[self.x_key.get()]
        pts = [(s.name, getter(s), s.epd, defects.usable_for_calibration(s, self.x_key.get()))
               for s in samples if getter(s) and s.epd]
        if pts:
            x = np.array([p[1] for p in pts], dtype=float)
            y = np.array([p[2] for p in pts], dtype=float)
            ok = np.array([p[3] for p in pts], dtype=bool)
            if ok.any():
                ax.loglog(x[ok], y[ok], "o", color="#1f6fb2", label="образцы")
            if (~ok).any():
                ax.loglog(x[~ok], y[~ok], "o", color="#1f6fb2", markerfacecolor="none",
                          label="XRD без поправок или ниже предела (7.3) — не в калибровке")
            for name, xi, yi, _ok in pts:
                ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
            lo = min(x.min(), y.min()) / 3
            hi = max(x.max(), y.max()) * 3
            grid = np.logspace(np.log10(lo), np.log10(hi), 50)
            if self.x_key.get() == "xrd_density":
                ax.loglog(grid, grid, "--", color="#555555", linewidth=0.8, label="EPD = N_XRD")
                ax.loglog(grid, defects.EPD_TO_XRD_111 * grid, ":", color="#555555", linewidth=0.8,
                          label="EPD = N_XRD/3 [KKA56]")
            if calib is not None and calib.n >= 2 and np.isfinite(calib.slope):
                gx = np.logspace(np.log10(x[ok].min()), np.log10(x[ok].max()), 30)
                ax.loglog(gx, [calib.predict(v) for v in gx], "-", color="#c0392b", label="калибровка")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "нет образцов с EPD и выбранной величиной", ha="center", va="center",
                    transform=ax.transAxes, color="#777777")
        ax.set_xlabel(label)
        ax.set_ylabel("EPD, см⁻²")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title("Ямки травления против неразрушающего метода", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def open_defects_window(app):
    window = getattr(app, "defects_window", None)
    if window is not None and window.winfo_exists():
        window.refresh()
        window.lift()
        return window
    app.defects_window = DefectsWindow(app)
    return app.defects_window
