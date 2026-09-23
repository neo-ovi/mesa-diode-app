# -*- coding: utf-8 -*-
"""Основное окно симулятора мезадиода Ge (ТЗ §8).

В окне — только варьируемые параметры модели (§5.3); всё справочное —
во вкладке «Справочник» окна «Формулы и параметры». Физика — в
simulator/physics.py, наборы образцов — simulator/presets.py.

Запуск:        python scripts/run_simulator.py
Сборка в exe:  см. mesa_diode/simulator/README.md
"""

import math
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mesa_diode.simulator import physics as ph
from mesa_diode.simulator import presets
from mesa_diode.simulator import reference as ref
from mesa_diode.simulator import formulas as formulas_module
from mesa_diode.simulator import help as help_module
from mesa_diode.simulator import hints
from mesa_diode.simulator.widgets import Tooltip, bind_wheel, format_value, wheel_step
from mesa_diode.simulator.io import load_xy_file
from mesa_diode.simulator.diagram import draw_mesa_diagram
from mesa_diode.simulator.plot_utils import (
    adaptive_point_count, adaptive_voltage_range, auto_scale, robust_value_limits,
)
from mesa_diode.simulator.style import MAX_DATASETS, dataset_style, remaining_slots

UM = 1e-4
SCENARIO_COLORS = {ph.SCENARIO_A: "#1f4fb2", ph.SCENARIO_B: "#c0392b"}
SINGLE_COLOR = "#1f6fb2"
COMPONENT_STYLE = {
    "emp": ("I_emp (6.3)", "#17becf"),
    "diff": ("I_diff", "#2ca02c"),
    "gr": ("I_gr", "#ff7f0e"),
    "sh": ("I_sh", "#7f7f7f"),
    "L": ("I_L", "#9467bd"),
}
JS_COLOR = "#9b870c"
FONT = ("Segoe UI", 9)

# Группы полей: (заголовок, ключи, строка, столбец).
GROUPS = [
    ("Геометрия", ["D_um", "D_inner_um", "h_um", "d_epi_um", "d_n_um", "d_sub_um"], 0, 0),
    ("Легирование и температура", ["ND_plus", "N_i", "rho_sub", "T"], 0, 1),
    ("Материал и дефекты (таблицы, подбор)",
     ["mu_n", "mu_p_i", "mu_p_nplus", "sigma_R_epi", "sigma_R_sub",
      "tau_n_bg", "tau_p_bg", "tau0_bg"], 0, 2),
    ("Дефекты: измерения", ["N_dis", "afm_rms_nm", "afm_defects", "xrd_fwhm"], 0, 3),
    ("Ток и утечки", ["n_emp", "J0_emp", "Rs", "Rsh", "n2", "I_L", "m_leak"], 1, 0),
]
# Короткие подписи в основном окне; полные — в таблице параметров окна формул.
SHORT_LABELS = {
    "D_um": "D (мезы)", "D_inner_um": "d кольца (схема)", "d_epi_um": "d_epi",
    "h_um": "h (травление)", "d_n_um": "d_n (n⁺)", "d_sub_um": "d_sub",
    "ND_plus": "N_D⁺", "N_i": "N_i", "rho_sub": "ρ_sub", "T": "T",
    "mu_n": "μ_n (электроны)", "mu_p_i": "μ_p (i-слой)", "mu_p_nplus": "μ_p (n⁺)",
    "tau_n_bg": "τ_n^bg", "tau_p_bg": "τ_p^bg", "tau0_bg": "τ₀^bg (ОПЗ)",
    "N_dis": "N_dis", "sigma_R_epi": "σ_R (i, n⁺)", "sigma_R_sub": "σ_R (подложка)",
    "n2": "n₂ (ОПЗ)", "Rs": "R_s", "Rsh": "R_sh", "I_L": "I_L", "m_leak": "m",
    "n_emp": "n (6.3)", "J0_emp": "J₀ (6.3)",
    "afm_rms_nm": "RMS (АСМ)", "afm_defects": "дефекты (АСМ)", "xrd_fwhm": "FWHM (XRD)",
}
# Подписи кривых для галочек: (график, ключ, подпись).
CURVES = {
    "iv": [("total", "модель"), ("emp", "I_emp"), ("diff", "I_diff"), ("gr", "I_gr"),
           ("sh", "I_sh"), ("L", "I_L"), ("emp_fit", "(6.3) по n_эксп"), ("exp", "эксперимент")],
    "cv": [("model_C", "модель"), ("exp", "эксперимент"), ("fit_C", "прямые 1/C²")],
    "jv": [("total", "|J| модели"), ("emp", "J_emp"), ("diff", "J_diff"), ("gr", "J_gr"),
           ("sh", "J_sh"), ("L", "J_L"), ("Js0", "J_s(0)"), ("emp_fit", "(6.3) по n_эксп"),
           ("js_exp", "J(S) эксп."), ("n_exp", "n(V) эксп."), ("n_mod", "n(V) модели")],
}
# Кривые, которых нет в данной модели (компоненты тока различаются).
CURVES_BY_MODEL = {ph.MODEL_EMPIRICAL: {"diff", "gr", "emp_fit"}, ph.MODEL_PHYSICAL: {"emp"}}
POSITIVE_KEYS = ("D_um", "d_epi_um", "h_um", "d_n_um", "d_sub_um", "ND_plus", "N_i",
                 "rho_sub", "T", "mu_n", "mu_p_i", "mu_p_nplus", "tau_n_bg", "tau_p_bg",
                 "tau0_bg", "n2", "Rsh", "m_leak", "D_inner_um", "n_emp", "J0_emp")
NONNEGATIVE_KEYS = ("N_dis", "sigma_R_epi", "sigma_R_sub", "Rs", "I_L",
                    "afm_rms_nm", "afm_defects", "xrd_fwhm")
WHEEL_DELAY_MS = 400   # пересчёт после паузы в прокрутке колеса
SPECS = {spec.key: spec for spec in presets.NUMERIC_PARAMS}
# Подвижности, которые не входят в модель при данном сценарии (§5.3: «активны нужные»)
UNUSED_BY_SCENARIO = {ph.SCENARIO_A: {"mu_p_i", "sigma_R_sub"}, ph.SCENARIO_B: {"mu_p_nplus"}}
BOUNDARY_FIELDS = {
    ph.SCENARIO_A: [("bc_A_n", "A: верхний контакт"), ("bc_A_p", "A: граница i/подложка")],
    ph.SCENARIO_B: [("bc_B_n", "B: граница n/n⁺"), ("bc_B_p", "B: тыльный контакт")],
}
BOUNDARY_BY_LABEL = {label: key for key, label in ph.BOUNDARY_LABELS.items()}
TAU0_V = -1.0   # напряжение оценки τ₀ по обратной ветви, В


def _fmt(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:g}"


class MesaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Симулятор мезадиода Ge — ВАХ, ВФХ, J–V")
        self.geometry("1850x1150")
        self.minsize(1500, 950)

        self.preset = presets.startup_preset()
        self.exp_iv = []   # {"label", "path", "voltage", "value"}
        self.exp_cv = []

        self.vars = {key: tk.StringVar() for key in SPECS}
        self.i_type = tk.StringVar(value=presets.I_TYPE_N)
        self.substrate = tk.StringVar(value="Ge")
        self.bc_vars = {key: tk.StringVar() for key in presets.CHOICE_FIELDS}
        self.flag_vars = {key: tk.BooleanVar() for key in presets.FLAG_FIELDS}
        self.mode = tk.StringVar(value=presets.MODE_BASIC)
        self.cv_view = tk.StringVar(value="C")
        self.curve_vars = {(plot, key): tk.BooleanVar(value=True)
                           for plot, curves in CURVES.items() for key, _label in curves}
        self.window_auto = tk.BooleanVar(value=True)
        self.n_exp_var = tk.StringVar()
        self.V1_var = tk.StringVar()
        self.V2_var = tk.StringVar()
        self.c2_from_var = tk.StringVar(value="-1.0")
        self.c2_to_var = tk.StringVar(value="0.0")

        self.structures = {}
        self.results = {}
        self.ideality = None
        self.ideality_notes = []     # причина неудачи n_эксп и замечания
        self._take_from_iv = False   # подставить n и J₀ из новой ВАХ при пересчёте
        self._wheel_after = None
        self.model_ideality = None
        self.comparison = None
        self.reference_data = None
        self.formulas_window = None

        self._build_menu()
        self._build_figure_area()
        self._build_panel()
        self._build_curve_bar()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))
        self._apply_preset(self.preset)
        self.after(100, self.recompute)

    # ------------------------------------------------------------ меню
    def _build_menu(self):
        menubar = tk.Menu(self)

        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Загрузить эксперим. ВАХ (файл)...", command=self.load_experimental_iv)
        data_menu.add_command(label="Загрузить эксперим. ВФХ (файл)...", command=self.load_experimental_cv)
        data_menu.add_separator()
        data_menu.add_command(label="Очистить экспериментальные данные", command=self.clear_experimental)
        menubar.add_cascade(label="Данные", menu=data_menu)

        sets_menu = tk.Menu(menubar, tearoff=0)
        sets_menu.add_command(label="Сохранить набор...", command=self.save_preset)
        sets_menu.add_command(label="Загрузить набор...", command=self.load_preset)
        sets_menu.add_command(label="Сбросить к опорному", command=self.reset_to_reference)
        menubar.add_cascade(label="Наборы", menu=sets_menu)

        self.help_menu = tk.Menu(menubar, tearoff=0)
        self.help_menu.add_command(label="Формулы и параметры", command=self.open_formulas_window)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="Методичка (PDF)",
                                   command=lambda: help_module.open_metodichka(self, "pdf"))
        self.help_menu.add_command(label="Методичка (Word)",
                                   command=lambda: help_module.open_metodichka(self, "docx"))
        menubar.add_cascade(label="Справка", menu=self.help_menu)
        self.config(menu=menubar)

        quick = ttk.Frame(self)
        quick.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 0))
        for text, command in (("Загрузить ВАХ", self.load_experimental_iv),
                              ("Загрузить ВФХ", self.load_experimental_cv),
                              ("Очистить данные", self.clear_experimental),
                              ("Сохранить набор", self.save_preset),
                              ("Загрузить набор", self.load_preset),
                              ("Сбросить к опорному", self.reset_to_reference),
                              ("Формулы и параметры", self.open_formulas_window)):
            ttk.Button(quick, text=text, command=command).pack(side=tk.LEFT, padx=(0, 6))
        self.preset_label = ttk.Label(quick, text="", foreground="#555555")
        self.preset_label.pack(side=tk.LEFT, padx=(12, 0))

    # --------------------------------------------------------- графики
    def _build_figure_area(self):
        self.fig = Figure(figsize=(16, 4.1), dpi=100)
        gs = self.fig.add_gridspec(1, 3, wspace=0.42)
        self.fig.subplots_adjust(left=0.05, right=0.96, bottom=0.13, top=0.91)
        self.ax_iv = self.fig.add_subplot(gs[0, 0])
        self.ax_cv = self.fig.add_subplot(gs[0, 1])
        self.ax_jv = self.fig.add_subplot(gs[0, 2])
        self.ax_jv_n = self.ax_jv.twinx()

        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        ttk.Label(bar, text="График 2:", font=FONT).pack(side=tk.LEFT)
        for value, text in (("C", "ВФХ"), ("invC2", "1/C²")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.cv_view,
                            command=self.recompute).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(bar, text="  участок прямой 1/C²: от", font=FONT).pack(side=tk.LEFT)
        for var in (self.c2_from_var, self.c2_to_var):
            e = ttk.Entry(bar, textvariable=var, width=6, justify="center")
            e.pack(side=tk.LEFT, padx=2)
            e.bind("<Return>", lambda _e: self.recompute())
        ttk.Label(bar, text="В", font=FONT).pack(side=tk.LEFT)

        # Холст упаковывается после панели (см. __init__): иначе он, растягиваясь,
        # забирает высоту первым и панель с кнопкой «Рассчитать» обрезается.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

    def _build_curve_bar(self):
        """Галочки кривых под каждым графиком; подсказка — что это за кривая."""
        bar = ttk.Frame(self)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(2, 0))
        self.curve_buttons = {}
        per_row = 4
        for column, plot in enumerate(("iv", "cv", "jv")):
            bar.columnconfigure(column, weight=1, uniform="curves")
            frame = ttk.Frame(bar)
            frame.grid(row=0, column=column, sticky="nw", padx=(40, 0))
            for index, (key, label) in enumerate(CURVES[plot]):
                button = ttk.Checkbutton(frame, text=label, variable=self.curve_vars[(plot, key)],
                                         command=self._redraw)
                button.grid(row=index // per_row, column=index % per_row, sticky="w", padx=(0, 6))
                Tooltip(button, hints.plain(hints.CURVE_HINTS[key]))
                self.curve_buttons[(plot, key)] = button

    def _shown(self, plot, key):
        return self.curve_vars[(plot, key)].get()

    # ---------------------------------------------------------- панель
    def _build_panel(self):
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        # Схема мезы — отдельный холст справа: текст на ней фиксированного
        # размера в пунктах, поэтому ей нужна своя, не сжимаемая площадь
        # (упаковывается первой, чтобы панель её не вытесняла).
        self.mesa_fig = Figure(figsize=(5.0, 3.9), dpi=100)
        self.mesa_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
        self.ax_mesa = self.mesa_fig.add_subplot(111)
        self.mesa_canvas = FigureCanvasTkAgg(self.mesa_fig, master=bottom)
        self.mesa_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=(8, 0))
        outer = ttk.Frame(bottom)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Подложка:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Ge", value="Ge", variable=self.substrate).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Si", value="Si", variable=self.substrate,
                        state="disabled").pack(side=tk.LEFT)
        ttk.Label(top, text="(Si — бэклог Б-1: нужно сродство к электрону)",
                  foreground="#777777", font=FONT).pack(side=tk.LEFT, padx=(2, 18))
        ttk.Label(top, text="Тип i-слоя (знак Холла):", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        for value, text in ((presets.I_TYPE_N, "n (R_H < 0)"), (presets.I_TYPE_P, "p (R_H > 0)"),
                            (presets.I_TYPE_BOTH, "оба")):
            ttk.Radiobutton(top, text=text, value=value, variable=self.i_type,
                            command=self._on_scenario_change).pack(side=tk.LEFT, padx=(4, 0))

        grid = ttk.Frame(outer)
        grid.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        self.entries = {}
        for title, keys, grid_row, grid_col in GROUPS:
            box = ttk.LabelFrame(grid, text=title, padding=(6, 2))
            box.grid(row=grid_row, column=grid_col, sticky="nsew", padx=(0, 6), pady=(0, 4))
            for row, key in enumerate(keys):
                self._param_row(box, row, key)
            if "d_sub_um" in keys:
                self.d_i_label = ttk.Label(box, text="", foreground="#555555", font=FONT)
                self.d_i_label.grid(row=len(keys), column=0, columnspan=3, sticky="w")
            if "tau0_bg" in keys:
                self.tau0_button = ttk.Button(box, text="Оценить τ₀ по обратной ветви",
                                              command=self._estimate_tau0)
                self.tau0_button.grid(row=len(keys), column=0, columnspan=3, sticky="w", pady=(2, 0))
                Tooltip(self.tau0_button, hints.plain(
                    f"τ₀ из тока обратной ветви при V = {TAU0_V:g} В: из измеренного тока "
                    "вычитаются шунт V/R_{sh} и диффузионный ток модели, остаток обращается по (5.4); "
                    "τ₀^{bg} — по (5.10). Нужны загруженная ВАХ и режим «Подгонка»."))
            if "N_dis" in keys:
                ttk.Label(box, text="N_dis — по ямкам травления (EPD);\nостальное хранится в наборе",
                          foreground="#555555", font=FONT, justify="left").grid(
                    row=len(keys), column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._build_extra_boxes(grid)

        buttons = ttk.Frame(outer)
        buttons.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        ttk.Label(buttons, text="Режим:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        for mode in presets.MODES:
            button = ttk.Radiobutton(buttons, text=presets.MODE_LABELS[mode], value=mode,
                                     variable=self.mode, command=self._on_mode_change)
            button.pack(side=tk.LEFT, padx=(4, 0))
            Tooltip(button, hints.plain(hints.MODE_HINTS[mode]))
        ttk.Button(buttons, text="Рассчитать", command=self.recompute).pack(side=tk.LEFT, padx=(16, 8))
        self.status_lbl = ttk.Label(buttons, text="", font=("Segoe UI", 9, "bold"), wraplength=820)
        self.status_lbl.pack(side=tk.LEFT)

    def _param_row(self, box, row, key):
        spec = SPECS[key]
        hint = hints.plain(hints.PARAM_HINTS[key])
        label = ttk.Label(box, text=SHORT_LABELS[key], font=FONT)
        label.grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(box, textvariable=self.vars[key], width=9, justify="center")
        entry.grid(row=row, column=1, padx=3, pady=1)
        entry.bind("<Return>", lambda _e: self.recompute())
        bind_wheel(entry, lambda direction, fine, key=key: self._on_wheel(key, direction, fine))
        ttk.Label(box, text=spec.unit, font=FONT, foreground="#555555").grid(row=row, column=2, sticky="w")
        for widget in (label, entry):
            Tooltip(widget, f"{spec.label}.\n{hint}")
        self.entries[key] = entry

    def _on_wheel(self, key, direction, fine):
        """Колесо над полем: шаг — десятая часть старшего разряда (Ctrl — мельче);
        пересчёт — после паузы в прокрутке."""
        entry = self.entries[key]
        if entry.instate(["disabled"]):
            return
        try:
            value = float(self.vars[key].get().strip().replace(",", "."))
        except ValueError:
            return
        self.vars[key].set(format_value(wheel_step(value, direction, fine, key)))
        if self._wheel_after is not None:
            self.after_cancel(self._wheel_after)
        self._wheel_after = self.after(WHEEL_DELAY_MS, self._wheel_recompute)

    def _wheel_recompute(self):
        self._wheel_after = None
        self.recompute()

    def _build_extra_boxes(self, grid):
        box = ttk.LabelFrame(grid, text="Граничные условия и опции", padding=(6, 2))
        box.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=(0, 4))
        self.bc_widgets = {}
        bc_hint = "\n".join(f"{ph.BOUNDARY_LABELS[key]} — {hints.plain(text)}"
                            for key, text in hints.BOUNDARY_HINTS.items())
        row = 0
        for scenario in (ph.SCENARIO_A, ph.SCENARIO_B):
            for key, label in BOUNDARY_FIELDS[scenario]:
                lbl = ttk.Label(box, text=label, font=FONT)
                combo = ttk.Combobox(box, textvariable=self.bc_vars[key], width=13, state="readonly",
                                     values=list(ph.BOUNDARY_LABELS.values()))
                combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())
                lbl.grid(row=row, column=0, sticky="w")
                combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=3)
                for widget in (lbl, combo):
                    Tooltip(widget, bc_hint, wraplength=520)
                self.bc_widgets[key] = (lbl, combo)
                row += 1
        self.flag_buttons = {}
        for key, text in (("sns_refinement", "уточнение SNS (5.5)"),
                          ("edge_area", "краевая добавка площади ОПЗ (1.3)")):
            button = ttk.Checkbutton(box, text=text, variable=self.flag_vars[key], command=self.recompute)
            button.grid(row=row, column=0, columnspan=3, sticky="w")
            self.flag_buttons[key] = button
            row += 1

        box = ttk.LabelFrame(grid, text="Коэффициент идеальности (§6.3)", padding=(6, 2))
        box.grid(row=1, column=2, columnspan=2, sticky="nsew", padx=(0, 6), pady=(0, 4))
        row = 0
        ttk.Label(box, text="n_эксп", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w")
        n_entry = ttk.Entry(box, textvariable=self.n_exp_var, width=10, justify="center", state="readonly")
        n_entry.grid(row=row, column=1, padx=3, sticky="w")
        Tooltip(n_entry, "Коэффициент идеальности по прямой ветви первой загруженной ВАХ (§6.3). "
                         "Считается автоматически; при загрузке ВАХ подставляется в n (6.3).")
        row += 1
        ttk.Label(box, text="окно V₁ … V₂, В", font=FONT).grid(row=row, column=0, sticky="w")
        window = ttk.Frame(box)
        window.grid(row=row, column=1, columnspan=2, sticky="w", padx=3)
        for var in (self.V1_var, self.V2_var):
            e = ttk.Entry(window, textvariable=var, width=6, justify="center")
            e.pack(side=tk.LEFT, padx=(0, 2))
            e.bind("<KeyRelease>", lambda _e: self.window_auto.set(False))
            e.bind("<Return>", lambda _e: self.recompute())
        ttk.Checkbutton(window, text="авто", variable=self.window_auto,
                        command=self.recompute).pack(side=tk.LEFT)
        row += 1
        actions = ttk.Frame(box)
        actions.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        ttk.Button(actions, text="n (6.3) ← n_эксп", command=self._n_emp_from_experiment).pack(
            side=tk.LEFT, padx=(0, 6))
        self.n2_button = ttk.Button(actions, text="n₂ ← n_эксп", command=self._n2_from_experiment)
        self.n2_button.pack(side=tk.LEFT)
        row += 1
        self.ideality_lbl = ttk.Label(box, text="", foreground="#b35c00", font=FONT,
                                      wraplength=420, justify="left")
        self.ideality_lbl.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))

    # ---------------------------------------------------- наборы образцов
    def _apply_preset(self, preset):
        self.preset = preset
        params = {**presets.DEFAULT_PARAMS, **preset.params}
        for key, var in self.vars.items():
            var.set(_fmt(float(params[key])))
        for key, var in self.bc_vars.items():
            var.set(ph.BOUNDARY_LABELS[params[key]])
        for key, var in self.flag_vars.items():
            var.set(bool(params[key]))
        self.i_type.set(params["i_type"])
        self.substrate.set(params.get("substrate", "Ge"))
        self.mode.set(params["mode"] if params["mode"] in presets.MODES else presets.MODE_FIT)
        v1, v2 = params.get("ideality_V1"), params.get("ideality_V2")
        self.window_auto.set(v1 is None or v2 is None)
        self.V1_var.set("" if v1 is None else _fmt(v1))
        self.V2_var.set("" if v2 is None else _fmt(v2))
        self.preset_label.config(text=f"Набор: {preset.name}")
        self._load_preset_files(preset)
        # n и J₀ из ВАХ набора подставляются, только если набор их не задаёт.
        self._take_from_iv = bool(self.exp_iv) and "n_emp" not in preset.params
        self._on_scenario_change(recompute=False)

    def _load_preset_files(self, preset):
        self.exp_iv, self.exp_cv = [], []
        missing = []
        for kind, target in (("iv", self.exp_iv), ("cv", self.exp_cv)):
            for path in preset.resolved_files(kind)[:MAX_DATASETS]:
                try:
                    voltage, value = load_xy_file(path)
                except (OSError, ValueError):
                    missing.append(str(path))
                    continue
                target.append({"label": Path(path).name, "path": str(path),
                               "voltage": voltage, "value": value})
        if missing:
            messagebox.showwarning("Файлы набора", "Не удалось загрузить:\n" + "\n".join(missing))

    def save_preset(self):
        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода параметров", str(e))
            return
        path = filedialog.asksaveasfilename(title="Сохранить набор образца", defaultextension=".json",
                                            filetypes=[("Набор образца", "*.json")])
        if not path:
            return
        preset = presets.Preset(name=Path(path).stem, params=params, metadata=self.preset.metadata,
                                files={"iv": [d["path"] for d in self.exp_iv],
                                       "cv": [d["path"] for d in self.exp_cv]},
                                notes=self.preset.notes)
        self.preset = presets.save_preset(preset, path)
        self.preset_label.config(text=f"Набор: {self.preset.name}")

    def load_preset(self):
        path = filedialog.askopenfilename(title="Загрузить набор образца",
                                          filetypes=[("Набор образца", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            preset = presets.load_preset(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Набор образца", str(e))
            return
        self._apply_preset(preset)
        self.recompute()

    def reset_to_reference(self):
        preset = presets.reference_preset()
        if preset is None:
            messagebox.showinfo(
                "Опорный набор",
                "Опорный набор хранится в каталоге данных (переменная MESA_DATA_DIR в .env) "
                "и сейчас не найден. Установлены нейтральные значения по умолчанию.")
            preset = presets.default_preset()
        self._apply_preset(preset)
        self.recompute()

    # -------------------------------------------------- сценарий, поля
    def _scenarios(self):
        i_type = self.i_type.get()
        if i_type == presets.I_TYPE_BOTH:
            return [ph.SCENARIO_A, ph.SCENARIO_B]
        return [presets.I_TYPE_TO_SCENARIO[i_type]]

    def _on_scenario_change(self, recompute=True):
        """Доступность полей: режим (какие параметры свободны) и сценарий
        (какие подвижности входят в модель). Значения полей не меняются."""
        scenarios = self._scenarios()
        mode = self.mode.get()
        editable = presets.editable_keys(mode)
        unused = set.intersection(*(UNUSED_BY_SCENARIO[sc] for sc in scenarios))
        for key, entry in self.entries.items():
            entry.state(["!disabled"] if key in editable and key not in unused else ["disabled"])
        physical = mode != presets.MODE_BASIC
        for scenario, fields in BOUNDARY_FIELDS.items():
            for key, _label in fields:
                state = ["!disabled"] if physical and scenario in scenarios else ["disabled"]
                for widget in self.bc_widgets[key]:
                    widget.state(state)
        fit = mode == presets.MODE_FIT
        for button in self.flag_buttons.values():
            button.state(["!disabled"] if fit else ["disabled"])
        for button in (self.tau0_button, self.n2_button):
            button.state(["!disabled"] if fit else ["disabled"])
        if recompute:
            self.recompute()

    def _on_mode_change(self):
        self._on_scenario_change()

    def _read_params(self):
        params = {"substrate": self.substrate.get(), "i_type": self.i_type.get(),
                  "mode": self.mode.get()}
        for key, var in self.vars.items():
            raw = var.get().strip().replace(",", ".")
            try:
                params[key] = float(raw)
            except ValueError:
                raise ValueError(f"«{SPECS[key].label}» задан некорректно: «{raw}»")
        for key, var in self.bc_vars.items():
            params[key] = BOUNDARY_BY_LABEL[var.get()]
        for key, var in self.flag_vars.items():
            params[key] = bool(var.get())
        if self.window_auto.get():
            params["ideality_V1"] = params["ideality_V2"] = None
        else:
            try:
                params["ideality_V1"] = float(self.V1_var.get().replace(",", "."))
                params["ideality_V2"] = float(self.V2_var.get().replace(",", "."))
            except ValueError:
                raise ValueError("Окно идеальности V₁ … V₂ задано некорректно.")
            if params["ideality_V1"] >= params["ideality_V2"]:
                raise ValueError("Окно идеальности: V₁ должно быть меньше V₂.")

        for key in POSITIVE_KEYS:
            if params[key] <= 0:
                raise ValueError(f"«{SPECS[key].label}» должно быть больше нуля.")
        for key in NONNEGATIVE_KEYS:
            if params[key] < 0:
                raise ValueError(f"«{SPECS[key].label}» не может быть отрицательным.")
        if params["d_n_um"] >= params["d_epi_um"]:
            raise ValueError("Толщина n⁺-слоя d_n должна быть меньше толщины эпитаксии d_epi.")
        if params["D_inner_um"] >= params["D_um"]:
            raise ValueError("Внутренний диаметр кольца должен быть меньше диаметра мезы D.")
        return params

    def _ideality_unavailable_text(self):
        """Почему n_эксп нет: нет ВАХ или расчёт не удался (с причиной)."""
        if not self.exp_iv:
            return "n_эксп ещё не определён: загрузите ВАХ."
        reason = "; ".join(n.rstrip(".") for n in self.ideality_notes) or "причина не установлена"
        return (f"n_эксп по ВАХ «{self.exp_iv[0]['label']}» не определён: {reason}.\n"
                "Проверьте R_s и окно V₁ … V₂ (снимите «авто» и задайте окно вручную).")

    def _copy_n_exp(self, key, title):
        if self.ideality is None:
            messagebox.showinfo(title, self._ideality_unavailable_text())
            return
        self.vars[key].set(f"{self.ideality.n:.3g}")
        self.recompute()

    def _n2_from_experiment(self):
        self._copy_n_exp("n2", "n₂ ← n_эксп")

    def _n_emp_from_experiment(self):
        self._copy_n_exp("n_emp", "n (6.3) ← n_эксп")

    def _estimate_tau0(self):
        title = "Оценка τ₀ по обратной ветви"
        if not self.exp_iv or not self.structures:
            messagebox.showinfo(title, "Загрузите ВАХ с обратной ветвью.")
            return
        data = self.exp_iv[0]
        s = self.structures[self._scenarios()[-1]]
        V_at = max(TAU0_V, float(np.min(data["voltage"])))
        notes = []
        estimate = ph.estimate_tau0_from_reverse(s, data["voltage"], data["value"], V_at, notes)
        if estimate is None:
            messagebox.showinfo(title, f"«{data['label']}»: {notes[0] if notes else 'оценка не удалась'}.")
            return
        text = (f"ВАХ «{data['label']}», V = {estimate.V:g} В, сценарий {s.scenario}.\n"
                f"Ток ОПЗ после вычета шунта и диффузии: {estimate.I_gr:.3g} А.\n"
                f"τ₀ = {estimate.tau0:.3g} с (обращение (5.4)).\n")
        if not np.isfinite(estimate.tau0_bg):
            messagebox.showinfo(title, text + notes[-1] + ".")
            return
        text += f"τ₀^bg = {estimate.tau0_bg:.3g} с по (5.10) с N_dis = {s.N_dis:g} см⁻².\n\nПодставить τ₀^bg?"
        if messagebox.askyesno(title, text):
            self.vars["tau0_bg"].set(f"{estimate.tau0_bg:.3g}")
            self.recompute()

    # --------------------------------------------- экспериментальные данные
    def _load_experimental_files(self, dialog_title, target_list, error_title):
        paths = filedialog.askopenfilenames(
            title=dialog_title, filetypes=[("Текст/CSV", "*.csv;*.txt;*.dat"), ("Все файлы", "*.*")])
        if not paths:
            return
        slots = remaining_slots(len(target_list))
        accepted, rejected = paths[:slots], paths[slots:]
        for path in accepted:
            try:
                voltage, value = load_xy_file(path)
            except Exception as e:
                messagebox.showerror(error_title, f"{Path(path).name}: {e}")
                continue
            target_list.append({"label": Path(path).name, "path": path,
                                "voltage": voltage, "value": value})
        if target_list is self.exp_iv and accepted:
            self._take_from_iv = True
        if rejected:
            messagebox.showwarning(
                "Достигнут лимит наборов данных",
                f"Загружено {len(accepted)} из {len(paths)} файлов — на графике уже "
                f"максимум {MAX_DATASETS} наборов. Уберите лишние через «Очистить данные».")
        self.recompute()

    def load_experimental_iv(self):
        self._load_experimental_files("Файлы экспериментальной ВАХ (V, I)", self.exp_iv, "Ошибка загрузки ВАХ")

    def load_experimental_cv(self):
        self._load_experimental_files("Файлы экспериментальной ВФХ (V, C)", self.exp_cv, "Ошибка загрузки ВФХ")

    def clear_experimental(self):
        self.exp_iv, self.exp_cv = [], []
        self.recompute()

    # -------------------------------------------------------- пересчёт
    def recompute(self):
        try:
            params = self._read_params()
            self._update_ideality(params)
            self.structures = {sc: presets.to_structure(params, sc) for sc in self._scenarios()}
            main = self.structures[self._scenarios()[-1]]
            if not np.isfinite(main.substrate[0]):
                raise ValueError(main.substrate[1][0])
        except ValueError as e:
            messagebox.showerror("Ошибка ввода параметров", str(e))
            return
        self.preset.params = params
        self.d_i_label.config(text=f"d_i = d_epi − d_n = {params['d_epi_um'] - params['d_n_um']:.3g} мкм (вычисляется)")

        v_min, v_max = adaptive_voltage_range([d["voltage"] for d in self.exp_iv])
        V = np.linspace(v_min, v_max, adaptive_point_count(v_min, v_max))
        self.results = {sc: ph.solve_iv(s, V) for sc, s in self.structures.items()}

        window = self._ideality_window(params)
        model_window = window or ((self.ideality.V1, self.ideality.V2) if self.ideality else None)
        main_result = self.results[main.scenario]
        self.model_ideality = ph.ideality_from_data(V, main_result.I, main.T, main.Rs, model_window)
        n_exp = self.ideality.n if self.ideality else None

        self.comparison = None
        if len(self.structures) == 2:
            exp_iv = (self.exp_iv[0]["voltage"], self.exp_iv[0]["value"]) if self.exp_iv else None
            exp_cv = (self.exp_cv[0]["voltage"], self.exp_cv[0]["value"]) if self.exp_cv else None
            self.comparison = ph.compare_scenarios(main, exp_iv, exp_cv, self._c2_window(), model_window)
        n_mod = self.model_ideality.n if self.model_ideality else None
        self.reference_data = ref.reference_table(main, iv=main_result, n_exp=n_exp, n_mod=n_mod,
                                                  metadata=self.preset.metadata)

        self._V, self._n_exp = V, n_exp
        self._redraw()
        self._plot_mesa(params, main)
        self._update_status(n_exp, n_mod)
        if self.formulas_window is not None:
            self.formulas_window.refresh()

    def _redraw(self):
        """Перерисовка графиков без пересчёта (галочки кривых)."""
        if not self.results:
            return
        self._update_curve_buttons()
        self._plot_iv(self._V, self._n_exp)
        self._plot_cv()
        self._plot_jv(self._V, self._n_exp)
        self.canvas.draw_idle()

    @staticmethod
    def _ideality_window(params):
        return None if params["ideality_V1"] is None else (params["ideality_V1"], params["ideality_V2"])

    def _update_ideality(self, params):
        """n_эксп по первой ВАХ (§6.3). После загрузки новой ВАХ n и J₀ модели
        (6.3) заменяются на n_эксп и I₀/A; дальше их можно менять вручную."""
        self.ideality, self.ideality_notes = None, []
        if self.exp_iv:
            first = self.exp_iv[0]
            self.ideality = ph.ideality_from_data(first["voltage"], first["value"], params["T"],
                                                  params["Rs"], self._ideality_window(params),
                                                  diagnostics=self.ideality_notes)
        if self.ideality:
            self.n_exp_var.set(f"{self.ideality.n:.3f}")
            if self.window_auto.get():
                self.V1_var.set(f"{self.ideality.V1:.3f}")
                self.V2_var.set(f"{self.ideality.V2:.3f}")
            text = "; ".join(self.ideality_notes)
        else:
            self.n_exp_var.set("—")
            text = self._ideality_unavailable_text() if self.exp_iv else ""
        self.ideality_lbl.config(text=text)
        if self._take_from_iv and self.ideality:
            area = presets.to_structure(params).area
            params["n_emp"] = float(f"{self.ideality.n:.3g}")
            params["J0_emp"] = float(f"{self.ideality.I0 / area:.3g}")
            self.vars["n_emp"].set(format_value(params["n_emp"]))
            self.vars["J0_emp"].set(format_value(params["J0_emp"]))
        self._take_from_iv = False

    def _c2_window(self):
        try:
            a = float(self.c2_from_var.get().replace(",", "."))
            b = float(self.c2_to_var.get().replace(",", "."))
        except ValueError:
            return None
        return (min(a, b), max(a, b))

    def _update_status(self, n_exp, n_mod):
        vbi = " / ".join(f"{s.Vbi:.3f} ({sc})" for sc, s in self.structures.items())
        parts = [f"V_bi = {vbi} В"]
        if n_exp:
            parts.append(f"n_эксп = {n_exp:.3f}")
        else:
            parts.append("n_эксп = — (не определён, см. блок идеальности)" if self.exp_iv
                         else "n_эксп = — (нет ВАХ)")
        parts.append(f"n_мод = {n_mod:.3f}" if n_mod else "n_мод = —")
        count = len(self.reference_data.warnings) if self.reference_data else 0
        if count:
            parts.append(f"⚠ {count} предупреждений — см. Справочник")
        self.status_lbl.config(text="   |   ".join(parts),
                               foreground="#b35c00" if count else "#222222")

    def _color(self, scenario):
        return SCENARIO_COLORS[scenario] if len(self.structures) == 2 else SINGLE_COLOR

    def _metric_suffix(self, scenario):
        if not self.comparison:
            return ""
        row = self.comparison[scenario]
        texts = [f"{name}={row[key]:.3g}" for key, name in (("delta_I", "δ_I"), ("delta_C", "δ_C"))
                 if np.isfinite(row[key])]
        return (", " + ", ".join(texts)) if texts else ""

    # ------------------------------------------------ галочки кривых
    def _model(self):
        return next(iter(self.structures.values())).model

    def _visible(self, plot, key):
        """Кривая показывается: галочка стоит и кривая есть в текущей модели."""
        return self._shown(plot, key) and key not in CURVES_BY_MODEL[self._model()]

    def _update_curve_buttons(self):
        absent = CURVES_BY_MODEL[self._model()]
        for (_plot, key), button in self.curve_buttons.items():
            button.state(["disabled"] if key in absent else ["!disabled"])

    @staticmethod
    def _legend(ax, handles=None, labels=None, **kwargs):
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, **kwargs)

    # ------------------------------------------------ График 1. ВАХ
    def _plot_iv(self, V, n_exp):
        ax = self.ax_iv
        ax.clear()
        exp_values = [d["value"] for d in self.exp_iv]
        any_I = next(iter(self.results.values())).I
        factor, unit = auto_scale(np.concatenate(exp_values) if exp_values else np.nan_to_num(any_I), "А")

        single = len(self.structures) == 1
        for sc, result in self.results.items():
            if self._visible("iv", "total"):
                label = "модель" if single else f"сценарий {sc}{self._metric_suffix(sc)}"
                ax.plot(V, result.I * factor, color=self._color(sc), linewidth=1.8, label=label)
            if single:
                for key, values in result.parts.items():
                    name, color = COMPONENT_STYLE[key]
                    if self._visible("iv", key) and np.any(values != 0):
                        ax.plot(V, values * factor, color=color, linewidth=0.9, linestyle="--", label=name)
        if self.ideality and n_exp and self._visible("iv", "emp_fit"):
            Vf = V[V > 0]
            T = next(iter(self.structures.values())).T
            ax.plot(Vf, ph.empirical_current(Vf, self.ideality.I0, n_exp, T) * factor, color="#888888",
                    linestyle=":", linewidth=1.3, label=f"(6.3) по n_эксп = {n_exp:.2f}")
        if self._visible("iv", "exp"):
            for index, dataset in enumerate(self.exp_iv):
                marker, color = dataset_style(index)
                ax.plot(dataset["voltage"], dataset["value"] * factor, marker=marker, linestyle="-",
                        linewidth=0.6, ms=3, markeredgewidth=0.7, color=color, markerfacecolor="none",
                        label=dataset["label"])
        y_lo, y_hi = robust_value_limits(exp_values, any_I)
        ax.set_ylim(y_lo * factor, y_hi * factor)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.axvline(0, color="#999999", linewidth=0.7)
        ax.set_title("График 1. ВАХ", fontsize=10)
        ax.set_xlabel("V, В")
        ax.set_ylabel(f"I, {unit}")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        self._legend(ax, fontsize=7, loc="best")

    # ------------------------------------------- График 2. ВФХ и 1/C²
    def _plot_cv(self):
        ax = self.ax_cv
        ax.clear()
        vbi_max = max(s.Vbi for s in self.structures.values())
        exp_V = [d["voltage"] for d in self.exp_cv]
        v_lo = min([-1.5] + [float(np.min(v)) for v in exp_V])
        V = np.linspace(v_lo, vbi_max, 200)
        single = len(self.structures) == 1
        show_model, show_exp = self._visible("cv", "model_C"), self._visible("cv", "exp")
        if self.cv_view.get() == "C":
            model = {sc: ph.capacitance(s, V) for sc, s in self.structures.items()}
            combo = np.concatenate([np.nan_to_num(c) for c in model.values()] + [d["value"] for d in self.exp_cv])
            factor, unit = auto_scale(combo, "Ф")
            if show_model:
                for sc, C in model.items():
                    ax.plot(V, C * factor, color=self._color(sc), linewidth=1.8,
                            label="модель" if single else f"сценарий {sc}")
            if show_exp:
                for index, dataset in enumerate(self.exp_cv):
                    marker, color = dataset_style(index)
                    mask = dataset["value"] > 0
                    ax.plot(dataset["voltage"][mask], dataset["value"][mask] * factor, marker=marker,
                            linestyle="-", linewidth=0.6, ms=3, markeredgewidth=0.7, color=color,
                            markerfacecolor="none", label=dataset["label"])
            ax.set_yscale("log")
            ax.set_ylabel(f"C, {unit} (лог.)")
            ax.set_title("График 2. ВФХ", fontsize=10)
        else:
            window = self._c2_window()
            if show_model:
                for sc, s in self.structures.items():
                    C = ph.capacitance(s, V)
                    ax.plot(V, 1e-24 / C ** 2, color=self._color(sc), linewidth=1.8,
                            label=f"модель{'' if single else ' ' + sc}: N_eff = {s.N_eff:.2e}, "
                                  f"V₀ = {ph.c2_cutoff(s):.3f} В")
            area = next(iter(self.structures.values())).area
            eps = next(iter(self.structures.values())).eps
            for index, dataset in enumerate(self.exp_cv):
                marker, color = dataset_style(index)
                Vd, C = dataset["voltage"], dataset["value"]
                mask = C > 0
                N, V0 = ph.fit_inv_c2(Vd, C, area, eps, window)
                label = dataset["label"]
                if np.isfinite(N):
                    label += f": N = {N:.2e} см⁻³, V₀ = {V0:.3f} В"
                    if self._visible("cv", "fit_C"):
                        sel = Vd[mask]
                        b = -2.0 / (ph.Q * eps * area ** 2 * N)
                        line_V = np.linspace(min(sel.min(), window[0]) if window else sel.min(), V0, 50)
                        ax.plot(line_V, 1e-24 * b * (line_V - V0), color=color, linewidth=0.8, linestyle="--")
                if show_exp:
                    ax.plot(Vd[mask], 1e-24 / C[mask] ** 2, marker=marker, linestyle="none", ms=3,
                            markeredgewidth=0.7, color=color, markerfacecolor="none", label=label)
            if window:
                for x in window:
                    ax.axvline(x, color="#bbbbbb", linewidth=0.7, linestyle=":")
            ax.set_ylabel("1/C², пФ⁻²")
            ax.set_title("График 2. 1/C² (прямые — на выбранном участке)", fontsize=10)
        ax.set_xlabel("V, В")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
        self._legend(ax, fontsize=7, loc="best")

    # ------------------------------------ График 3. Плотность тока J–V
    def _plot_jv(self, V, n_exp):
        ax, axn = self.ax_jv, self.ax_jv_n
        ax.clear()
        axn.clear()
        single = len(self.structures) == 1
        values = []
        for sc, result in self.results.items():
            s = self.structures[sc]
            J = np.abs(result.I) / s.area
            if self._visible("jv", "total"):
                values.append(J)
                ax.plot(V, J, color=self._color(sc), linewidth=1.8,
                        label="|J| модели" if single else f"|J|, сценарий {sc}")
            if single:
                for key, part in result.parts.items():
                    name, color = COMPONENT_STYLE[key]
                    comp = np.abs(part) / s.area
                    if self._visible("jv", key) and np.any(comp > 0):
                        values.append(comp)
                        ax.plot(V, comp, color=color, linewidth=0.9, linestyle="--",
                                label=name.replace("I_", "J_", 1))
                if self._visible("jv", "Js0"):
                    if s.model == ph.MODEL_EMPIRICAL:
                        Js0, name = s.J0_emp, "J₀"
                    else:
                        Js0, name = float(ph.saturation_current_density(s, 0.0)), "J_s(0)"
                    ax.axhline(Js0, color="#555555", linestyle="--", linewidth=0.8, label=f"{name} = {Js0:.2e}")
        area = next(iter(self.structures.values())).area
        T = next(iter(self.structures.values())).T
        if self._visible("jv", "js_exp"):
            for index, dataset in enumerate(self.exp_iv):
                J = np.abs(dataset["value"]) / area
                values.append(J)
                ax.plot(dataset["voltage"], J, marker="o", linestyle="none", ms=3, markeredgewidth=0.7,
                        color=JS_COLOR, markerfacecolor="none",
                        label="J(S) = |I_эксп|/A" if index == 0 else None)
        if self.ideality and n_exp:
            if self._visible("jv", "emp_fit"):
                Vf = V[V > 0]
                ax.plot(Vf, self.ideality.I0 / area * np.exp(Vf / (n_exp * ph.thermal_voltage(T))),
                        color="#888888", linestyle=":", linewidth=1.3, label=f"(6.3), n_эксп = {n_exp:.2f}")
            if self._visible("jv", "n_exp"):
                axn.plot(self.ideality.V_local, self.ideality.n_local, marker=".", linestyle="none",
                         color="#444444", ms=3, label="n(V) эксп.")
        if self.model_ideality and self._visible("jv", "n_mod"):
            axn.plot(self.model_ideality.V_local, self.model_ideality.n_local, color="#999999",
                     linewidth=0.8, label="n(V) модели")
        axn.yaxis.tick_right()
        axn.yaxis.set_label_position("right")
        axn.set_ylim(0.5, 4.0)
        axn.set_ylabel("локальный n (6.5)", fontsize=8)
        ax.set_yscale("log")
        finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in values]) if values else np.array([])
        if finite.size:
            ax.set_ylim(max(finite.min() * 0.5, finite.max() * 1e-12), finite.max() * 2)
        ax.set_title("График 3. Плотность тока |J|–V", fontsize=10)
        ax.set_xlabel("V, В")
        ax.set_ylabel("|J|, А/см²")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = axn.get_legend_handles_labels()
        self._legend(ax, handles + h2, labels + l2, fontsize=6.5, loc="upper left")

    # --------------------------------------------------- схема мезы
    def _plot_mesa(self, params, s):
        self.ax_mesa.clear()
        self.mesa_canvas.draw_idle()
        diagram_params = type("DiagramParams", (), {
            "D_um": params["D_um"], "d_um": params["D_inner_um"], "h_um": params["h_um"],
            "ND_plus": params["ND_plus"], "N_i": params["N_i"], "N_sub": s.substrate[0],
            "T": params["T"], "i_type": {"n": "i-слой n", "p": "i-слой p", "both": "тип i: оба"}[params["i_type"]],
        })
        draw_mesa_diagram(self.ax_mesa, diagram_params)
        self.mesa_canvas.draw_idle()

    # ------------------------------------------- окно формул
    def open_formulas_window(self):
        formulas_module.open_formulas_window(self)


def main():
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    app = MesaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
