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
    "diff": ("I_diff", "#2ca02c"),
    "gr": ("I_gr", "#ff7f0e"),
    "sh": ("I_sh", "#7f7f7f"),
    "L": ("I_L", "#9467bd"),
}
JS_COLOR = "#9b870c"
FONT = ("Segoe UI", 9)

# Группы полей: (заголовок, ключи, строка, столбец) — две строки по три группы.
GROUPS = [
    ("Геометрия", ["D_um", "D_inner_um", "d_epi_um", "h_um", "d_n_um", "d_sub_um"], 0, 0),
    ("Легирование и температура", ["ND_plus", "N_i", "rho_sub", "T"], 0, 1),
    ("Неосновные носители и дефекты",
     ["mu_n", "mu_p_i", "mu_p_nplus", "tau_n_bg", "tau_p_bg", "tau0_bg",
      "N_dis", "sigma_R_epi", "sigma_R_sub"], 0, 2),
    ("Ток и утечки", ["n2", "Rs", "Rsh", "I_L", "m_leak"], 1, 0),
]
# Короткие подписи в основном окне; полные — в таблице параметров окна формул.
SHORT_LABELS = {
    "D_um": "D (мезы)", "D_inner_um": "d кольца (схема)", "d_epi_um": "d_epi",
    "h_um": "h (травление)", "d_n_um": "d_n (n⁺)", "d_sub_um": "d_sub",
    "ND_plus": "N_D⁺", "N_i": "N_i", "rho_sub": "ρ_sub", "T": "T",
    "mu_n": "μ_n (электроны)", "mu_p_i": "μ_p (i-слой)", "mu_p_nplus": "μ_p (n⁺)",
    "tau_n_bg": "τ_n^bg", "tau_p_bg": "τ_p^bg", "tau0_bg": "τ₀^bg (ОПЗ)",
    "N_dis": "N_dis", "sigma_R_epi": "σ_R (i, n⁺)", "sigma_R_sub": "σ_R (подложка)",
    "n2": "n₂", "Rs": "R_s", "Rsh": "R_sh", "I_L": "I_L", "m_leak": "m",
}
SPECS = {spec.key: spec for spec in presets.NUMERIC_PARAMS}
# Подвижности, которые не входят в модель при данном сценарии (§5.3: «активны нужные»)
UNUSED_BY_SCENARIO = {ph.SCENARIO_A: {"mu_p_i", "sigma_R_sub"}, ph.SCENARIO_B: {"mu_p_nplus"}}
BOUNDARY_FIELDS = {
    ph.SCENARIO_A: [("bc_A_n", "A: верхний контакт"), ("bc_A_p", "A: граница i/подложка")],
    ph.SCENARIO_B: [("bc_B_n", "B: граница n/n⁺"), ("bc_B_p", "B: тыльный контакт")],
}
BOUNDARY_BY_LABEL = {label: key for key, label in ph.BOUNDARY_LABELS.items()}
N_I_RANGE = (15.0, 17.0)      # лог. ползунок N_i, 10¹⁵–10¹⁷ см⁻³
D_SUB_RANGE = (100.0, 1000.0)  # ползунок d_sub, мкм


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
        self.cv_view = tk.StringVar(value="C")
        self.js_var = tk.BooleanVar(value=True)
        self.window_auto = tk.BooleanVar(value=True)
        self.n_manual = tk.BooleanVar(value=False)
        self.n_exp_var = tk.StringVar()
        self.V1_var = tk.StringVar()
        self.V2_var = tk.StringVar()
        self.c2_from_var = tk.StringVar(value="-1.0")
        self.c2_to_var = tk.StringVar(value="0.0")
        self.n_slider = tk.DoubleVar()
        self.dsub_slider = tk.DoubleVar()

        self.structures = {}
        self.results = {}
        self.ideality = None
        self.model_ideality = None
        self.comparison = None
        self.reference_data = None
        self.formulas_window = None

        self._build_menu()
        self._build_figure_area()
        self._build_panel()
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
        ttk.Checkbutton(bar, text="График 3: точки J(S) = |I_эксп|/A", variable=self.js_var,
                        command=self.recompute).pack(side=tk.LEFT, padx=(24, 0))

        # Холст упаковывается после панели (см. __init__): иначе он, растягиваясь,
        # забирает высоту первым и панель с кнопкой «Рассчитать» обрезается.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

    # ---------------------------------------------------------- панель
    def _build_panel(self):
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        outer = ttk.Frame(bottom)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Схема мезы — отдельный холст справа: текст на ней фиксированного
        # размера в пунктах, поэтому ей нужна своя, не сжимаемая площадь.
        self.mesa_fig = Figure(figsize=(5.0, 3.9), dpi=100)
        self.mesa_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
        self.ax_mesa = self.mesa_fig.add_subplot(111)
        self.mesa_canvas = FigureCanvasTkAgg(self.mesa_fig, master=bottom)
        self.mesa_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=(8, 0))

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
                self.d_i_label.grid(row=len(keys), column=0, columnspan=4, sticky="w")
        self._build_extra_boxes(grid)

        buttons = ttk.Frame(outer)
        buttons.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        ttk.Button(buttons, text="Рассчитать", command=self.recompute).pack(side=tk.LEFT, padx=(0, 8))
        self.status_lbl = ttk.Label(buttons, text="", font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(side=tk.LEFT)

    def _param_row(self, box, row, key):
        spec = SPECS[key]
        ttk.Label(box, text=SHORT_LABELS[key], font=FONT).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(box, textvariable=self.vars[key], width=9, justify="center")
        entry.grid(row=row, column=1, padx=3, pady=1)
        entry.bind("<Return>", lambda _e: self.recompute())
        ttk.Label(box, text=spec.unit, font=FONT, foreground="#555555").grid(row=row, column=2, sticky="w")
        self.entries[key] = entry
        if key == "N_i":
            scale = ttk.Scale(box, from_=N_I_RANGE[0], to=N_I_RANGE[1], variable=self.n_slider,
                              command=lambda _v: self.vars["N_i"].set(f"{10 ** self.n_slider.get():.3g}"))
            scale.grid(row=row, column=3, sticky="ew", padx=(4, 0))
            scale.bind("<ButtonRelease-1>", lambda _e: self.recompute())
        if key == "d_sub_um":
            scale = ttk.Scale(box, from_=D_SUB_RANGE[0], to=D_SUB_RANGE[1], variable=self.dsub_slider,
                              command=lambda _v: self.vars["d_sub_um"].set(f"{self.dsub_slider.get():.0f}"))
            scale.grid(row=row, column=3, sticky="ew", padx=(4, 0))
            scale.bind("<ButtonRelease-1>", lambda _e: self.recompute())

    def _build_extra_boxes(self, grid):
        box = ttk.LabelFrame(grid, text="Граничные условия и опции", padding=(6, 2))
        box.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=(0, 4))
        self.bc_widgets = {}
        row = 0
        for scenario in (ph.SCENARIO_A, ph.SCENARIO_B):
            for key, label in BOUNDARY_FIELDS[scenario]:
                lbl = ttk.Label(box, text=label, font=FONT)
                combo = ttk.Combobox(box, textvariable=self.bc_vars[key], width=13, state="readonly",
                                     values=list(ph.BOUNDARY_LABELS.values()))
                combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())
                lbl.grid(row=row, column=0, sticky="w")
                combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=3)
                self.bc_widgets[key] = (lbl, combo)
                row += 1
        for key, text in (("sns_refinement", "уточнение SNS (5.5)"),
                          ("edge_area", "краевая добавка площади ОПЗ (1.3)")):
            ttk.Checkbutton(box, text=text, variable=self.flag_vars[key],
                            command=self.recompute).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1

        box = ttk.LabelFrame(grid, text="Коэффициент идеальности (§6.3)", padding=(6, 2))
        box.grid(row=1, column=2, sticky="nsew", padx=(0, 6), pady=(0, 4))
        row = 0
        ttk.Label(box, text="n_эксп", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.n_exp_var, width=10, justify="center").grid(row=row, column=1, padx=3)
        ttk.Checkbutton(box, text="вручную", variable=self.n_manual).grid(row=row, column=2, sticky="w")
        row += 1
        ttk.Label(box, text="окно V₁ … V₂, В", font=FONT).grid(row=row, column=0, sticky="w")
        window = ttk.Frame(box)
        window.grid(row=row, column=1, columnspan=2, sticky="w", padx=3)
        for var in (self.V1_var, self.V2_var):
            e = ttk.Entry(window, textvariable=var, width=6, justify="center")
            e.pack(side=tk.LEFT, padx=(0, 2))
            e.bind("<KeyRelease>", lambda _e: self.window_auto.set(False))
        ttk.Checkbutton(window, text="авто", variable=self.window_auto).pack(side=tk.LEFT)
        row += 1
        ttk.Button(box, text="n₂ ← n_эксп (эмпирика)", command=self._n2_from_experiment).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))

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
        self.n_slider.set(min(max(math.log10(float(params["N_i"])), N_I_RANGE[0]), N_I_RANGE[1]))
        self.dsub_slider.set(min(max(float(params["d_sub_um"]), D_SUB_RANGE[0]), D_SUB_RANGE[1]))
        v1, v2 = params.get("ideality_V1"), params.get("ideality_V2")
        self.window_auto.set(v1 is None or v2 is None)
        self.V1_var.set("" if v1 is None else _fmt(v1))
        self.V2_var.set("" if v2 is None else _fmt(v2))
        self.preset_label.config(text=f"Набор: {preset.name}")
        self._load_preset_files(preset)
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
        scenarios = self._scenarios()
        unused = set.intersection(*(UNUSED_BY_SCENARIO[sc] for sc in scenarios))
        for key, entry in self.entries.items():
            entry.state(["disabled"] if key in unused else ["!disabled"])
        for scenario, fields in BOUNDARY_FIELDS.items():
            for key, _label in fields:
                state = ["!disabled"] if scenario in scenarios else ["disabled"]
                for widget in self.bc_widgets[key]:
                    widget.state(state)
        if recompute:
            self.recompute()

    def _read_params(self):
        params = {"substrate": self.substrate.get(), "i_type": self.i_type.get()}
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

        positive = ("D_um", "d_epi_um", "h_um", "d_n_um", "d_sub_um", "ND_plus", "N_i",
                    "rho_sub", "T", "mu_n", "mu_p_i", "mu_p_nplus", "tau_n_bg", "tau_p_bg",
                    "tau0_bg", "n2", "Rsh", "m_leak", "D_inner_um")
        for key in positive:
            if params[key] <= 0:
                raise ValueError(f"«{SPECS[key].label}» должно быть больше нуля.")
        for key in ("N_dis", "sigma_R_epi", "sigma_R_sub", "Rs", "I_L"):
            if params[key] < 0:
                raise ValueError(f"«{SPECS[key].label}» не может быть отрицательным.")
        if params["d_n_um"] >= params["d_epi_um"]:
            raise ValueError("Толщина n⁺-слоя d_n должна быть меньше толщины эпитаксии d_epi.")
        if params["D_inner_um"] >= params["D_um"]:
            raise ValueError("Внутренний диаметр кольца должен быть меньше диаметра мезы D.")
        return params

    def _n2_from_experiment(self):
        try:
            n = float(self.n_exp_var.get().replace(",", "."))
        except ValueError:
            messagebox.showinfo("n₂ ← n_эксп", "n_эксп ещё не определён: загрузите ВАХ.")
            return
        self.vars["n2"].set(f"{n:.3g}")
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

        window = None if params["ideality_V1"] is None else (params["ideality_V1"], params["ideality_V2"])
        self.ideality = None
        if self.exp_iv:
            first = self.exp_iv[0]
            self.ideality = ph.ideality_from_data(first["voltage"], first["value"], main.T, main.Rs, window)
        model_window = window or ((self.ideality.V1, self.ideality.V2) if self.ideality else None)
        main_result = self.results[main.scenario]
        self.model_ideality = ph.ideality_from_data(V, main_result.I, main.T, main.Rs, model_window)
        if self.ideality:
            if self.window_auto.get():
                self.V1_var.set(f"{self.ideality.V1:.3f}")
                self.V2_var.set(f"{self.ideality.V2:.3f}")
            if not self.n_manual.get():
                self.n_exp_var.set(f"{self.ideality.n:.3f}")
        n_exp = self._n_exp_value()

        self.comparison = None
        if len(self.structures) == 2:
            exp_iv = (self.exp_iv[0]["voltage"], self.exp_iv[0]["value"]) if self.exp_iv else None
            exp_cv = (self.exp_cv[0]["voltage"], self.exp_cv[0]["value"]) if self.exp_cv else None
            self.comparison = ph.compare_scenarios(main, exp_iv, exp_cv, self._c2_window(), model_window)
        n_mod = self.model_ideality.n if self.model_ideality else None
        self.reference_data = ref.reference_table(main, iv=main_result, n_exp=n_exp, n_mod=n_mod,
                                                  metadata=self.preset.metadata)

        self._plot_iv(V, n_exp)
        self._plot_cv()
        self._plot_jv(V, n_exp)
        self._plot_mesa(params, main)
        self._update_status(n_exp, n_mod)
        self.canvas.draw_idle()
        if self.formulas_window is not None:
            self.formulas_window.refresh()

    def _n_exp_value(self):
        try:
            return float(self.n_exp_var.get().replace(",", "."))
        except ValueError:
            return None

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
        parts.append(f"n_эксп = {n_exp:.3f}" if n_exp else "n_эксп = —")
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

    # ------------------------------------------------ График 1. ВАХ
    def _plot_iv(self, V, n_exp):
        ax = self.ax_iv
        ax.clear()
        exp_values = [d["value"] for d in self.exp_iv]
        any_I = next(iter(self.results.values())).I
        factor, unit = auto_scale(np.concatenate(exp_values) if exp_values else np.nan_to_num(any_I), "А")

        single = len(self.structures) == 1
        for sc, result in self.results.items():
            label = "сумма" if single else f"сценарий {sc}{self._metric_suffix(sc)}"
            ax.plot(V, result.I * factor, color=self._color(sc), linewidth=1.8, label=label)
            if single:
                for key, (name, color) in COMPONENT_STYLE.items():
                    values = result.parts[key]
                    if np.any(values != 0):
                        ax.plot(V, values * factor, color=color, linewidth=0.9, linestyle="--", label=name)
        if self.ideality and n_exp:
            Vf = V[V > 0]
            T = next(iter(self.structures.values())).T
            ax.plot(Vf, ph.empirical_current(Vf, self.ideality.I0, n_exp, T) * factor, color="#888888",
                    linestyle=":", linewidth=1.3, label=f"эмпирика (6.3), n = {n_exp:.2f}")
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
        ax.legend(fontsize=7, loc="best")

    # ------------------------------------------- График 2. ВФХ и 1/C²
    def _plot_cv(self):
        ax = self.ax_cv
        ax.clear()
        vbi_max = max(s.Vbi for s in self.structures.values())
        exp_V = [d["voltage"] for d in self.exp_cv]
        v_lo = min([-1.5] + [float(np.min(v)) for v in exp_V])
        V = np.linspace(v_lo, vbi_max, 200)
        single = len(self.structures) == 1
        if self.cv_view.get() == "C":
            model = {sc: ph.capacitance(s, V) for sc, s in self.structures.items()}
            combo = np.concatenate([np.nan_to_num(c) for c in model.values()] + [d["value"] for d in self.exp_cv])
            factor, unit = auto_scale(combo, "Ф")
            for sc, C in model.items():
                ax.plot(V, C * factor, color=self._color(sc), linewidth=1.8,
                        label="модель" if single else f"сценарий {sc}")
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
                    sel = Vd[mask]
                    b = -2.0 / (ph.Q * eps * area ** 2 * N)
                    line_V = np.linspace(min(sel.min(), window[0]) if window else sel.min(), V0, 50)
                    ax.plot(line_V, 1e-24 * b * (line_V - V0), color=color, linewidth=0.8, linestyle="--")
                ax.plot(Vd[mask], 1e-24 / C[mask] ** 2, marker=marker, linestyle="none", ms=3,
                        markeredgewidth=0.7, color=color, markerfacecolor="none", label=label)
            if window:
                for x in window:
                    ax.axvline(x, color="#bbbbbb", linewidth=0.7, linestyle=":")
            ax.set_ylabel("1/C², пФ⁻²")
            ax.set_title("График 2. 1/C² (прямые — на выбранном участке)", fontsize=10)
        ax.set_xlabel("V, В")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=7, loc="best")

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
            values.append(J)
            ax.plot(V, J, color=self._color(sc), linewidth=1.8,
                    label="|J| модели" if single else f"|J|, сценарий {sc}")
            if single:
                for key, (name, color) in COMPONENT_STYLE.items():
                    comp = np.abs(result.parts[key]) / s.area
                    if np.any(comp > 0):
                        ax.plot(V, comp, color=color, linewidth=0.9, linestyle="--", label=name)
                Js0 = float(ph.saturation_current_density(s, 0.0))
                ax.axhline(Js0, color="#555555", linestyle="--", linewidth=0.8, label=f"J_s(0) = {Js0:.2e}")
        area = next(iter(self.structures.values())).area
        T = next(iter(self.structures.values())).T
        if self.js_var.get():
            for index, dataset in enumerate(self.exp_iv):
                J = np.abs(dataset["value"]) / area
                values.append(J)
                ax.plot(dataset["voltage"], J, marker="o", linestyle="none", ms=3, markeredgewidth=0.7,
                        color=JS_COLOR, markerfacecolor="none",
                        label="J(S) = |I_эксп|/A" if index == 0 else None)
        if self.ideality and n_exp:
            Vf = V[V > 0]
            ax.plot(Vf, self.ideality.I0 / area * np.exp(Vf / (n_exp * ph.thermal_voltage(T))),
                    color="#888888", linestyle=":", linewidth=1.3, label=f"(6.3), n_эксп = {n_exp:.2f}")
            axn.plot(self.ideality.V_local, self.ideality.n_local, marker=".", linestyle="none",
                     color="#444444", ms=3, label="n(V) эксп.")
        if self.model_ideality:
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
        ax.legend(handles + h2, labels + l2, fontsize=6.5, loc="upper left")

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
