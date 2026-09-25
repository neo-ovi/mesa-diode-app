# -*- coding: utf-8 -*-
"""Основное окно симулятора мезадиода Ge (ТЗ §8; компоновка — версия 3.2).

Компоновка по порядку работы: слева — шаги (1. режим, 2. образец,
3. параметры — только используемые в режиме, 4. расчёт и подгонка);
справа — графики и вкладки результатов (подгонка и отклонения модели от
ВАХ, идеальность, предупреждения, схема мезы); внизу — строка статуса.
Всё справочное — в окне «Формулы и параметры». Физика — simulator/physics.py,
автоподгонка — simulator/fitting.py, наборы образцов — simulator/presets.py.

Запуск:        python scripts/run_simulator.py
Сборка в exe:  см. mesa_diode/simulator/README.md
"""

import math
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mesa_diode.simulator import fitting
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator import presets
from mesa_diode.simulator import reference as ref
from mesa_diode.simulator import formulas as formulas_module
from mesa_diode.simulator import help as help_module
from mesa_diode.simulator import hints
from mesa_diode.simulator.widgets import Tooltip, bind_wheel, format_value, wheel_step
from mesa_diode.simulator import datafile
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

# Разделы боковой панели по порядку работы: (заголовок, ключи) — hints.FIELD_GROUPS.
SIDEBAR_GROUPS = [(title, keys) for title, keys, _about in hints.FIELD_GROUPS]
BOUNDARY_TITLE = "Граничные условия и опции"
SIDEBAR_WIDTH = 400
# Короткие подписи в основном окне; полные — в таблице параметров окна формул.
SHORT_LABELS = {
    "D_um": "D (мезы)", "D_inner_um": "d кольца (схема)", "d_epi_um": "d_epi",
    "h_um": "h (травление)", "d_n_um": "d_n (n⁺)", "d_sub_um": "d_sub",
    "ND_plus": "N_D⁺", "N_i": "N_i", "rho_sub": "ρ_sub", "T": "T",
    "mu_n": "μ_n (электроны)", "mu_p_i": "μ_p (i-слой)", "mu_p_nplus": "μ_p (n⁺)",
    "tau_n_bg": "τ_n^bg", "tau_p_bg": "τ_p^bg", "tau0_bg": "τ₀^bg (ОПЗ)",
    "N_dis": "N_dis", "sigma_R_epi": "σ_R (i, n⁺)", "sigma_R_sub": "σ_R (подложка)",
    "n2": "n₂ (ОПЗ)", "Rs": "R_s", "Rsh": "R_sh", "I_L": "I_L", "m_leak": "m",
    "n_emp": "n (6.3)", "J0_emp": "J₀ (6.3)", "I_mod": "I_mod (R_s(I))",
    "afm_rms_nm": "RMS (АСМ)", "afm_defects": "дефекты (АСМ)", "xrd_fwhm": "FWHM (XRD)",
    "hall_mu": "μ Холла (i-слой)", "implant_dose": "доза n⁺ Q", "implant_energy": "энергия ионов",
    "anneal_T": "T отжига", "growth_T": "T роста",
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
                    "afm_rms_nm", "afm_defects", "xrd_fwhm", "hall_mu", "implant_dose",
                    "implant_energy")
AUTO_COLOR = "#1a5fd0"   # поля, заполненные автофитом
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
    if value is None:
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:g}"


class MesaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Симулятор мезадиода Ge — ВАХ, ВФХ, J–V")
        self.geometry("1700x1020")
        self.minsize(1250, 800)

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
        self.show_all = tk.BooleanVar(value=False)
        self.n_exp_var = tk.StringVar()
        self.V1_var = tk.StringVar()
        self.V2_var = tk.StringVar()
        self.c2_from_var = tk.StringVar(value="-1.0")
        self.c2_to_var = tk.StringVar(value="0.0")

        self.structures = {}
        self.results = {}
        self.ideality = None
        self.ideality_notes = []     # причина неудачи n_эксп и замечания
        self.ideality_short = ""     # коротко для строки статуса
        self.ideality_message = ""   # причина, связь с R_s и что сделать
        self._take_from_iv = False   # подставить n и J₀ из новой ВАХ при пересчёте
        self._wheel_after = None
        self.autofilled = {}         # ключ → источник значения, подставленного автофитом
        self.model_ideality = None
        self.comparison = None
        self.reference_data = None
        self.formulas_window = None
        self.fit_result = None       # fitting.FitResult последней автоподгонки
        self._fit_backup = None      # значения полей до подгонки (для «Вернуть»)
        self._fit_thread = None
        self._fit_progress = ""
        self.deviation = None        # (V, отклонение модели от эксперимента, δ)

        self._build_layout()
        self._apply_preset(self.preset)
        self.after(100, self.recompute)

    # ------------------------------------------------------ компоновка окна
    def _build_layout(self):
        """Окно: панель инструментов; слева — шаги работы (режим, образец,
        поля, действия); справа — графики и вкладки результатов; внизу — статус."""
        self._build_menu()
        self._build_toolbar()
        self.status_lbl = ttk.Label(self, text="", font=("Segoe UI", 9, "bold"), anchor="w",
                                    padding=(8, 3), relief="sunken")
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)
        Tooltip(self.status_lbl, hints.plain(hints.with_reference(hints.STATUS_HINT, "vbi")),
                wraplength=520)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(4, 4))
        sidebar = ttk.Frame(paned)
        right = ttk.PanedWindow(paned, orient=tk.VERTICAL)
        paned.add(sidebar, weight=0)
        paned.add(right, weight=1)
        self._build_sidebar(sidebar)
        plots = ttk.Frame(right)
        results = ttk.Frame(right)
        right.add(plots, weight=3)
        right.add(results, weight=2)
        self._build_figure_area(plots)
        self._build_results(results)
        ttk.Style(self).configure("Auto.TEntry", foreground=AUTO_COLOR)
        ttk.Style(self).configure("Section.TButton", anchor="w", font=("Segoe UI", 9, "bold"))

    def _build_menu(self):
        menubar = tk.Menu(self)
        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Загрузить эксперим. ВАХ (файл)...", command=self.load_experimental_iv)
        data_menu.add_command(label="Загрузить эксперим. ВФХ (файл)...", command=self.load_experimental_cv)
        data_menu.add_command(label="Формат файлов данных и примеры", command=self.show_data_format)
        data_menu.add_separator()
        data_menu.add_command(label="Очистить экспериментальные данные", command=self.clear_experimental)
        menubar.add_cascade(label="Данные", menu=data_menu)
        sets_menu = tk.Menu(menubar, tearoff=0)
        sets_menu.add_command(label="Сохранить набор...", command=self.save_preset)
        sets_menu.add_command(label="Загрузить набор...", command=self.load_preset)
        sets_menu.add_command(label="Сбросить к опорному", command=self.reset_to_reference)
        sets_menu.add_command(label="Новый образец (пустые поля)", command=self.new_sample)
        menubar.add_cascade(label="Наборы", menu=sets_menu)
        self.help_menu = tk.Menu(menubar, tearoff=0)
        self.help_menu.add_command(label="Формулы и параметры", command=self.open_formulas_window)
        self.help_menu.add_command(label="Формат файлов данных", command=self.show_data_format)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="Методичка (PDF)",
                                   command=lambda: help_module.open_metodichka(self, "pdf"))
        self.help_menu.add_command(label="Методичка (Word)",
                                   command=lambda: help_module.open_metodichka(self, "docx"))
        menubar.add_cascade(label="Справка", menu=self.help_menu)
        self.config(menu=menubar)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(6, 6, 6, 0))
        bar.pack(side=tk.TOP, fill=tk.X)
        groups = (
            (("Загрузить ВАХ", self.load_experimental_iv, "CSV, TXT или Excel: столбцы V, I (вольты, "
              "амперы; другие единицы — подписью в заголовке). Формат — «Данные → Формат файлов»."),
             ("Загрузить ВФХ", self.load_experimental_cv, "CSV, TXT или Excel: столбцы V, C (вольты, "
              "фарады; другие единицы — подписью в заголовке). Формат — «Данные → Формат файлов»."),
             ("Очистить данные", self.clear_experimental, "Убрать загруженные ВАХ и ВФХ.")),
            (("Сохранить набор", self.save_preset, "Все поля, файлы измерений и метаданные — в JSON."),
             ("Загрузить набор", self.load_preset, "Открыть сохранённый набор образца."),
             ("Опорный", self.reset_to_reference, "Опорный набор из каталога данных."),
             ("Новый образец", self.new_sample, "Все поля пустые, режим «Расширенная модель».")),
            (("Формулы и параметры", self.open_formulas_window, "Что означает каждый параметр, формулы "
              "модели и справочные величины."),
             ("Методичка", lambda: help_module.open_metodichka(self, "pdf"), "Открыть методичку (PDF).")),
        )
        for index, group in enumerate(groups):
            if index:
                ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
            for text, command, tip in group:
                button = ttk.Button(bar, text=text, command=command)
                button.pack(side=tk.LEFT, padx=(0, 4))
                Tooltip(button, tip)
        self.preset_label = ttk.Label(bar, text="", foreground="#555555")
        self.preset_label.pack(side=tk.LEFT, padx=(14, 0))

    # ----------------------------------------------------- боковая панель
    def _build_sidebar(self, parent):
        canvas = tk.Canvas(parent, width=SIDEBAR_WIDTH, highlightthickness=0,
                           background=ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0")
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        body = ttk.Frame(canvas, padding=(4, 2, 8, 8))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        self.sidebar_canvas = canvas

        # Шаг 1. Режим
        step = self._step(body, "1. Режим модели")
        for mode in presets.MODES:
            button = ttk.Radiobutton(step, text=presets.MODE_LABELS[mode], value=mode,
                                     variable=self.mode, command=self._on_mode_change)
            button.pack(anchor="w")
            Tooltip(button, hints.plain(hints.with_reference(hints.MODE_HINTS[mode], "modes")))
        self.mode_lbl = ttk.Label(step, text="", foreground="#555555", font=FONT,
                                  wraplength=SIDEBAR_WIDTH - 40, justify="left")
        self.mode_lbl.pack(anchor="w", fill=tk.X, pady=(2, 0))

        # Шаг 2. Образец
        step = self._step(body, "2. Образец")
        row = ttk.Frame(step)
        row.pack(anchor="w", fill=tk.X)
        ttk.Label(row, text="Подложка:", font=FONT).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Ge", value="Ge", variable=self.substrate).pack(side=tk.LEFT)
        si = ttk.Radiobutton(row, text="Si", value="Si", variable=self.substrate, state="disabled")
        si.pack(side=tk.LEFT)
        Tooltip(si, "Подложка Si — в разработке (бэклог Б-1): нужно сродство к электрону.")
        row = ttk.Frame(step)
        row.pack(anchor="w", fill=tk.X, pady=(2, 0))
        label = ttk.Label(row, text="Тип i-слоя:", font=FONT)
        label.pack(side=tk.LEFT)
        Tooltip(label, "Знак эффекта Холла i-слоя. n — переход i/подложка (сценарий B), p — переход n⁺/i "
                       "(сценарий A), «оба» — сравнение сценариев (п. 10.4 методички).")
        for value, text in ((presets.I_TYPE_N, "n"), (presets.I_TYPE_P, "p"), (presets.I_TYPE_BOTH, "оба")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.i_type,
                            command=self._on_scenario_change).pack(side=tk.LEFT, padx=(4, 0))

        # Шаг 3. Параметры
        step = self._step(body, "3. Параметры")
        show_all = ttk.Checkbutton(step, text="показать все поля", variable=self.show_all,
                                   command=self._refresh_fields)
        show_all.pack(anchor="w")
        Tooltip(show_all, "Показать и поля, не используемые в текущем режиме (они серые). "
                          "Их значения сохраняются в наборе.")
        self.entries, self.field_rows, self.sections = {}, {}, {}
        for title, keys in SIDEBAR_GROUPS:
            section = self._section(step, title)
            for row_index, key in enumerate(keys):
                self._param_row(section.body, row_index, key)
            extra = len(keys)
            if "d_sub_um" in keys:
                self.d_i_label = ttk.Label(section.body, text="", foreground="#555555", font=FONT)
                self.d_i_label.grid(row=extra, column=0, columnspan=3, sticky="w")
            if "tau0_bg" in keys:
                self.tau0_button = ttk.Button(section.body, text="Оценить τ₀ по обратной ветви",
                                              command=self._estimate_tau0)
                self.tau0_button.grid(row=extra, column=0, columnspan=3, sticky="w", pady=(2, 0))
                Tooltip(self.tau0_button, hints.plain(
                    f"τ₀ из тока обратной ветви при V = {TAU0_V:g} В: из измеренного тока "
                    "вычитаются шунт V/R_{sh} и диффузионный ток модели, остаток обращается по (5.4); "
                    "τ₀^{bg} — по (5.10). Нужна загруженная ВАХ."))
            self.sections[title] = (section, keys)
        section = self._section(step, BOUNDARY_TITLE)
        self._build_boundary_box(section.body)
        self.sections[BOUNDARY_TITLE] = (section, [])

        # Шаг 4. Действия
        step = self._step(body, "4. Расчёт и подгонка")
        row = ttk.Frame(step)
        row.pack(anchor="w", fill=tk.X)
        ttk.Button(row, text="Рассчитать", command=self.recompute).pack(side=tk.LEFT, padx=(0, 4))
        self.fit_button = ttk.Button(row, text="Подогнать к ВАХ", command=self.start_fit)
        self.fit_button.pack(side=tk.LEFT, padx=(0, 4))
        Tooltip(self.fit_button, hints.plain(hints.with_reference(hints.FIT_HINT, "fit")))
        row = ttk.Frame(step)
        row.pack(anchor="w", fill=tk.X, pady=(3, 0))
        self.autofit_button = ttk.Button(row, text="Заполнить пустые поля", command=self.autofit)
        self.autofit_button.pack(side=tk.LEFT)
        Tooltip(self.autofit_button, hints.plain(hints.with_reference(hints.AUTOFIT_HINT, "autofit")))
        self._bind_sidebar_scroll(body)

    def _step(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=(6, 2, 6, 4))
        frame.pack(fill=tk.X, pady=(0, 6))
        return frame

    def _section(self, parent, title):
        """Сворачиваемый раздел: заголовок-кнопка и тело с полями."""
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.X, pady=(3, 0))
        state = {"open": True}
        header = ttk.Button(outer, text=f"▾ {title}", style="Section.TButton")
        header.pack(fill=tk.X)
        body = ttk.Frame(outer, padding=(6, 2, 0, 2))
        body.pack(fill=tk.X)

        def toggle():
            state["open"] = not state["open"]
            header.configure(text=f"{'▾' if state['open'] else '▸'} {title}")
            if state["open"]:
                body.pack(fill=tk.X)
            else:
                body.pack_forget()

        header.configure(command=toggle)
        outer.body, outer.header = body, header
        return outer

    def _bind_sidebar_scroll(self, widget):
        """Колесо над панелью прокручивает её; над полем ввода — меняет значение."""
        if not isinstance(widget, (ttk.Entry, ttk.Combobox)):
            bind_wheel(widget, lambda direction, _fine: self.sidebar_canvas.yview_scroll(-direction, "units"))
        for child in widget.winfo_children():
            self._bind_sidebar_scroll(child)

    def _param_row(self, box, row, key):
        spec = SPECS[key]
        hint = hints.plain(hints.with_reference(hints.PARAM_HINTS[key], key))
        label = ttk.Label(box, text=SHORT_LABELS[key], font=FONT, width=16)
        label.grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(box, textvariable=self.vars[key], width=11, justify="center")
        entry.grid(row=row, column=1, padx=3, pady=1)
        entry.bind("<Return>", lambda _e: self.recompute())
        entry.bind("<KeyRelease>", lambda e, key=key: self._clear_autofilled(key)
                   if e.keysym not in ("Return", "Tab", "ISO_Left_Tab") else None, add="+")
        bind_wheel(entry, lambda direction, fine, key=key: self._on_wheel(key, direction, fine))
        unit = ttk.Label(box, text=spec.unit, font=FONT, foreground="#555555")
        unit.grid(row=row, column=2, sticky="w")

        def tooltip_text(key=key, head=f"{spec.label}.\n{hint}"):
            source = self.autofilled.get(key)
            return f"{head}\nПодставлено: {source}." if source else head

        for widget in (label, entry):
            Tooltip(widget, tooltip_text)
        self.entries[key] = entry
        self.field_rows[key] = (label, entry, unit)

    def _clear_autofilled(self, key):
        if self.autofilled.pop(key, None) is not None:
            self.entries[key].configure(style="TEntry")

    def _set_autofilled(self, filled):
        for key in list(self.autofilled):
            self._clear_autofilled(key)
        for key, (value, source) in filled.items():
            self.vars[key].set(format_value(value))
            self.autofilled[key] = source
            self.entries[key].configure(style="Auto.TEntry")

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
        self._clear_autofilled(key)
        if self._wheel_after is not None:
            self.after_cancel(self._wheel_after)
        self._wheel_after = self.after(WHEEL_DELAY_MS, self._wheel_recompute)

    def _wheel_recompute(self):
        self._wheel_after = None
        self.recompute()

    def _build_boundary_box(self, box):
        self.bc_widgets = {}
        bc_hint = "\n".join(f"{ph.BOUNDARY_LABELS[key]} — {hints.plain(text)}"
                            for key, text in hints.BOUNDARY_HINTS.items())
        row = 0
        for scenario in (ph.SCENARIO_A, ph.SCENARIO_B):
            for key, label in BOUNDARY_FIELDS[scenario]:
                lbl = ttk.Label(box, text=label, font=FONT, width=20)
                combo = ttk.Combobox(box, textvariable=self.bc_vars[key], width=13, state="readonly",
                                     values=list(ph.BOUNDARY_LABELS.values()))
                combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())
                lbl.grid(row=row, column=0, sticky="w")
                combo.grid(row=row, column=1, sticky="w", padx=3)
                for widget in (lbl, combo):
                    Tooltip(widget, bc_hint + "\n" + hints.plain(hints.reference("boundary")), wraplength=520)
                self.bc_widgets[key] = (lbl, combo)
                row += 1
        self.flag_buttons = {}
        for key, text in (("sns_refinement", "уточнение SNS (5.5)"),
                          ("edge_area", "краевая добавка площади ОПЗ (1.3)")):
            button = ttk.Checkbutton(box, text=text, variable=self.flag_vars[key], command=self.recompute)
            button.grid(row=row, column=0, columnspan=2, sticky="w")
            self.flag_buttons[key] = button
            row += 1

    # ------------------------------------------------------------ графики
    def _build_figure_area(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Label(bar, text="График 2:", font=FONT).pack(side=tk.LEFT)
        for value, text in (("C", "ВФХ"), ("invC2", "1/C²")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.cv_view,
                            command=self.recompute).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(bar, text="   участок прямой 1/C²: от", font=FONT).pack(side=tk.LEFT)
        for var in (self.c2_from_var, self.c2_to_var):
            e = ttk.Entry(bar, textvariable=var, width=6, justify="center")
            e.pack(side=tk.LEFT, padx=2)
            e.bind("<Return>", lambda _e: self.recompute())
        ttk.Label(bar, text="В", font=FONT).pack(side=tk.LEFT)

        self.fig = Figure(figsize=(12, 3.6), dpi=100)
        gs = self.fig.add_gridspec(1, 3, wspace=0.40)
        self.fig.subplots_adjust(left=0.06, right=0.95, bottom=0.15, top=0.90)
        self.ax_iv = self.fig.add_subplot(gs[0, 0])
        self.ax_cv = self.fig.add_subplot(gs[0, 1])
        self.ax_jv = self.fig.add_subplot(gs[0, 2])
        self.ax_jv_n = self.ax_jv.twinx()
        self._build_curve_bar(parent)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_curve_bar(self, parent):
        """Галочки кривых под каждым графиком; подсказка — что это за кривая."""
        bar = ttk.Frame(parent)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        self.curve_buttons = {}
        per_row = 4
        for column, plot in enumerate(("iv", "cv", "jv")):
            bar.columnconfigure(column, weight=1, uniform="curves")
            frame = ttk.Frame(bar)
            frame.grid(row=0, column=column, sticky="nw", padx=(30, 0))
            for index, (key, label) in enumerate(CURVES[plot]):
                button = ttk.Checkbutton(frame, text=label, variable=self.curve_vars[(plot, key)],
                                         command=self._redraw)
                button.grid(row=index // per_row, column=index % per_row, sticky="w", padx=(0, 6))
                Tooltip(button, hints.plain(hints.with_reference(hints.CURVE_HINTS[key], "curves")))
                self.curve_buttons[(plot, key)] = button

    def _shown(self, plot, key):
        return self.curve_vars[(plot, key)].get()

    # --------------------------------------------------- вкладки результатов
    def _build_results(self, parent):
        self.results_tabs = ttk.Notebook(parent)
        self.results_tabs.pack(fill=tk.BOTH, expand=True)
        self._build_fit_tab()
        self._build_ideality_tab()
        self._build_warnings_tab()
        self._build_mesa_tab()

    def _build_fit_tab(self):
        tab = ttk.Frame(self.results_tabs, padding=4)
        self.results_tabs.add(tab, text="Подгонка и отклонения")
        left = ttk.Frame(tab)
        left.pack(side=tk.LEFT, fill=tk.Y)
        columns = ("value", "error", "unit")
        self.fit_table = ttk.Treeview(left, columns=columns, height=8, selectmode="none")
        self.fit_table.heading("#0", text="параметр")
        for column, text, width in (("value", "значение", 90), ("error", "±", 60), ("unit", "ед.", 60)):
            self.fit_table.heading(column, text=text)
            self.fit_table.column(column, width=width, anchor="center")
        self.fit_table.column("#0", width=140)
        self.fit_table.pack(side=tk.TOP, fill=tk.Y, expand=True)
        buttons = ttk.Frame(left)
        buttons.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        self.undo_fit_button = ttk.Button(buttons, text="Вернуть значения до подгонки",
                                          command=self.undo_fit, state="disabled")
        self.undo_fit_button.pack(side=tk.LEFT)
        middle = ttk.Frame(tab)
        middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        self.fit_text = tk.Text(middle, wrap="word", height=9, width=60, font=FONT, relief="flat",
                                background=ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0")
        self.fit_text.pack(fill=tk.BOTH, expand=True)
        self._set_fit_text(hints.FIT_EMPTY_TEXT)
        self.resid_fig = Figure(figsize=(4.2, 2.3), dpi=100)
        self.resid_fig.subplots_adjust(left=0.2, right=0.97, bottom=0.22, top=0.86)
        self.ax_resid = self.resid_fig.add_subplot(111)
        self.resid_canvas = FigureCanvasTkAgg(self.resid_fig, master=tab)
        self.resid_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.Y)
        Tooltip(self.resid_canvas.get_tk_widget(), hints.plain(hints.RESIDUAL_HINT))

    def _set_fit_text(self, text):
        self.fit_text.configure(state="normal")
        self.fit_text.delete("1.0", "end")
        self.fit_text.insert("end", text)
        self.fit_text.configure(state="disabled")

    def _build_ideality_tab(self):
        box = ttk.Frame(self.results_tabs, padding=6)
        self.results_tabs.add(box, text="Идеальность n")
        row = 0
        ttk.Label(box, text="n_эксп", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w")
        n_entry = ttk.Entry(box, textvariable=self.n_exp_var, width=10, justify="center", state="readonly")
        n_entry.grid(row=row, column=1, padx=3, sticky="w")
        Tooltip(n_entry, "Коэффициент идеальности по прямой ветви первой загруженной ВАХ (§6.3). "
                         "Считается автоматически; при загрузке ВАХ подставляется в n (6.3). "
                         "Точнее n по всей ВАХ даёт «Подогнать к ВАХ».")
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
        self.n_emp_button = ttk.Button(actions, text="n (6.3) ← n_эксп", command=self._n_emp_from_experiment)
        self.n_emp_button.pack(side=tk.LEFT, padx=(0, 6))
        self.n2_button = ttk.Button(actions, text="n₂ ← n_эксп", command=self._n2_from_experiment)
        self.n2_button.pack(side=tk.LEFT)
        row += 1
        self.ideality_lbl = ttk.Label(box, text="", foreground="#b35c00", font=FONT,
                                      wraplength=900, justify="left")
        self.ideality_lbl.grid(row=row, column=0, columnspan=4, sticky="w", pady=(4, 0))
        Tooltip(self.ideality_lbl, hints.plain(hints.reference("ideality")))

    def _build_warnings_tab(self):
        tab = ttk.Frame(self.results_tabs, padding=6)
        self.results_tabs.add(tab, text="Предупреждения")
        self.warnings_text = tk.Text(tab, wrap="word", height=8, font=FONT, relief="flat",
                                     background=ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0")
        self.warnings_text.pack(fill=tk.BOTH, expand=True)
        self.warnings_tab = tab

    def _build_mesa_tab(self):
        tab = ttk.Frame(self.results_tabs, padding=2)
        self.results_tabs.add(tab, text="Схема мезы")
        self.mesa_fig = Figure(figsize=(5.0, 2.8), dpi=100)
        self.mesa_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
        self.ax_mesa = self.mesa_fig.add_subplot(111)
        self.mesa_canvas = FigureCanvasTkAgg(self.mesa_fig, master=tab)
        self.mesa_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------------------------------------------- наборы образцов
    def _apply_preset(self, preset):
        self.preset = preset
        params = {**presets.DEFAULT_PARAMS, **preset.params}
        for key, var in self.vars.items():
            value = params[key]
            var.set(_fmt(None if value is None else float(value)))
        self._set_autofilled({})
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
        problems = []
        for kind, target in (("iv", self.exp_iv), ("cv", self.exp_cv)):
            for path in preset.resolved_files(kind)[:MAX_DATASETS]:
                report = datafile.analyze(path, kind)
                if report.errors or report.warnings:
                    problems.append(f"{Path(path).name}:\n" + report.text(("error", "warning")))
                if report.ok:
                    target.append(self._dataset(path, report))
        if problems:
            messagebox.showwarning("Файлы набора", "\n\n".join(problems))

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

    def new_sample(self):
        """Новый образец: все числовые поля пустые, режим «Расширенная модель».
        Введите измеренное, остальное заполнит «Заполнить пустые»."""
        self._apply_preset(presets.Preset(name="новый образец",
                                          params={**{key: None for key in SPECS},
                                                  "mode": presets.MODE_EXTENDED}))
        self.status_lbl.config(text="Новый образец: введите измеренные значения, затем «Заполнить пустые» "
                                    "заполнит остальные.", foreground="#222222")

    def autofit(self):
        """Заполняет пустые поля текущего режима (см. presets.autofill)."""
        values = {}
        for key, var in self.vars.items():
            try:
                values[key] = float(var.get().strip().replace(",", "."))
            except ValueError:
                values[key] = None
        ideality = self.ideality if self.exp_iv else None
        if ideality is None and self.exp_iv:
            first = self.exp_iv[0]
            T = values["T"] if values["T"] else presets.DEFAULT_PARAMS["T"]
            Rs = values["Rs"] if values["Rs"] is not None else presets.DEFAULT_PARAMS["Rs"]
            ideality = ph.ideality_from_data(first["voltage"], first["value"], T, Rs)
        # Заполняются все пустые поля модели, в том числе затемнённые в текущем
        # режиме: так видно, с какими значениями идёт расчёт.
        filled = presets.autofill(values, list(SPECS), ideality)
        if not filled:
            messagebox.showinfo("Заполнить пустые", "Пустых полей нет — заполнять нечего.")
            return
        self._set_autofilled(filled)
        self.recompute()

    # -------------------------------------------------- сценарий, поля
    def _scenarios(self):
        i_type = self.i_type.get()
        if i_type == presets.I_TYPE_BOTH:
            return [ph.SCENARIO_A, ph.SCENARIO_B]
        return [presets.I_TYPE_TO_SCENARIO[i_type]]

    def _on_scenario_change(self, recompute=True):
        self._refresh_fields()
        if recompute:
            self.recompute()

    def _on_mode_change(self):
        self._on_scenario_change()

    def _refresh_fields(self):
        """Поля боковой панели: видны только используемые в режиме (или все,
        если включено «показать все поля» — тогда лишние серые). Значения не
        меняются. Подвижности, не входящие в сценарий, — серые."""
        scenarios = self._scenarios()
        mode = self.mode.get()
        self.mode_lbl.config(text=hints.plain(hints.MODE_HINTS[mode]))
        editable = presets.editable_keys(mode)
        unused = set.intersection(*(UNUSED_BY_SCENARIO[sc] for sc in scenarios))
        show_all = self.show_all.get()
        for title, (section, keys) in self.sections.items():
            visible = 0
            for key in keys:
                active = key in editable and key not in unused
                self.entries[key].state(["!disabled"] if active else ["disabled"])
                shown = show_all or key in editable
                for widget in self.field_rows[key]:
                    widget.grid() if shown else widget.grid_remove()
                visible += shown
            if title == BOUNDARY_TITLE:
                visible = show_all or mode != presets.MODE_BASIC
            if visible:
                section.pack(fill=tk.X, pady=(3, 0))
            else:
                section.pack_forget()
        physical = mode != presets.MODE_BASIC
        for scenario, fields in BOUNDARY_FIELDS.items():
            for key, _label in fields:
                state = ["!disabled"] if physical and scenario in scenarios else ["disabled"]
                for widget in self.bc_widgets[key]:
                    widget.state(state)
        fit = mode == presets.MODE_FIT
        for button in self.flag_buttons.values():
            button.state(["!disabled"] if fit else ["disabled"])
        self.n2_button.state(["!disabled"] if fit else ["disabled"])
        self.n_emp_button.state(["!disabled"] if mode == presets.MODE_BASIC else ["disabled"])
        self.autofit_button.state(["!disabled"] if physical else ["disabled"])

    def _read_params(self):
        params = {"substrate": self.substrate.get(), "i_type": self.i_type.get(),
                  "mode": self.mode.get()}
        editable = presets.editable_keys(self.mode.get())
        missing = []
        for key, var in self.vars.items():
            raw = var.get().strip().replace(",", ".")
            if not raw:
                # Пустое поле: «не измерено» (хранимые поля) или ещё не задано.
                params[key] = None
                if key in editable and key not in presets.STORED_KEYS:
                    missing.append(SPECS[key].label)
                continue
            try:
                params[key] = float(raw)
            except ValueError:
                raise ValueError(f"«{SPECS[key].label}» задан некорректно: «{raw}»")
        if missing:
            hint = ("введите значения или нажмите «Заполнить пустые» — программа подставит значения по умолчанию "
                    "и оценки по данным." if self.mode.get() != presets.MODE_BASIC
                    else "введите значения (или переключитесь в «Расширенную модель» и нажмите «Заполнить пустые»).")
            raise ValueError("Не заданы: " + ", ".join(missing) + ".\n" + hint)
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
            if params[key] is not None and params[key] <= 0:
                raise ValueError(f"«{SPECS[key].label}» должно быть больше нуля.")
        for key in NONNEGATIVE_KEYS:
            if params[key] is not None and params[key] < 0:
                raise ValueError(f"«{SPECS[key].label}» не может быть отрицательным.")
        value = self._value_or_default
        if value(params, "d_n_um") >= value(params, "d_epi_um"):
            raise ValueError("Толщина n⁺-слоя d_n должна быть меньше толщины эпитаксии d_epi.")
        if value(params, "D_inner_um") >= value(params, "D_um"):
            raise ValueError("Внутренний диаметр кольца должен быть меньше диаметра мезы D.")
        return params

    @staticmethod
    def _value_or_default(params, key):
        """Значение поля; для пустого (недоступного в режиме) — значение по умолчанию."""
        return presets.DEFAULT_PARAMS[key] if params[key] is None else params[key]

    def _ideality_unavailable_text(self):
        """Почему n_эксп нет: нет ВАХ или расчёт не удался (с причиной и решением)."""
        if not self.exp_iv:
            return "n_эксп ещё не определён: загрузите ВАХ."
        return self.ideality_message or "n_эксп не определён: причина не установлена."

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

    # ------------------------------------------------------ автоподгонка
    def start_fit(self):
        """«Подогнать к ВАХ»: в базовом режиме — эмпирическая модель (J₀, n),
        иначе — физическая (τ₀^bg, τ^bg); вместе с эквивалентной схемой.
        Считается в фоне, окно не замирает."""
        if self._fit_thread is not None:
            return
        if not self.exp_iv:
            messagebox.showinfo("Подгонка", "Загрузите ВАХ: подгонка ищет параметры, при которых модель "
                                            "совпадает с ней.")
            return
        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода параметров", str(e))
            return
        s = presets.to_structure(params, self._scenarios()[-1])
        model = fitting.EMPIRICAL if self.mode.get() == presets.MODE_BASIC else fitting.PHYSICAL
        data = self.exp_iv[0]
        self._fit_backup = {key: var.get() for key, var in self.vars.items()}
        self._fit_box = {"progress": ""}
        self.fit_button.state(["disabled"])

        def work():
            try:
                self._fit_box["result"] = fitting.fit_iv(
                    s, data["voltage"], data["value"], model,
                    progress=lambda text: self._fit_box.__setitem__("progress", text))
            except Exception as error:     # сообщение показывается в главном потоке
                self._fit_box["error"] = str(error)

        self._fit_thread = threading.Thread(target=work, daemon=True)
        self._fit_thread.start()
        self._poll_fit()

    def _poll_fit(self):
        if self._fit_thread.is_alive():
            self.status_lbl.config(text=f"Подгонка к ВАХ… {self._fit_box['progress']}", foreground=AUTO_COLOR)
            self.after(200, self._poll_fit)
            return
        self._fit_thread = None
        self.fit_button.state(["!disabled"])
        if "error" in self._fit_box:
            messagebox.showerror("Подгонка", f"Подгонка не удалась: {self._fit_box['error']}")
            self.recompute()
            return
        self._apply_fit(self._fit_box["result"])

    def _apply_fit(self, result):
        """Найденные значения — в поля (выделены синим), результат — во вкладку."""
        self.fit_result = result
        source = "автоподгонка к ВАХ (§6.7)"
        values = {key: float(f"{value:.5g}") if np.isfinite(value) else value
                  for key, value in result.structure_values().items() if key in self.vars}
        self._set_autofilled({key: (value, source) for key, value in values.items()})
        self._take_from_iv = False      # n и J₀ уже найдены подгонкой — не подставлять n_эксп
        self.undo_fit_button.state(["!disabled"])
        self._show_fit_result(result)
        self.results_tabs.select(0)
        self.recompute()

    def undo_fit(self):
        if not self._fit_backup:
            return
        for key, value in self._fit_backup.items():
            self.vars[key].set(value)
        self._set_autofilled({})
        self._fit_backup = None
        self.undo_fit_button.state(["disabled"])
        self._set_fit_text("Значения до подгонки восстановлены. " + hints.FIT_EMPTY_TEXT)
        self.recompute()

    def _show_fit_result(self, result):
        self.fit_table.delete(*self.fit_table.get_children())
        for key, value in result.params.items():
            log_scale = fitting.PARAMS[key][1]
            sigma = result.stderr.get(key, float("nan"))
            if not np.isfinite(value) or value == 0:
                text, error = ("выкл." if key in ("I_L", "I_mod") else _fmt(value)), ""
            else:
                text = f"{value:.4g}"
                error = "" if not np.isfinite(sigma) else (f"{100 * sigma:.1f} %" if log_scale else f"{sigma:.2g}")
            self.fit_table.insert("", "end", text=fitting.LABELS[key],
                                  values=(text, error, fitting.UNITS[key]))
        model = "эмпирическая (6.3)" if result.model == fitting.EMPIRICAL else "физическая (§4–§6)"
        head = (f"Модель: {model}. Ошибка (6.10): {100 * result.error:.2f} %. "
                "Найденные значения подставлены в поля и выделены синим.\n")
        self._set_fit_text(head + "\n".join("• " + note for note in result.notes))

    # --------------------------------------------- экспериментальные данные
    @staticmethod
    def _dataset(path, report):
        return {"label": Path(path).name, "path": str(path), "voltage": report.voltage,
                "value": report.value, "report": report}

    def _load_experimental_files(self, dialog_title, target_list, error_title, kind):
        paths = filedialog.askopenfilenames(title=dialog_title, filetypes=datafile.FILE_TYPES)
        if not paths:
            return
        slots = remaining_slots(len(target_list))
        accepted, rejected = paths[:slots], paths[slots:]
        for path in accepted:
            try:
                report = datafile.analyze(path, kind)
            except Exception as e:  # непредвиденный сбой чтения — сообщить, не падать
                messagebox.showerror(error_title, f"{Path(path).name}: {e}\n"
                                     f"Подробнее: методичка, {datafile.REF}.")
                continue
            if not report.ok:
                messagebox.showerror(error_title, f"{Path(path).name} не загружен.\n\n"
                                     + report.text(("error", "warning")))
                continue
            if report.warnings:
                messagebox.showwarning(error_title.replace("Ошибка", "Проверка"),
                                       f"{Path(path).name} загружен, но проверьте данные.\n\n"
                                       + report.text(("warning", "info")))
            target_list.append(self._dataset(path, report))
        if target_list is self.exp_iv and accepted:
            self._take_from_iv = True
        if rejected:
            messagebox.showwarning(
                "Достигнут лимит наборов данных",
                f"Загружено {len(accepted)} из {len(paths)} файлов — на графике уже "
                f"максимум {MAX_DATASETS} наборов. Уберите лишние через «Очистить данные».")
        self.recompute()

    def load_experimental_iv(self):
        self._load_experimental_files("Файлы экспериментальной ВАХ (V, I)", self.exp_iv,
                                      "Ошибка загрузки ВАХ", datafile.KIND_IV)

    def load_experimental_cv(self):
        self._load_experimental_files("Файлы экспериментальной ВФХ (V, C)", self.exp_cv,
                                      "Ошибка загрузки ВФХ", datafile.KIND_CV)

    def show_data_format(self):
        """Окно «Формат файлов данных»: описание и открытие каталога примеров."""
        window = tk.Toplevel(self)
        window.title("Формат файлов данных")
        text = tk.Text(window, wrap="word", width=78, height=34, font=FONT, relief="flat")
        text.insert("end", datafile.FORMAT_HELP + f"\nПодробнее: методичка, {datafile.REF}.\n\n"
                    "Примеры (синтетические данные):\n"
                    + "\n".join(f"  {name}" for name in datafile.example_files()))
        text.configure(state="disabled")
        buttons = ttk.Frame(window, padding=(8, 0, 8, 8))
        buttons.pack(side=tk.BOTTOM, fill=tk.X)
        scroll = ttk.Scrollbar(window, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        text.pack(fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        ttk.Button(buttons, text="Открыть каталог примеров",
                   command=self._open_examples).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Проверить файл...", command=self.check_data_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Закрыть", command=window.destroy).pack(side=tk.RIGHT)

    def _open_examples(self):
        error = help_module.open_with_system(datafile.EXAMPLES_DIR)
        if error:
            messagebox.showinfo("Примеры", f"Каталог примеров: {datafile.EXAMPLES_DIR}\n({error})")

    def check_data_file(self):
        """Разбор файла без загрузки: все замечания, включая «сделано»."""
        path = filedialog.askopenfilename(title="Проверить файл данных", filetypes=datafile.FILE_TYPES)
        if not path:
            return
        kind = datafile.KIND_CV if messagebox.askyesno(
            "Проверить файл", "Это ВФХ (ёмкость)?\n«Нет» — ВАХ (ток).") else datafile.KIND_IV
        report = datafile.analyze(path, kind)
        head = (f"{Path(path).name}: {len(report.voltage)} точек, V от {report.voltage.min():.3g} до "
                f"{report.voltage.max():.3g} В." if report.ok else f"{Path(path).name} не загрузится.")
        body = report.text() or "Замечаний нет."
        (messagebox.showinfo if report.ok else messagebox.showerror)("Проверка файла", head + "\n\n" + body)

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
        d_i = self._value_or_default(params, "d_epi_um") - self._value_or_default(params, "d_n_um")
        self.d_i_label.config(text=f"d_i = d_epi − d_n = {d_i:.3g} мкм (вычисляется)")

        v_min, v_max = adaptive_voltage_range([d["voltage"] for d in self.exp_iv])
        V = np.linspace(v_min, v_max, adaptive_point_count(v_min, v_max))
        self.results = {sc: ph.solve_iv(s, V) for sc, s in self.structures.items()}

        window = self._ideality_window(params)
        model_window = window or ((self.ideality.V1, self.ideality.V2) if self.ideality else None)
        main_result = self.results[main.scenario]
        self.model_ideality = ph.ideality_from_data(V, main_result.I, main.T, main.Rs, model_window,
                                                    I_mod=main.I_mod)
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
        self._update_deviation(main)
        self._redraw()
        self._plot_mesa(params, main)
        self._update_status(n_exp, n_mod)
        if self.formulas_window is not None:
            self.formulas_window.refresh()

    def _update_deviation(self, s):
        """Отклонение модели от первой ВАХ в её точках: (V, отклонение, δ (6.10))."""
        self.deviation = None
        if self.exp_iv:
            V, I = fitting.prepare_data(self.exp_iv[0]["voltage"], self.exp_iv[0]["value"], max_points=600)
            if V.size:
                I_model = ph.solve_iv(s, V).I
                floor = 10.0 * fitting.current_scale(I)
                deviation = (I_model - I) / np.maximum(np.abs(I), floor)
                self.deviation = (V, deviation, fitting.relative_error(I_model, I, floor))
        self._plot_deviation()

    def _plot_deviation(self):
        ax = self.ax_resid
        ax.clear()
        ax.set_title("Отклонение модели от ВАХ", fontsize=9)
        ax.set_xlabel("V, В", fontsize=8)
        ax.set_ylabel("(I_мод − I_эксп)/|I_эксп|, %", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="#999999", linewidth=0.7)
        if self.deviation is not None:
            V, deviation, delta = self.deviation
            ax.plot(V, 100 * deviation, ".", ms=2, color=SINGLE_COLOR)
            limit = max(1.0, min(100.0, float(np.nanpercentile(np.abs(100 * deviation), 98)) * 1.2))
            ax.set_ylim(-limit, limit)
            ax.set_title(f"Отклонение модели от ВАХ: δ = {100 * delta:.2f} %", fontsize=9)
        else:
            ax.text(0.5, 0.5, "загрузите ВАХ", ha="center", va="center", transform=ax.transAxes,
                    color="#888888")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        self.resid_canvas.draw_idle()

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
        self.ideality_short, self.ideality_message = "", ""
        T, Rs = self._value_or_default(params, "T"), self._value_or_default(params, "Rs")
        I_mod = self._value_or_default(params, "I_mod")
        if self.exp_iv:
            first = self.exp_iv[0]
            self.ideality = ph.ideality_from_data(first["voltage"], first["value"], T, Rs,
                                                  self._ideality_window(params),
                                                  diagnostics=self.ideality_notes, I_mod=I_mod)
            rs_limit = ph.series_resistance_limit(first["voltage"], first["value"])
            # с предельным наклоном dV/dI сравнивается дифференциальное сопротивление
            # d(I·R_s(I))/dI = R_s/(1 + |I|/I_mod)² на верху прямой ветви (6.9)
            Rs_top = Rs / (1.0 + float(np.max(np.abs(first["value"]))) / I_mod) ** 2
            self.ideality_short, self.ideality_message = hints.ideality_failure_text(
                first["label"], self.ideality_notes, Rs_top, rs_limit, self.ideality is not None)
        if self.ideality:
            self.n_exp_var.set(f"{self.ideality.n:.3f}")
            if self.window_auto.get():
                self.V1_var.set(f"{self.ideality.V1:.3f}")
                self.V2_var.set(f"{self.ideality.V2:.3f}")
        else:
            self.n_exp_var.set("—")
        self.ideality_lbl.config(text=self.ideality_message)
        if self._take_from_iv and self.ideality:
            area = presets.to_structure(params).area
            params["n_emp"] = float(f"{self.ideality.n:.3g}")
            params["J0_emp"] = float(f"{self.ideality.I0 / area:.3g}")
            self.vars["n_emp"].set(format_value(params["n_emp"]))
            self.vars["J0_emp"].set(format_value(params["J0_emp"]))
            for key in ("n_emp", "J0_emp"):
                self._clear_autofilled(key)
            # Флаг снимается только после подстановки: если n_эксп пока не найден
            # (например, R_s завышено), n и J₀ подставятся после исправления.
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
            flag = f" ({self.ideality_short})" if self.ideality_short else ""
            parts.append(f"n_эксп = {n_exp:.3f}{flag}")
        elif self.exp_iv:
            parts.append(f"n_эксп = — ({self.ideality_short or 'см. блок идеальности'})")
        else:
            parts.append("n_эксп = — (нет ВАХ)")
        parts.append(f"n_мод = {n_mod:.3f}" if n_mod else "n_мод = —")
        if self.deviation is not None:
            parts.append(f"отклонение от ВАХ δ = {100 * self.deviation[2]:.2f} %")
        warnings = list(self.reference_data.warnings) if self.reference_data else []
        for data in self.exp_iv + self.exp_cv:
            report = data.get("report")
            warnings += [f"{data['label']}: {issue.text} {issue.hint} (методичка, {issue.ref})"
                         for issue in (report.warnings if report else [])]
        count = len(warnings)
        self.warnings_text.configure(state="normal")
        self.warnings_text.delete("1.0", "end")
        self.warnings_text.insert("end", "\n".join("⚠ " + hints.plain(w) for w in warnings)
                                  or "Предупреждений нет.")
        self.warnings_text.configure(state="disabled")
        self.results_tabs.tab(self.warnings_tab, text=f"Предупреждения ({count})")
        if count:
            word = ("предупреждение" if count % 10 == 1 and count % 100 != 11 else
                    "предупреждения" if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14 else
                    "предупреждений")
            parts.append(f"⚠ {count} {word} — вкладка «Предупреждения»")
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
        value = lambda key: self._value_or_default(params, key)  # noqa: E731
        diagram_params = type("DiagramParams", (), {
            "D_um": value("D_um"), "d_um": value("D_inner_um"), "h_um": value("h_um"),
            "ND_plus": value("ND_plus"), "N_i": value("N_i"), "N_sub": s.substrate[0],
            "T": value("T"), "i_type": {"n": "i-слой n", "p": "i-слой p", "both": "тип i: оба"}[params["i_type"]],
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
