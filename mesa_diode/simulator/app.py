# -*- coding: utf-8 -*-
"""Графический интерфейс симулятора мезоструктуры Ge/Si — ВАХ и ВФХ.

Версия 2. Возможности:
  - загрузка экспериментальных данных ВАХ и ВФХ из нескольких файлов сразу
    (до simulator.style.MAX_DATASETS штук на график) для сравнения с моделью;
  - логарифмическая шкала ёмкости на графике ВФХ (оценка резкости перехода);
  - отдельное окно «Формулы и параметры» (см. simulator/formulas.py);
  - подробные комментарии в местах, отвечающих за отрисовку/масштаб/расположение —
    см. блоки "### ЗДЕСЬ ... ###" внутри методов _plot_iv, _plot_cv, _build_input_row;
    схема мезы вынесена в simulator/diagram.py (draw_mesa_diagram).

Запуск:        python scripts/run_simulator.py
Сборка в exe:  см. mesa_diode/simulator/README.md
"""

from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mesa_diode.simulator.physics import (
    MesaParams, auto_scale, capacitance, estimate_grading_m, solve_iv,
)
from mesa_diode.simulator.formulas import build_formulas_text
from mesa_diode.simulator.io import load_xy_file
from mesa_diode.simulator.diagram import draw_mesa_diagram
from mesa_diode.simulator.style import MAX_DATASETS, dataset_style, remaining_slots

PARAM_SPECS = [
    # (ключ, обозначение, единицы, значение_по_умолчанию)
    ("D",   "D",     "мкм",     "50"),
    ("h",   "h",     "мкм",     "2"),
    ("ND",  "N_D",   "см⁻³",    "1e17"),
    ("NA",  "N_A",   "см⁻³",    "5e17"),
    ("T",   "T",     "К",       "300"),
    ("Rs",  "R_s",   "Ом",      "10"),
    ("Rsh", "R_sh",  "Ом",      "1e6"),
    ("n2",  "n₂",    "б/р",     "2.0"),
]


class MesaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Симулятор мезы Ge/Si — ВАХ и ВФХ")
        self.geometry("1500x980")
        self.minsize(1200, 820)

        # Экспериментальные данные, загруженные из файлов — до MAX_DATASETS
        # штук на каждый график. Каждый элемент: {"label", "voltage", "value"}
        self.exp_iv = []
        self.exp_cv = []

        self._build_menu()
        self._build_figure_area()
        self._build_input_row()

        self.entries["D"].focus_set()
        self.after(100, self.recompute)

    # ---------------- меню: загрузка данных, окно формул ----------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Загрузить эксперим. ВАХ (файл)...",
                               command=self.load_experimental_iv)
        data_menu.add_command(label="Загрузить эксперим. ВФХ (файл)...",
                               command=self.load_experimental_cv)
        data_menu.add_separator()
        data_menu.add_command(label="Очистить экспериментальные данные",
                               command=self.clear_experimental)
        menubar.add_cascade(label="Данные", menu=data_menu)

        info_menu = tk.Menu(menubar, tearoff=0)
        info_menu.add_command(label="Формулы и параметры модели",
                               command=self.open_formulas_window)
        menubar.add_cascade(label="Справка", menu=info_menu)

        self.config(menu=menubar)

        # Дублирующая строка кнопок для быстрого доступа (то же самое, что в меню)
        quick_row = ttk.Frame(self)
        quick_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 0))
        ttk.Button(quick_row, text="Загрузить эксперим. ВАХ",
                   command=self.load_experimental_iv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(quick_row, text="Загрузить эксперим. ВФХ",
                   command=self.load_experimental_cv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(quick_row, text="Очистить эксперим. данные",
                   command=self.clear_experimental).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(quick_row, text="Формулы и параметры",
                   command=self.open_formulas_window).pack(side=tk.LEFT, padx=(0, 6))

    # ---------------- верхняя область: графики + схема ----------------
    def _build_figure_area(self):
        self.fig = Figure(figsize=(13, 6.5), dpi=100)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0],
                                    hspace=0.45, wspace=0.28)
        self.ax_iv = self.fig.add_subplot(gs[0, 0])
        self.ax_cv = self.fig.add_subplot(gs[0, 1])
        self.ax_mesa = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                                          padx=8, pady=(8, 0))

    # ---------------- нижняя область: ввод параметров ----------------
    def _build_input_row(self):
        outer = ttk.Frame(self)
        outer.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        title = ttk.Label(outer, text="Параметры мезы:", font=("Segoe UI", 10, "bold"))
        title.pack(side=tk.TOP, anchor="w")

        row = ttk.Frame(outer)
        row.pack(side=tk.TOP, fill=tk.X, pady=(4, 4))

        # ### ЗДЕСЬ формируется РЯД полей "обозначение - поле - единицы" ###
        # Если поля "наезжают" друг на друга или обрезаются при уменьшении
        # окна — причина в том, что этот Frame не переносит элементы на
        # новую строку (pack(side=LEFT) укладывает всё в один ряд без wrap).
        # Быстрые способы поправить:
        #   (а) уменьшить width= у Entry/Label ниже;
        #   (б) заменить pack на grid с фиксированным числом столбцов и
        #       расчётом row = index // N_COLS, column = index % N_COLS;
        #   (в) обернуть row в Canvas с горизontal Scrollbar.
        self.entries = {}
        for key, symbol, unit, default in PARAM_SPECS:
            group = ttk.Frame(row, padding=(6, 2))
            group.pack(side=tk.LEFT)

            lbl = ttk.Label(group, text=symbol, width=4,          # <- ширина подписи-обозначения
                             font=("Segoe UI", 10, "bold"), anchor="e")
            lbl.pack(side=tk.LEFT, padx=(0, 3))

            entry = ttk.Entry(group, width=8, justify="center")   # <- ширина поля ввода
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            entry.bind("<Return>", lambda e: self.recompute())

            unit_lbl = ttk.Label(group, text=unit, width=6, anchor="w")  # <- ширина подписи единиц
            unit_lbl.pack(side=tk.LEFT, padx=(3, 0))

            self.entries[key] = entry

        btn_row = ttk.Frame(outer)
        btn_row.pack(side=tk.TOP, fill=tk.X)

        calc_btn = ttk.Button(btn_row, text="Рассчитать", command=self.recompute)
        calc_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.status_lbl = ttk.Label(btn_row, text="", foreground="#555555")
        self.status_lbl.pack(side=tk.LEFT)

    # ---------------- окно «Формулы и параметры» ----------------
    def open_formulas_window(self):
        """
        Открывает отдельное окно (Toplevel) с описанием всех формул и
        параметров модели.

        ЧТОБЫ ИЗМЕНИТЬ СОДЕРЖИМОЕ ОКНА: правьте список FORMULA_SECTIONS
        в simulator/formulas.py — эта функция лишь отображает то, что
        вернёт build_formulas_text(); сама функция ничего не решает по
        содержанию и трогать её для правки текста формул не нужно.

        ЧТОБЫ ИЗМЕНИТЬ ВНЕШНИЙ ВИД ОКНА (размер, шрифт, перенос строк):
          - geometry("800x700")   -> размер окна при открытии;
          - font=("Consolas", 10) -> шрифт текста (моноширинный удобен для формул);
          - wrap="word"           -> перенос по словам ("none" — без переноса,
                                      тогда появится горизonтальная прокрутка).
        """
        win = tk.Toplevel(self)
        win.title("Формулы и параметры модели")
        win.geometry("800x700")

        text_widget = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 10))
        text_widget.pack(fill="both", expand=True, padx=8, pady=8)

        content = build_formulas_text()
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")  # только чтение, чтобы не редактировали случайно

        ttk.Button(win, text="Закрыть", command=win.destroy).pack(side="bottom", pady=(0, 8))

    # ---------------- загрузка/очистка экспериментальных данных ----------------
    def _load_experimental_files(self, dialog_title, target_list, error_title):
        """Общая логика загрузки для ВАХ и ВФХ: несколько файлов за раз,
        до MAX_DATASETS штук суммарно на график (см. simulator/style.py)."""
        paths = filedialog.askopenfilenames(
            title=dialog_title,
            filetypes=[("Текст/CSV", "*.csv;*.txt;*.dat"), ("Все файлы", "*.*")])
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
            target_list.append({"label": Path(path).name, "voltage": voltage, "value": value})

        if rejected:
            messagebox.showwarning(
                "Достигнут лимит наборов данных",
                f"Загружено {len(accepted)} из {len(paths)} файлов — уже "
                f"на графике максимум {MAX_DATASETS} наборов. Остальные "
                f"{len(rejected)} не загружены. Уберите лишние через "
                f"«Очистить эксперим. данные» и загрузите заново."
            )

        self.recompute()

    def load_experimental_iv(self):
        self._load_experimental_files(
            "Выберите файлы экспериментальной ВАХ (V, I)", self.exp_iv, "Ошибка загрузки ВАХ")

    def load_experimental_cv(self):
        self._load_experimental_files(
            "Выберите файлы экспериментальной ВФХ (V, C)", self.exp_cv, "Ошибка загрузки ВФХ")

    def clear_experimental(self):
        self.exp_iv = []
        self.exp_cv = []
        self.recompute()

    # ---------------- чтение и валидация введённых значений ----------------
    def _read_params(self):
        vals = {}
        for key, symbol, unit, _ in PARAM_SPECS:
            raw = self.entries[key].get().strip().replace(",", ".")
            try:
                vals[key] = float(raw)
            except ValueError:
                raise ValueError(f"Параметр {symbol} задан некорректно: «{raw}»")

        if vals["D"] <= 0 or vals["h"] <= 0:
            raise ValueError("Диаметр D и глубина h должны быть положительными.")
        if vals["ND"] <= 0 or vals["NA"] <= 0:
            raise ValueError("Концентрации N_D и N_A должны быть положительными.")
        if vals["T"] <= 0:
            raise ValueError("Температура T должна быть положительной (в кельвинах).")
        if vals["Rs"] < 0 or vals["Rsh"] <= 0:
            raise ValueError("R_s должно быть ≥ 0, R_sh должно быть > 0.")
        if vals["n2"] <= 0:
            raise ValueError("Коэффициент неидеальности n₂ должен быть положительным.")

        return MesaParams(
            D_um=vals["D"], h_um=vals["h"],
            ND_si=vals["ND"], NA_ge=vals["NA"],
            T_K=vals["T"], Rs_ohm=vals["Rs"], Rsh_ohm=vals["Rsh"], n2=vals["n2"],
        )

    # ---------------- основной пересчёт и перерисовка ----------------
    def recompute(self):
        try:
            p = self._read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода параметров", str(e))
            return

        V_iv = np.linspace(-1.0, 0.6, 121)
        I = solve_iv(V_iv, p)

        Vc_max = max(0.05, 0.9 * p.Vbi)
        V_cv = np.linspace(-1.5, Vc_max, 121)
        C = capacitance(V_cv, p)

        self._plot_iv(V_iv, I)
        self._plot_cv(V_cv, C, p.Vbi)
        self._plot_mesa(p)

        m, slope = estimate_grading_m(V_cv, C, p.Vbi)
        m_text = f",  m≈{m:.2f} (резкость по модели)" if m is not None else ""
        self.status_lbl.config(
            text=(f"Расчёт выполнен.  V_bi = {p.Vbi:.3f} В,  "
                  f"A = {p.A:.3e} см²,  N_eff = {p.Neff:.3e} см⁻³{m_text}")
        )
        self.canvas.draw_idle()

    # ---------------- отрисовка ВАХ ----------------
    def _plot_iv(self, V, I):
        ax = self.ax_iv
        ax.clear()

        # ### ЗДЕСЬ задаётся МАСШТАБ отображения тока (авто-приставка А/мА/мкА/нА) ###
        # Если нужен фиксированный масштаб — замените вызов auto_scale(...)
        # на конкретную пару, например: factor, unit = 1e3, "мА"
        combo = np.concatenate([I] + [d["value"] for d in self.exp_iv])
        factor, unit = auto_scale(combo, "А")

        # ### ЗДЕСЬ задаётся сама КРИВАЯ МОДЕЛИ ВАХ ###
        ax.plot(V, I * factor, color="#1f6fb2", linewidth=1.8, label="модель")

        # ### ЗДЕСЬ добавляются ЭКСПЕРИМЕНТАЛЬНЫЕ точки ВАХ (если загружены) ###
        # У каждого набора — свой маркер и цвет (simulator/style.py), в
        # легенде — имя файла, чтобы несколько кривых были различимы.
        for index, dataset in enumerate(self.exp_iv):
            marker, color = dataset_style(index)
            ax.plot(dataset["voltage"], dataset["value"] * factor, marker, ms=5,
                    color=color, markerfacecolor="none", label=dataset["label"])

        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.axvline(0, color="#999999", linewidth=0.7)
        ax.set_title("Вольт-амперная характеристика (ВАХ)", fontsize=10)
        ax.set_xlabel("Напряжение V, В")
        ax.set_ylabel(f"Ток I, {unit}")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        if self.exp_iv:
            ax.legend(fontsize=8, loc="best")

    # ---------------- отрисовка ВФХ ----------------
    def _plot_cv(self, V, C, Vbi):
        ax = self.ax_cv
        ax.clear()

        C_safe = np.nan_to_num(C, nan=0.0)
        combo = np.concatenate([C_safe] + [d["value"] for d in self.exp_cv])
        factor, unit = auto_scale(combo, "Ф")

        # ### ЗДЕСЬ задаётся сама КРИВАЯ МОДЕЛИ ВФХ ###
        ax.plot(V, C * factor, color="#b2401f", linewidth=1.8, label="модель")

        # ### ЗДЕСЬ добавляются ЭКСПЕРИМЕНТАЛЬНЫЕ точки ВФХ (если загружены) ###
        # У каждого набора — свой маркер и цвет (simulator/style.py), в
        # легенде — имя файла, чтобы несколько кривых были различимы.
        for index, dataset in enumerate(self.exp_cv):
            marker, color = dataset_style(index)
            mask = dataset["value"] > 0  # логарифмическая шкала требует C > 0
            ax.plot(dataset["voltage"][mask], dataset["value"][mask] * factor, marker, ms=5,
                    color=color, markerfacecolor="none", label=dataset["label"])

        # ### ЗДЕСЬ настраивается ЛОГАРИФМИЧЕСКИЙ МАСШТАБ оси ёмкости ###
        # Лог. шкала по оси Y нужна, чтобы визуально оценивать резкость
        # p-n перехода: на логарифмической шкале степенная зависимость
        # C ~ (Vbi-V)^(-1/(m+2)) выглядит как ПРЯМАЯ линия, и по её наклону
        # (см. функцию estimate_grading_m) можно отличить резкий переход
        # (m≈0) от плавного/градиентного (m≈1). Если нужно вернуть линейную
        # шкалу — замените "log" на "linear" строкой ниже.
        ax.set_yscale("log")

        ax.set_title("Вольт-фарадная характеристика (ВФХ), лог. шкала C", fontsize=10)
        ax.set_xlabel("Напряжение V, В")
        ax.set_ylabel(f"Ёмкость C, {unit} (лог. шкала)")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
        if self.exp_cv:
            ax.legend(fontsize=8, loc="best")

    # ---------------- схематическое изображение мезы ----------------
    def _plot_mesa(self, p):
        ax = self.ax_mesa
        ax.clear()
        draw_mesa_diagram(ax, p)


def main():
    # Cyrillic-совместимый шрифт по умолчанию (DejaVu Sans идёт с matplotlib)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    app = MesaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
