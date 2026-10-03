# -*- coding: utf-8 -*-
"""Обучение: первое моделирование в базовом режиме на модельном образце
(методичка, п. 1А.21).

Шаги — список STEPS: на что указать (ключ app.tutorial_targets), что сказать
и какое действие предложить. Окно TutorialWindow подсвечивает элемент рамкой
и показывает рядом текст с кнопками «Назад», «Далее», «Закончить». Обучение
запускается само при первом запуске (флаг в SETTINGS_PATH; переменная
окружения MESA_NO_TUTORIAL=1 отключает автозапуск) и из «Справка → Обучение».
Пример ВАХ — синтетика по модельной структуре (presets.DEFAULT_PARAMS), не
данные образца."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesa_diode.simulator import presets
from mesa_diode.simulator import physics as ph

SETTINGS_PATH = Path.home() / ".mesa_diode" / "settings.json"
DISABLE_ENV = "MESA_NO_TUTORIAL"
EXAMPLE_LABEL = "пример ВАХ (модельный образец)"
# Элементы окна, на которые указывает обучение (app.tutorial_targets).
TARGET_KEYS = ("mode_bar", "params", "D_um", "N_i", "n_emp", "Rs", "load_iv", "graph", "scheme",
               "fit", "results")

# Действия шагов: имя → метод MesaApp (выполняется по кнопке шага).
PREPARE, LOAD_EXAMPLE, FIT = "prepare", "load_example", "fit"


@dataclass(frozen=True)
class Step:
    target: str          # ключ app.tutorial_targets
    title: str
    text: str
    action: str = ""     # подпись кнопки действия (пусто — нет)
    run: str = ""        # действие: PREPARE, LOAD_EXAMPLE, FIT


STEPS = [
    Step("mode_bar", "Первое моделирование",
         "За 2–3 минуты пройдём базовую модель на модельном образце — выдуманной структуре с типичными "
         "значениями (D = 500 мкм, окно кольца 300 мкм, ρ подложки 50 Ом·см), не на реальном образце.\n\n"
         "Поля окна заменятся модельным образцом; свои значения потом можно вернуть, загрузив набор.",
         "Начать с модельного образца", PREPARE),
    Step("mode_bar", "Три режима",
         "Базовая — простой диод: грубо, но каждую точку можно посчитать вручную. Начинаем здесь.\n"
         "Расширенная — модель тока на выбор и всё, что измеряется.\n"
         "Подгонка — ещё и то, что не измеряется (подвижности, времена жизни).\n\n"
         "Компоновка во всех режимах одна: слева параметры, в центре график, справа схема модели."),
    Step("params", "Параметры — слева",
         "В базовом режиме открыто около десяти полей — всё, без чего простой диод не считается. "
         "Остальные скрыты: этой модели они не нужны. Наведите на любое поле — подсказка скажет, что это, "
         "откуда взять значение и на что оно влияет."),
    Step("D_um", "Диаметр мезы D",
         "D задаёт площадь A = πD²/4 — масштаб всех токов и ёмкости. Для своего образца введите D по "
         "фотошаблону или микроскопу. Enter или колесо мыши — пересчёт."),
    Step("N_i", "Концентрации — для ВФХ",
         "N_D⁺, N_i, ρ подложки и N_As в базовом режиме нужны только для ВФХ и V_bi: ток диода от них не "
         "зависит. N_i — по эффекту Холла, ρ — из паспорта пластины."),
    Step("n_emp", "Параметры диода n и J₀",
         "Ток диода: I = A·J₀·(e^(qV/nkT) − 1). n — от 1 (диффузия) до 2 (рекомбинация в ОПЗ), J₀ — ток "
         "насыщения на см². Если загрузить ВАХ, программа подставит их сама по прямой ветви."),
    Step("Rs", "R_s и R_sh",
         "R_s — последовательное сопротивление (контакты, подложка): выпрямляет верх прямой ветви. "
         "R_sh — шунт в обход перехода: прямая линия на обратной ветви. Их найдёт подгонка."),
    Step("load_iv", "Загрузите ВАХ",
         "Своя ВАХ — кнопка «Загрузить ВАХ»: CSV, TXT или Excel со столбцами V и I. Для обучения загрузим "
         "пример — синтетическую ВАХ модельного образца с шумом.",
         "Загрузить пример ВАХ", LOAD_EXAMPLE),
    Step("graph", "График — в центре",
         "Вкладки сверху: ВАХ, ВФХ и 1/C², плотность тока. Точки — эксперимент, линия — модель. Под "
         "графиком — галочки кривых этой модели; у простого диода их немного: сам диод и шунт."),
    Step("scheme", "Схема модели — справа",
         "Справа — как считается модель: эквивалентная схема (наведите на элемент — что это и почему он "
         "здесь), формулы модели с пояснением символов и ссылками на методичку и расчёт одной точки "
         "вручную на текущих числах: формула → числа → результат. Лист с формулами прокручивается."),
    Step("fit", "Подгонка",
         "«Подогнать к ВАХ» подберёт n, J₀, R_s и R_sh так, чтобы линия прошла через точки.",
         "Подогнать сейчас", FIT),
    Step("results", "Результат — внизу",
         "Найденные значения с погрешностью, пояснения «почему так» и отклонение модели от точек. "
         "Базовая модель грубая — большая ошибка здесь нормальна.\n\nДальше — режим «Расширенная»: модель "
         "тока на выбор и больше параметров. Повторить обучение — «Справка → Обучение»."),
]


def should_autostart():
    """Показывать ли обучение при запуске: первый запуск и автозапуск не отключён."""
    if os.environ.get(DISABLE_ENV):
        return False
    try:
        return not json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("tutorial_done")
    except (OSError, ValueError):
        return True


def mark_done():
    """Запомнить, что обучение пройдено (или закрыто) — больше само не открывается."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if SETTINGS_PATH.is_file():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        data["tutorial_done"] = True
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass                               # настройки не сохранились — обучение покажется снова


def example_iv(seed=7):
    """Пример ВАХ для обучения: физическая модель модельной структуры с
    нелинейной утечкой, шум 1 % и дискретность прибора 1 нА. Возвращает (V, I)."""
    params = {**presets.DEFAULT_PARAMS, "mode": presets.MODE_FIT, "tau0_bg": 3e-8,
              "I_L": 3e-7, "m_leak": 2.0}
    s = presets.to_structure(params)
    V = np.round(np.linspace(-3.0, 0.8, 191), 4)
    I = ph.solve_iv(s, V).I
    I = I * (1.0 + 0.01 * np.random.default_rng(seed).standard_normal(V.size))
    return V, np.round(I / 1e-9) * 1e-9


class TutorialWindow:
    """Окно шага и рамка вокруг элемента, на который указывает шаг."""

    BORDER = 3
    COLOR = "#e8590c"

    def __init__(self, app):
        import tkinter as tk
        from tkinter import ttk

        self.app, self.index = app, 0
        self.window = tk.Toplevel(app)
        self.window.title("Обучение")
        self.window.transient(app)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.finish)
        body = ttk.Frame(self.window, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        self.counter = ttk.Label(body, foreground="#777777", font=("Segoe UI", 8))
        self.counter.pack(anchor="w")
        self.title = ttk.Label(body, font=("Segoe UI", 11, "bold"))
        self.title.pack(anchor="w", pady=(0, 4))
        self.text = ttk.Label(body, wraplength=380, justify="left", font=("Segoe UI", 9))
        self.text.pack(anchor="w", fill=tk.X)
        self.action = ttk.Button(body, command=self._run_action)
        self.action.pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(10, 0))
        self.back = ttk.Button(buttons, text="← Назад", command=lambda: self.show(self.index - 1))
        self.back.pack(side=tk.LEFT)
        self.next = ttk.Button(buttons, text="Далее →", command=lambda: self.show(self.index + 1))
        self.next.pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Закончить обучение", command=self.finish).pack(side=tk.RIGHT)
        self.frames = [tk.Frame(app, background=self.COLOR) for _ in range(4)]
        self.show(0)

    def _highlight(self, widget):
        """Рамка из четырёх полос вокруг widget (координаты — в главном окне)."""
        app, b = self.app, self.BORDER
        app.update_idletasks()
        x = widget.winfo_rootx() - app.winfo_rootx()
        y = widget.winfo_rooty() - app.winfo_rooty()
        w, h = widget.winfo_width(), widget.winfo_height()
        for frame, (fx, fy, fw, fh) in zip(self.frames, ((x - b, y - b, w + 2 * b, b), (x - b, y + h, w + 2 * b, b),
                                                         (x - b, y, b, h), (x + w, y, b, h))):
            frame.place(x=fx, y=fy, width=fw, height=fh)
            frame.lift()
        # окно шага — справа от элемента, если там есть место, иначе слева;
        # у широкого элемента (строка режимов) — под ним, чтобы не закрывать его
        sx = widget.winfo_rootx() + w + 20
        sy = min(max(10, widget.winfo_rooty()), app.winfo_screenheight() - 320)
        if w > app.winfo_width() // 2:
            sx, sy = widget.winfo_rootx() + 420, widget.winfo_rooty() + h + 12
        elif sx + 420 > app.winfo_screenwidth():
            sx = max(10, widget.winfo_rootx() - 440)
        self.window.geometry(f"+{sx}+{sy}")

    def show(self, index):
        index = max(0, min(index, len(STEPS) - 1))
        self.index = index
        step = STEPS[index]
        self.counter.config(text=f"Шаг {index + 1} из {len(STEPS)}")
        self.title.config(text=step.title)
        self.text.config(text=step.text)
        if step.action and step.run != PREPARE:
            self.action.config(text=step.action)
            self.action.pack(anchor="w", pady=(8, 0))
        else:
            self.action.pack_forget()
        self.back.state(["!disabled"] if index else ["disabled"])
        if step.run == PREPARE:              # первый шаг: «Далее» открывает модельный образец
            self.next.config(text=step.action + " →", command=self._run_action)
        elif index < len(STEPS) - 1:
            self.next.config(text="Далее →", command=lambda: self.show(self.index + 1))
        else:
            self.next.config(text="Готово", command=self.finish)
        target = self.app.tutorial_targets.get(step.target)
        if target is not None:
            self._highlight(target)
        self.window.lift()

    def _run_action(self):
        run = STEPS[self.index].run
        {PREPARE: self.app.tutorial_prepare, LOAD_EXAMPLE: self.app.tutorial_load_example,
         FIT: self.app.start_fit}[run]()
        if run != FIT:
            self.show(self.index + 1)

    def finish(self):
        mark_done()
        for frame in self.frames:
            frame.destroy()
        self.window.destroy()
        self.app.tutorial_window = None
