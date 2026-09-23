# -*- coding: utf-8 -*-
"""Вспомогательные элементы окна: всплывающие подсказки и изменение
значения поля колесом мыши."""

import math

# Шаг от нуля вверх для параметров, где ноль допустим.
ZERO_STEPS = {"N_dis": 1e3, "I_L": 1e-6}
ZERO_STEP_DEFAULT = 0.1


def wheel_step(value, direction, fine=False, key=None):
    """Новое значение поля после одного щелчка колеса.

    Шаг — одна десятая старшего разряда: 10^(⌊lg|x|⌋ − 1) (для 500 — 10,
    для 5·10¹⁵ — 10¹⁴); с Ctrl — в 10 раз мельче. direction > 0 — вверх.
    Значение не уходит в ноль и ниже: шаг всегда не больше десятой части x.
    Для нуля вверх берётся ZERO_STEPS[key], вниз — ноль остаётся."""
    if value == 0:
        if direction <= 0:
            return 0.0
        step = ZERO_STEPS.get(key, ZERO_STEP_DEFAULT)
        return step / 10 if fine else step
    if not math.isfinite(value) or value < 0:
        return value
    step = 10.0 ** (math.floor(math.log10(value)) - 1)
    if fine:
        step /= 10.0
    new = value + step if direction > 0 else value - step
    return float(f"{new:.12g}")             # без хвостов 0.30000000000000004


def format_value(value):
    return f"{value:.6g}"


class Tooltip:
    """Подсказка у виджета: появляется при наведении через delay мс."""

    def __init__(self, widget, text, delay=500, wraplength=380):
        self.widget, self.text, self.delay, self.wraplength = widget, text, delay, wraplength
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        import tkinter as tk

        self._after = None
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(tip, text=self.text, justify="left", background="#ffffe8", relief="solid",
                 borderwidth=1, wraplength=self.wraplength, font=("Segoe UI", 9),
                 padx=6, pady=4).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def bind_wheel(widget, callback):
    """callback(direction, fine) на прокрутку колеса над виджетом:
    <MouseWheel> — Windows, macOS и Tk ≥ 8.7; Button-4/5 — X11 в Tk 8.6."""
    def on_wheel(event):
        fine = bool(event.state & 0x0004)   # Ctrl
        if getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            direction = 1 if event.delta > 0 else -1
        callback(direction, fine)
        return "break"

    widget.bind("<MouseWheel>", on_wheel)
    widget.bind("<Button-4>", on_wheel)
    widget.bind("<Button-5>", on_wheel)
