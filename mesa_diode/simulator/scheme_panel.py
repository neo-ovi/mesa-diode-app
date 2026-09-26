# -*- coding: utf-8 -*-
"""Правая панель окна: рисунок эквивалентной схемы с подсказками у элементов
и текст формулы (содержание — scheme.py; методичка, п. 1А.20)."""

import tkinter as tk
from tkinter import ttk

from mesa_diode.simulator import hints
from mesa_diode.simulator.scheme import elements, formula_lines, model_summary


class SchemePanel(ttk.Frame):
    """Схема (Canvas с подсказками у элементов) и формула с расчётом точки."""

    WIDTH, HEIGHT = 410, 250

    def __init__(self, parent):
        super().__init__(parent, padding=(6, 4, 6, 4))
        self.title = ttk.Label(self, text="", font=("Segoe UI", 10, "bold"))
        self.title.pack(anchor="w")
        self.summary = ttk.Label(self, text="", wraplength=self.WIDTH, justify="left",
                                 font=("Segoe UI", 9), foreground="#444444")
        self.summary.pack(anchor="w", fill=tk.X, pady=(0, 4))
        self.canvas = tk.Canvas(self, width=self.WIDTH, height=self.HEIGHT, background="white",
                                highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(anchor="w")
        ttk.Label(self, text="Наведите на элемент схемы — что это и почему он здесь.",
                  font=("Segoe UI", 8), foreground="#777777").pack(anchor="w")
        self.text = tk.Text(self, wrap="word", width=62, height=22, font=("DejaVu Sans Mono", 8),
                            relief="flat", background="#fbfbf6")
        self.text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._tip = None

    # --- подсказки у элементов Canvas
    def _show_tip(self, event, text):
        self._hide_tip()
        tip = tk.Toplevel(self)
        tip.wm_overrideredirect(True)
        tk.Label(tip, text=hints.plain(text), justify="left", background="#ffffe8", relief="solid",
                 borderwidth=1, wraplength=380, font=("Segoe UI", 9), padx=6, pady=4).pack()
        tip.update_idletasks()
        x, y = event.x_root + 14, event.y_root + 10
        if x + tip.winfo_reqwidth() > self.winfo_screenwidth():        # у правого края — левее курсора
            x = max(0, event.x_root - tip.winfo_reqwidth() - 10)
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide_tip(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _bind(self, tag, text):
        self.canvas.tag_bind(tag, "<Enter>", lambda e: self._show_tip(e, text))
        self.canvas.tag_bind(tag, "<Leave>", self._hide_tip)

    # --- символы
    def _symbol(self, element, x, y, tag, horizontal=False):
        c = self.canvas
        color = "#222222" if element.active else "#aaaaaa"
        dash = () if element.active else (3, 2)
        kw = dict(outline=color, width=1.5, dash=dash, tags=tag)
        if element.symbol == "diode":
            c.create_polygon(x - 11, y - 9, x + 11, y - 9, x, y + 9, fill="#dde8f7", **kw)
            c.create_line(x - 11, y + 9, x + 11, y + 9, fill=color, width=2, tags=tag)
        elif element.symbol == "source":
            c.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#f3ecdb", **kw)
            c.create_line(x, y - 7, x, y + 7, arrow=tk.LAST, fill=color, width=1.5, tags=tag)
        elif element.symbol == "box":
            c.create_rectangle(x - 16, y - 14, x + 16, y + 14, fill="#eeeeee", **kw)
        else:                                   # resistor, varres
            half_w, half_h = (18, 7) if horizontal else (7, 18)
            c.create_rectangle(x - half_w, y - half_h, x + half_w, y + half_h, fill="#f7f7f7", **kw)
            if element.symbol == "varres":
                c.create_line(x - 14, y + 16, x + 14, y - 16, arrow=tk.LAST, fill=color, tags=tag)

    def _label(self, x, y, text, color, width, tag):
        """Подпись на белом фоне — читается поверх линий схемы."""
        c = self.canvas
        item = c.create_text(x, y, text=text, font=("Segoe UI", 8), fill=color, width=width, tags=tag)
        x0, y0, x1, y1 = c.bbox(item)
        back = c.create_rectangle(x0 - 1, y0, x1 + 1, y1, fill="white", outline="", tags=tag)
        c.tag_raise(item, back)

    def draw(self, s, mode):
        c = self.canvas
        c.delete("all")
        self._hide_tip()
        branches, series = elements(s, mode)
        top, bottom, mid = 45, 205, 125
        x_node = 150
        step = max(52, min(80, (self.WIDTH - x_node - 30) // max(len(branches) - 1, 1)))
        xs = [x_node + i * step for i in range(len(branches))]
        # клеммы и R_s
        c.create_text(14, top, text="+", font=("Segoe UI", 12, "bold"))
        c.create_text(14, bottom, text="−", font=("Segoe UI", 12, "bold"))
        c.create_line(24, top, 60, top, width=1.5)
        c.create_line(24, bottom, xs[-1], bottom, width=1.5)
        c.create_line(100, top, xs[-1], top, width=1.5)
        tag = "el_rs"
        c.create_line(60, top, 100, top, width=1.5, tags=tag)
        self._symbol(series, 80, top, tag, horizontal=True)
        self._label(80, top - 24, series.label, "#222222", 70, tag)
        self._bind(tag, series.tip)
        # ветви
        for x, element in zip(xs, branches):
            tag = f"el_{element.key}"
            color = "#222222" if element.active else "#aaaaaa"
            dash = () if element.active else (3, 2)
            c.create_line(x, top, x, mid - 20, fill=color, width=1.5, dash=dash, tags=tag)
            c.create_line(x, mid + 20, x, bottom, fill=color, width=1.5, dash=dash, tags=tag)
            c.create_oval(x - 3, top - 3, x + 3, top + 3, fill="#222222", tags=tag)
            c.create_oval(x - 3, bottom - 3, x + 3, bottom + 3, fill="#222222", tags=tag)
            self._symbol(element, x, mid, tag)
            self._label(x, mid + 36, element.label, color, step - 6, tag)
            self._bind(tag, element.tip)
        # напряжения
        c.create_text(40, (top + bottom) / 2, text="V", font=("Segoe UI", 10, "italic"))
        c.create_line(40, top + 14, 40, bottom - 14, arrow=tk.BOTH, fill="#888888")
        c.create_text(xs[0] - 22, bottom - 16, text="V_д", font=("Segoe UI", 8, "italic"), fill="#555555")
        c.create_text(self.WIDTH / 2, bottom + 28, text="V = V_д + I·R_s;  I — сумма токов ветвей",
                      font=("Segoe UI", 8), fill="#555555")

    def show(self, s, mode):
        """Обновить панель по структуре s в режиме mode."""
        self.title.config(text="Как считается модель")
        self.summary.config(text=model_summary(s, mode))
        self.draw(s, mode)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join(formula_lines(s, mode)))
        self.text.configure(state="disabled")
