# -*- coding: utf-8 -*-
"""Правая панель окна: рисунок эквивалентной схемы с подсказками у элементов
и формулы модели, набранные как в учебнике (содержание — scheme.py, набор —
typeset.py; методичка, п. 1А.20).

Под схемой — прокручиваемый лист: формулы с номерами, пояснением «где» и
сноской (источник, пункт методички), расчёт точки вручную и ВФХ. Лист
перестраивается целиком, только когда меняется набор формул (режим, модель
тока); при пересчёте с теми же формулами обновляются лишь строки с числами."""

import tkinter as tk
from tkinter import ttk

from mesa_diode.simulator import hints, typeset
from mesa_diode.simulator.scheme import (Eq, Heading, Remark, Step, elements, model_summary, panel_blocks,
                                         panel_title)

PAPER = "#fbfbf6"           # фон листа с формулами
HEADING_FONT = ("Segoe UI", 9, "bold")
HEADING_COLOR = "#1a4fa0"
LABEL_FONT = ("Segoe UI", 8)
EQ_PT = 14                  # формулы модели (в узкой панели чуть меньше, чем в окне формул)


class SchemePanel(ttk.Frame):
    """Схема (Canvas с подсказками у элементов) и лист с формулами и расчётом точки."""

    WIDTH, HEIGHT = 410, 250
    INDENT = 10             # отступ формул слева (выключка влево, как fleqn в LaTeX)

    def __init__(self, parent):
        super().__init__(parent, padding=(6, 4, 6, 4))
        self.title = ttk.Label(self, text="Как считается модель", font=("Segoe UI", 10, "bold"))
        self.title.pack(anchor="w")
        self.model_name = ttk.Label(self, text="", font=HEADING_FONT, foreground=HEADING_COLOR,
                                    wraplength=self.WIDTH)
        self.model_name.pack(anchor="w")
        self.bg = typeset.background(self)
        self.summary = typeset.RichText(self, "", bg=self.bg, font=("Segoe UI", 9), fg="#444444")
        self.summary.configure(width=self.WIDTH)
        self.summary.pack(anchor="w", fill=tk.X, pady=(0, 4))
        self.canvas = tk.Canvas(self, width=self.WIDTH, height=self.HEIGHT, background="white",
                                highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(anchor="w")
        ttk.Label(self, text="Наведите на элемент схемы — что это и почему он здесь.",
                  font=("Segoe UI", 8), foreground="#777777").pack(anchor="w")
        self.sheet = typeset.ScrollFrame(self, PAPER)
        self.sheet.canvas.configure(width=self.WIDTH, height=300, highlightthickness=1,
                                    highlightbackground="#dddddd")
        self.sheet.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.sheet.bind_wheel()
        self._tip = None
        self._layout = None     # набор формул, по которому построен лист
        self._live = []         # (блок-образец, обновитель) — строки с числами

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

    def _math(self, x, y, tex, color, tag):
        """Подпись формулой на белом фоне — как в формулах под схемой. tex — строка
        или кортеж строк (подпись в несколько строк, первая — в точке y)."""
        c = self.canvas
        for line in ((tex,) if isinstance(tex, str) else tex):
            image = typeset.photo(c, line, size=11, fg=color, bg="white")
            self._images.append(image)
            item = c.create_image(x, y, image=image, anchor="n", tags=tag)
            x0, y0, x1, y1 = c.bbox(item)
            back = c.create_rectangle(x0 - 1, y0, x1 + 1, y1, fill="white", outline="", tags=tag)
            c.tag_raise(item, back)
            y = y1

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
        self._images = []        # картинки подписей: ссылки, иначе их удалит сборщик мусора
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
        if series.tex:
            self._math(80, top - 32, series.tex, "#222222", tag)
        else:
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
            if element.tex:
                self._math(x, mid + 24, element.tex, color, tag)
            else:
                self._label(x, mid + 36, element.label, color, step - 6, tag)
            self._bind(tag, element.tip)
        # напряжения
        self._math(40, (top + bottom) / 2 - 10, r"$V$", "#222222", "volt")
        c.create_line(40, top + 14, 40, bottom - 14, arrow=tk.BOTH, fill="#888888")
        self._math(xs[0] - 34, mid - 10, r"$V_{\mathrm{д}}$", "#555555", "vd")
        self._math(self.WIDTH / 2, bottom + 14, r"$V=V_{\mathrm{д}}+IR_{\mathrm{s}},\qquad I=\sum I_{\mathrm{ветвей}}$",
                   "#555555", "caption")

    # --- лист с формулами
    def _width(self):
        return max(self.sheet.canvas.winfo_width(), self.WIDTH) - 2 * self.INDENT - 8

    def _heading(self, body, text):
        frame = tk.Frame(body, background=PAPER)
        frame.pack(fill=tk.X, padx=6, pady=(10, 2))
        typeset.RichText(frame, text, bg=PAPER, font=HEADING_FONT, fg=HEADING_COLOR, spacing=0).pack(fill=tk.X)
        tk.Frame(frame, background="#c9d6ea", height=1).pack(fill=tk.X, pady=(1, 0))

    def _equation(self, body, eq):
        width = self._width()
        frame = typeset.equation(body, eq.tex, eq.number, bg=PAPER, size=EQ_PT, max_width=width - 70,
                                 align="left")
        frame.pack(fill=tk.X, padx=(self.INDENT, 6), pady=(4, 0))
        where = typeset.where_list(body, eq.where, PAPER)
        if where is not None:
            where.pack(fill=tk.X, padx=(self.INDENT + 6, 6), pady=(1, 0))
        note = typeset.note(body, eq.note, PAPER)
        if note is not None:
            note.pack(fill=tk.X, padx=(self.INDENT + 6, 6), pady=(0, 4))

    def _step_lines(self, step):
        """Подстановка и результат одной строкой, если помещается, иначе двумя."""
        if not step.result:
            return [step.tex]
        joined = step.tex[:-1] + step.result[1:]
        dpi = typeset.dpi_of(self)
        if typeset.raster(joined, typeset.CALC_PT, dpi)[0].shape[1] <= self._width():
            return [joined]
        return [step.tex, step.result]

    def _step(self, body, step):
        frame = tk.Frame(body, background=PAPER)
        frame.pack(fill=tk.X, padx=(self.INDENT, 6), pady=(2, 1))
        typeset.RichText(frame, step.label, bg=PAPER, font=LABEL_FONT, fg=typeset.NOTE_COLOR,
                         spacing=0).pack(fill=tk.X)
        holder = tk.Frame(frame, background=PAPER)
        holder.pack(fill=tk.X)

        def update(new_step):
            for child in holder.winfo_children():
                child.destroy()
            for index, tex in enumerate(self._step_lines(new_step)):
                label = typeset.MathLabel(holder, tex, size=typeset.CALC_PT, bg=PAPER, max_width=self._width())
                label.pack(anchor="w", padx=(18 if index else 0, 0))
            self.sheet.bind_wheel(holder)

        update(step)
        self._live.append(update)

    def _remark(self, body, remark):
        widget = typeset.RichText(body, remark.text, bg=PAPER, font=typeset.TEXT_FONT, fg="#444444")
        widget.pack(fill=tk.X, padx=(self.INDENT, 6), pady=(2, 2))
        self._live.append(lambda new: widget.set(new.text))

    @staticmethod
    def _signature(blocks):
        """Что определяет вид листа: формулы, заголовки и подписи строк — но не числа."""
        sign = []
        for block in blocks:
            if isinstance(block, Eq):
                sign.append(("eq", block.tex, block.number, block.where, block.note))
            elif isinstance(block, Heading):
                sign.append(("h", block.text))
            elif isinstance(block, Step):
                sign.append(("step", block.label))
            else:
                sign.append(("remark",))
        return tuple(sign)

    def _fill(self, blocks):
        body = self.sheet.body
        layout = self._signature(blocks)
        if layout != self._layout:
            for child in body.winfo_children():
                child.destroy()
            self._live = []
            for block in blocks:
                if isinstance(block, Heading):
                    self._heading(body, block.text)
                elif isinstance(block, Eq):
                    self._equation(body, block)
                elif isinstance(block, Step):
                    self._step(body, block)
                elif isinstance(block, Remark):
                    self._remark(body, block)
            tk.Frame(body, background=PAPER, height=8).pack()
            self._layout = layout
            self.sheet.bind_wheel()
            return
        live = iter(self._live)
        for block in blocks:
            if isinstance(block, (Step, Remark)):
                next(live)(block)

    def show(self, s, mode):
        """Обновить панель по структуре s в режиме mode."""
        self.model_name.config(text=panel_title(s, mode))
        self.summary.set(model_summary(s, mode))
        self.draw(s, mode)
        self._fill(panel_blocks(s, mode))
