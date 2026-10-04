# -*- coding: utf-8 -*-
"""Формулы как в учебнике: mathtext → картинка для Tk (методичка, пп. 1А.13, 1А.20).

Правила набора — ISO 80000-2 и NIST SP 811, гл. 10 (шрифты символов),
ГОСТ 2.105 и ГОСТ 7.32 (формулы в тексте):
  * величины — курсивом; описательные индексы (sh, bi, eff, diff, …), функции
    (exp, ln), числа и единицы — прямым шрифтом. Индекс из двух и более
    латинских букв считается описательным и выпрямляется сам — upright();
  * формула стоит отдельной строкой по центру, её номер — в круглых скобках
    у правого края (equation);
  * пояснения символов — сразу под формулой, со слова «где» без двоеточия,
    каждый символ с новой строки в порядке появления в формуле (where_list);
  * сноска под формулой — источник и пункт методички, мелким шрифтом (note).
Размеры: формула 15 pt, строки расчёта 13 pt, символ в «где» 12 pt, текст
пояснений 9 pt, сноска 8 pt — главное крупнее, справочное мельче.

Шрифт формул — STIX (начертание Times, как в учебниках); кириллица — только
внутри \\mathrm{…} (индексы «д», «эф», единицы «А», «см»), её рисует DejaVu.
Картинка — PPM без промежуточного PNG и фигуры matplotlib; растр кэшируется
по строке, размеру и разрешению, поэтому повторная отрисовка почти бесплатна.
"""

import base64
import io
import re
from functools import lru_cache

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

try:  # без Tk (CI) модуль импортируется: растр и upright() проверяются тестами
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = tkfont = ttk = None

FONTSET = "stix"
DISPLAY_PT = 15         # выключная формула
CALC_PT = 13            # строки расчёта
SYMBOL_PT = 12          # символ в пояснении «где»
MIN_PT = 9              # меньше формула не сжимается при подгонке по ширине
# Шрифты текста — именованные шрифты окна (desktop.py: в Linux нет Segoe UI);
# формулы растут с масштабом экрана сами — разрешение берётся у окна (dpi_of).
TEXT_FONT = "MesaText"
NOTE_FONT = "MesaSmall"
NUMBER_FONT = "MesaLarge"
TEXT_COLOR = "#222222"
NOTE_COLOR = "#666666"
NUMBER_COLOR = "#333333"

_PARSER = MathTextParser("agg")
_DESCRIPTIVE = re.compile(r"([_^])\{([A-Za-z]{2,})\}")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_MATHRM = re.compile(r"\\mathrm\{[^{}]*\}")


def upright(tex):
    """Описательные индексы из двух и более латинских букв — прямым шрифтом:
    R_{sh} → R_{\\mathrm{sh}}, τ^{bg} → τ^{\\mathrm{bg}} (ISO 80000-2)."""
    return _DESCRIPTIVE.sub(lambda m: f"{m.group(1)}{{\\mathrm{{{m.group(2)}}}}}", tex)


_TEXT_FRACTION = re.compile(r"\\frac")      # в \dfrac перед «frac» стоит «d», а не «\»


def display(tex):
    """Выключная формула: дроби в полный размер (\\frac → \\dfrac), как displaystyle в TeX."""
    return _TEXT_FRACTION.sub(r"\\dfrac", tex)


def cyrillic_outside_mathrm(tex):
    """Кириллица вне \\mathrm{…} внутри $…$ (её не должно быть) — для тестов."""
    found = []
    for math in re.findall(r"\$(.*?)\$", tex):
        found += _CYRILLIC.findall(_MATHRM.sub("", math))
    return found


def _rgb(color):
    """Цвет (#rrggbb, #rgb или имя: white) → (r, g, b) 0…255."""
    return tuple(round(255 * c) for c in to_rgb(color))


@lru_cache(maxsize=4096)
def raster(tex, size, dpi):
    """Растр формулы: (покрытие uint8 [h, w], глубина под базовой линией, px)."""
    prop = FontProperties(size=size)
    prop.set_math_fontfamily(FONTSET)
    result = _PARSER.parse(upright(tex), dpi=dpi, prop=prop)
    return np.asarray(result.image, dtype=np.uint8), int(round(result.depth))


@lru_cache(maxsize=4096)
def ppm(tex, size, dpi, fg="#000000", bg="#ffffff"):
    """Формула в формате PPM (P6) для tk.PhotoImage: (данные, ширина, высота)."""
    alpha, _depth = raster(tex, size, dpi)
    a = alpha.astype(np.float32)[..., None] / 255.0
    rgb = np.array(_rgb(bg), np.float32) * (1.0 - a) + np.array(_rgb(fg), np.float32) * a
    rgb = np.clip(rgb.round(), 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (width, height) + rgb.tobytes(), width, height


def fit_size(tex, size, dpi, max_width=None):
    """Размер шрифта, при котором формула не шире max_width (не меньше MIN_PT)."""
    if not max_width:
        return size
    width = raster(tex, size, dpi)[0].shape[1]
    if width <= max_width:
        return size
    return max(MIN_PT, int(size * max_width / width))


def dpi_of(widget):
    """Пикселей на дюйм у окна (с учётом масштаба Tk): формула и текст в pt одного размера."""
    try:
        return int(round(widget.winfo_fpixels("1i")))
    except Exception:  # noqa: BLE001 — окно ещё не создано
        return 96


def photo(master, tex, size=DISPLAY_PT, fg="#000000", bg="#ffffff", max_width=None):
    """tk.PhotoImage с формулой; ширина ограничена max_width (формула сжимается)."""
    dpi = dpi_of(master)
    size = fit_size(tex, size, dpi, max_width)
    data, _w, _h = ppm(tex, size, dpi, fg, bg)
    try:
        return tk.PhotoImage(master=master, data=data, format="ppm")
    except tk.TclError:  # pragma: no cover — запасной путь через PNG
        import matplotlib.image as mimage

        buffer = io.BytesIO()
        alpha, _depth = raster(tex, size, dpi)
        mimage.imsave(buffer, 255 - alpha, cmap="gray", format="png")
        return tk.PhotoImage(master=master, data=base64.b64encode(buffer.getvalue()))


def background(widget):
    """Цвет фона ttk-рамки в виде #rrggbb (под него рисуются формулы)."""
    color = ttk.Style(widget).lookup("TFrame", "background") or "#f0f0f0"
    r, g, b = widget.winfo_rgb(color)
    return "#%02x%02x%02x" % (r >> 8, g >> 8, b >> 8)


# ------------------------------------------------------------- виджеты --

class MathLabel(tk.Label if tk else object):
    """Подпись-картинка с формулой; set() меняет формулу."""

    def __init__(self, parent, tex, size=DISPLAY_PT, fg="#000000", bg="#ffffff", max_width=None):
        super().__init__(parent, background=bg, borderwidth=0, highlightthickness=0)
        self._style = (size, fg, bg, max_width)
        self.tex = None
        self.set(tex)

    def set(self, tex):
        if tex == self.tex:
            return
        size, fg, bg, max_width = self._style
        self.image = photo(self, tex, size, fg, bg, max_width)  # ссылка, иначе картинку удалит сборщик мусора
        self.configure(image=self.image)
        self.tex = tex


_INDEX_MARKUP = re.compile(r"([_^])\{([^{}]*)\}")


def split_index_markup(text):
    """Обычный текст → куски [(фрагмент, None|"sub"|"sup"), ...].
    Понимает только _{...} и ^{...}; одиночные подчёркивания (solve_iv) — как есть."""
    parts, pos = [], 0
    for m in _INDEX_MARKUP.finditer(text):
        if m.start() > pos:
            parts.append((text[pos:m.start()], None))
        parts.append((m.group(2), "sub" if m.group(1) == "_" else "sup"))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], None))
    return parts


class RichText(tk.Frame if tk else object):
    """Нередактируемый текст с переносом по ширине, индексами _{…}/^{…} и
    формулами внутри строки; высота подстраивается под текст после переноса.

    Содержание — список строк-абзацев; абзац — строка с разметкой индексов или
    список кусков: строка (текст) или ("math", tex[, size]) (формула). Высоту
    держит рамка в пикселях: у Text высота задаётся в строках шрифта, а строки
    с индексами и формулами выше — последняя строка обрезалась бы."""

    def __init__(self, parent, content="", bg="#ffffff", font=TEXT_FONT, fg=TEXT_COLOR,
                 indent=0, hanging=0, spacing=2):
        self._base = tkfont.Font(parent, font=font)
        size = abs(self._base.actual("size"))
        self._small = tkfont.Font(parent, font=font)
        self._small.configure(size=max(round(size * 0.75), 6))
        self._offset = max(size // 3, 2)
        super().__init__(parent, background=bg, borderwidth=0, highlightthickness=0,
                         height=self._base.metrics("linespace") + 2 * self._offset)
        self.pack_propagate(False)
        self.bg, self.fg, self.math_size = bg, fg, max(size + 3, SYMBOL_PT)
        # width=1: ширину задаёт pack/grid рамки, а не число символов по умолчанию.
        self.text = tk.Text(self, wrap="word", width=1, height=1, borderwidth=0, highlightthickness=0,
                            padx=0, pady=self._offset, background=bg, foreground=fg, font=self._base,
                            cursor="arrow", takefocus=0, spacing1=spacing, spacing2=self._offset)
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("sub", offset=-self._offset, font=self._small)
        self.text.tag_configure("sup", offset=self._offset, font=self._small)
        self.text.tag_configure("para", lmargin1=indent, lmargin2=indent + hanging)
        self.images = []
        self.text.bind("<Configure>", self.fit_height)
        self.set(content)

    def _insert(self, text, tags):
        for chunk, tag in split_index_markup(text):
            self.text.insert("end", chunk, tags + ((tag,) if tag else ()))

    def set(self, content):
        text = self.text
        text.configure(state="normal")
        text.delete("1.0", "end")
        self.images = []
        paragraphs = [content] if isinstance(content, str) else list(content)
        for index, paragraph in enumerate(paragraphs):
            if index:
                text.insert("end", "\n", ("para",))
            pieces = [paragraph] if isinstance(paragraph, str) else paragraph
            for piece in pieces:
                if isinstance(piece, str):
                    self._insert(piece, ("para",))
                else:
                    size = piece[2] if len(piece) > 2 else self.math_size
                    image = photo(text, piece[1], size, self.fg, self.bg)
                    self.images.append(image)
                    text.image_create("end", image=image, align="center", padx=1)
        text.configure(state="disabled")
        self.fit_height()       # пока ширина неизвестна, подгонку сделает <Configure>

    def fit_height(self, _event=None):
        # Ширина ещё не известна (скрытая вкладка, окно не размещено): при ширине в
        # 1–2 пикселя перенос дал бы по слову в строке, а сумма таких высот больше
        # 32767 пикселей не помещается в окно X11. Подгонка — когда ширина появится.
        if self.text.winfo_width() <= 2:
            return
        # -update: Tk считает переносы строк лениво, без него высота ещё неизвестна;
        # -ypixels до "end" — высота всех строк текста.
        pixels = int(self.text.tk.call(self.text._w, "count", "-update", "-ypixels", "1.0", "end"))
        height = max(pixels, self._base.metrics("linespace")) + 2 * self._offset
        if int(self.cget("height")) != height:
            self.configure(height=height)


def where_list(parent, rows, bg, width_hint=None):
    """Пояснение символов под формулой (ГОСТ 2.105): «где A — площадь мезы, см²;».
    rows — [(символ mathtext, пояснение)], пояснение — текст с разметкой индексов."""
    if not rows:
        return None
    font = tkfont.Font(parent, font=TEXT_FONT)
    lead = font.measure("где ") + 2
    paragraphs = []
    for index, (symbol, description) in enumerate(rows):
        end = "." if index == len(rows) - 1 else ";"
        head = "где\t" if index == 0 else "\t"
        paragraphs.append([head, ("math", symbol, SYMBOL_PT), f" — {description}{end}"])
    widget = RichText(parent, paragraphs, bg=bg, font=TEXT_FONT, fg=TEXT_COLOR, hanging=lead + 12)
    widget.text.configure(tabs=(lead,))
    return widget


def note(parent, text, bg):
    """Сноска под формулой: источник и пункт методички, мелким шрифтом."""
    return RichText(parent, text, bg=bg, font=NOTE_FONT, fg=NOTE_COLOR) if text else None


def equation(parent, tex, number="", bg="#ffffff", size=DISPLAY_PT, max_width=None, badge="",
             badge_color="#e6e6e6", boxed=False, align="center"):
    """Выключная формула и номер в скобках у правого края (ГОСТ 7.32).
    tex — строка или кортеж строк (система: строки одна под другой, номер один).
    align="center" — формула по центру строки; "left" — с отступом слева
    (выключка влево, как fleqn в LaTeX: для узкой панели)."""
    frame = tk.Frame(parent, background=bg)
    if align == "center":
        frame.columnconfigure(0, weight=1, uniform="side")
        frame.columnconfigure(2, weight=1, uniform="side")
        column, sticky = 1, ""
    else:
        frame.columnconfigure(1, weight=1)
        column, sticky = 0, "w"
    lines = (tex,) if isinstance(tex, str) else tuple(tex)
    body = tk.Frame(frame, background=bg, highlightthickness=1 if boxed else 0,
                    highlightbackground="#1a4fa0", padx=4 if boxed else 0, pady=2 if boxed else 0)
    body.grid(row=0, column=column, sticky=sticky, pady=2)
    frame.math = []
    for line in lines:
        label = MathLabel(body, display(line), size=size, bg=bg, max_width=max_width)
        label.pack(anchor="w", pady=1)
        frame.math.append(label)
    side = tk.Frame(frame, background=bg)
    side.grid(row=0, column=2, sticky="e", padx=(8, 0))
    if badge:
        tk.Label(side, text=badge, background=badge_color, font=NOTE_FONT, padx=3).pack(side="left", padx=(6, 6))
    if number:
        tk.Label(side, text=number, background=bg, font=NUMBER_FONT, foreground=NUMBER_COLOR).pack(side="left")
    return frame


class ScrollFrame(ttk.Frame if ttk else object):
    """Вертикально прокручиваемая рамка: canvas + внутренняя рамка body.
    bind_wheel() привязывает колесо мыши ко всем дочерним виджетам."""

    def __init__(self, parent, bg):
        super().__init__(parent)
        self.bg = bg
        self.canvas = tk.Canvas(self, highlightthickness=0, background=bg, borderwidth=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.body = tk.Frame(self.canvas, background=bg)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))

    def scroll_to(self, widget):
        self.canvas.update_idletasks()
        total = max(self.body.winfo_height(), 1)
        y = widget.winfo_rooty() - self.body.winfo_rooty()
        self.canvas.yview_moveto(max(0.0, y / total))

    def _on_wheel(self, event):
        if getattr(event, "num", None) == 4:
            step = -3
        elif getattr(event, "num", None) == 5:
            step = 3
        else:
            step = -3 if event.delta > 0 else 3
        self.canvas.yview_scroll(step, "units")
        return "break"

    def bind_wheel(self, widget=None):
        """Колесо над canvas и над любым его потомком прокручивает рамку."""
        widget = widget or self.canvas
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._on_wheel)
        for child in widget.winfo_children():
            self.bind_wheel(child)
        if widget is self.canvas:
            self.bind_wheel(self.body)
