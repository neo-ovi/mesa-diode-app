# -*- coding: utf-8 -*-
"""Чтение и проверка файлов измерений (ВАХ, ВФХ).

Поддерживаются текстовые таблицы (.csv, .txt, .dat, .tsv) и Excel (.xlsx,
.xlsm; .xls — если установлен xlrd). Формат описан в FORMAT_HELP и в
методичке (п. 1А.17); примеры — в каталоге examples рядом с модулем.

Главная функция — analyze(path, kind): читает файл, выбирает столбцы,
переводит единицы в СИ и проверяет данные. Каждое замечание (Issue) говорит,
что не так, что сделать и где об этом в методичке. Ошибки (level="error")
делают файл непригодным; предупреждения — данные прочитаны, но их стоит
проверить; сведения («info») — что программа сделала сама (перевела единицы,
отсортировала точки).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KIND_IV = "iv"
KIND_CV = "cv"

TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".tsv"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
LEGACY_EXCEL_SUFFIXES = {".xls"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | LEGACY_EXCEL_SUFFIXES
FILE_TYPES = [("Данные измерений", "*.csv *.txt *.dat *.tsv *.xlsx *.xlsm *.xls"),
              ("Текст/CSV", "*.csv *.txt *.dat *.tsv"), ("Excel", "*.xlsx *.xlsm *.xls"),
              ("Все файлы", "*.*")]

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
REF = "п. 1А.17"

MIN_POINTS = 20          # меньше — подгонка и n_эксп ненадёжны
MAX_ABS_VOLTAGE = 100.0  # больше — вероятно, милливольты без подписи
MAX_ABS_CURRENT = 1.0    # больше — вероятно, миллиамперы без подписи
MAX_CAPACITANCE = 1e-6   # больше — вероятно, пФ/нФ без подписи

FORMAT_HELP = """\
Файл измерения — таблица из двух столбцов: напряжение и измеренная величина.

  ВАХ: 1-й столбец — V, вольты; 2-й — I, амперы.
  ВФХ: 1-й столбец — V, вольты; 2-й — C, фарады.

Форматы: .csv, .txt, .dat, .tsv (текст) и .xlsx, .xlsm (Excel).
Старый формат .xls читается, только если установлен пакет xlrd;
иначе сохраните файл как .xlsx или CSV.

Текст: разделитель — точка с запятой, табуляция, пробелы или запятая;
десятичный разделитель — точка или запятая (запятая — только если
столбцы разделены не запятой). Кодировка UTF-8 или Windows-1251.

Excel: берётся лист, на котором больше всего числовых строк; ячейки —
числа (или текст-числа). Формулы допустимы, если файл сохранён в Excel
(хранится вычисленное значение).

Первая строка может быть заголовком. По заголовку программа узнаёт
столбцы и единицы: «V», «U», «Напряжение», «I», «Ток», «C», «Ёмкость»;
единицы — в скобках или через запятую: «I (мА)», «I, uA», «C [pF]».
Поддерживаются В, мВ; А, мА, мкА (uA), нА, пА; Ф, мкФ, нФ, пФ, фФ.
Если столбцов больше двух (например, время, V, I), заголовок нужен,
иначе берутся первые два.

Знак тока: прямая ветвь (V > 0) — ток положительный, обратная —
отрицательный. Если обратный ток записан по модулю, программа
поставит знак сама и предупредит.

Порядок точек любой — программа сортирует по напряжению. Строки,
которые не читаются как числа (комментарии, пустые), пропускаются.
"""


# ------------------------------------------------------------ замечания
@dataclass
class Issue:
    level: str        # "error" | "warning" | "info"
    text: str         # что обнаружено
    hint: str = ""    # что сделать
    ref: str = REF    # пункт методички

    def format(self):
        parts = [self.text]
        if self.hint:
            parts.append(f"Что сделать: {self.hint}")
        if self.ref:
            parts.append(f"Подробнее: методичка, {self.ref}.")
        return "\n".join(parts)


@dataclass
class DataReport:
    path: Path
    kind: str
    voltage: np.ndarray = field(default_factory=lambda: np.empty(0))
    value: np.ndarray = field(default_factory=lambda: np.empty(0))
    issues: list = field(default_factory=list)
    columns: tuple = ()     # (название столбца V, название столбца величины)
    sheet: str = ""         # лист Excel

    @property
    def ok(self):
        return not self.errors

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self):
        return [i for i in self.issues if i.level == "info"]

    def add(self, level, text, hint="", ref=REF):
        self.issues.append(Issue(level, text, hint, ref))

    def text(self, levels=("error", "warning", "info")):
        """Замечания нужных уровней одним текстом для окна сообщения."""
        titles = {"error": "Ошибка", "warning": "Проверьте", "info": "Сделано"}
        blocks = [f"{titles[i.level]}: {i.format()}" for i in self.issues if i.level in levels]
        return "\n\n".join(blocks)


class DataFileError(ValueError):
    """Файл не прочитан или данные непригодны; report — полный разбор."""

    def __init__(self, report):
        self.report = report
        super().__init__(report.text(("error",)))


# ------------------------------------------------------------ единицы
_UNITS = {
    "voltage": {"v": 1.0, "в": 1.0, "mv": 1e-3, "мв": 1e-3},
    "current": {"a": 1.0, "а": 1.0, "ma": 1e-3, "ма": 1e-3, "ua": 1e-6, "µa": 1e-6, "μa": 1e-6,
                "мка": 1e-6, "na": 1e-9, "на": 1e-9, "pa": 1e-12, "па": 1e-12},
    "capacitance": {"f": 1.0, "ф": 1.0, "uf": 1e-6, "µf": 1e-6, "μf": 1e-6, "мкф": 1e-6,
                    "nf": 1e-9, "нф": 1e-9, "pf": 1e-12, "пф": 1e-12, "ff": 1e-15, "фф": 1e-15},
}
_PREFIXES = {1.0: "", 1e-3: "м", 1e-6: "мк", 1e-9: "н", 1e-12: "п", 1e-15: "ф"}

_QUANTITY_PATTERNS = {
    "voltage": r"^(v|u|vbias|bias|voltage|напряжение|напр|смещение|в)\b",
    "current": r"^(i|j|id|current|ток|а)\b",
    "capacitance": r"^(c|cp|cs|cap|capacitance|ёмкость|емкость|ф)\b",
    "time": r"^(t|time|время|с|s)\b",
}
_BASE_UNIT = {"voltage": "В", "current": "А", "capacitance": "Ф"}
_QUANTITY_RU = {"voltage": "напряжение", "current": "ток", "capacitance": "ёмкость"}


def _header_tokens(name):
    return [t for t in re.split(r"[\s,;_()\[\]{}:=]+", name.strip().lower()) if t]


def column_quantity(name):
    """Какая величина в столбце по его заголовку: voltage/current/capacitance/time/None."""
    tokens = _header_tokens(name)
    if not tokens:
        return None
    for quantity, pattern in _QUANTITY_PATTERNS.items():
        if re.match(pattern, tokens[0]):
            return quantity
    # заголовок из одной единицы: «В», «А», «пФ»
    for quantity, units in _UNITS.items():
        if tokens[0] in units and len(tokens) == 1:
            return quantity
    return None


def column_scale(name, quantity):
    """Множитель перевода в СИ по единице в заголовке; None — единица не указана.

    Возвращает math.nan, если указана плотность тока (А/см²)."""
    text = name.strip().lower()
    if re.search(r"/\s*(cm|см|m|м)\s*(\^?2|²)", text):
        return math.nan
    tokens = _header_tokens(text)
    units = _UNITS.get(quantity, {})
    for token in tokens[1:] if len(tokens) > 1 else tokens:
        if token in units:
            return units[token]
    return None


# ------------------------------------------------------------ числа
def parse_number(cell):
    """Число из ячейки: float/int или текст с точкой либо запятой; иначе None."""
    if cell is None or isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip().replace("−", "-").replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


# ------------------------------------------------------------ чтение таблиц
def _decode(raw):
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def _split(line, delimiter):
    if delimiter is None:
        return line.split()
    return [c.strip() for c in line.split(delimiter)]


def _numeric_count(cells):
    return sum(parse_number(c) is not None for c in cells)


def _choose_delimiter(lines):
    """Разделитель, при котором больше всего строк дают ≥ 2 числа.

    При равенстве порядок: «;», таб, пробелы, «,» — запятая последней, потому
    что может быть десятичным разделителем."""
    best, best_score = None, -1
    for delimiter in (";", "\t", None, ","):
        score = sum(_numeric_count(_split(line, delimiter)) >= 2 for line in lines if line.strip())
        if score > best_score:
            best, best_score = delimiter, score
    return best


def read_text_table(path, report):
    raw = Path(path).read_bytes()
    if b"\x00" in raw[:4096]:
        report.add("error", "Файл двоичный, а не текстовый: это не таблица чисел.",
                   "Сохраните данные как CSV или .xlsx (формат — в методичке).")
        return []
    text, encoding = _decode(raw)
    if encoding == "latin-1":
        report.add("warning", "Кодировка файла не распознана (не UTF-8 и не Windows-1251); "
                   "заголовки могут читаться неверно.", "Сохраните файл в UTF-8.")
    lines = text.splitlines()
    delimiter = _choose_delimiter(lines)
    rows = [_split(line, delimiter) for line in lines]
    if delimiter == ",":
        _check_decimal_comma(rows, report)
    return rows


def _check_decimal_comma(rows, report):
    """Запятая и разделитель столбцов, и десятичный: «0,5,1,2e-6» читается неоднозначно."""
    numeric = [r for r in rows if _numeric_count(r) >= 2]
    if numeric and all(len(r) % 2 == 0 and len(r) >= 4 for r in numeric) and all(
            re.fullmatch(r"\d+(e[-+]?\d+)?", r[1].lower()) for r in numeric):
        report.add("error", "Запятая служит и разделителем столбцов, и десятичным знаком — "
                   "столбцы не определить однозначно.",
                   "Разделяйте столбцы точкой с запятой или табуляцией, либо пишите "
                   "дробную часть через точку.")


def read_excel_table(path, report):
    suffix = Path(path).suffix.lower()
    if suffix in LEGACY_EXCEL_SUFFIXES:
        return _read_xls(path, report)
    try:
        import openpyxl
    except ImportError:
        report.add("error", "Для чтения Excel нужен пакет openpyxl, он не установлен.",
                   "Установите зависимости (pip install -r requirements.txt) или сохраните "
                   "данные как CSV.")
        return []
    try:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as error:  # повреждённый файл, не Excel, защищён паролем
        report.add("error", f"Файл Excel не открывается: {error}.",
                   "Откройте его в Excel и сохраните заново как .xlsx или CSV.")
        return []
    sheets = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in book.worksheets}
    book.close()
    return _pick_sheet(sheets, report)


def _read_xls(path, report):
    try:
        import xlrd
    except ImportError:
        report.add("error", "Формат .xls (Excel 97–2003) без пакета xlrd не читается.",
                   "Сохраните файл в Excel как .xlsx («Файл → Сохранить как → Книга Excel») "
                   "или как CSV.")
        return []
    try:
        book = xlrd.open_workbook(path)
    except Exception as error:
        report.add("error", f"Файл .xls не открывается: {error}.",
                   "Сохраните его заново как .xlsx или CSV.")
        return []
    sheets = {sh.name: [sh.row_values(i) for i in range(sh.nrows)] for sh in book.sheets()}
    return _pick_sheet(sheets, report)


def _pick_sheet(sheets, report):
    scores = {name: sum(_numeric_count(r) >= 2 for r in rows) for name, rows in sheets.items()}
    if not scores or max(scores.values()) == 0:
        if not any(c is not None for rows in sheets.values() for r in rows for c in r):
            report.add("error", "В книге Excel нет значений: листы пустые или в ячейках формулы "
                       "без сохранённых результатов (файл создан не в Excel).",
                       "Откройте файл в Excel и сохраните его заново — или сохраните как CSV.")
            return []
        return next(iter(sheets.values()))
    name = max(scores, key=scores.get)
    report.sheet = name
    others = [n for n, s in scores.items() if s and n != name]
    if others:
        report.add("warning", f"Данные есть на нескольких листах; взят лист «{name}» "
                   f"(больше всего числовых строк). Не взяты: {', '.join(others)}.",
                   "Если нужен другой лист, сохраните его отдельным файлом.")
    return sheets[name]


def read_table(path, report):
    suffix = Path(path).suffix.lower()
    if suffix in EXCEL_SUFFIXES | LEGACY_EXCEL_SUFFIXES:
        return read_excel_table(path, report)
    if suffix in {".ods", ".numbers", ".xlsb", ".json", ".xml", ".mat", ".opj"}:
        report.add("error", f"Формат {suffix} не поддерживается.",
                   "Сохраните данные как .xlsx или CSV (формат — в методичке).")
        return []
    if suffix not in TEXT_SUFFIXES:
        report.add("info", f"Расширение «{suffix or 'нет'}» не из списка поддерживаемых; "
                   "файл прочитан как текстовая таблица.")
    return read_text_table(path, report)


# ------------------------------------------------------------ выбор столбцов
def _find_header(rows, first_data):
    """Последняя строка перед данными, где есть нечисловые ячейки, — заголовок."""
    for row in reversed(rows[:first_data]):
        cells = ["" if c is None else str(c).strip() for c in row]
        if sum(bool(c) for c in cells) >= 2 and _numeric_count(cells) < 2:
            return cells
    return None


def _select_columns(header, width, report, kind):
    """Индексы столбцов (напряжение, величина) по заголовку или первые два."""
    target = "current" if kind == KIND_IV else "capacitance"
    if header:
        found = {}
        for index, name in enumerate(header[:width]):
            quantity = column_quantity(name)
            if quantity and quantity not in found:
                found[quantity] = index
        if "voltage" in found and target in found:
            if width > 2 or (found["voltage"], found[target]) != (0, 1):
                report.add("info", f"Столбцы выбраны по заголовку: «{header[found['voltage']]}» "
                           f"и «{header[found[target]]}».")
            return found["voltage"], found[target]
        other = "capacitance" if kind == KIND_IV else "current"
        if other in found and target not in found:
            what = "ВАХ" if kind == KIND_IV else "ВФХ"
            report.add("warning", f"По заголовку в файле {_QUANTITY_RU[other]}, а загружается {what}.",
                       "Проверьте, что выбрали нужную кнопку загрузки (ВАХ или ВФХ).")
    if width > 2:
        report.add("warning", f"В файле {width} числовых столбцов; взяты первые два.",
                   "Подпишите столбцы заголовком («V», «I» или «C») — программа выберет нужные "
                   "сама — или оставьте в файле только два столбца.")
    return 0, 1


# ------------------------------------------------------------ разбор
def analyze(path, kind=KIND_IV):
    """Прочитать файл измерения и проверить данные. Возвращает DataReport."""
    path = Path(path)
    report = DataReport(path=path, kind=kind)
    if not path.is_file():
        report.add("error", f"Файл не найден: {path}.",
                   "Проверьте путь; в наборе образца пути к файлам задаются относительно файла набора.")
        return report
    try:
        rows = read_table(path, report)
    except OSError as error:
        report.add("error", f"Файл не читается: {error}.", "Проверьте, что он не открыт с блокировкой "
                   "в другой программе и что есть права на чтение.")
        return report
    if report.errors:
        return report

    numeric_rows = [i for i, r in enumerate(rows) if _numeric_count(r) >= 2]
    if not numeric_rows:
        report.add("error", "В файле не найдено ни одной строки с двумя числами.",
                   "Нужны минимум два числовых столбца: V и I (ВАХ) или V и C (ВФХ). Проверьте "
                   "разделитель столбцов и десятичный знак.")
        return report

    first = numeric_rows[0]
    header = _find_header(rows, first)
    width = max(len(rows[i]) for i in numeric_rows)
    widths = {sum(parse_number(c) is not None for c in rows[i]) for i in numeric_rows}
    width = max(widths) if widths else width
    iv, iy = _select_columns(header, width, report, kind)

    points, skipped = [], []
    for i in range(first, len(rows)):
        row = rows[i]
        x = parse_number(row[iv]) if iv < len(row) else None
        y = parse_number(row[iy]) if iy < len(row) else None
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            if any(c not in (None, "") for c in row):
                skipped.append(i + 1)
            continue
        points.append((x, y))
    if skipped:
        shown = ", ".join(map(str, skipped[:5])) + (" …" if len(skipped) > 5 else "")
        report.add("warning", f"Среди данных {len(skipped)} строк не прочитаны как числа "
                   f"и пропущены (строки {shown}).",
                   "Проверьте эти строки: пустые ячейки, текст, «nan», «inf».")
    if len(points) < 2:
        report.add("error", "Числовых точек меньше двух — строить нечего.",
                   "Проверьте, что выбраны нужные столбцы.")
        return report

    data = np.array(points, dtype=float)
    quantity = "current" if kind == KIND_IV else "capacitance"
    names = (header[iv], header[iy]) if header and max(iv, iy) < len(header) else ("", "")
    report.columns = names
    _apply_units(data, names, ("voltage", quantity), report)
    if report.errors:
        return report

    if np.any(np.diff(data[:, 0]) < 0):
        report.add("info", "Точки отсортированы по напряжению.")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    report.voltage, report.value = data[:, 0].copy(), data[:, 1].copy()

    _check_common(report)
    if kind == KIND_IV:
        _check_iv(report, bool(header))
    else:
        _check_cv(report, bool(header))
    return report


def _apply_units(data, names, quantities, report):
    for column, (name, quantity) in enumerate(zip(names, quantities)):
        if not name:
            continue
        scale = column_scale(name, quantity)
        if scale is None or scale == 1.0:
            continue
        if math.isnan(scale):
            report.add("error", f"Столбец «{name}» — плотность тока (на единицу площади), "
                       "а нужен ток в амперах.",
                       "Умножьте плотность на площадь мезы A = πD²/4 (п. 2.4) или загрузите "
                       "исходный файл с током.")
            return
        data[:, column] *= scale
        unit = _BASE_UNIT[quantity]
        report.add("info", f"Столбец «{name}» ({_QUANTITY_RU[quantity]}): значения в "
                   f"{_PREFIXES[scale]}{unit} переведены в {unit} (×{scale:g}).")


def _check_common(report):
    V, Y = report.voltage, report.value
    n = len(V)
    if n < MIN_POINTS:
        report.add("warning", f"Мало точек: {n}. Для n_эксп и автоподгонки нужно хотя бы "
                   f"{MIN_POINTS}, лучше 100 и больше.",
                   "Снимите кривую с меньшим шагом по напряжению.")
    if np.ptp(V) == 0:
        report.add("error", "Все напряжения одинаковые — это не кривая.",
                   "Проверьте, что первым столбцом идёт напряжение.")
        return
    if np.ptp(Y) == 0:
        report.add("error", "Второй столбец постоянен — измеренная величина не меняется.",
                   "Проверьте, что выбран нужный столбец и что прибор был подключён.")
        return
    unique, counts = np.unique(V, return_counts=True)
    repeated = unique[counts > 1]
    if repeated.size:
        spread = max(np.ptp(Y[V == v]) / max(np.max(np.abs(Y[V == v])), 1e-300) for v in repeated)
        text = (f"Напряжения повторяются ({repeated.size} значений) — вероятно, прямой и обратный "
                "проход развёртки или несколько измерений.")
        if spread > 0.1:
            text += f" В одной точке значения расходятся до {100 * spread:.0f} % (гистерезис)."
        report.add("warning", text, "Для подгонки лучше оставить один проход: гистерезис указывает "
                   "на перезарядку ловушек или нагрев.")
    if np.max(np.abs(V)) > MAX_ABS_VOLTAGE:
        report.add("warning", f"Напряжение до {np.max(np.abs(V)):.3g} В — для этой структуры "
                   "слишком много; возможно, это милливольты.",
                   "Подпишите столбец «V (мВ)» — программа переведёт единицы сама.")


def _is_sweep(values):
    """Значения — равномерная сетка (как напряжение развёртки)."""
    steps = np.diff(np.unique(values))
    return steps.size >= 4 and np.std(steps) < 0.05 * np.mean(steps)


def _check_iv(report, has_header):
    V, I = report.voltage, report.value
    vmax_abs, imax_abs = np.max(np.abs(V)), np.max(np.abs(I))
    if not has_header and ((vmax_abs < 1e-2 and imax_abs > 0.1)
                           or (_is_sweep(I) and not _is_sweep(V))):
        report.add("error", "Похоже, столбцы перепутаны: второй столбец выглядит как напряжение "
                   f"развёртки (равномерная сетка или значения до {imax_abs:.3g}), а первый — "
                   f"как ток (до {vmax_abs:.2g}).",
                   "Поставьте напряжение первым столбцом или подпишите столбцы «V» и «I».")
        return
    if imax_abs > MAX_ABS_CURRENT:
        report.add("warning", f"Ток до {imax_abs:.3g} А — для мезадиода слишком много; "
                   "возможно, ток записан в мА или мкА.",
                   "Подпишите столбец «I (мА)» или «I (мкА)» — программа переведёт единицы сама.")
    reverse, forward = V < -1e-3, V > 1e-3
    if not reverse.any():
        report.add("warning", "Нет обратной ветви (V < 0): утечку (R_sh, I_L·|V|^m) по этим данным "
                   "не определить.", "Для подгонки снимайте ВАХ в обе стороны от нуля.",
                   ref="п. 1А.17; гл. 9")
    if not forward.any():
        report.add("warning", "Нет прямой ветви (V > 0): n, J₀ и R_s по этим данным не определить.",
                   "Для подгонки снимайте ВАХ в обе стороны от нуля.", ref="п. 1А.17; гл. 8")
    # Знак: прямой ток > 0, обратный < 0 (I·V ≥ 0 вне шума у нуля).
    mask = np.abs(V) > 0.05 * vmax_abs
    signs = np.sign(I[mask] * V[mask])
    if reverse.any() and forward.any() and signs.size and np.mean(signs < 0) > 0.8:
        report.value = -I
        report.add("warning", "Знак тока обратный (при V > 0 ток отрицательный): ток умножен на −1.",
                   "Проверьте полярность подключения: «+» прибора — к p-стороне.")
    elif reverse.any() and np.all(I[reverse] >= 0) and np.any(I[reverse] > 0):
        report.value = np.where(reverse, -np.abs(I), I)
        report.add("warning", "Обратный ток записан по модулю (положительный при V < 0); "
                   "программа поставила знак «−».",
                   "Если это не так — проверьте файл: при V < 0 ток диода отрицателен.")
    if forward.any() and reverse.any():
        i_fwd = np.max(np.abs(report.value[forward]))
        i_rev = np.max(np.abs(report.value[reverse]))
        v_fwd, v_rev = np.max(V[forward]), np.max(-V[reverse])
        if v_fwd >= 0.3 and v_rev >= 0.3 and i_fwd < i_rev:
            report.add("warning", "Прямой ток меньше обратного — выпрямления нет. Это похоже на "
                       "резистор, пробой или перепутанную полярность.",
                       "Проверьте контакты и полярность; для такого образца модель p-n перехода "
                       "не применима.", ref="п. 1А.17; гл. 9")


def _check_cv(report, has_header):
    C = report.value
    if np.all(C <= 0):
        report.add("error", "Все значения ёмкости неположительные.",
                   "Проверьте столбец: нужна ёмкость C (Ф), а не фаза, проводимость или ток.")
        return
    if np.any(C <= 0):
        report.add("warning", f"{int(np.sum(C <= 0))} значений ёмкости ≤ 0 — на графике 1/C² они "
                   "не показываются.", "Проверьте эти точки: часто это сбой измерителя у нуля.")
    if np.max(C) > MAX_CAPACITANCE:
        report.add("warning", f"Ёмкость до {np.max(C):.3g} Ф — для мезадиода слишком много; "
                   "возможно, записана в пФ или нФ.",
                   "Подпишите столбец «C (пФ)» или «C (нФ)» — программа переведёт единицы сама.")


# ------------------------------------------------------------ загрузка
def load(path, kind=KIND_IV):
    """(V, величина, report); при ошибке — DataFileError с разбором."""
    report = analyze(path, kind)
    if not report.ok:
        raise DataFileError(report)
    return report.voltage, report.value, report


def example_files():
    """Примеры файлов, поставляемые с программой (имя → путь)."""
    if not EXAMPLES_DIR.is_dir():
        return {}
    return {p.name: p for p in sorted(EXAMPLES_DIR.iterdir())
            if p.suffix.lower() in SUPPORTED_SUFFIXES}
