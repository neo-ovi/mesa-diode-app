# -*- coding: utf-8 -*-
"""Окно на разных системах: шрифты, масштаб экрана, тема, сочетания клавиш,
внешние программы, значок и пункт меню приложений (методичка, п. 1А.22).

Программа сначала делалась под Windows. В Linux (Linux Mint, Ubuntu, Debian)
по-другому, и это учтено здесь:
- шрифта Segoe UI нет: все шрифты окна — именованные шрифты Tk (FONTS), их
  семейство один раз выбирает setup() — первое доступное из UI_FAMILIES;
- масштаб экрана (HiDPI, «Масштаб интерфейса» и размер шрифта в настройках
  рабочего стола) Tk сам не знает: читается Xft.dpi из ресурсов X; вручную —
  «Настройки → Масштаб интерфейса» или переменная MESA_UI_SCALE. Размеры в
  пикселях пересчитывает px();
- тема ttk по умолчанию («default») устарела — ставится «clam»;
- на русской раскладке Ctrl+C/V/X/A/Z в полях Tk не срабатывают: событие
  выбирается по коду клавиши (CLIPBOARD_KEYCODES);
- в диалоге файлов Tk расширения чувствительны к регистру (*.csv не находит
  DATA.CSV), «*.*» не находит файлы без точки, видны скрытые файлы —
  file_types() и setup();
- собранная программа (PyInstaller) подменяет LD_LIBRARY_PATH своими
  библиотеками: внешние программы (xdg-open и то, что он запускает) получают
  исходное окружение — system_env().
На Windows всё как раньше: Segoe UI, системный масштаб, тема ttk по умолчанию.
tkinter импортируется внутри функций: чистые функции проверяют тесты без Tk."""

import json
import os
import subprocess
import sys
from pathlib import Path

APP_ID = "mesa-simulator"          # имя файлов пункта меню и значка
WM_CLASS = "MesaSimulator"         # класс окна X11 = StartupWMClass пункта меню
APP_NAME = "Симулятор мезадиода Ge"
APP_COMMENT = "ВАХ, ВФХ и J–V мезадиода Ge: модель и подгонка к эксперименту"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SETTINGS_PATH = Path.home() / ".mesa_diode" / "settings.json"
LOG_PATH = Path.home() / ".mesa_diode" / "errors.log"
SCALE_ENV = "MESA_UI_SCALE"
SCALE_KEY = "ui_scale"
SCALE_CHOICES = (1.0, 1.25, 1.5, 1.75, 2.0)
SCALE_LIMITS = (0.75, 3.0)
BASE_DPI = 96.0                    # масштаб 1 — 96 точек на дюйм, как в Windows при 100 %

# Шрифты окна: имя шрифта Tk → (размер, пт; жирный; курсив; моноширинный).
TEXT, BOLD = "MesaText", "MesaBold"
SMALL, SMALL_BOLD, SMALL_ITALIC = "MesaSmall", "MesaSmallBold", "MesaSmallItalic"
LARGE, LARGE_BOLD, LARGE_ITALIC = "MesaLarge", "MesaLargeBold", "MesaLargeItalic"
HEADING, BIG_BOLD, TITLE, MONO = "MesaHeading", "MesaBigBold", "MesaTitle", "MesaMono"
PART = "MesaPart"                  # заголовок части вкладки «Модель» (с версии 3.7)
FONTS = {
    TEXT: (9, False, False, False), BOLD: (9, True, False, False),
    SMALL: (8, False, False, False), SMALL_BOLD: (8, True, False, False), SMALL_ITALIC: (8, False, True, False),
    LARGE: (10, False, False, False), LARGE_BOLD: (10, True, False, False), LARGE_ITALIC: (10, False, True, False),
    HEADING: (11, True, False, False), BIG_BOLD: (12, True, False, False), TITLE: (13, True, False, False),
    PART: (15, True, False, False),
    MONO: (8, False, False, True),
}
# Стандартные шрифты Tk (кнопки, поля, меню, таблицы, диалоги): в Linux — тем же
# семейством; в Windows они и так Segoe UI 9. TkCaptionFont — текст окон сообщений
# Tk в Linux: не жирный, сообщения программы длинные.
TK_FONTS = {
    "TkDefaultFont": (9, False, False, False), "TkTextFont": (9, False, False, False),
    "TkMenuFont": (9, False, False, False), "TkHeadingFont": (9, True, False, False),
    "TkCaptionFont": (10, False, False, False), "TkSmallCaptionFont": (8, False, False, False),
    "TkIconFont": (9, False, False, False), "TkTooltipFont": (9, False, False, False),
    "TkFixedFont": (9, False, False, True),
}
# Семейства по порядку: Windows; Linux Mint и Ubuntu; Debian и LMDE; остальные.
UI_FAMILIES = ("Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", "Liberation Sans", "FreeSans",
               "Helvetica")
MONO_FAMILIES = ("DejaVu Sans Mono", "Ubuntu Mono", "Noto Sans Mono", "Liberation Mono", "Consolas",
                 "Courier New", "Courier")

# Ctrl+клавиша на нелатинской раскладке: код клавиши X11 (evdev) → событие Tk.
CLIPBOARD_KEYCODES = {54: "<<Copy>>", 55: "<<Paste>>", 53: "<<Cut>>", 38: "<<SelectAll>>", 52: "<<Undo>>"}
EDIT_CLASSES = frozenset({"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"})
# Символы, из-за которых аргумент строки Exec берётся в кавычки (спецификация Desktop Entry).
EXEC_RESERVED = frozenset(" \t\n\"'\\><~|&;$*?#()`")

# Итог setup(): масштаб (px() умножает на него размеры в пикселях), откуда он взят
# ("env", "settings", "xft", "default") и выбранные семейства шрифтов.
SCALE = 1.0
SCALE_SOURCE = "default"
UI_FAMILY = MONO_FAMILY = ""


def px(value):
    """Размер в пикселях с учётом масштаба окна (вёрстка рассчитана на 96 точек на дюйм)."""
    return int(round(value * SCALE))


def is_frozen():
    """Собранная программа (PyInstaller), а не запуск из исходников."""
    return bool(getattr(sys, "frozen", False))


# ---------------------------------------------------------- настройки --

def read_settings(path=None):
    """Настройки окна (~/.mesa_diode/settings.json) или {}."""
    path = SETTINGS_PATH if path is None else Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_setting(key, value, path=None):
    """Записать одну настройку (None — удалить), остальные сохраняются. True — записано."""
    path = SETTINGS_PATH if path is None else Path(path)
    data = read_settings(path)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True


# -------------------------------------------------- внешние программы --

def system_env(environ=None, frozen=None, bundle=None):
    """Окружение для внешних программ.

    Собранная программа (PyInstaller) ставит в LD_LIBRARY_PATH каталог своих
    библиотек, а прежнее значение хранит в LD_LIBRARY_PATH_ORIG; программа
    просмотра PDF, запущенная через xdg-open, со старыми библиотеками сборки
    может не открыться. Здесь — исходное окружение: прежний LD_LIBRARY_PATH,
    без TCL_LIBRARY/TK_LIBRARY сборки и служебных переменных PyInstaller."""
    env = dict(os.environ if environ is None else environ)
    if not (is_frozen() if frozen is None else frozen):
        return env
    original = env.pop("LD_LIBRARY_PATH_ORIG", None)
    if original is not None:
        env["LD_LIBRARY_PATH"] = original
    else:
        env.pop("LD_LIBRARY_PATH", None)
    bundle = getattr(sys, "_MEIPASS", None) if bundle is None else bundle
    for key in ("TCL_LIBRARY", "TK_LIBRARY"):
        if bundle and env.get(key, "").startswith(str(bundle)):
            env.pop(key)
    for key in [key for key in env if key.startswith("_PYI") or key == "_MEIPASS2"]:
        env.pop(key)
    return env


def run_quiet(args, timeout=3.0):
    """Вывод внешней команды или "" (нет такой команды, ошибка, тайм-аут)."""
    try:
        result = subprocess.run(args, env=system_env(), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def open_path(path):
    """Открыть файл системным приложением. Возвращает текст ошибки или None."""
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 — штатный способ Windows
            return None
        command = ["open", path] if sys.platform == "darwin" else ["xdg-open", path]
        subprocess.Popen(command, env=system_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except (OSError, AttributeError) as e:
        return str(e)
    return None


def system_summary():
    """Одна строка о системе для журнала ошибок и проверки --check."""
    import platform

    libc = " ".join(platform.libc_ver()).strip()
    kind = "собранная программа" if is_frozen() else "из исходников"
    return f"{platform.platform()}; {libc or 'libc ?'}; Python {platform.python_version()} ({kind})"


def log_error(exc_type, value, tb):
    """Записать трассировку в ~/.mesa_diode/errors.log и в stderr. Возвращает путь к журналу."""
    import datetime
    import traceback

    text = "".join(traceback.format_exception(exc_type, value, tb))
    try:
        sys.stderr.write(text)
    except (AttributeError, OSError):    # у программы без консоли stderr может не быть
        pass
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}; {system_summary()}\n{text}\n")
    except OSError:
        pass
    return LOG_PATH


def show_startup_error(log_path):
    """Окно ошибки, если главное окно не открылось (запуск двойным щелчком — без терминала)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP_NAME, f"Программа не запустилась. Подробности записаны в {log_path}.",
                             parent=root)
        root.destroy()
    except Exception:  # noqa: BLE001 — окна нет совсем: остаётся журнал и stderr
        pass


# ------------------------------------------------------------- масштаб --

def parse_xft_dpi(resources):
    """Xft.dpi из текста ресурсов X (строка «Xft.dpi: 120») или None."""
    for line in resources.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "Xft.dpi":
            try:
                dpi = float(value.strip())
            except ValueError:
                return None
            return dpi if dpi > 0 else None
    return None


def x_resources():
    """Ресурсы X — свойство RESOURCE_MANAGER корневого окна. Его заполняет рабочий
    стол (Cinnamon, MATE, Xfce, GNOME): Xft.dpi = 96 × масштаб интерфейса × масштаб
    шрифта. Читается через libX11, без внешних программ; если не вышло — xrdb -query."""
    try:
        import ctypes
        lib = ctypes.CDLL("libX11.so.6")
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XResourceManagerString.restype = ctypes.c_char_p
        lib.XResourceManagerString.argtypes = [ctypes.c_void_p]
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = lib.XOpenDisplay(None)
        if display:
            try:
                data = lib.XResourceManagerString(display)
            finally:
                lib.XCloseDisplay(display)
            return (data or b"").decode("utf-8", "replace")
    except (OSError, AttributeError):
        pass
    return run_quiet(["xrdb", "-query"])


def _parse_scale(raw):
    if raw is None or str(raw).strip().lower() in ("", "auto"):
        return None
    try:
        value = float(str(raw).replace(",", ".").rstrip("% "))
    except ValueError:
        return None
    if value > 10:               # «125» или «125 %» — проценты
        value /= 100
    return value if value > 0 else None


def choose_scale(environ=None, settings=None, dpi=None):
    """Масштаб окна и откуда он взят: (масштаб, источник).

    Порядок: переменная MESA_UI_SCALE («env») → «Настройки → Масштаб
    интерфейса» («settings») → Xft.dpi рабочего стола / 96 («xft») → 1
    («default»). dpi — число или функция без аргументов (читается, только
    если до неё дошло)."""
    environ = os.environ if environ is None else environ
    for source, raw in (("env", environ.get(SCALE_ENV)), ("settings", (settings or {}).get(SCALE_KEY))):
        value = _parse_scale(raw)
        if value:
            return _clamp_scale(value), source
    dpi = dpi() if callable(dpi) else dpi
    if dpi:
        return _clamp_scale(dpi / BASE_DPI), "xft"
    return 1.0, "default"


def _clamp_scale(value):
    low, high = SCALE_LIMITS
    return round(min(max(value, low), high), 2)


# ------------------------------------------------------- настройка окна --

def setup(root):
    """Настроить окно под систему: масштаб, шрифты, тема, клавиши, диалог файлов.
    Вызывать сразу после создания корневого окна, до остальных виджетов.
    Возвращает (масштаб, источник)."""
    global SCALE, SCALE_SOURCE
    x11 = root.tk.call("tk", "windowingsystem") == "x11"
    scale, source = choose_scale(settings=read_settings(),
                                 dpi=(lambda: parse_xft_dpi(x_resources())) if x11 else None)
    if x11 or source in ("env", "settings"):
        root.tk.call("tk", "scaling", BASE_DPI * scale / 72.0)
        SCALE = scale
    else:
        SCALE = 1.0                  # Windows, «Авто»: масштаб — системный, как раньше
    SCALE_SOURCE = source
    _create_fonts(root, x11)
    if x11:
        _setup_x11(root)
    return SCALE, SCALE_SOURCE


def _pick(families, preferred, fallback):
    return next((name for name in preferred if name in families), fallback)


def _create_fonts(root, x11):
    """Именованные шрифты окна (FONTS), в Linux — и стандартные шрифты Tk."""
    import tkinter as tk
    from tkinter import font as tkfont

    global UI_FAMILY, MONO_FAMILY
    families = set(tkfont.families(root))
    UI_FAMILY = _pick(families, UI_FAMILIES,
                      tkfont.Font(root=root, name="TkDefaultFont", exists=True).actual("family"))
    MONO_FAMILY = _pick(families, MONO_FAMILIES,
                        tkfont.Font(root=root, name="TkFixedFont", exists=True).actual("family"))
    specs = dict(FONTS, **(TK_FONTS if x11 else {}))
    keep = root.__dict__.setdefault("_mesa_fonts", {})   # ссылки: без них Python удалит шрифты Tk
    for name, (size, bold, italic, mono) in specs.items():
        options = dict(family=MONO_FAMILY if mono else UI_FAMILY, size=size,
                       weight="bold" if bold else "normal", slant="italic" if italic else "roman")
        try:
            font = tkfont.Font(root=root, name=name, exists=True)
            font.configure(**options)
        except tk.TclError:
            font = tkfont.Font(root=root, name=name, **options)
        keep[name] = font


def _setup_x11(root):
    """Linux: тема, высота строк таблиц, меню без отрывной черты, диалог файлов, Ctrl+клавиши."""
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk

    style = ttk.Style(root)
    if style.theme_use() == "default" and "clam" in style.theme_names():
        style.theme_use("clam")
        # кнопки clam — не уже 11 символов и с широкими полями: панель инструментов
        # не помещалась бы на экране ноутбука
        style.configure("TButton", width=-7, padding=(px(7), px(3)))
        # размеры элементов clam заданы в пикселях — при HiDPI растут вместе со шрифтом
        style.configure("TCheckbutton", indicatorsize=px(10))
        style.configure("TRadiobutton", indicatorsize=px(10))
        style.configure("TScrollbar", arrowsize=px(14))
        style.configure("TCombobox", arrowsize=px(14))
        style.configure("Sash", sashthickness=px(6))
    root.configure(background=style.lookup("TFrame", "background") or "#dcdad5")
    linespace = tkfont.Font(root=root, name=TEXT, exists=True).metrics("linespace")
    style.configure("Treeview", rowheight=linespace + px(6))
    root.option_add("*tearOff", False)
    # Диалог файлов Tk: скрытые файлы (.cache, .config …) не показывать, кнопка
    # «показать скрытые» — есть. Первый вызов загружает код диалога, иначе его
    # переменных ещё нет.
    try:
        root.tk.call("tk_getOpenFile", "-foobarbaz")
    except tk.TclError:
        pass
    try:
        root.tk.call("set", "::tk::dialog::file::showHiddenBtn", "1")
        root.tk.call("set", "::tk::dialog::file::showHiddenVar", "0")
    except tk.TclError:
        pass
    root.bind_all("<Control-KeyPress>", lambda event: _on_control_key(root, event), add="+")


def keycode_event(keysym, keycode):
    """Событие Tk для Ctrl+клавиша на нелатинской раскладке или None.

    На русской раскладке Tk видит Cyrillic_es вместо c, и стандартные
    привязки Ctrl+C/V/X/A/Z не срабатывают; на латинской всё работает и так."""
    if len(keysym) == 1 and keysym.isascii():
        return None
    return CLIPBOARD_KEYCODES.get(keycode)


def _on_control_key(root, event):
    import tkinter as tk

    virtual = keycode_event(event.keysym, event.keycode)
    if virtual is None:
        return None
    path = str(event.widget)          # виджет диалога Tk может быть без объекта Python
    try:
        if root.tk.call("winfo", "class", path) not in EDIT_CLASSES:
            return None
        root.tk.call("event", "generate", path, virtual)
    except tk.TclError:
        return None
    return "break"


def file_types(types, linux=None):
    """Фильтры диалога файлов. В Linux Tk сравнивает расширения с учётом регистра,
    а «*.*» не находит файлы без точки: к «*.csv» добавляется «*.CSV», «*.*» → «*»."""
    if not (sys.platform.startswith("linux") if linux is None else linux):
        return list(types)
    result = []
    for label, patterns in types:
        items = ["*" if item == "*.*" else item for item in patterns.split()]
        items += [item.upper() for item in items if item.upper() not in items]
        result.append((label, " ".join(items)))
    return result


def fit_window(window, width, height, min_width=None, min_height=None, maximize=False):
    """Размер окна (в точках вёрстки, см. px) не больше экрана. Если окно не
    помещается и maximize — развернуть его (экран ноутбука 1366×768)."""
    screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
    avail_w, avail_h = screen_w - px(16), screen_h - px(90)      # рамка окна и панель задач
    w, h = px(width), px(height)
    window.geometry(f"{min(w, avail_w)}x{min(h, avail_h)}")
    if min_width and min_height:
        window.minsize(min(px(min_width), avail_w), min(px(min_height), avail_h))
    if maximize and (w > avail_w or h > avail_h):
        _maximize(window)


def _maximize(window):
    import tkinter as tk

    try:
        if window.tk.call("tk", "windowingsystem") == "x11":
            window.attributes("-zoomed", True)
        else:
            window.state("zoomed")
    except tk.TclError:
        pass


# ------------------------------------------------- значок и меню приложений --

def _icon_pixel(u, v):
    """Цвет точки значка 64×64 (u, v — координаты в точках значка) или None — прозрачно.
    Синий квадрат со скруглёнными углами, на нём белый символ диода."""
    r = 12.0
    cx, cy = min(max(u, r), 64 - r), min(max(v, r), 64 - r)
    if (u - cx) ** 2 + (v - cy) ** 2 > r * r:
        return None
    white = "#ffffff"
    if 42 <= u <= 47 and 16 <= v <= 48:                     # черта катода
        return white
    if 20 <= u <= 42 and abs(v - 32) <= (42 - u) * 16 / 22:  # треугольник анода
        return white
    if 8 <= u <= 56 and 30 <= v <= 34:                      # выводы
        return white
    return "#1f4fb2"


def icon_image(root, size=64):
    """Значок программы — рисуется в коде, отдельного файла нет."""
    import tkinter as tk

    image = tk.PhotoImage(master=root, width=size, height=size)
    k = 64.0 / size
    pixels = [[_icon_pixel((x + 0.5) * k, (y + 0.5) * k) for x in range(size)] for y in range(size)]
    image.put(" ".join("{" + " ".join(color or "#1f4fb2" for color in row) + "}" for row in pixels))
    for y, row in enumerate(pixels):
        for x, color in enumerate(row):
            if color is None:
                image.transparency_set(x, y, True)
    return image


def set_icon(root):
    """Значок окна и панели задач."""
    import tkinter as tk

    try:
        images = [icon_image(root, size) for size in (64, 32, 16)]
        root.iconphoto(True, *images)
        root.__dict__["_mesa_icons"] = images
    except tk.TclError:
        pass


def launch_command():
    """Команда запуска этой программы: собранный файл или python -m из исходников."""
    if is_frozen():
        return [sys.executable]
    return [sys.executable, "-m", "mesa_diode.simulator.app"]


def _exec_arg(arg):
    """Аргумент строки Exec: % удваивается; с пробелом или спецсимволом — в кавычках
    (\\ " ` $ экранируются); затем общее правило строк .desktop удваивает \\."""
    arg = arg.replace("%", "%%")
    if not any(ch in EXEC_RESERVED for ch in arg):
        return arg
    quoted = '"' + "".join("\\" + ch if ch in '"`$\\' else ch for ch in arg) + '"'
    return quoted.replace("\\", "\\\\")


def desktop_entry(command, icon, workdir=None):
    """Текст пункта меню приложений (.desktop по спецификации freedesktop.org)."""
    lines = ["[Desktop Entry]", "Type=Application", "Version=1.0", f"Name={APP_NAME}",
             f"Comment={APP_COMMENT}", "Exec=" + " ".join(_exec_arg(str(arg)) for arg in command),
             f"Icon={icon}", "Terminal=false", "Categories=Education;Science;Physics;",
             f"StartupWMClass={WM_CLASS}", "StartupNotify=true"]
    if workdir:
        lines.append(f"Path={workdir}")
    return "\n".join(lines) + "\n"


def data_home(environ=None):
    """~/.local/share или XDG_DATA_HOME."""
    environ = os.environ if environ is None else environ
    return Path(environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def menu_entry_paths(environ=None):
    """(файл пункта меню .desktop, файл значка .png)."""
    share = data_home(environ)
    return (share / "applications" / f"{APP_ID}.desktop",
            share / "icons" / "hicolor" / "128x128" / "apps" / f"{APP_ID}.png")


def on_removable_drive(path):
    """Файл на съёмном диске (флешке): в Linux они монтируются в /media или /run/media."""
    return str(Path(path).resolve()).startswith(("/media/", "/run/media/", "/mnt/"))


def install_menu_entry(root):
    """Пункт «Симулятор мезадиода Ge» в меню приложений (Linux) со значком.
    Возвращает путь к файлу .desktop."""
    entry, icon = menu_entry_paths()
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon_image(root, 128).write(str(icon), format="png")
    entry.parent.mkdir(parents=True, exist_ok=True)
    workdir = None if is_frozen() else PROJECT_ROOT
    entry.write_text(desktop_entry(launch_command(), icon, workdir), encoding="utf-8")
    entry.chmod(0o755)
    run_quiet(["update-desktop-database", str(entry.parent)])
    return entry


def remove_menu_entry():
    """Убрать пункт меню и значок. True — что-то было удалено."""
    removed = False
    for path in menu_entry_paths():
        try:
            path.unlink()
            removed = True
        except OSError:
            pass
    return removed
