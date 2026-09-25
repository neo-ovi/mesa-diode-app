# -*- coding: utf-8 -*-
"""Реестр моделей тока p-n перехода.

Модель тока перехода — это всё, что отличает одну модель от другой:
какие поля окна ей нужны, по какой формуле считается ток при напряжении на
переходе V_d, какие параметры подбирает автоподгонка и с каких значений она
начинает. Эквивалентная схема (R_s(I), R_sh, утечка I_L·|V|^m, формула (6.1))
общая для всех моделей и живёт в physics.

Чтобы добавить свою (например, упрощённую) модель:

1. Напишите функцию parts(s, Vd) → {ключ компоненты: ток, А} — ток перехода
   без эквивалентной схемы; ключи компонент — из COMPONENTS (или добавьте
   свой ключ туда и в app.COMPONENT_STYLE / app.CURVES).
2. Опишите её параметры: поля окна (presets.ParamSpec, поле Structure) и
   параметры подгонки (FitParam в FIT_PARAMS).
3. Зарегистрируйте CurrentModel через register(...). basic=True — модель
   появится в списке «Модель тока» базового режима.
4. Добавьте формулу с источником в окно формул (formulas.py), подсказки
   полей (hints.py) и пункт методички.

Остальное — решатель (6.1), графики, автоподгонка с выбором механизмов,
фиксация параметров — работает с любой зарегистрированной моделью.
"""

from dataclasses import dataclass

import numpy as np

from mesa_diode.simulator import physics as ph

# Компоненты тока перехода (ключи словаря parts) — подписи и цвета в app.
COMPONENTS = ("emp", "d1", "d2", "diff", "gr")


@dataclass(frozen=True)
class FitParam:
    """Параметр автоподгонки: поле Structure (None — особое правило в
    fitting.apply), логарифмическая шкала, допустимые границы, единица, подпись."""

    field: object
    log: bool
    lo: float
    hi: float
    unit: str
    label: str


@dataclass(frozen=True)
class CurrentModel:
    key: str               # значение Structure.model
    label: str             # название в окне
    formula: str           # номер формулы в окне «Формулы и параметры»
    summary: str           # одна-две фразы для подсказки
    fields: frozenset      # поля окна, которые нужны только этой модели
    fit_core: tuple        # параметры подгонки тока перехода (без схемы)
    components: tuple      # ключи компонент, которые возвращает parts (из COMPONENTS)
    parts: object          # parts(s, Vd) → {компонента: ток, А}
    saturation: object     # saturation(s) → (J при V → 0, А/см², подпись)
    start: object          # start(s, V, I) → начальные значения fit_core
    basic: bool = False    # доступна в базовом режиме
    ref: str = ""          # пункт методички


MODELS = {}


def register(model):
    MODELS[model.key] = model
    return model


def get(key):
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"неизвестная модель тока: {key!r}; есть: {', '.join(MODELS)}") from None


def basic_models():
    return [m for m in MODELS.values() if m.basic]


def model_fields():
    """Поля всех моделей тока (скрываются, если модель не выбрана)."""
    return frozenset().union(*(m.fields for m in MODELS.values()))


# ------------------------------------------------------------ параметры
# Параметры тока перехода; параметры схемы (R_s, R_sh, I_L, m, I_mod) — в fitting.
FIT_PARAMS = {
    # n: 1 — диффузия [Зи, с. 94], 2 — рекомбинация в ОПЗ [СНШ57]; вне [1, 2]
    # формула (6.3) теряет физический смысл — упор в границу сообщает о другом механизме.
    "J0_emp": FitParam("J0_emp", True, 1e-15, 1e3, "А/см²", "J₀"),
    "n_emp": FitParam("n_emp", False, 1.0, 2.0, "—", "n"),
    "J01_2d": FitParam("J01_2d", True, 1e-20, 1e3, "А/см²", "J₀₁ (n = 1)"),
    "J02_2d": FitParam("J02_2d", True, 1e-15, 1e3, "А/см²", "J₀₂ (n = 2)"),
    "tau0_bg": FitParam("tau0_bg", True, 1e-13, 1e-2, "с", "τ₀^bg"),
    "tau_bg": FitParam(None, True, 1e-12, 1e-1, "с", "τ_n^bg = τ_p^bg"),   # τ_n^bg = τ_p^bg
}


def _reverse_start(V, I, area):
    """|I| при −0.1 В, А/см² — порядок тока насыщения."""
    i0 = np.interp(-0.1, V, I) if V[0] < -0.1 else -1e-3 * np.max(np.abs(I))
    return max(abs(i0), 1e-12) / area


# ------------------------------------------------ эмпирическая (6.3)
def _empirical_parts(s, Vd):
    Vd = np.asarray(Vd, dtype=float)
    return {"emp": s.area * s.J0_emp * ph._expm1(Vd / (s.n_emp * s.Vt))}


register(CurrentModel(
    key=ph.MODEL_EMPIRICAL, label="эмпирическая (6.3)", formula="(6.3)",
    summary="I = A·J₀·(e^{qV/nkT} − 1): два параметра, n и J₀, без связи со структурой.",
    fields=frozenset({"n_emp", "J0_emp"}), fit_core=("J0_emp", "n_emp"),
    components=("emp",), parts=_empirical_parts,
    saturation=lambda s: (s.J0_emp, "J₀"),
    start=lambda s, V, I: {"J0_emp": _reverse_start(V, I, s.area), "n_emp": 1.5},
    basic=True, ref="п. 1А.7; гл. 8"))


# ------------------------------------------------ двухдиодная (6.3а)
def _two_diode_parts(s, Vd):
    Vd = np.asarray(Vd, dtype=float)
    return {"d1": s.area * s.J01_2d * ph._expm1(Vd / s.Vt),
            "d2": s.area * s.J02_2d * ph._expm1(Vd / (2.0 * s.Vt))}


def _two_diode_start(s, V, I):
    j0 = _reverse_start(V, I, s.area)
    return {"J02_2d": j0, "J01_2d": 1e-3 * j0}


register(CurrentModel(
    key=ph.MODEL_TWO_DIODE, label="двухдиодная (6.3а)", formula="(6.3а)",
    summary="I = A·[J₀₁(e^{qV/kT} − 1) + J₀₂(e^{qV/2kT} − 1)]: диффузия (n = 1) и рекомбинация "
            "в ОПЗ (n = 2) по отдельности [Зи, с. 99, ур. (55)].",
    fields=frozenset({"J01_2d", "J02_2d"}), fit_core=("J01_2d", "J02_2d"),
    components=("d1", "d2"), parts=_two_diode_parts,
    saturation=lambda s: (s.J01_2d + s.J02_2d, "J₀₁ + J₀₂"),
    start=_two_diode_start,
    basic=True, ref="п. 8.4"))


# ------------------------------------------------ физическая (§4–§5)
def _physical_parts(s, Vd):
    return {"diff": ph.diffusion_current(s, Vd), "gr": ph.gr_current(s, Vd)}


register(CurrentModel(
    key=ph.MODEL_PHYSICAL, label="физическая (§4–§6)", formula="(4.4) + (5.4)",
    summary="Диффузия (4.4) и ток ОПЗ (5.4) по геометрии, легированию и временам жизни.",
    fields=frozenset(), fit_core=("tau0_bg", "tau_bg"),
    components=("diff", "gr"), parts=_physical_parts,
    saturation=lambda s: (float(ph.saturation_current_density(s, 0.0)), "J_s(0)"),
    start=lambda s, V, I: {"tau0_bg": s.tau0_bg, "tau_bg": float(np.sqrt(s.tau_n_bg * s.tau_p_bg))},
    ref="гл. 4–7"))
