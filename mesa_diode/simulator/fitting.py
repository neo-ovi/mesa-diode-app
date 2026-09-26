# -*- coding: utf-8 -*-
"""Автоматическая подгонка модели к экспериментальной ВАХ (§6.7).

Нелинейные наименьшие квадраты с границами (trust region reflective,
scipy.optimize.least_squares) [BCL99] по невязке arsinh(I/I_ref); механизмы
(утечка, модуляция R_s) подключаются по необходимости, выбор варианта —
по (6.10), (6.11). Что минимизируется и как выбирается вариант — методичка,
пп. 9А.1–9А.3; параметры тока перехода — из реестра models (п. 13.6),
параметры схемы — CIRCUIT_PARAMS.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from mesa_diode.simulator import models
from mesa_diode.simulator import physics as ph

# Модели тока — ключи реестра models (значения Structure.model).
EMPIRICAL, TWO_DIODE, PHYSICAL = ph.MODEL_EMPIRICAL, ph.MODEL_TWO_DIODE, ph.MODEL_PHYSICAL

# Механизмы, которые подключаются по необходимости: ключ → (название, параметры).
TERM_LEAK = "leak"
TERM_MOD = "mod"
TERMS = {
    TERM_LEAK: ("нелинейная утечка I_L·|V|^m", ("I_L", "m_leak")),
    TERM_MOD: ("модуляция R_s током I_mod (6.9)", ("I_mod",)),
}
# Эквивалентная схема (6.1) — общая для всех моделей тока.
CIRCUIT_PARAMS = {
    "Rs": models.FitParam("Rs", True, 1e-3, 1e6, "Ом", "R_s"),
    "Rsh": models.FitParam("Rsh", True, 1.0, 1e12, "Ом", "R_sh"),
    "I_L": models.FitParam("I_L", True, 1e-14, 1.0, "А", "I_L"),
    "m_leak": models.FitParam("m_leak", False, 1.0, 8.0, "—", "m"),
    "I_mod": models.FitParam("I_mod", True, 1e-7, 1e2, "А", "I_mod"),
}
_ALL = {**models.FIT_PARAMS, **CIRCUIT_PARAMS}


def core(model):
    """Всегда подбираемые параметры: ток перехода модели + R_s и R_sh."""
    return tuple(models.get(model).fit_core) + ("Rs", "Rsh")


CORE = {key: core(key) for key in models.MODELS}

# Параметр → (поле Structure, логарифмическая шкала, нижняя, верхняя граница).
PARAMS = {k: (p.field, p.log, p.lo, p.hi) for k, p in _ALL.items()}
UNITS = {k: p.unit for k, p in _ALL.items()}
LABELS = {k: p.label for k, p in _ALL.items()}
SCALES = {k: p.scale for k, p in _ALL.items()}     # значение в окне = значение в Structure / scale

# Поле окна (presets) → параметр подгонки; τ_n^bg и τ_p^bg подбираются общим
# множителем tau_bg, поэтому фиксируются вместе.
FIELD_TO_PARAM = {**{(p.key or p.field): k for k, p in _ALL.items() if p.field},
                  "tau_n_bg": "tau_bg", "tau_p_bg": "tau_bg"}


def display_value(key, value):
    """Значение параметра подгонки в единицах окна (d_s — мкм, остальные как в Structure)."""
    return value / SCALES.get(key, 1.0) if value == value else value


def fittable_fields(model):
    """Поля окна, которые подгонка модели model может изменить."""
    params = set(core(model)) | {p for _name, keys in TERMS.values() for p in keys}
    return {f for f, p in FIELD_TO_PARAM.items() if p in params}


# Достаточная точность описания (критерий проекта): физическая модель, дающая
# δ (6.10) в пределах 4–8 %, адекватна; из таких вариантов берётся самый простой.
TARGET_ERROR = 0.05
LOOSE_STDERR = 1.0     # относительная погрешность > 100 % — параметр данными не определён

BIC_GAIN = 10.0        # механизм добавляется, если BIC падает больше чем на 10 [KR95]
MAX_POINTS = 400       # точки ВАХ для подгонки (равномерно по напряжению)


@dataclass
class Candidate:
    terms: tuple
    params: dict
    error: float
    bic: float
    accepted: bool = False


@dataclass
class FitResult:
    model: str
    params: dict                    # ключ интерфейса → значение
    stderr: dict                    # ключ → относительная погрешность (доля) или nan
    terms: tuple                    # подключённые механизмы (TERM_*)
    error: float                    # (6.10) ошибка по всей ВАХ, доля
    error_forward: float
    error_reverse: float
    bic: float
    V: np.ndarray
    I_exp: np.ndarray
    I_model: np.ndarray
    candidates: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    locked: tuple = ()              # параметры подгонки, зафиксированные пользователем
    target: float = TARGET_ERROR    # достаточная ошибка (6.10) при выборе варианта
    at_bounds: tuple = ()           # параметры, упёршиеся в границу допустимых значений

    def structure_values(self):
        """Значения для полей окна (ключи presets): τ_bg → τ_n^bg и τ_p^bg."""
        values = {}
        for key, value in self.params.items():
            if key == "tau_bg":
                values["tau_n_bg"] = values["tau_p_bg"] = value
            else:
                values[_ALL[key].key or key] = display_value(key, value)
        return values


# ------------------------------------------------------------ подготовка --

def prepare_data(V, I, max_points=MAX_POINTS):
    """Конечные точки, сортировка по V, без повторов; не больше max_points
    (прореживание равномерно по напряжению)."""
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = np.isfinite(V) & np.isfinite(I)
    V, first = np.unique(V[mask], return_index=True)
    I = I[mask][first]
    if V.size > max_points:
        idx = np.unique(np.round(np.linspace(0, V.size - 1, max_points)).astype(int))
        V, I = V[idx], I[idx]
    return V, I


def current_scale(I):
    """I_ref невязки: шаг квантования тока прибора (×3) или 10⁻⁴ от максимума."""
    values = np.unique(np.abs(I[I != 0]))
    step = float(np.min(np.diff(values))) if values.size > 1 else 0.0
    return max(3.0 * step, 1e-4 * float(np.max(np.abs(I))), 1e-15)


def relative_error(I_model, I_exp, floor):
    """(6.10) δ = √(Σ((I_мод − I_эксп)/max(|I_эксп|, I_min))²/N)."""
    denom = np.maximum(np.abs(I_exp), floor)
    return float(np.sqrt(np.mean(((I_model - I_exp) / denom) ** 2)))


def bic(residuals, k):
    """(6.11) BIC = N·ln(RSS/N) + k·ln N [Sch78]."""
    n = residuals.size
    rss = max(float(np.sum(residuals ** 2)), 1e-300)
    return n * np.log(rss / n) + k * np.log(n)


# --------------------------------------------------------------- модель --

def apply(s, values):
    """Structure с подставленными значениями параметров подгонки."""
    changes = {}
    for key, value in values.items():
        name = PARAMS[key][0]
        if name is None:           # tau_bg — общий множитель
            changes["tau_n_bg"] = changes["tau_p_bg"] = value
        else:
            changes[name] = value
    return s.updated(**changes)


def model_current(s, V, values):
    return ph.solve_iv(apply(s, values), V).I


def _start_values(model, s, V, I):
    """Начальное приближение по форме ВАХ (без подгонки)."""
    fwd = (V > 0) & (I > 0)
    rs = ph.series_resistance_limit(V, I)
    rs = 0.9 * rs if np.isfinite(rs) else max(s.Rs, 1.0)
    rev = V < -0.1
    rsh = 1e9
    if rev.sum() >= 3:
        slope = np.polyfit(V[rev], I[rev], 1)[0]
        rsh = 1.0 / slope if slope > 0 else 1e9
    start = {"Rs": rs, "Rsh": min(max(rsh, 10.0), 1e11), "I_L": 1e-3 * np.max(np.abs(I)),
             "m_leak": 2.0, "I_mod": 10.0 * np.max(np.abs(I))}
    start.update(models.get(model).start(s, V, I))
    if fwd.sum() < 3:
        start["Rs"] = max(s.Rs, 1.0)
    return start


def _fit_keys(model, s, V, I, keys, start, I_ref):
    """Подгонка параметров keys; остальные — из s. Возвращает (values, J, residuals)."""
    from scipy.optimize import least_squares

    logs = [PARAMS[k][1] for k in keys]
    lo = np.array([np.log10(PARAMS[k][2]) if lg else PARAMS[k][2] for k, lg in zip(keys, logs)])
    hi = np.array([np.log10(PARAMS[k][3]) if lg else PARAMS[k][3] for k, lg in zip(keys, logs)])

    def unpack(x):
        return {k: (10.0 ** v if lg else float(v)) for k, v, lg in zip(keys, x, logs)}

    x0 = np.array([np.log10(start[k]) if lg else start[k] for k, lg in zip(keys, logs)])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    target = np.arcsinh(I / I_ref)

    def residuals(x):
        Im = model_current(s, V, unpack(x))
        r = np.arcsinh(np.nan_to_num(Im, nan=0.0) / I_ref) - target
        return np.where(np.isfinite(Im), r, 50.0)

    if not keys:                   # всё зафиксировано — только расчёт невязок
        return {}, np.empty((V.size, 0)), residuals(x0), logs

    result = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
    return unpack(result.x), result.jac, result.fun, logs


def _stderr(keys, logs, jac, res):
    """Относительные погрешности из ковариации (JᵀJ)⁻¹·RSS/(N − k)."""
    n, k = res.size, len(keys)
    out = {key: float("nan") for key in keys}
    if k == 0 or n <= k:
        return out
    # Направления в пространстве параметров, вдоль которых невязки почти не меняются
    # (сингулярные числа якобиана ≤ 10⁻⁷ от наибольшего), данными не определены.
    # Параметр, лежащий в основном в таких направлениях, получает погрешность ∞
    # (псевдообратная матрица дала бы ложный ноль).
    try:
        _u, sv, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return out
    weak = sv <= 1e-7 * max(float(sv.max()), 1e-300)
    blind = (vt[weak] ** 2).sum(axis=0) > 0.5
    inv = np.where(weak, 0.0, 1.0 / np.maximum(sv, 1e-300) ** 2)
    cov = (vt.T * inv) @ vt * float(np.sum(res ** 2)) / (n - k)
    for i, (key, lg) in enumerate(zip(keys, logs)):
        if blind[i]:
            out[key] = float("inf")
            continue
        sigma = float(np.sqrt(max(cov[i, i], 0.0)))
        out[key] = sigma * np.log(10.0) if lg else sigma   # log10 → доля; линейный — абсолютная
    return out


def _variants(s, lock):
    """Проверяемые наборы механизмов с учётом фиксированных параметров.

    Зафиксированный I_L = 0 (или I_mod = ∞) выключает механизм, ненулевой
    (конечный) — включает его во всех вариантах."""
    variants = [(), (TERM_LEAK,), (TERM_MOD,), (TERM_LEAK, TERM_MOD)]
    if "I_L" in lock:
        forced = s.I_L > 0
        variants = [v for v in variants if (TERM_LEAK in v) == forced]
    if "I_mod" in lock:
        forced = np.isfinite(s.I_mod)
        variants = [v for v in variants if (TERM_MOD in v) == forced]
    return variants


def _locked_values(s, lock):
    """Значения зафиксированных параметров (для таблицы результата)."""
    values = {}
    for key in lock:
        if key == "tau_bg":
            if s.tau_n_bg == s.tau_p_bg:
                values[key] = s.tau_n_bg
        else:
            values[key] = getattr(s, PARAMS[key][0])
    return values


def select_variant(candidates, target=TARGET_ERROR):
    """Выбор варианта модели (§6.7).

    1. Физичность прежде точности: если есть варианты с ошибкой δ ≤ target,
       берётся самый простой из них (меньше механизмов; при равенстве — меньший
       BIC (6.11)). Лишний механизм не добавляется ради долей процента.
    2. Если ни один вариант не достиг target, механизм добавляется, только если
       BIC падает больше чем на BIC_GAIN [KR95]; сама недостижимость target —
       признак того, что в модели чего-то не хватает (сообщается в пояснениях).
    target = 0 — только критерий BIC."""
    adequate = [c for c in candidates if c.error <= target]
    if adequate:
        return min(adequate, key=lambda c: (len(c.terms), c.bic))
    best = None
    for cand in candidates:
        if best is None or cand.bic < best.bic - (BIC_GAIN if len(cand.terms) > len(best.terms) else 0.0):
            best = cand
    return best


def _at_bounds(keys, logs, values, rtol=1e-3):
    """Параметры, значения которых легли на границу допустимой области."""
    out = []
    for key, lg in zip(keys, logs):
        lo, hi = PARAMS[key][2], PARAMS[key][3]
        v = values[key]
        a, b, x = (np.log10(lo), np.log10(hi), np.log10(v)) if lg else (lo, hi, v)
        if abs(x - a) <= rtol * max(1.0, abs(b - a)) or abs(x - b) <= rtol * max(1.0, abs(b - a)):
            out.append(key)
    return tuple(out)


def fit_iv(s, V, I, model=EMPIRICAL, progress=None, locked=(), target=TARGET_ERROR):
    """Подгонка ВАХ с выбором механизмов (§6.7).

    s — Structure с текущими параметрами (геометрия, легирование и т. д.);
    model — EMPIRICAL (эмпирическая модель (6.3)) или PHYSICAL (§4–§6).
    progress(text) — необязательный вызов для строки состояния.
    target — достаточная ошибка (6.10) для выбора варианта (select_variant).
    locked — поля окна (ключи presets), зафиксированные пользователем: их
    значения берутся из s и не меняются."""
    base = replace(s, model=models.get(model).key)
    V, I = prepare_data(V, I)
    if V.size < 8:
        raise ValueError("для подгонки нужно не меньше 8 точек ВАХ")
    I_ref = current_scale(I)
    floor = 10.0 * I_ref
    start = _start_values(model, base, V, I)
    lock = {FIELD_TO_PARAM[f] for f in locked if f in FIELD_TO_PARAM}
    held = _locked_values(base, lock)

    # Выключенные механизмы: утечка I_L = 0, модуляция I_mod = ∞.
    off = {"I_L": 0.0, "I_mod": float("inf")}
    candidates = []
    best = None                      # лучший по BIC — только для начального приближения
    for terms in _variants(base, lock):
        if progress:
            progress("подгонка: " + (", ".join(TERMS[t][0] for t in terms) or "минимальная модель"))
        # d_s подбирается, только если обогащённый слой задан (N_As > 0)
        used = [k for k in core(model) if k != "d_s" or base.has_surface_layer]
        used += [p for t in terms for p in TERMS[t][1]]
        keys = [k for k in used if k not in lock]
        fixed = {k: v for k, v in off.items() if k not in used}
        s_run = apply(base, fixed)
        # старт — лучший предыдущий результат (кроме выключенных механизмов: 0 и ∞)
        seed = dict(start)
        if best is not None:
            seed.update({k: v for k, v in best.params.items() if np.isfinite(v) and v > 0})
        values, jac, res, logs = _fit_keys(model, s_run, V, I, keys, seed, I_ref)
        values_all = {**fixed, **{k: v for k, v in held.items() if k in used}, **values}
        Im = model_current(s_run, V, values)
        cand = Candidate(terms=terms, params=values_all, error=relative_error(Im, I, floor),
                         bic=bic(res, len(keys)))
        cand._fit = (keys, logs, jac, res, Im)
        candidates.append(cand)
        if best is None or cand.bic < best.bic - (BIC_GAIN if len(terms) > len(best.terms) else 0.0):
            best = cand
    best = select_variant(candidates, target)
    best.accepted = True
    keys, logs, jac, res, Im = best._fit
    for cand in candidates:
        del cand._fit
    fwd, rev = V > 0.05, V < -0.05
    result = FitResult(
        model=model, params=best.params, stderr=_stderr(keys, logs, jac, res), terms=best.terms,
        error=best.error,
        error_forward=relative_error(Im[fwd], I[fwd], floor) if fwd.any() else float("nan"),
        error_reverse=relative_error(Im[rev], I[rev], floor) if rev.any() else float("nan"),
        bic=best.bic, V=V, I_exp=I, I_model=Im, candidates=candidates, locked=tuple(sorted(lock)),
        target=target, at_bounds=_at_bounds(keys, logs, best.params))
    result.notes = explain(result, base)
    return result


# ------------------------------------------------------------ пояснения --

def _selection_notes(result):
    """Почему выбран этот вариант: достаточная точность или BIC."""
    target = 100 * result.target
    chosen = next(c for c in result.candidates if c.accepted)
    if result.target and chosen.error <= result.target:
        better = [c for c in result.candidates
                  if c.error < 0.8 * chosen.error and len(c.terms) > len(chosen.terms)]
        text = (f"Выбор: самый простой вариант с ошибкой не больше {target:.0f} % — физичность важнее "
                "долей процента.")
        if better:
            best = min(better, key=lambda c: c.error)
            names = ", ".join(TERMS[t][0] for t in best.terms)
            text += (f" С механизмом «{names}» ошибка была бы {100 * best.error:.2f} %; он не нужен для "
                     "описания в пределах точности, но может быть реальным — проверьте его отдельным "
                     "измерением или зафиксируйте, если он известен.")
        return [text]
    if result.target:
        return [f"Ни один вариант не описывает ВАХ с ошибкой ≤ {target:.0f} % — в модели не хватает "
                "механизма или неверны измеряемые параметры (геометрия, концентрации, T). Выбран вариант "
                "по критерию BIC (6.11). Смотрите график отклонений: где систематическая волна — там и "
                "недостающий механизм."]
    return ["Выбор по критерию BIC (6.11)."]


def _quality_notes(result):
    """Параметры на границе и неопределённые — признаки неполноты модели."""
    notes = []
    for key in result.at_bounds:
        lo, hi = PARAMS[key][2], PARAMS[key][3]
        value = result.params[key]
        side = "нижнюю" if abs(value - lo) <= abs(value - hi) else "верхнюю"
        text = f"{LABELS[key]} = {value:.3g} упёрся в {side} границу допустимых значений ({lo:g} … {hi:g})."
        if key == "n_emp":
            text += (" n вне [1, 2] диффузия и рекомбинация в ОПЗ не дают: ток определяет другой механизм "
                     "(туннелирование, утечка по поверхности, высокий уровень инжекции) — модель (6.3) "
                     "неполна для этой ВАХ.")
        else:
            text += " Подгонка не нашла физического значения: данные требуют механизма, которого в модели нет."
        notes.append(text)
    loose = [LABELS[k] for k, v in result.stderr.items() if not np.isnan(v) and v > LOOSE_STDERR]
    if loose:
        notes.append("Данными почти не определяются (погрешность больше 100 %): " + ", ".join(loose)
                     + ". Их значения условны — зафиксируйте известные параметры (галочка справа от поля) "
                       "или добавьте данные (другой диапазон V, T).")
    return notes


def _consistency_notes(result, s):
    """Согласованность найденных значений с физической моделью структуры."""
    notes = []
    p = result.params
    if result.model == EMPIRICAL and "J0_emp" in p:
        phys = replace(s, model=ph.MODEL_PHYSICAL)
        try:
            js = float(ph.saturation_current_density(phys, 0.0))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            js = float("nan")
        if np.isfinite(js) and js > 0:
            ratio = p["J0_emp"] / js
            if ratio > 100:
                notes.append(f"J₀ = {p['J0_emp']:.3g} А/см² в {ratio:.3g} раз больше диффузионного J_s "
                             f"= {js:.3g} А/см² структуры (4.3) при текущих N и τ: прямой ток — не диффузия "
                             "(рекомбинация в ОПЗ, утечка) или времена жизни короче заданных.")
            elif ratio < 0.01:
                notes.append(f"J₀ = {p['J0_emp']:.3g} А/см² в {1 / ratio:.3g} раз меньше диффузионного J_s "
                             f"= {js:.3g} А/см² структуры (4.3): проверьте площадь D, T и концентрации — "
                             "меньше диффузионного предела ток идеального диода не бывает.")
    if result.model == TWO_DIODE and "J02_2d" in p:
        # (5.4) при n₂ = 2: J_gr = q·n_i·W/(2τ₀) → τ₀ ≈ q·n_i·W(0)/(2·J₀₂)
        W0 = float(ph.depletion_width(s, 0.0))
        tau0 = ph.Q * s.ni * W0 / (2.0 * p["J02_2d"])
        notes.append(f"J₀₂ = {p['J02_2d']:.3g} А/см² по (5.4) соответствует τ₀ ≈ q·n_i·W/(2J₀₂) = {tau0:.3g} с "
                     f"(W(0) = {W0 * 1e4:.3g} мкм при текущих концентрациях).")
        phys = replace(s, model=PHYSICAL)
        try:
            js = float(ph.saturation_current_density(phys, 0.0))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            js = float("nan")
        if np.isfinite(js) and js > 0:
            ratio = p["J01_2d"] / js
            text = (f"J₀₁ = {p['J01_2d']:.3g} А/см² против диффузионного J_s = {js:.3g} А/см² структуры "
                    f"(4.3): отношение {ratio:.3g}.")
            if ratio > 10:
                text += " J₀₁ больше: диффузионные длины короче заданных или ток n = 1 идёт не через объём."
            elif ratio < 0.1:
                text += (" J₀₁ меньше: концентрации сторон или T в полях не соответствуют образцу "
                         "(J_s ∝ n_i²/N слабой стороны) либо времена жизни длиннее заданных.")
            notes.append(text)
        V_mid = 0.2
        i1 = p["J01_2d"] * np.expm1(V_mid / s.Vt)
        i2 = p["J02_2d"] * np.expm1(V_mid / (2 * s.Vt))
        notes.append(f"При V_d = {V_mid:g} В доля тока ОПЗ (n = 2) — {100 * i2 / (i1 + i2):.0f} %.")
    Rs = p.get("Rs")
    if Rs is not None and np.isfinite(Rs):
        try:
            geo = ph.series_resistance_estimate(s).total
        except (ValueError, ZeroDivisionError):
            geo = float("nan")
        if np.isfinite(geo) and geo > 0:
            if Rs < 0.5 * geo:
                notes.append(f"R_s = {Rs:.3g} Ом меньше объёмной оценки по геометрии {geo:.3g} Ом (6.12)–(6.15): "
                             "ток растекается шире, чем в оценке (проверьте N_As и d_s обогащённого слоя), "
                             "или ρ и толщины слоёв заданы неверно (методичка, п. 9.5).")
            elif Rs > 10 * geo:
                notes.append(f"R_s = {Rs:.3g} Ом больше объёмной оценки по геометрии {geo:.3g} Ом в "
                             f"{Rs / geo:.0f} раз: основное сопротивление — контакты [Кур74, с. 238] "
                             "(методичка, п. 9.5).")
    if result.model == PHYSICAL and "tau0_bg" in p and "tau_bg" in p:
        if p["tau0_bg"] > 10 * p["tau_bg"]:
            notes.append(f"τ₀^bg = {p['tau0_bg']:.3g} с больше τ^bg = {p['tau_bg']:.3g} с в "
                         f"{p['tau0_bg'] / p['tau_bg']:.0f} раз, хотя обе величины задают одни и те же "
                         "центры рекомбинации (5.9), (5.10): вероятно, ток ОПЗ занижает другой механизм "
                         "или τ^bg описывает не объём, а границу.")
    return notes

def reverse_exponent(V, I, v_from=-1.0):
    """Показатель m в |I| ∝ |V|^m на обратной ветви при V < v_from (наклон
    ln|I| от ln|V|); nan, если точек мало."""
    sel = (V < v_from) & (I < 0)
    if sel.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log(-V[sel]), np.log(-I[sel]), 1)[0])


def forward_resistance_trend(V, I, parts=3, n=None, T=300.0):
    """dV/dI по участкам верха прямой ветви (выше 40 % максимума V), Ом.

    n — если задан, из dV/dI вычитается дифференциальное сопротивление самого
    диода n·kT/(q·I) (из (6.3)): остаётся сопротивление последовательной цепи."""
    sel = (V > 0.4 * V.max()) & (I > 0)
    if sel.sum() < 3 * parts:
        return []
    chunks = [c for c in np.array_split(np.flatnonzero(sel), parts) if c.size >= 3]
    out = []
    for c in chunks:
        r = float(np.polyfit(I[c], V[c], 1)[0])
        if n is not None:
            r -= n * ph.thermal_voltage(T) / float(np.mean(I[c]))
        out.append(r)
    return out


def explain(result, s):
    """«Почему так»: что показывают данные и что выбрала подгонка."""
    V, I = result.V, result.I_exp
    notes = []
    m_data = reverse_exponent(V, I)
    if np.isfinite(m_data):
        if m_data > 1.2:
            notes.append(f"Обратная ветвь сверхлинейна: при V < −1 В |I| ∝ |V|^{m_data:.2f}. Шунт R_sh "
                         "даёт только прямую линию, поэтому нужна нелинейная утечка I_L·|V|^m.")
        elif m_data < 0.8:
            notes.append(f"Обратная ветвь сублинейна (|I| ∝ |V|^{m_data:.2f}): так растёт ток генерации "
                         "в ОПЗ (∝ W ∝ √V) — ориентир для τ₀.")
        else:
            notes.append(f"Обратная ветвь почти линейна (|I| ∝ |V|^{m_data:.2f}): её задаёт шунт R_sh.")
    # Сопротивление цепи = dV/dI минус сопротивление самого диода n·kT/(qI); n = 2 —
    # наибольшее для (6.3), поэтому остаток — нижняя оценка R_s и не путает спад
    # сопротивления диода с модуляцией.
    n_diode = result.params.get("n_emp", 2.0)
    trend = forward_resistance_trend(V, I, n=n_diode, T=s.T)
    if len(trend) >= 2 and trend[0] > 0:
        change = (trend[-1] - trend[0]) / trend[0]
        text = " → ".join(f"{r:.0f}" for r in trend)
        if change < -0.15:
            notes.append(f"Верх прямой ветви: сопротивление цепи (dV/dI минус n·kT/qI диода, n = "
                         f"{n_diode:.2g}) падает с ростом тока: {text} Ом. Так проявляется модуляция "
                         "проводимости высокоомной базы инжектированными носителями — R_s(I) по (6.9).")
        else:
            notes.append(f"Верх прямой ветви: сопротивление цепи (dV/dI минус n·kT/qI диода) ≈ {text} Ом — "
                         "последовательное сопротивление почти постоянно.")
    for cand in result.candidates:
        name = ", ".join(TERMS[t][0] for t in cand.terms) or "минимальная модель"
        mark = "выбрана" if cand.accepted else "отклонена"
        notes.append(f"Вариант «{name}»: ошибка {100 * cand.error:.2f} %, BIC = {cand.bic:.0f} — {mark}.")
    notes += _selection_notes(result)
    notes += _quality_notes(result)
    notes += _consistency_notes(result, s)
    p = result.params
    if TERM_LEAK in result.terms:
        m = p["m_leak"]
        if 1.7 <= m <= 2.4:
            notes.append(f"m = {m:.2f} ≈ 2: так растёт ток, ограниченный пространственным зарядом "
                         "(закон Мотта–Гёрни, J ∝ V²) [MG40] — гипотеза о канале утечки через "
                         "высокоомную область; проверка — по температуре и по мезам разного диаметра.")
        elif m >= 3.0:
            notes.append(f"m = {m:.2f}: «мягкая» обратная ВАХ I ∝ U^n, 3 < n < 7 — микромостики вдоль "
                         "сквозных дислокаций [Кур74, с. 167–168].")
    if result.model == PHYSICAL and "tau0_bg" in p:
        tau_dis = ph.dislocation_lifetime(ph.scr_sigma_R(s), s.N_dis)
        tau_bg = p.get("tau_bg", float("nan"))
        text = f"τ₀^bg = {p['tau0_bg']:.3g} с, τ^bg = {tau_bg:.3g} с"
        if np.isfinite(tau_dis):
            text += (f"; дислокации при N_dis = {s.N_dis:g} см⁻² дали бы τ_dis = {tau_dis:.3g} с (5.10)")
            if tau_dis > 100 * p["tau0_bg"]:
                text += " — ток ОПЗ определяют другие центры, не дислокации"
        notes.append(text + ".")
    if result.model == EMPIRICAL and "n_emp" in p:
        n = p["n_emp"]
        share = ph.gr_share_from_ideality(n) if 1.0 <= n <= 2.0 else float("nan")
        text = f"n = {n:.3g} по всей ВАХ с учётом R_s"
        if np.isfinite(share):
            text += f": по (6.6) доля тока ОПЗ ≈ {100 * share:.0f} %, остальное — диффузия"
        notes.append(text + ".")
    if result.locked:
        names = ", ".join(LABELS[k] for k in result.locked)
        notes.append(f"Зафиксированы пользователем (не подбирались): {names}.")
    notes.append(f"Итог: ошибка (6.10) {100 * result.error:.2f} % (прямая ветвь "
                 f"{100 * result.error_forward:.2f} %, обратная {100 * result.error_reverse:.2f} %).")
    return notes
