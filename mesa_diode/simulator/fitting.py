# -*- coding: utf-8 -*-
"""Автоматическая подгонка модели к экспериментальной ВАХ (§6.7).

Подбираются параметры, которые экспериментатор не задаёт напрямую, так,
чтобы модель (6.1) с последовательным сопротивлением (6.9) совпала с
измеренной ВАХ. Механизмы тока подключаются по необходимости: сначала
подгоняется минимальная модель, затем к ней по очереди добавляются
нелинейная утечка и модуляция R_s; механизм остаётся, только если он
заметно улучшает описание по информационному критерию (6.11).

Метод — нелинейные наименьшие квадраты с границами (алгоритм trust region
reflective, scipy.optimize.least_squares) [BCL99]. Невязка — разность
arsinh(I/I_ref): для малых токов она близка к относительной ошибке на
линейной шкале около нуля, для больших — к разности логарифмов, поэтому
прямая и обратная ветви весят сопоставимо.

Модели:
  * эмпирическая (базовый режим): J₀, n формулы (6.3) + R_s, R_sh, I_L, m, I_mod;
  * физическая (расширенный режим и «Подгонка»): τ₀^bg, τ^bg (общий множитель
    τ_n^bg = τ_p^bg) + те же элементы эквивалентной схемы.

Результат — FitResult: значения с погрешностями, ошибка (6.10), список
проверенных вариантов модели и пояснения «почему так» (notes).
"""

from dataclasses import dataclass, field, replace

import numpy as np

from mesa_diode.simulator import physics as ph

EMPIRICAL, PHYSICAL = "empirical", "physical"

# Механизмы, которые подключаются по необходимости: ключ → (название, параметры).
TERM_LEAK = "leak"
TERM_MOD = "mod"
TERMS = {
    TERM_LEAK: ("нелинейная утечка I_L·|V|^m", ("I_L", "m_leak")),
    TERM_MOD: ("модуляция R_s током I_mod (6.9)", ("I_mod",)),
}
CORE = {
    EMPIRICAL: ("J0_emp", "n_emp", "Rs", "Rsh"),
    PHYSICAL: ("tau0_bg", "tau_bg", "Rs", "Rsh"),
}

# Параметр → (поле Structure, логарифмическая шкала, нижняя, верхняя граница).
PARAMS = {
    "J0_emp": ("J0_emp", True, 1e-15, 1e3),
    "n_emp": ("n_emp", False, 0.5, 8.0),
    "Rs": ("Rs", True, 1e-3, 1e6),
    "Rsh": ("Rsh", True, 1.0, 1e12),
    "I_L": ("I_L", True, 1e-14, 1.0),
    "m_leak": ("m_leak", False, 1.0, 8.0),
    "I_mod": ("I_mod", True, 1e-7, 1e2),
    "tau0_bg": ("tau0_bg", True, 1e-13, 1e-2),
    "tau_bg": (None, True, 1e-12, 1e-1),      # τ_n^bg = τ_p^bg
}
UNITS = {"J0_emp": "А/см²", "n_emp": "—", "Rs": "Ом", "Rsh": "Ом", "I_L": "А", "m_leak": "—",
         "I_mod": "А", "tau0_bg": "с", "tau_bg": "с"}
LABELS = {"J0_emp": "J₀", "n_emp": "n", "Rs": "R_s", "Rsh": "R_sh", "I_L": "I_L", "m_leak": "m",
          "I_mod": "I_mod", "tau0_bg": "τ₀^bg", "tau_bg": "τ_n^bg = τ_p^bg"}

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

    def structure_values(self):
        """Значения для полей окна (ключи presets): τ_bg → τ_n^bg и τ_p^bg."""
        values = dict(self.params)
        if "tau_bg" in values:
            tau = values.pop("tau_bg")
            values["tau_n_bg"] = values["tau_p_bg"] = tau
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
    return replace(s, **changes)


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
    if model == EMPIRICAL:
        i0 = np.interp(-0.1, V, I) if V[0] < -0.1 else -1e-3 * np.max(np.abs(I))
        start["J0_emp"] = max(abs(i0), 1e-12) / s.area
        start["n_emp"] = 1.5
    else:
        start["tau0_bg"] = s.tau0_bg
        start["tau_bg"] = np.sqrt(s.tau_n_bg * s.tau_p_bg)
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

    result = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
    return unpack(result.x), result.jac, result.fun, logs


def _stderr(keys, logs, jac, res):
    """Относительные погрешности из ковариации (JᵀJ)⁻¹·RSS/(N − k)."""
    n, k = res.size, len(keys)
    out = {key: float("nan") for key in keys}
    if n <= k:
        return out
    try:
        cov = np.linalg.pinv(jac.T @ jac) * float(np.sum(res ** 2)) / (n - k)
    except np.linalg.LinAlgError:
        return out
    for i, (key, lg) in enumerate(zip(keys, logs)):
        sigma = float(np.sqrt(max(cov[i, i], 0.0)))
        out[key] = sigma * np.log(10.0) if lg else sigma   # log10 → доля; линейный — абсолютная
    return out


def fit_iv(s, V, I, model=EMPIRICAL, progress=None):
    """Подгонка ВАХ с выбором механизмов (§6.7).

    s — Structure с текущими параметрами (геометрия, легирование и т. д.);
    model — EMPIRICAL (эмпирическая модель (6.3)) или PHYSICAL (§4–§6).
    progress(text) — необязательный вызов для строки состояния."""
    base = replace(s, model=ph.MODEL_EMPIRICAL if model == EMPIRICAL else ph.MODEL_PHYSICAL)
    V, I = prepare_data(V, I)
    if V.size < 8:
        raise ValueError("для подгонки нужно не меньше 8 точек ВАХ")
    I_ref = current_scale(I)
    floor = 10.0 * I_ref
    start = _start_values(model, base, V, I)

    # Выключенные механизмы: утечка I_L = 0, модуляция I_mod = ∞.
    off = {"I_L": 0.0, "I_mod": float("inf")}
    variants = [(), (TERM_LEAK,), (TERM_MOD,), (TERM_LEAK, TERM_MOD)]
    candidates = []
    best = None
    for terms in variants:
        if progress:
            progress("подгонка: " + (", ".join(TERMS[t][0] for t in terms) or "минимальная модель"))
        keys = list(CORE[model]) + [p for t in terms for p in TERMS[t][1]]
        fixed = {k: v for k, v in off.items() if k not in keys}
        s_run = apply(base, fixed)
        # старт — лучший предыдущий результат (кроме выключенных механизмов: 0 и ∞)
        seed = dict(start)
        if best is not None:
            seed.update({k: v for k, v in best.params.items() if np.isfinite(v) and v > 0})
        values, jac, res, logs = _fit_keys(model, s_run, V, I, keys, seed, I_ref)
        values_all = {**fixed, **values}
        Im = model_current(s_run, V, values)
        cand = Candidate(terms=terms, params=values_all, error=relative_error(Im, I, floor),
                         bic=bic(res, len(keys)))
        cand._fit = (keys, logs, jac, res, Im)
        candidates.append(cand)
        if best is None or cand.bic < best.bic - (BIC_GAIN if len(terms) > len(best.terms) else 0.0):
            best = cand
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
        bic=best.bic, V=V, I_exp=I, I_model=Im, candidates=candidates)
    result.notes = explain(result, base)
    return result


# ------------------------------------------------------------ пояснения --

def reverse_exponent(V, I, v_from=-1.0):
    """Показатель m в |I| ∝ |V|^m на обратной ветви при V < v_from (наклон
    ln|I| от ln|V|); nan, если точек мало."""
    sel = (V < v_from) & (I < 0)
    if sel.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log(-V[sel]), np.log(-I[sel]), 1)[0])


def forward_resistance_trend(V, I, parts=3):
    """dV/dI по участкам верха прямой ветви (выше 40 % максимума V), Ом."""
    sel = (V > 0.4 * V.max()) & (I > 0)
    if sel.sum() < 3 * parts:
        return []
    chunks = np.array_split(np.flatnonzero(sel), parts)
    return [float(np.polyfit(I[c], V[c], 1)[0]) for c in chunks if c.size >= 3]


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
    trend = forward_resistance_trend(V, I)
    if len(trend) >= 2:
        change = (trend[-1] - trend[0]) / trend[0]
        text = " → ".join(f"{r:.0f}" for r in trend)
        if change < -0.05:
            notes.append(f"Верх прямой ветви: dV/dI падает с ростом тока ({text} Ом) быстрее, чем даёт сам "
                         "диод. Так проявляется модуляция проводимости высокоомной базы "
                         "инжектированными носителями — R_s(I) по (6.9).")
        else:
            notes.append(f"Верх прямой ветви: dV/dI ≈ {text} Ом — последовательное сопротивление почти "
                         "постоянно.")
    for cand in result.candidates:
        name = ", ".join(TERMS[t][0] for t in cand.terms) or "минимальная модель"
        mark = "выбрана" if cand.accepted else "отклонена"
        notes.append(f"Вариант «{name}»: ошибка {100 * cand.error:.2f} %, BIC = {cand.bic:.0f} — {mark}.")
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
        text = f"τ₀^bg = {p['tau0_bg']:.3g} с, τ^bg = {p['tau_bg']:.3g} с"
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
    notes.append(f"Итог: ошибка (6.10) {100 * result.error:.2f} % (прямая ветвь "
                 f"{100 * result.error_forward:.2f} %, обратная {100 * result.error_reverse:.2f} %).")
    return notes
