# -*- coding: utf-8 -*-
"""Наборы образцов (ТЗ §5.4): сохранение и загрузка в JSON.

Набор хранит все параметры основного окна (§5.3) в единицах интерфейса,
пути к файлам ВАХ/ВФХ и метаданные образца. Параметры реальных образцов
в этом репозитории не хранятся: наборы лежат в каталоге данных
(MESA_DATA_DIR/samples/<образец>/preset.json), опорный помечен
"reference": true. Без каталога данных программа стартует с нейтральных
значений DEFAULT_PARAMS.
"""

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from mesa_diode import config
from mesa_diode.simulator import models
from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.materials import GE

FORMAT = "mesa-diode-preset/1"
PRESET_FILENAME = "preset.json"

I_TYPE_N, I_TYPE_P, I_TYPE_BOTH = "n", "p", "both"
I_TYPE_TO_SCENARIO = {I_TYPE_N: ph.SCENARIO_B, I_TYPE_P: ph.SCENARIO_A}

UM = 1e-4


@dataclass(frozen=True)
class ParamSpec:
    key: str            # ключ в наборе и в интерфейсе
    label: str
    unit: str
    field: str | None   # поле physics.Structure; None — не входит в модель
    scale: float = 1.0  # множитель «интерфейс → расчёт»


# Порядок совпадает с таблицей §5.3.
NUMERIC_PARAMS = [
    ParamSpec("D_um", "диаметр мезы D", "мкм", "D", UM),
    ParamSpec("D_inner_um", "внутренний диаметр кольца (только схема)", "мкм", None),
    ParamSpec("d_epi_um", "толщина эпитаксии d_epi", "мкм", "d_epi", UM),
    ParamSpec("h_um", "высота мезы h", "мкм", "h", UM),
    ParamSpec("d_n_um", "толщина n⁺-слоя d_n", "мкм", "d_n", UM),
    ParamSpec("ND_plus", "концентрация n⁺ N_D⁺", "см⁻³", "ND_plus"),
    ParamSpec("N_i", "концентрация i-слоя N_i", "см⁻³", "N_i"),
    ParamSpec("rho_sub", "удельное сопротивление подложки ρ_sub", "Ом·см", "rho_sub"),
    ParamSpec("d_sub_um", "толщина подложки d_sub", "мкм", "d_sub", UM),
    ParamSpec("T", "температура T", "К", "T"),
    ParamSpec("T_rho", "температура измерения ρ_sub", "К", "T_rho"),
    ParamSpec("mu_n", "μ_n электронов (неосновные)", "см²/(В·с)", "mu_n"),
    ParamSpec("mu_p_i", "μ_p дырок в i-слое", "см²/(В·с)", "mu_p_i"),
    ParamSpec("mu_p_nplus", "μ_p дырок в n⁺", "см²/(В·с)", "mu_p_nplus"),
    ParamSpec("tau_n_bg", "τ_n^bg", "с", "tau_n_bg"),
    ParamSpec("tau_p_bg", "τ_p^bg", "с", "tau_p_bg"),
    ParamSpec("tau0_bg", "τ₀^bg (ОПЗ)", "с", "tau0_bg"),
    ParamSpec("N_dis", "плотность дислокаций N_dis", "см⁻²", "N_dis"),
    ParamSpec("sigma_R_epi", "σ_R: i-слой и n⁺", "см²/с", "sigma_R_epi"),
    ParamSpec("sigma_R_sub", "σ_R: подложка", "см²/с", "sigma_R_sub"),
    ParamSpec("n2", "показатель тока ОПЗ n₂", "—", "n2"),
    ParamSpec("Rs", "последовательное сопротивление R_s", "Ом", "Rs"),
    ParamSpec("I_mod", "ток модуляции R_s (inf — R_s постоянно)", "А", "I_mod"),
    ParamSpec("Rsh", "шунт R_sh", "Ом", "Rsh"),
    ParamSpec("I_L", "нелинейная утечка I_L", "А", "I_L"),
    ParamSpec("m_leak", "показатель утечки m", "—", "m_leak"),
    # Эмпирическая модель (6.3) — базовый режим.
    ParamSpec("n_emp", "коэффициент идеальности n", "—", "n_emp"),
    ParamSpec("J0_emp", "плотность тока насыщения J₀", "А/см²", "J0_emp"),
    ParamSpec("J01_2d", "ток насыщения диффузии J₀₁ (n = 1)", "А/см²", "J01_2d"),
    ParamSpec("J02_2d", "ток насыщения ОПЗ J₀₂ (n = 2)", "А/см²", "J02_2d"),
    # Измерения и технология: хранятся в наборе, в расчёт не входят (кроме
    # оценки N_D⁺ = Q/d_n автофитом). Пустое поле — «не измерено» (None).
    ParamSpec("afm_rms_nm", "шероховатость RMS (АСМ)", "нм", None),
    ParamSpec("afm_defects", "плотность дефектов (АСМ)", "см⁻²", None),
    ParamSpec("xrd_fwhm", "полуширина кривой качания (XRD)", "угл. с", None),
    ParamSpec("xrd_instrument", "ширина прибора φ (XRD)", "угл. с", None),
    ParamSpec("xrd_intrinsic", "естественная ширина отражения (XRD)", "угл. с", None),
    ParamSpec("hall_mu", "холловская подвижность i-слоя", "см²/(В·с)", None),
    ParamSpec("implant_dose", "доза имплантации n⁺ Q", "см⁻²", None),
    ParamSpec("implant_energy", "энергия имплантации", "кэВ", None),
    ParamSpec("anneal_T", "температура отжига", "°C", None),
    ParamSpec("growth_T", "температура подложки при росте", "°C", None),
]
# Ключи, которые только хранятся в наборе (в physics.Structure не входят).
# D_inner_um в расчёт тоже не входит, но нужен для схемы мезы — он обычный параметр.
STORED_KEYS = frozenset(spec.key for spec in NUMERIC_PARAMS
                        if spec.field is None and spec.key != "D_inner_um")
CHOICE_FIELDS = ("bc_A_n", "bc_A_p", "bc_B_n", "bc_B_p")
FLAG_FIELDS = ("sns_refinement", "edge_area")

# Режимы окна. Значения при переключении сохраняются, меняется только то,
# какие поля доступны для правки:
#   базовый — эмпирическая модель (6.3): минимум параметров, строится сразу;
#   расширенный — физическая модель по всему, что экспериментатор измеряет
#     (ВАХ, ВФХ, Холл, ρ, геометрия, технология, EPD, АСМ, XRD);
#   «Подгонка» — плюс параметры формул, которые не измеряются (подвижности
#     неосновных носителей, σ_R, времена жизни, n₂, нелинейная утечка).
# n и J₀ формулы (6.3) есть только в базовом режиме: физическая модель их
# не использует.
MODE_BASIC, MODE_EXTENDED, MODE_FIT = "basic", "extended", "fit"
MODES = (MODE_BASIC, MODE_EXTENDED, MODE_FIT)
MODE_LABELS = {MODE_BASIC: "Базовая модель", MODE_EXTENDED: "Расширенная модель",
               MODE_FIT: "Подгонка"}
# Модель тока в режиме: базовый — выбранная из models.basic_models()
# (поле basic_model), расширенный и «Подгонка» — физическая.
MODE_MODEL = {MODE_BASIC: ph.MODEL_EMPIRICAL, MODE_EXTENDED: ph.MODEL_PHYSICAL,
              MODE_FIT: ph.MODEL_PHYSICAL}
BASIC_MODELS = tuple(m.key for m in models.basic_models())
EMPIRICAL_KEYS = models.get(ph.MODEL_EMPIRICAL).fields
MODEL_KEYS = models.model_fields()           # поля, нужные только своей модели тока
# Эквивалентная схема (R_s, его модуляция, шунт, нелинейная утечка) нужна
# и эмпирической, и физической модели: без неё не описать изгиб ветвей ВАХ.
CIRCUIT_KEYS = frozenset({"Rs", "I_mod", "Rsh", "I_L", "m_leak"})
# ρ подложки — в базовом режиме: в сценарии B подложка — p-сторона перехода,
# и по ρ_sub (2.8) считаются V_bi, ширина ОПЗ и ВФХ.
BASIC_KEYS = (frozenset({"D_um", "D_inner_um", "h_um", "ND_plus", "N_i", "rho_sub", "T_rho", "T"})
              | CIRCUIT_KEYS | MODEL_KEYS)
MEASURED_KEYS = (BASIC_KEYS - MODEL_KEYS) | {"d_epi_um", "d_n_um", "d_sub_um",
                                                 "N_dis"} | STORED_KEYS
FIT_ONLY_KEYS = frozenset({"mu_n", "mu_p_i", "mu_p_nplus", "sigma_R_epi", "sigma_R_sub",
                           "tau_n_bg", "tau_p_bg", "tau0_bg", "n2"})
MODE_KEYS = {MODE_BASIC: BASIC_KEYS, MODE_EXTENDED: MEASURED_KEYS,
             MODE_FIT: MEASURED_KEYS | FIT_ONLY_KEYS}


def editable_keys(mode, basic_model=None):
    """Числовые параметры, доступные для правки в режиме mode. basic_model —
    модель тока базового режима: поля других моделей тогда не нужны."""
    keys = MODE_KEYS.get(mode, MODE_KEYS[MODE_FIT])
    if mode == MODE_BASIC and basic_model is not None:
        keys = keys - (MODEL_KEYS - models.get(basic_model).fields)
    return keys


def current_model(params):
    """Ключ модели тока по параметрам набора (режим и basic_model)."""
    mode = params.get("mode")
    if mode == MODE_BASIC:
        return params.get("basic_model") or ph.MODEL_EMPIRICAL
    return MODE_MODEL.get(mode, ph.MODEL_PHYSICAL)

# Нейтральные значения по умолчанию (не параметры какого-либо образца).
DEFAULT_PARAMS = {
    "substrate": "Ge",
    "i_type": I_TYPE_N,
    "D_um": 500.0,
    "D_inner_um": 300.0,
    "d_epi_um": 3.2,
    "h_um": 3.2,
    "d_n_um": 0.4,
    "ND_plus": 5e17,
    "N_i": 5e15,
    "rho_sub": 5.0,
    "d_sub_um": 300.0,
    "T": 300.0,
    "T_rho": 300.0,
    "mu_n": GE.mu_n_max,
    "mu_p_i": GE.mu_p_max,
    "mu_p_nplus": GE.mu_p_max,
    "tau_n_bg": 1e-6,
    "tau_p_bg": 1e-6,
    "tau0_bg": 1e-7,
    "N_dis": 0.0,
    "sigma_R_epi": 3.5e-3,
    "sigma_R_sub": 5.5e-4,
    "n2": 2.0,
    "Rs": 10.0,
    "I_mod": float("inf"),
    "Rsh": 1e6,
    "I_L": 0.0,
    "m_leak": 3.0,
    "n_emp": 1.5,
    "J0_emp": 1e-6,
    "J01_2d": 1e-7,
    "J02_2d": 1e-5,
    "basic_model": ph.MODEL_EMPIRICAL,
    "afm_rms_nm": None,
    "afm_defects": None,
    "xrd_fwhm": None,
    "xrd_instrument": None,
    "xrd_intrinsic": None,
    "hall_mu": None,
    "implant_dose": None,
    "implant_energy": None,
    "anneal_T": None,
    "growth_T": None,
    "mode": MODE_BASIC,
    "bc_A_n": ph.SINK,
    "bc_A_p": ph.LAYER,
    "bc_B_n": ph.LAYER,
    "bc_B_p": ph.SINK,
    "sns_refinement": False,
    "edge_area": False,
    "ideality_V1": None,
    "ideality_V2": None,
}


AUTO_DEFAULT = "значение по умолчанию"
AUTO_FROM_IV = "из ВАХ (n_эксп, подгонка (6.3))"
AUTO_FROM_DOSE = "оценка Q/d_n по дозе имплантации (полная активация)"


def autofill(values, keys, ideality=None):
    """«Заполнить пустые»: значения для пустых полей (None) из keys.

    n и J₀ — из n_эксп и I₀/A по загруженной ВАХ (ideality —
    physics.IdealityResult или None); N_D⁺ — оценка Q/d_n, если доза и d_n
    заданы пользователем; остальное — DEFAULT_PARAMS. Поля, которые только
    хранятся в наборе (STORED_KEYS), не заполняются: пустое — «не измерено».
    Возвращает {ключ: (значение, источник)}."""
    empty = [key for key in keys if values.get(key) is None and key not in STORED_KEYS]
    filled = {}
    D_um = values.get("D_um") if values.get("D_um") is not None else DEFAULT_PARAMS["D_um"]
    area = math.pi * (D_um * UM) ** 2 / 4.0     # (1.1), как physics.Structure.area
    for key in empty:
        if key == "n_emp" and ideality is not None:
            filled[key] = (float(f"{ideality.n:.3g}"), AUTO_FROM_IV)
        elif key == "J0_emp" and ideality is not None:
            filled[key] = (float(f"{ideality.I0 / area:.3g}"), AUTO_FROM_IV)
        elif key == "ND_plus" and values.get("implant_dose") and values.get("d_n_um"):
            filled[key] = (float(f"{values['implant_dose'] / (values['d_n_um'] * UM):.3g}"), AUTO_FROM_DOSE)
        else:
            filled[key] = (DEFAULT_PARAMS[key], AUTO_DEFAULT)
    return filled


@dataclass
class Preset:
    name: str
    params: dict
    metadata: dict = field(default_factory=dict)
    files: dict = field(default_factory=lambda: {"iv": [], "cv": []})
    notes: dict = field(default_factory=dict)   # пояснения к разделам окна формул
    reference: bool = False
    path: Path | None = None

    def to_json(self):
        return {"format": FORMAT, "name": self.name, "reference": self.reference,
                "params": self.params, "metadata": self.metadata,
                "files": self.files, "notes": self.notes}

    def resolved_files(self, kind):
        """Пути к файлам измерений; относительные — от каталога набора."""
        base = self.path.parent if self.path else Path.cwd()
        return [p if Path(p).is_absolute() else base / p for p in self.files.get(kind, [])]


def default_preset():
    return Preset(name="по умолчанию", params=dict(DEFAULT_PARAMS))


def to_structure(params, scenario=None):
    """physics.Structure по параметрам набора; scenario — "A"/"B"
    (по умолчанию — из переключателя «тип i-слоя», для «оба» — B)."""
    merged = {**DEFAULT_PARAMS, **params}
    # Пустое поле (None), недоступное в текущем режиме, берётся по умолчанию.
    kwargs = {spec.field: float(DEFAULT_PARAMS[spec.key] if merged[spec.key] is None
                                else merged[spec.key]) * spec.scale
              for spec in NUMERIC_PARAMS if spec.field}
    kwargs.update({name: merged[name] for name in CHOICE_FIELDS})
    kwargs.update({name: bool(merged[name]) for name in FLAG_FIELDS})
    if scenario is None:
        scenario = I_TYPE_TO_SCENARIO.get(merged["i_type"], ph.SCENARIO_B)
    model = current_model(merged)
    return ph.Structure(scenario=scenario, model=model, **kwargs)


def save_preset(preset, path):
    path = Path(path)
    path.write_text(json.dumps(preset.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return replace(preset, path=path)


def load_preset(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path.name}: не набор образца (ожидается format = {FORMAT!r})")
    params = dict(data.get("params", {}))
    # Наборы, сохранённые до появления режимов, заданы для физической модели
    # целиком — открываются в «Подгонке».
    params.setdefault("mode", MODE_FIT)
    return Preset(name=data.get("name", path.stem), params=params,
                  metadata=data.get("metadata", {}),
                  files=data.get("files", {"iv": [], "cv": []}),
                  notes=data.get("notes", {}), reference=bool(data.get("reference")),
                  path=path)


def data_presets():
    """Наборы из каталога данных: MESA_DATA_DIR/samples/*/preset.json."""
    try:
        root = config.data_dir() / "samples"
    except RuntimeError:
        return []
    return sorted(root.glob(f"*/{PRESET_FILENAME}"))


def reference_preset():
    """Опорный набор из каталога данных или None."""
    for path in data_presets():
        try:
            preset = load_preset(path)
        except (ValueError, json.JSONDecodeError):
            continue
        if preset.reference:
            return preset
    return None


def startup_preset():
    """Набор при запуске: опорный из каталога данных, иначе нейтральный."""
    return reference_preset() or default_preset()
