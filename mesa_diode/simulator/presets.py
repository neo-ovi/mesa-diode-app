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
from dataclasses import dataclass, field, replace
from pathlib import Path

from mesa_diode import config
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
    ParamSpec("Rsh", "шунт R_sh", "Ом", "Rsh"),
    ParamSpec("I_L", "нелинейная утечка I_L", "А", "I_L"),
    ParamSpec("m_leak", "показатель утечки m", "—", "m_leak"),
]
CHOICE_FIELDS = ("bc_A_n", "bc_A_p", "bc_B_n", "bc_B_p")
FLAG_FIELDS = ("sns_refinement", "edge_area")

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
    "Rsh": 1e6,
    "I_L": 0.0,
    "m_leak": 3.0,
    "bc_A_n": ph.SINK,
    "bc_A_p": ph.SINK,
    "bc_B_n": ph.REFLECT,
    "bc_B_p": ph.SINK,
    "sns_refinement": False,
    "edge_area": False,
    "ideality_V1": None,
    "ideality_V2": None,
}


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
    kwargs = {spec.field: float(merged[spec.key]) * spec.scale
              for spec in NUMERIC_PARAMS if spec.field}
    kwargs.update({name: merged[name] for name in CHOICE_FIELDS})
    kwargs.update({name: bool(merged[name]) for name in FLAG_FIELDS})
    if scenario is None:
        scenario = I_TYPE_TO_SCENARIO.get(merged["i_type"], ph.SCENARIO_B)
    return ph.Structure(scenario=scenario, **kwargs)


def save_preset(preset, path):
    path = Path(path)
    path.write_text(json.dumps(preset.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return replace(preset, path=path)


def load_preset(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path.name}: не набор образца (ожидается format = {FORMAT!r})")
    return Preset(name=data.get("name", path.stem), params=data.get("params", {}),
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
