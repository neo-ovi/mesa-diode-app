# -*- coding: utf-8 -*-
"""§6. Оценка последовательного сопротивления по геометрии мезы и ρ(N) слоёв.

Формулы (6.12)–(6.15); вывод, допущения и пример — методичка, п. 9.5.
Это оценка объёмной части R_s без контактного сопротивления: она служит
начальным значением и проверкой для R_s, а R_s модели остаётся полем окна
(его уточняет подгонка)."""

import math
from dataclasses import dataclass, field

from mesa_diode.simulator.physics.carriers import resistivity
from mesa_diode.simulator.physics.constants import SCENARIO_A


@dataclass
class ResistanceEstimate:
    parts: dict                     # составляющая → Ом
    total: float                    # сумма, Ом
    rho: dict                       # слой → ρ при T, Ом·см
    spread_length: float = 0.0      # λ растекания по обогащённому слою, см
    notes: list = field(default_factory=list)


def layer_resistivities(s):
    """ρ(N, T) слоёв по (2.8), (2.9), Ом·см: n⁺, i, обогащённый слой, подложка."""
    i_kind = "p" if s.scenario == SCENARIO_A else "n"
    rho = {"n+": resistivity(s.ND_plus, s.T, s.material, kind="n"),
           "i": resistivity(s.N_i, s.T, s.material, kind=i_kind),
           "sub": resistivity(s.substrate[0], s.T, s.material)}
    if s.has_surface_layer:
        rho["surf"] = resistivity(s.N_As, s.T, s.material)
    return rho


def series_resistance_estimate(s):
    """Объёмная часть R_s, Ом (оценка):

    (6.12) окно кольцевого контакта: R_ок = (r_ок/r)⁴·R_□,n⁺/(8π), R_□ = ρ_n⁺/d_n —
           ток, равномерно проходящий через переход под окном радиуса r_ок,
           собирается по n⁺-слою к кольцу (доля тока под окном (r_ок/r)²);
    (6.13) слои поперёк: (ρ_n⁺·d_n + ρ_i·d_i + ρ_s·d_s)/A;
    (6.14) растекание по обогащённому слою: λ = √(ρ_sub·t·d_s/ρ_s), a_эфф = a + λ;
    (6.15) подложка толщиной t: R_sub = min(ρ_sub/(4a_эфф), ρ_sub·t/(π·a_эфф²)).
    Контактное сопротивление не входит."""
    rho = layer_resistivities(s)
    a = s.D / 2.0
    r_win = min(s.D_inner, s.D) / 2.0
    parts, notes = {}, []
    sheet_n = rho["n+"] / s.d_n
    parts["n⁺ под окном кольца (6.12)"] = (r_win / a) ** 4 * sheet_n / (8.0 * math.pi) if r_win else 0.0
    vertical = rho["n+"] * s.d_n + rho["i"] * s.d_i
    t = s.d_sub
    lam = 0.0
    if s.has_surface_layer:
        vertical += rho["surf"] * s.d_s
        t = max(s.d_sub - s.d_s, 1e-7)
        lam = math.sqrt(rho["sub"] * t * s.d_s / rho["surf"])
    parts["слои поперёк (6.13)"] = vertical / s.area

    def spreading(radius):
        return min(rho["sub"] / (4.0 * radius), rho["sub"] * t / (math.pi * radius ** 2))

    parts["подложка, растекание (6.15)"] = spreading(a + lam)
    if lam:
        notes.append(f"Обогащённый слой растекает ток на λ ≈ {lam * 1e4:.3g} мкм за край мезы (6.14): "
                     f"сопротивление подложки {spreading(a + lam):.3g} Ом вместо {spreading(a):.3g} Ом "
                     "без слоя.")
    notes.append("Контактное сопротивление в оценку не входит: R_s из подгонки должно быть не меньше.")
    return ResistanceEstimate(parts=parts, total=sum(parts.values()), rho=rho, spread_length=lam, notes=notes)
