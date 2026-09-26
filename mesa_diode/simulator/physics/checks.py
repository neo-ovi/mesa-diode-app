# -*- coding: utf-8 -*-
"""Проверки Справочника: низкая инжекция (§6.4), изоляция (1.2),
рекомендации граничных условий (6.2); методичка, п. 2.4, п. 6.7, гл. 11."""

import numpy as np

from mesa_diode.simulator.materials import Q
from mesa_diode.simulator.physics.carriers import equilibrium_carriers
from mesa_diode.simulator.physics.constants import SCENARIO_A
from mesa_diode.simulator.physics.electrostatics import depletion_edges


def low_injection_voltage(s):
    """§6.4. Граница низкой инжекции V_LI = (kT/q)·ln(0.1·M_B/m_B0), В: M_B, m_B0 — равновесные основные
    и неосновные носители более слабой стороны (2.4). Условие —
    [Зи, с. 94, перед ур. (38)]; выше V_LI модель неприменима [Зи, с. 97, п. 4]."""
    weak = s.n_side if s.n_side.N < s.p_side.N else s.p_side
    M, m = equilibrium_carriers(weak.N, s.ni)
    return s.Vt * np.log(0.1 * M / m)


ISOLATED, EDGE, NOT_ISOLATED = "ok", "warn", "error"


def isolation_status(s, V):
    """(1.2) Изоляция перехода травлением мезы при смещении V.
    Возвращает (уровень, текст): уровень ok / warn / error."""
    if s.scenario == SCENARIO_A:
        if s.h > s.d_n:
            return ISOLATED, "переход изолирован (h > d_n)"
        return NOT_ISOLATED, "не изолирован: h ≤ d_n"
    xn, xp = depletion_edges(s, V)
    xn, xp = float(xn), float(xp)
    if s.h < s.d_epi - xn:
        return NOT_ISOLATED, "не изолирован: n-слой соединяет мезу с полем"
    if s.h < s.d_epi + xp:
        return EDGE, "ОПЗ выходит на поверхность поля у подножия мезы"
    return ISOLATED, "ОПЗ внутри мезы"


RECOMMEND_REFLECT = "отражение"
RECOMMEND_SINK = "сток"
RECOMMEND_INTERMEDIATE = "промежуточный"


def _recommend(N_layer, N_neighbour):
    """Правило ТЗ (6.2): соседний слой того же типа легирован в ≥ 10 раз
    сильнее → отражение; в ≥ 10 раз слабее или металл → сток; иначе промежуточный."""
    if N_neighbour is None or N_neighbour <= N_layer / 10.0:
        return RECOMMEND_SINK
    if N_neighbour >= 10.0 * N_layer:
        return RECOMMEND_REFLECT
    return RECOMMEND_INTERMEDIATE


def boundary_recommendations(s):
    """(6.2) Рекомендации для дальних границ n- и p-стороны текущего сценария.
    Возвращает {"n": (граница, рекомендация), "p": (...)}; None — металл."""
    below = s.N_As if s.has_surface_layer else s.substrate[0]
    if s.scenario == SCENARIO_A:
        name = "граница i/обогащённый слой" if s.has_surface_layer else "граница i/подложка"
        return {"n": ("верхний контакт", _recommend(s.ND_plus, None)),
                "p": (name, _recommend(s.N_i, below))}
    if s.has_surface_layer:
        return {"n": ("граница n/n⁺", _recommend(s.N_i, s.ND_plus)),
                "p": ("обогащённый слой/подложка — всегда «соседний слой»",
                      _recommend(s.N_As, s.substrate[0]))}
    return {"n": ("граница n/n⁺", _recommend(s.N_i, s.ND_plus)),
            "p": ("тыльный контакт", _recommend(s.substrate[0], None))}


def drude_mean_free_path(vth, mu, m_rel):
    """λ = v_th·μ·m_cc/q, см — оценка (соотношение Друде); источника со
    страницей в проекте нет, только для сравнения с L."""
    from mesa_diode.simulator.materials import M0
    return (vth * 1e-2) * (mu * 1e-4) * (m_rel * M0) / Q * 1e2
