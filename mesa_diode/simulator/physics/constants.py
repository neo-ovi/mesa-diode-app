# -*- coding: utf-8 -*-
"""Константы модели: сценарии, граничные условия, модели тока, метод V_bi.

Смысл — методичка, п. 1А.1, п. 1А.8, гл. 13."""

SCENARIO_A = "A"   # i-слой p: переход n⁺/i на глубине d_n
SCENARIO_B = "B"   # i-слой n (по умолчанию): переход i/подложка на дне мезы
SCENARIOS = (SCENARIO_A, SCENARIO_B)

SINK, REFLECT, LONG, LAYER = "sink", "reflect", "long", "layer"
BOUNDARY_LABELS = {SINK: "сток", REFLECT: "отражение", LONG: "длинная база", LAYER: "соседний слой"}

MODEL_PHYSICAL = "physical"     # физическая модель §2–§6
MODEL_EMPIRICAL = "empirical"   # эмпирическая модель (6.3): J₀ и n задаются (базовый режим)
MODEL_TWO_DIODE = "two_diode"   # двухдиодная модель (6.3а): J₀₁ (n = 1) и J₀₂ (n = 2)
# Формулы тока перехода каждой модели — в реестре simulator/models.py.

VBI_DEGENERATE = "degenerate"   # (3.1а), используется в расчёте всегда
VBI_BOLTZMANN = "boltzmann"     # (3.1), для сравнения и эталонов §9
