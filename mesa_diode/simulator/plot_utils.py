# -*- coding: utf-8 -*-
"""Помощники отрисовки графиков: приставки единиц, диапазон напряжений,
границы оси по экспериментальным данным. Физики здесь нет."""

import numpy as np


def auto_scale(array, base_unit):
    """Подбор удобной приставки (базовая -> м -> мк -> н -> п) для отображения.
    base_unit — строка базовой единицы ('А' для тока, 'Ф' для ёмкости)."""
    finite = array[np.isfinite(array)]
    maxval = np.max(np.abs(finite)) if finite.size else 0.0
    for factor, prefix in ((1, ""), (1e3, "м"), (1e6, "мк"),
                            (1e9, "н"), (1e12, "п")):
        if maxval * factor >= 1.0 or prefix == "п":
            return factor, prefix + base_unit
    return 1e12, "п" + base_unit


def robust_value_limits(primary_arrays, fallback_array, margin_fraction=0.2):
    """Границы оси Y по «содержательным» данным, а не по хвосту модели.

    Экспоненциальные модели (Шокли, двухдиодная) при широком диапазоне
    напряжений могут давать физически нереалистичные значения на краях
    (реальный диод либо ограничен Rs, либо электрически пробивается задолго
    до таких V) — если строить ось Y по ним, содержательная часть графика
    (там, где есть эксперимент) сжимается в незаметную линию у нуля.

    Если ``primary_arrays`` непусты (обычно — экспериментальные значения),
    границы считаются по ним с запасом margin_fraction, а модельная кривая
    может выходить за пределы видимой области — это ожидаемо. Если
    ``primary_arrays`` пуст (данных не загружено), используются границы
    ``fallback_array`` (обычно — сама модельная кривая) целиком.
    """
    values = [np.asarray(a) for a in primary_arrays if np.asarray(a).size]
    if values:
        combined = np.concatenate(values)
        vmin, vmax = float(np.min(combined)), float(np.max(combined))
    else:
        fb = np.asarray(fallback_array)
        finite = fb[np.isfinite(fb)]
        vmin, vmax = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)

    span = vmax - vmin
    margin = span * margin_fraction if span > 0 else max(abs(vmin), abs(vmax), 1.0) * margin_fraction
    return vmin - margin, vmax + margin


def adaptive_voltage_range(voltage_arrays, default_min=-1.0, default_max=0.6,
                            margin_fraction=0.05):
    """Диапазон напряжений для расчёта модели: покрывает значения по
    умолчанию и все переданные экспериментальные массивы (с запасом по
    краям), чтобы модельная кривая не обрывалась раньше точек эксперимента.
    """
    vmin, vmax = default_min, default_max
    for arr in voltage_arrays:
        arr = np.asarray(arr)
        if arr.size:
            vmin = min(vmin, float(np.min(arr)))
            vmax = max(vmax, float(np.max(arr)))

    span = vmax - vmin
    margin = span * margin_fraction if span > 0 else 0.05
    return vmin - margin, vmax + margin


def adaptive_point_count(v_min, v_max, base_count=121, resolution_V=0.01, max_count=400):
    """Число точек расчёта ВАХ: гуще на широком диапазоне напряжений, но не
    в ущерб отклику интерфейса (physics.solve_iv — метод Ньютона, до 100 итераций на
    точку)."""
    span = max(v_max - v_min, 0.0)
    return int(np.clip(np.ceil(span / resolution_V), base_count, max_count))
