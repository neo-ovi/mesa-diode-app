"""Модель ВАХ мезадиодной структуры.

Двухдиодная модель: диффузионный ток (n = 1) и рекомбинационный ток
в области пространственного заряда (n = 2), опционально с шунтом.

Shockley W. // Bell Syst. Tech. J. 28, 435 (1949)
Sah C.T., Noyce R.N., Shockley W. // Proc. IRE 45, 1228 (1957)
"""

import numpy as np

ELEMENTARY_CHARGE = 1.602176634e-19  # Кл
BOLTZMANN = 1.380649e-23  # Дж/К

DIFFUSION_IDEALITY = 1.0
RECOMBINATION_IDEALITY = 2.0


def thermal_voltage(temperature_K: float = 300.0) -> float:
    """Тепловой потенциал kT/q, В."""
    return BOLTZMANN * temperature_K / ELEMENTARY_CHARGE


def shockley_current(voltage_V, saturation_current_A, ideality, temperature_K=300.0):
    """Один член уравнения Шокли: I = I0 * [exp(qU / n·kT) − 1]."""
    voltage_V = np.asarray(voltage_V, dtype=float)
    argument = voltage_V / (ideality * thermal_voltage(temperature_K))
    # expm1 сохраняет точность при малых аргументах, где exp(x) − 1 теряет разряды
    return saturation_current_A * np.expm1(argument)


def two_diode_current(
    voltage_V,
    diffusion_saturation_A,
    recombination_saturation_A,
    shunt_resistance_ohm=None,
    temperature_K=300.0,
):
    """Суммарный ток двухдиодной модели.

    Последовательное сопротивление пока не учитывается: при его включении
    уравнение становится неявным относительно тока.
    """
    current = shockley_current(
        voltage_V, diffusion_saturation_A, DIFFUSION_IDEALITY, temperature_K
    ) + shockley_current(
        voltage_V, recombination_saturation_A, RECOMBINATION_IDEALITY, temperature_K
    )

    if shunt_resistance_ohm is not None:
        current = current + np.asarray(voltage_V, dtype=float) / shunt_resistance_ohm

    return current


def ideality_factor(voltage_V, current_A, voltage_window_V=(0.10, 0.30), temperature_K=300.0):
    """Эффективный коэффициент идеальности по наклону ln(I) от U.

    Наклон берётся на прямой ветви в окне voltage_window_V. Нижняя граница
    окна должна заметно превышать n·kT/q (иначе член «−1» в уравнении Шокли
    ещё значим и наклон занижает n), верхняя — оставаться ниже участка, где
    ход искажает последовательное сопротивление.
    """
    voltage_V = np.asarray(voltage_V, dtype=float)
    current_A = np.asarray(current_A, dtype=float)

    low, high = voltage_window_V
    selected = (voltage_V >= low) & (voltage_V <= high) & (current_A > 0)
    if selected.sum() < 2:
        raise ValueError(
            f"В окне {voltage_window_V} В меньше двух точек с положительным током"
        )

    slope, _ = np.polyfit(voltage_V[selected], np.log(current_A[selected]), 1)
    return 1.0 / (slope * thermal_voltage(temperature_K))


def mesa_area_cm2(meta: dict) -> float:
    """Площадь мезы по её диаметру из паспорта образца."""
    diameter_um = meta["geometry"]["mesa_diameter_um"]
    return np.pi * (diameter_um * 1e-4 / 2.0) ** 2


def ring_contact_area_cm2(meta: dict) -> float:
    """Площадь металлизации кольцевого верхнего контакта."""
    contact = meta["geometry"]["top_contact"]
    outer_cm = contact["outer_diameter_um"] * 1e-4
    inner_cm = contact["inner_diameter_um"] * 1e-4
    return np.pi * (outer_cm**2 - inner_cm**2) / 4.0


def optical_window_area_cm2(meta: dict) -> float:
    """Площадь открытого окна внутри кольцевого контакта — под ввод излучения."""
    inner_cm = meta["geometry"]["top_contact"]["inner_diameter_um"] * 1e-4
    return np.pi * (inner_cm / 2.0) ** 2
