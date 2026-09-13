"""Подгонка двухдиодной модели под измеренную ВАХ.

Токи насыщения различаются на несколько порядков, поэтому подгонка ведётся
по десятичным логарифмам параметров, а невязка считается в логарифме тока —
иначе точки вблизи нуля вклада почти не дают.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from mesa_diode.diode import ideality_factor, two_diode_current


@dataclass
class TwoDiodeFit:
    """Результат подгонки."""

    diffusion_saturation_A: float
    recombination_saturation_A: float
    shunt_resistance_ohm: float | None
    temperature_K: float
    rms_log_residual: float
    points_used: int

    def current(self, voltage_V):
        """Модельный ток при заданном напряжении."""
        return two_diode_current(
            voltage_V,
            self.diffusion_saturation_A,
            self.recombination_saturation_A,
            self.shunt_resistance_ohm,
            self.temperature_K,
        )

    def effective_ideality(self, voltage_window_V=(0.05, 0.30)):
        """Эффективный n модельной кривой — для сравнения с измеренным."""
        voltage_V = np.linspace(*voltage_window_V, 64)
        return ideality_factor(
            voltage_V, self.current(voltage_V), voltage_window_V, self.temperature_K
        )


def fit_two_diode(
    voltage_V,
    current_A,
    temperature_K=300.0,
    with_shunt=False,
    min_voltage_V=0.02,
    initial_guess=(1e-12, 1e-8, 1e6),
) -> TwoDiodeFit:
    """Подогнать токи насыщения (и шунт) под прямую ветвь ВАХ."""
    voltage_V = np.asarray(voltage_V, dtype=float)
    current_A = np.asarray(current_A, dtype=float)

    selected = (voltage_V >= min_voltage_V) & (current_A > 0)
    if selected.sum() < 3:
        raise ValueError(
            f"Для подгонки нужно минимум 3 точки прямой ветви при U >= {min_voltage_V} В, "
            f"найдено {int(selected.sum())}"
        )

    fit_voltage = voltage_V[selected]
    log_measured = np.log(current_A[selected])

    diffusion_guess, recombination_guess, shunt_guess = initial_guess
    start = [np.log10(diffusion_guess), np.log10(recombination_guess)]
    if with_shunt:
        start.append(np.log10(shunt_guess))

    def residuals(log_params):
        shunt = 10 ** log_params[2] if with_shunt else None
        model = two_diode_current(
            fit_voltage, 10 ** log_params[0], 10 ** log_params[1], shunt, temperature_K
        )
        # Модель может уйти в неположительные значения на промежуточной итерации
        model = np.clip(model, 1e-30, None)
        return np.log(model) - log_measured

    solution = least_squares(residuals, start, method="lm")

    return TwoDiodeFit(
        diffusion_saturation_A=10 ** solution.x[0],
        recombination_saturation_A=10 ** solution.x[1],
        shunt_resistance_ohm=10 ** solution.x[2] if with_shunt else None,
        temperature_K=temperature_K,
        rms_log_residual=float(np.sqrt(np.mean(solution.fun**2))),
        points_used=int(selected.sum()),
    )
