# -*- coding: utf-8 -*-
"""Физическое ядро симулятора мезадиода (Ge, гомопереход).

Пакет разбит по разделам ТЗ; все имена доступны как physics.<имя>
(импорт в программе — ``from mesa_diode.simulator import physics as ph``).

| модуль           | раздел | что внутри                                             |
|------------------|--------|--------------------------------------------------------|
| constants        | —      | сценарии, граничные условия, модели тока, метод V_bi   |
| carriers         | §2     | E_g, n_i, статистика, подвижность μ(N,T), ρ ↔ N         |
| structure        | §5     | Side, Structure: слои, стороны перехода, сценарии      |
| electrostatics   | §1, §3 | площадь, V_bi, ОПЗ, ёмкость, краевая площадь           |
| lifetimes        | §5А    | времена жизни с дислокациями                           |
| diffusion        | §4     | диффузионный ток, граничные условия базы, (4.3а)       |
| scr              | §5     | ток генерации–рекомбинации в ОПЗ                       |
| circuit          | §6     | эквивалентная схема, R_s(I), решатель (6.1)            |
| resistance       | §6     | оценка R_s по геометрии и ρ(N) слоёв (6.12)–(6.15)     |
| checks           | —      | низкая инжекция, изоляция, рекомендации границ         |
| ideality         | §6.3   | коэффициент идеальности из эксперимента                |
| scenarios        | §6.6   | сравнение сценариев A и B                              |
| estimates        | —      | τ₀ по обратной ветви                                   |

Номера формул (раздел.номер) совпадают с окном «Формулы и параметры».
Единицы: см, см⁻³, с, В, А, Ф, К; E_g — эВ. V > 0 — прямое смещение,
I > 0 — прямой ток [Ш49, с. 460]. Устройство и порядок расчёта —
методичка, гл. 13.
"""

from mesa_diode.simulator.materials import EPS0, GE, K_B, Q  # noqa: F401
from mesa_diode.simulator.physics.constants import *  # noqa: F401,F403
from mesa_diode.simulator.physics.carriers import *  # noqa: F401,F403
from mesa_diode.simulator.physics.carriers import _expm1  # noqa: F401
from mesa_diode.simulator.physics.electrostatics import *  # noqa: F401,F403
from mesa_diode.simulator.physics.structure import *  # noqa: F401,F403
from mesa_diode.simulator.physics.lifetimes import *  # noqa: F401,F403
from mesa_diode.simulator.physics.diffusion import *  # noqa: F401,F403
from mesa_diode.simulator.physics.scr import *  # noqa: F401,F403
from mesa_diode.simulator.physics.circuit import *  # noqa: F401,F403
from mesa_diode.simulator.physics.resistance import *  # noqa: F401,F403
from mesa_diode.simulator.physics.checks import *  # noqa: F401,F403
from mesa_diode.simulator.physics.ideality import *  # noqa: F401,F403
from mesa_diode.simulator.physics.scenarios import *  # noqa: F401,F403
from mesa_diode.simulator.physics.estimates import *  # noqa: F401,F403
