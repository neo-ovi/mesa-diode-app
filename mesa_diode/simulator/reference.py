# -*- coding: utf-8 -*-
"""Вкладка «Справочник» (ТЗ §7.2): вычисляемые величины и предупреждения.

Все числа считаются теми же функциями physics.py, что и графики, —
отдельных формул здесь нет.
"""

from dataclasses import dataclass, field

import numpy as np

from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.materials import SIGMA_R_VALID_RANGE

UM = 1e-4



@dataclass
class Row:
    label: str
    value: object          # число, строка или None
    unit: str = ""
    note: str = ""


@dataclass
class Group:
    title: str
    rows: list = field(default_factory=list)

    def add(self, label, value, unit="", note=""):
        self.rows.append(Row(label, value, unit, note))


@dataclass
class ReferenceData:
    groups: list
    warnings: list


def _layer_group(s, side, title, recommendation):
    g = Group(title)
    Nc, Nv = ph.effective_dos(s.T, s.material)
    M, m = ph.equilibrium_carriers(side.N, s.ni)
    n0, p0 = (M, m) if side.kind == "n" else (m, M)
    eta = ph.reduced_fermi_level(M, Nc if side.kind == "n" else Nv)
    ratio = ph.boltzmann_ratio(eta)
    edge = "E_F − E_C" if side.kind == "n" else "E_V − E_F"
    g.add("N (ионизованная примесь)", side.N, "см⁻³")
    g.add("тип", f"{side.kind}  (слой {side.layer})")
    g.add("n₀ / p₀ (2.4)", f"{n0:.3e} / {p0:.3e}", "см⁻³")
    g.add(f"η = ({edge})/kT (2.7)", eta)
    g.add(edge, f"{eta * s.Vt * 1e3:.1f} мэВ = {eta:.2f} kT")
    g.add("ошибка Больцмана (2/√π)F½(η)e^{−η}", ratio, "", f"{(1 - ratio) * 100:.2f} %")
    g.add("статус вырождения", ph.degeneracy_status(eta))
    D = ph.diffusion_coefficient(side.mu_minority, s.T)
    tau = ph.minority_lifetime(side, s.N_dis)
    L = ph.diffusion_length(D, tau)
    minority = "дырки" if side.kind == "n" else "электроны"
    g.add(f"D неосновных ({minority}) (2.5)", D, "см²/с")
    g.add("τ с учётом N_dis (5.9)", tau, "с")
    g.add("τ_dis = 1/(σ_R·N_dis)", ph.dislocation_lifetime(side.sigma_R, s.N_dis), "с")
    g.add("L (2.6)", L / UM, "мкм")
    wn, wp, _ = ph.neutral_widths(s, 0.0)
    w = float(wn if side.kind == "n" else wp)
    g.add("w/L при V = 0", w / L)
    if side.kind == "p" and s.material.m_cc:
        lam = ph.drude_mean_free_path(s.material.vth_n, side.mu_minority, s.material.m_cc)
        g.add("λ электронов (оценка, Друде)", lam * 1e7, "нм")
    else:
        g.add("λ (оценка)", "—", "", "нет m_cc для дырок")
    g.add(f"граничное условие: {recommendation[0]}",
          ph.BOUNDARY_LABELS[side.boundary], "", f"рекомендация (6.2): {recommendation[1]}")
    return g, eta


def reference_table(s, iv=None, n_exp=None, n_mod=None, metadata=None):
    """Все величины §7.2 для структуры ``s`` и список предупреждений.

    iv — результат physics.solve_iv (для проверки сходимости);
    n_exp, n_mod — коэффициенты идеальности (§6.3) для доли тока ОПЗ (6.6);
    metadata — метаданные набора образца (имплантация)."""
    warnings = []
    groups = []

    g = Group("Материал")
    Nc, Nv = ph.effective_dos(s.T, s.material)
    g.add("E_g(T) (2.1)", ph.band_gap(s.T, s.material), "эВ")
    g.add("N_C / N_V (2.2)", f"{Nc:.3e} / {Nv:.3e}", "см⁻³")
    g.add("n_i (2.3)", s.ni, "см⁻³")
    groups.append(g)

    rec = ph.boundary_recommendations(s)
    gn, eta_n = _layer_group(s, s.n_side, "n-сторона перехода", rec["n"])
    gp, eta_p = _layer_group(s, s.p_side, "p-сторона перехода", rec["p"])
    groups += [gn, gp]
    for side, eta in ((s.n_side, eta_n), (s.p_side, eta_p)):
        status = ph.degeneracy_status(eta)
        if status != "невырожден":
            warnings.append(f"Слой {side.layer}: {status} (η = {eta:.2f}), "
                            f"ошибка Больцмана {(1 - ph.boltzmann_ratio(eta)) * 100:.1f} %.")
        if side.N < 10.0 * s.ni:
            warnings.append(f"Слой {side.layer}: N < 10·n_i — приближение обеднения "
                            "грубое (почти собственный слой).")

    g = Group("Подложка")
    N_sub, sub_warn = s.substrate
    warnings += sub_warn
    p0, n0 = ph.equilibrium_carriers(N_sub, s.ni)
    g.add("N_A из ρ_sub (2.8)", N_sub, "см⁻³")
    g.add("n₀ / p₀", f"{n0:.3e} / {p0:.3e}", "см⁻³")
    g.add("R_sub = ρ·d_sub/A (верхняя оценка)", s.rho_sub * s.d_sub / s.area, "Ом")
    groups.append(g)

    g = Group(f"Переход (сценарий {s.scenario})")
    g.add("z_j", s.z_j / UM, "мкм")
    g.add("V_bi (3.1)", ph.built_in_potential(s, ph.VBI_BOLTZMANN), "В")
    g.add("V_bi (3.1а) — в расчёте", ph.built_in_potential(s, ph.VBI_DEGENERATE), "В")
    for V in (0.0, -1.0):
        xn, xp = ph.depletion_edges(s, V)
        g.add(f"W({V:g} В) / x_n / x_p",
              f"{float(ph.depletion_width(s, V)) / UM:.4g} / {float(xn) / UM:.4g} / {float(xp) / UM:.4g}",
              "мкм")
    g.add("N_eff", s.N_eff, "см⁻³")
    g.add("отсечка 1/C² = V_bi − 2kT/q", ph.c2_cutoff(s), "В")
    for V in (0.0, -1.0):
        level, text = ph.isolation_status(s, V)
        g.add(f"изоляция (1.2) при {V:g} В", text)
        if level != ph.ISOLATED:
            warnings.append(f"Изоляция при {V:g} В: {text}.")
    g.add("ΔA(−1 В) (1.3, оценка)", float(ph.edge_area(s, -1.0)), "см²",
          "включено" if s.edge_area else "опция выключена")
    g.add("S_side = πDh (справочно)", ph.side_wall_area(s), "см²")
    groups.append(g)
    if ph.neutral_widths(s, -1.0)[2]:
        warnings.append("Смыкание: ОПЗ при −1 В достигает дальней границы слоя.")

    g = Group("Ток")
    holes, electrons = ph.saturation_current_density_parts(s, 0.0)
    g.add("J_s(0): дырки в n-области", float(holes), "А/см²")
    g.add("J_s(0): электроны в p-области", float(electrons), "А/см²")
    g.add("I_gr(−1 В) (5.4)", float(ph.gr_current(s, -1.0)), "А")
    tau0 = ph.scr_lifetime(s)
    g.add("τ₀ (5.10)", tau0, "с")
    g.add("σ·v_th·N_t = 1/τ₀", 1.0 / tau0, "с⁻¹")
    v_li = ph.low_injection_voltage(s)
    g.add("V_LI (6.4)", v_li, "В")
    for name, n in (("n_эксп", n_exp), ("n_мод", n_mod)):
        if n is not None and np.isfinite(n):
            g.add(f"доля тока ОПЗ по (6.6), {name} = {n:.3f}", 2.0 * (1.0 - 1.0 / n) * 100, "%")
    groups.append(g)
    if v_li < 0.05:
        warnings.append(f"V_LI = {v_li:.3f} В: прямая ветвь модели неприменима почти "
                        "сразу (высокий уровень инжекции); сравнивайте обратную ветвь и ВФХ.")

    if metadata and metadata.get("implant"):
        imp = metadata["implant"]
        g = Group("Имплантация (из набора образца)")
        g.add("доза", imp.get("dose_cm2"), "см⁻²")
        g.add("энергия", imp.get("energy_keV"), "кэВ")
        g.add("отжиг", imp.get("anneal", "—"))
        if imp.get("dose_cm2"):
            g.add("N_D ≤ Q/d_n", imp["dose_cm2"] / s.d_n, "см⁻³")
        # Пробег ионов берётся из набора образца вместе с источником
        # (например, [Кур74, табл. 7.1] для Si с пометкой «в Ge меньше»).
        if imp.get("range_Rp_A") is not None:
            g.add("R_p / ΔR_p ионов", f"{imp['range_Rp_A']:g} / {imp.get('range_dRp_A', float('nan')):g}",
                  "Å", imp.get("range_note", ""))
        groups.append(g)

    if s.n2 != 2.0:
        warnings.append(f"n₂ = {s.n2:g} ≠ 2 — эмпирика (5.6).")
    lo, hi = SIGMA_R_VALID_RANGE
    if s.N_dis > 0 and not lo <= s.N_dis <= hi:
        warnings.append(f"N_dis = {s.N_dis:.2e} см⁻² вне диапазона измерений σ_R "
                        f"({lo:.0e}–{hi:.0e}): экстраполяция.")
    if iv is not None:
        warnings += iv.warnings

    g = Group(f"Пороги вырождения при T = {s.T:g} К (2.7)")
    table = ph.degeneracy_thresholds(s.T, s.material)
    for carrier, label in (("electrons", "электроны"), ("holes", "дырки")):
        g.add(label, " / ".join(f"η={eta:g}: {n:.2e}" for eta, n in table[carrier].items()), "см⁻³")
    groups.append(g)

    return ReferenceData(groups=groups, warnings=warnings)
