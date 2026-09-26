# -*- coding: utf-8 -*-
"""§5А. Дислокации и время жизни: (5.9), (5.10); методичка, гл. 10А."""


def dislocation_lifetime(sigma_R, N_dis):
    """τ_dis = 1/(σ_R·N_dis), с — из [KKA56, с. 1289, ур. (3)]: λ = σ_R·N_D·ΔP."""
    return float("inf") if N_dis <= 0 else 1.0 / (sigma_R * N_dis)


def minority_lifetime(side, N_dis):
    """(5.9) 1/τ = 1/τ^bg + σ_R·N_dis: каналы рекомбинации складываются;
    второе слагаемое — [KKA56, ур. (3)]."""
    return 1.0 / (1.0 / side.tau_bg + side.sigma_R * N_dis)


def scr_sigma_R(s):
    """σ_R слоя, в котором лежит бо́льшая часть ОПЗ (x_p > x_n ⇔ N_D > N_A)."""
    return s.p_side.sigma_R if s.n_side.N > s.p_side.N else s.n_side.sigma_R


def scr_lifetime(s):
    """(5.10) 1/τ₀ = 1/τ₀^bg + σ_R·N_dis — допущение модели: перенос
    [KKA56, ур. (3)] (объёмное время жизни) на ОПЗ."""
    return 1.0 / (1.0 / s.tau0_bg + scr_sigma_R(s) * s.N_dis)
