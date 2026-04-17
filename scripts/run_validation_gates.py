#!/usr/bin/env python3
"""
Phase 16 Validation — Analytic vs Finite Difference Derivatives

Compares analytic derivatives from theta function formulas against
finite differences on a Bonnet torus.  Uses k=5 seed for testing.

Expected improvements:
  - FD first derivatives:  ~10⁻⁶ accuracy (O(h²), h = 2π/N_u)
  - Analytic first derivatives: ~10⁻¹² accuracy (theta function precision)
  - FD mean curvature: ~10⁻⁴ to 10⁻⁶
  - Analytic mean curvature: ~10⁻¹⁰+
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.isothermic_torus import compute_torus
from src.theorem7_periodicity import build_theorem7_torus_parameters
from src.seed_catalog import SEEDS, TAU_IMAG
from src.analytic_derivatives import (
    compute_analytic_derivatives,
    gamma_derivatives_vec,
)
from src.bonnet_pair import (
    compute_mean_curvature_fd,
    compute_metric_tensor_periodic_central,
)

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

SEED_K = 5
U_RES = 24       # modest resolution to show FD error clearly
V_RES = 120
SEED = SEEDS[SEED_K]


def main():
    print("=" * 70)
    print(f"  Phase 16 Validation — k={SEED_K} seed, {U_RES}×{V_RES} grid")
    print("=" * 70)

    # ── Step 1: Build torus ──
    t0 = time.time()
    params, profile = build_theorem7_torus_parameters(
        tau_imag=TAU_IMAG,
        delta=SEED["delta"],
        s1=SEED["s1"],
        s2=SEED["s2"],
        symmetry_fold=SEED_K,
        u_res=U_RES,
        v_res=V_RES,
    )
    torus = compute_torus(params)
    t_torus = time.time() - t0
    print(f"\n  Torus built in {t_torus:.1f}s — ω = {torus.omega:.6f}")
    print(f"  Grid: {U_RES} × {V_RES}, V_period = {torus.metrics['V_period']:.4f}")

    # ── Step 2: FD derivatives (baseline) ──
    u_grid = torus.u_grid
    v_grid = torus.v_grid
    f_grid = torus.f_grid

    fd_metric = compute_metric_tensor_periodic_central(f_grid, u_grid, v_grid)
    fd_fu = fd_metric['f_u']   # (n_u, n_v, 3)
    fd_fv = fd_metric['f_v']

    fd_H = compute_mean_curvature_fd(f_grid, u_grid, v_grid)
    print(f"\n  FD derivatives computed.")
    print(f"  FD H range: [{fd_H.min():.6f}, {fd_H.max():.6f}]")

    # ── Step 3: Analytic derivatives ──
    t0 = time.time()
    ad = compute_analytic_derivatives(torus, verbose=True)
    t_analytic = time.time() - t0
    print(f"\n  Analytic derivatives computed in {t_analytic:.1f}s")
    print(f"  Analytic H range: [{ad.H.min():.6f}, {ad.H.max():.6f}]")

    # ── Step 4: Compare first derivatives ──
    # The FD f_u is (n_u, n_v, 3) from ℝ³ coordinates
    # The analytic f_u is (n_u, n_v, 4) quaternion — extract ℝ³ from slots 1:4
    an_fu3 = ad.f_u[:, :, 1:4]
    an_fv3 = ad.f_v[:, :, 1:4]

    # Pointwise difference
    err_fu = np.sqrt(np.sum((an_fu3 - fd_fu) ** 2, axis=2))
    err_fv = np.sqrt(np.sum((an_fv3 - fd_fv) ** 2, axis=2))

    scale_fu = np.sqrt(np.sum(an_fu3 ** 2, axis=2))
    scale_fv = np.sqrt(np.sum(an_fv3 ** 2, axis=2))

    rel_fu = err_fu / np.maximum(scale_fu, 1e-30)
    rel_fv = err_fv / np.maximum(scale_fv, 1e-30)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: First Derivatives")
    print(f"{'=' * 70}")
    print(f"  |f_u(analytic) − f_u(FD)|:")
    print(f"    max absolute: {err_fu.max():.3e}")
    print(f"    max relative: {rel_fu.max():.3e}")
    print(f"    mean relative: {rel_fu.mean():.3e}")
    print(f"  |f_v(analytic) − f_v(FD)|:")
    print(f"    max absolute: {err_fv.max():.3e}")
    print(f"    max relative: {rel_fv.max():.3e}")
    print(f"    mean relative: {rel_fv.mean():.3e}")

    # ── Step 5: Metric tensor comparison ──
    E_an = np.sum(an_fu3 ** 2, axis=2)
    F_an = np.sum(an_fu3 * an_fv3, axis=2)
    G_an = np.sum(an_fv3 ** 2, axis=2)

    E_fd = fd_metric['E']
    F_fd = fd_metric['F']
    G_fd = fd_metric['G']

    print(f"\n  Metric tensor E (isothermic: E ≈ G, F ≈ 0):")
    print(f"    E(analytic) range: [{E_an.min():.6f}, {E_an.max():.6f}]")
    print(f"    E(FD)      range: [{E_fd.min():.6f}, {E_fd.max():.6f}]")
    print(f"    max |E_an − E_fd|/E_an: {(np.abs(E_an - E_fd) / np.maximum(E_an, 1e-30)).max():.3e}")

    # Conformality check
    conf_an = np.abs(E_an - G_an) / np.maximum(0.5 * (E_an + G_an), 1e-30)
    conf_fd = np.abs(E_fd - G_fd) / np.maximum(0.5 * (E_fd + G_fd), 1e-30)
    print(f"\n  Conformality |E−G|/(E+G):")
    print(f"    analytic max: {conf_an.max():.3e}")
    print(f"    FD max:       {conf_fd.max():.3e}")

    # ── Step 6: Pure imaginary check for f_u, f_v ──
    scalar_fu = np.abs(ad.f_u[:, :, 0])
    scalar_fv = np.abs(ad.f_v[:, :, 0])
    print(f"\n  Pure imaginary check (scalar part should ≈ 0):")
    print(f"    max |Re(f_u)|: {scalar_fu.max():.3e}")
    print(f"    max |Re(f_v)|: {scalar_fv.max():.3e}")

    # ── Step 7: Mean curvature comparison ──
    err_H = np.abs(ad.H - fd_H)
    scale_H = np.maximum(np.abs(ad.H), 1e-30)
    rel_H = err_H / scale_H

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: Mean Curvature H")
    print(f"{'=' * 70}")
    print(f"  H(analytic) range: [{ad.H.min():.8f}, {ad.H.max():.8f}]")
    print(f"  H(FD)       range: [{fd_H.min():.8f}, {fd_H.max():.8f}]")
    print(f"  |H(an) − H(FD)|:")
    print(f"    max absolute: {err_H.max():.3e}")
    print(f"    max relative: {rel_H.max():.3e}")
    print(f"    mean relative: {rel_H.mean():.3e}")

    # ── Step 8: Self-consistency checks ──
    # For isothermic surfaces: H should be constant along u
    H_u_var_an = np.std(ad.H, axis=0) / np.maximum(np.abs(np.mean(ad.H, axis=0)), 1e-30)
    H_u_var_fd = np.std(fd_H, axis=0) / np.maximum(np.abs(np.mean(fd_H, axis=0)), 1e-30)

    print(f"\n  H variation along u (should be small for isothermic):")
    print(f"    analytic: max σ/|μ| = {H_u_var_an.max():.3e}")
    print(f"    FD:       max σ/|μ| = {H_u_var_fd.max():.3e}")

    # Second derivatives — check f_uu scalar part
    scalar_fuu = np.abs(ad.f_uu[:, :, 0])
    scalar_fuv = np.abs(ad.f_uv[:, :, 0])
    scalar_fvv = np.abs(ad.f_vv[:, :, 0])
    print(f"\n  Second derivatives scalar parts (purity check):")
    print(f"    max |Re(f_uu)|: {scalar_fuu.max():.3e}")
    print(f"    max |Re(f_uv)|: {scalar_fuv.max():.3e}")
    print(f"    max |Re(f_vv)|: {scalar_fvv.max():.3e}")

    # ── Step 9: Gamma derivative spot-check at a few points ──
    print(f"\n{'=' * 70}")
    print(f"  Gamma Derivative Spot Check (FD vs analytic)")
    print(f"{'=' * 70}")
    omega = torus.omega
    tau = params.tau
    n_v_grid = len(v_grid)
    w_test = params.w_func(v_grid[n_v_grid // 4])  # quarter-period w value
    u_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    n_v = len(v_grid)

    g, g_u, g_uu = gamma_derivatives_vec(u_test, w_test, omega, tau)

    # FD check
    h = 1e-7
    u_plus = u_test + h
    u_minus = u_test - h
    from src.theta_functions import gamma_curve_vec
    g_p = gamma_curve_vec(u_plus, w_test, omega, tau)
    g_m = gamma_curve_vec(u_minus, w_test, omega, tau)
    g_fd = (g_p - g_m) / (2 * h)
    g_uu_fd = (g_p - 2 * g + g_m) / h ** 2

    print(f"\n  w = {w_test:.6f}, h = {h}")
    print(f"  {'u':>6} | {'|γ_u(an)|':>12} | {'|γ_u(FD)|':>12} | {'rel err':>10} | "
          f"{'|γ_uu_an|':>12} | {'|γ_uu_FD|':>12} | {'rel err':>10}")
    print(f"  {'-' * 90}")
    for i in range(len(u_test)):
        rel_u = abs(g_u[i] - g_fd[i]) / max(abs(g_u[i]), 1e-30)
        rel_uu = abs(g_uu[i] - g_uu_fd[i]) / max(abs(g_uu[i]), 1e-30)
        print(f"  {u_test[i]:6.2f} | {abs(g_u[i]):12.6e} | {abs(g_fd[i]):12.6e} | {rel_u:10.3e} | "
              f"{abs(g_uu[i]):12.6e} | {abs(g_uu_fd[i]):12.6e} | {rel_uu:10.3e}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Grid: {U_RES}×{V_RES}, k={SEED_K}, computation: {t_analytic:.0f}s")
    print(f"  First derivatives:  FD→analytic error {rel_fu.mean():.1e} (mean)")
    print(f"  Mean curvature:     FD→analytic error {rel_H.mean():.1e} (mean)")
    print(f"  Conformality (E≈G): analytic {conf_an.max():.1e} vs FD {conf_fd.max():.1e}")
    print(f"  H u-uniformity:     analytic {H_u_var_an.max():.1e} vs FD {H_u_var_fd.max():.1e}")

    improvement = rel_H.mean() / max(conf_an.max(), 1e-30)
    print(f"\n  DONE")


if __name__ == "__main__":
    main()
