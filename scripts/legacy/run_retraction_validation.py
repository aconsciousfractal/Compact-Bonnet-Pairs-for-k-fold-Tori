#!/usr/bin/env python3
"""
LEGACY / auxiliary runner.

Phase 16.5+16.6 — Retraction Form Analytic + Multi-Seed Validation

Compares the FD-based retraction form pipeline vs the new analytic pipeline
on seeds k=3,4,5,6,7.  Measures all Phase 12 gates:
  - conformality |E−G|/(E+G)  on S³
  - closure dω=0
  - cross condition ω̄∧dx=0
  - exactness (path independence F± integration)
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
    compute_analytic_retraction_form,
)
from src.retraction_form import compute_retraction_bonnet

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

SEED_KS = [3, 4, 5, 6, 7]
U_RES = 24
V_RES = 120


def run_seed(k):
    """Run both FD and analytic retraction pipelines on seed k."""
    seed = SEEDS[k]
    print(f"\n{'=' * 70}")
    print(f"  Seed k={k}  delta={seed['delta']:.4f}  s1={seed['s1']:.2f}  s2={seed['s2']:.4f}")
    print(f"{'=' * 70}")

    # Build torus
    t0 = time.time()
    params, profile = build_theorem7_torus_parameters(
        tau_imag=TAU_IMAG,
        delta=seed["delta"],
        s1=seed["s1"],
        s2=seed["s2"],
        symmetry_fold=k,
        u_res=U_RES,
        v_res=V_RES,
    )
    torus = compute_torus(params)
    t_torus = time.time() - t0
    print(f"  Torus: {t_torus:.1f}s  omega={torus.omega:.6f}  V={torus.metrics['V_period']:.4f}")

    # -- FD retraction (baseline, 'dual' method = Christoffel dual) --
    t0 = time.time()
    fd_result = compute_retraction_bonnet(torus, method='dual', verbose=False)
    t_fd = time.time() - t0

    # -- FD retraction ('direct' method) --
    t0 = time.time()
    fd_direct = compute_retraction_bonnet(torus, method='direct', verbose=False)
    t_fd_d = time.time() - t0

    # -- Analytic retraction --
    t0 = time.time()
    an_result = compute_analytic_retraction_form(torus, verbose=False)
    t_an = time.time() - t0

    return {
        'k': k,
        't_torus': t_torus,
        't_fd': t_fd,
        't_fd_direct': t_fd_d,
        't_analytic': t_an,
        'fd_dual': fd_result,
        'fd_direct': fd_direct,
        'analytic': an_result,
    }


def main():
    print("=" * 70)
    print("  Phase 16.5+16.6 — Retraction Form: Analytic vs FD")
    print(f"  Seeds: k={SEED_KS}  Grid: {U_RES}×{V_RES}")
    print("=" * 70)

    results = []
    for k in SEED_KS:
        results.append(run_seed(k))

    # ── Summary Table ──
    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY TABLE -- RETRACTION FORM GATES")
    print(f"{'=' * 100}")
    print(f"  {'k':>2} | {'Method':>10} | {'Conform':>10} | {'Closure':>10} | "
          f"{'Cross':>10} | {'Exact':>10} | {'Time':>6}")
    print(f"  {'-' * 80}")

    for r in results:
        k = r['k']
        # FD dual
        fd = r['fd_dual']
        print(f"  {k:2d} | {'FD-dual':>10} | {fd.conformality_error:10.3e} | "
              f"{fd.closure_error:10.3e} | {fd.cross_error:10.3e} | "
              f"{fd.exactness_error:10.3e} | {r['t_fd']:5.1f}s")

        # FD direct
        fdd = r['fd_direct']
        print(f"  {k:2d} | {'FD-direct':>10} | {fdd.conformality_error:10.3e} | "
              f"{fdd.closure_error:10.3e} | {fdd.cross_error:10.3e} | "
              f"{fdd.exactness_error:10.3e} | {r['t_fd_direct']:5.1f}s")

        # Analytic
        an = r['analytic']
        print(f"  {k:2d} | {'Analytic':>10} | {an['conformality_error']:10.3e} | "
              f"{an['closure_error']:10.3e} | {an['cross_error']:10.3e} | "
              f"{an['exactness_error']:10.3e} | {r['t_analytic']:5.1f}s")
        print(f"  {'-' * 80}")

    # ── Improvement ratios ──
    print(f"\n  IMPROVEMENT RATIOS (FD-direct / Analytic):")
    print(f"  {'k':>2} | {'Conform':>12} | {'Closure':>12} | {'Cross':>12} | {'Exact':>12}")
    print(f"  {'-' * 60}")
    for r in results:
        k = r['k']
        fdd = r['fd_direct']
        an = r['analytic']
        def ratio(fd_val, an_val):
            if an_val < 1e-30:
                return float('inf')
            return fd_val / an_val
        print(f"  {k:2d} | {ratio(fdd.conformality_error, an['conformality_error']):12.1f}× | "
              f"{ratio(fdd.closure_error, an['closure_error']):12.1f}× | "
              f"{ratio(fdd.cross_error, an['cross_error']):12.1f}× | "
              f"{ratio(fdd.exactness_error, an['exactness_error']):12.1f}×")

    # ── Additional diagnostics ──
    print(f"\n  ADDITIONAL DIAGNOSTICS:")
    for r in results:
        an = r['analytic']
        print(f"  k={r['k']}: tangency={an['tangency_error']:.2e}  "
              f"unit={an['unit_error']:.2e}  "
              f"orth={an['orthogonality_error']:.2e}")

    # -- Analytic derivatives extra diagnostics --
    print(f"\n  CONFORMALITY (analytic f-level):")
    for r in results:
        an = r['analytic']
        ad = an['analytic_derivatives']
        fu3 = ad.f_u[:, :, 1:4]
        fv3 = ad.f_v[:, :, 1:4]
        E = np.sum(fu3**2, axis=-1)
        G = np.sum(fv3**2, axis=-1)
        f_conf = float(np.max(np.abs(E - G)) / max(float(np.max(np.maximum(E, G))), 1e-30))
        print(f"  k={r['k']}: |E-G|/(E+G)_f = {f_conf:.2e}")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
