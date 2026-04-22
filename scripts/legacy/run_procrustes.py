#!/usr/bin/env python3
"""
LEGACY / auxiliary runner.

Phase 15.6 -- Procrustes Decay Rate with Analytic Derivatives

Measures Procrustes disparity between retraction F+/F- and Eq.49 f+/f-
on seeds k=3..12, using the new analytic retraction form (Phase 16).

Fits competing models:
  (A) Power law:   d(k) = a / k^b
  (B) Exponential: d(k) = a * exp(-b*k)
  (C) Stretched:   d(k) = a * exp(-b * sqrt(k))

Reports R^2 and AIC for model selection.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.isothermic_torus import compute_torus
from src.theorem7_periodicity import build_theorem7_torus_parameters
from src.seed_catalog import SEEDS, TAU_IMAG
from src.retraction_form import (
    compute_retraction_bonnet,
    compare_retraction_vs_eq49,
)
from src.bonnet_pair import compute_bonnet_pair

# =====================================================================
# Configuration
# =====================================================================

SEED_KS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
U_RES = 40
V_RES = 120
EPSILON = 0.3


# =====================================================================
# Models to fit
# =====================================================================

def model_power(k, a, b):
    return a / np.power(k, b)

def model_exp(k, a, b):
    return a * np.exp(-b * k)

def model_stretched(k, a, b):
    return a * np.exp(-b * np.sqrt(k))


def fit_model(ks, ds, func, name, p0=None):
    """Fit d(k) to func, return params, R^2, AIC."""
    try:
        popt, pcov = curve_fit(func, ks, ds, p0=p0, maxfev=10000)
        d_pred = func(ks, *popt)
        ss_res = np.sum((ds - d_pred)**2)
        ss_tot = np.sum((ds - np.mean(ds))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n = len(ds)
        k_params = len(popt)
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k_params
        return dict(name=name, params=popt, r2=r2, aic=aic, ok=True)
    except Exception as e:
        return dict(name=name, params=None, r2=-1, aic=1e30, ok=False, err=str(e))


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("  Phase 15.6 -- Procrustes Decay with Analytic Retraction")
    print(f"  Seeds: k={SEED_KS}  Grid: {U_RES}x{V_RES}  eps={EPSILON}")
    print("=" * 70)

    ks_data = []
    procrustes_best = []
    procrustes_direct = []
    procrustes_swap = []
    conf_errors = []
    cross_errors = []

    for k in SEED_KS:
        seed = SEEDS[k]
        print(f"\n--- k={k} delta={seed['delta']:.4f} ---")

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
        t1 = time.time()

        # Analytic retraction (default)
        R = compute_retraction_bonnet(torus, method='analytic', verbose=False)
        t2 = time.time()

        # Eq.49 Bonnet pair
        P = compute_bonnet_pair(torus, epsilon=EPSILON)
        t3 = time.time()

        # Procrustes comparison
        comp = compare_retraction_vs_eq49(R, P, verbose=True)
        t4 = time.time()

        ks_data.append(k)
        procrustes_best.append(comp['best_match'])
        procrustes_direct.append(comp['procrustes_Fp_fp'] + comp['procrustes_Fm_fm'])
        procrustes_swap.append(comp['procrustes_Fp_fm'] + comp['procrustes_Fm_fp'])
        conf_errors.append(R.conformality_error)
        cross_errors.append(R.cross_error)

        print(f"  Time: torus={t1-t0:.1f}s retraction={t2-t1:.1f}s "
              f"bonnet={t3-t2:.1f}s procrustes={t4-t3:.1f}s")
        print(f"  Conform={R.conformality_error:.2e} Cross={R.cross_error:.2e}")

    # =====================================================================
    # Summary table
    # =====================================================================
    ks_arr = np.array(ks_data, dtype=float)
    pb = np.array(procrustes_best)

    print(f"\n\n{'=' * 70}")
    print(f"  PROCRUSTES DECAY TABLE (analytic retraction)")
    print(f"{'=' * 70}")
    print(f"  {'k':>3} | {'Best match':>12} | {'Direct sum':>12} | "
          f"{'Swapped sum':>12} | {'Conform':>10} | {'Cross':>10}")
    print(f"  {'-' * 72}")
    for i, k in enumerate(ks_data):
        print(f"  {k:3d} | {procrustes_best[i]:12.6e} | "
              f"{procrustes_direct[i]:12.6e} | {procrustes_swap[i]:12.6e} | "
              f"{conf_errors[i]:10.2e} | {cross_errors[i]:10.2e}")

    # =====================================================================
    # Model fitting
    # =====================================================================
    print(f"\n\n{'=' * 70}")
    print(f"  MODEL FITTING: d(k) = ?")
    print(f"{'=' * 70}")

    fits = [
        fit_model(ks_arr, pb, model_power, "a/k^b", p0=[1.0, 1.0]),
        fit_model(ks_arr, pb, model_exp, "a*exp(-bk)", p0=[1.0, 0.1]),
        fit_model(ks_arr, pb, model_stretched, "a*exp(-b*sqrt(k))", p0=[1.0, 1.0]),
    ]

    for f in fits:
        if f['ok']:
            print(f"  {f['name']:>20s}: params=({f['params'][0]:.6f}, {f['params'][1]:.6f})  "
                  f"R^2={f['r2']:.8f}  AIC={f['aic']:.2f}")
        else:
            print(f"  {f['name']:>20s}: FIT FAILED -- {f.get('err', '?')}")

    # Best model
    best = min(fits, key=lambda f: f['aic'])
    print(f"\n  >> Best model (min AIC): {best['name']}  R^2={best['r2']:.8f}")

    # Running exponent b(k) = -d(log d)/d(log k)
    if len(ks_arr) >= 3:
        print(f"\n  RUNNING EXPONENT (consecutive pairs):")
        print(f"  {'k_mid':>6} | {'b(k)':>8}")
        print(f"  {'-' * 20}")
        for i in range(len(ks_arr) - 1):
            if pb[i] > 0 and pb[i+1] > 0:
                b_run = -(np.log(pb[i+1]) - np.log(pb[i])) / (np.log(ks_arr[i+1]) - np.log(ks_arr[i]))
                k_mid = 0.5 * (ks_arr[i] + ks_arr[i+1])
                print(f"  {k_mid:6.1f} | {b_run:8.3f}")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
