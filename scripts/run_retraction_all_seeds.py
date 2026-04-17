"""
Phase 12.8 — Retraction form validation on ALL seeds k=3..7 + shallow variants.

Fire test: does the retraction form hold on compressed higher-fold tori?
"""
import sys
from pathlib import Path
_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project))

import numpy as np
from src.isothermic_torus import TorusParameters, compute_torus
from src.bonnet_pair import compute_bonnet_pair
from src.retraction_form import (
    compute_retraction_bonnet,
    compare_retraction_vs_eq49,
    retraction_validation_gate,
    verify_retraction_isometry,
)
from src.seed_catalog import SEEDS, TAU_IMAG

# ── Seed catalog ──
VARIANTS = {
    "5fold_shallow": {"k": 5, "delta": 1.464690, "s1": -3.00, "s2": 0.628483},
    "6fold_shallow": {"k": 6, "delta": 1.370707, "s1": -3.00, "s2": 0.752906},
    "7fold_shallow": {"k": 7, "delta": 1.415894, "s1": -3.50, "s2": 1.261727},
}

EPSILON = 0.3
RES = 80  # resolution for fire test


def run_seed(label: str, k: int, delta: float, s1: float, s2: float,
             res: int = RES, epsilon: float = EPSILON):
    """Run retraction form pipeline on one seed."""
    print(f"\n{'═' * 70}")
    print(f"  {label}  (k={k}, δ={delta:.6f}, s₁={s1:.2f}, s₂={s2:.6f})")
    print(f"  Resolution: {res}×{res},  ε={epsilon}")
    print(f"{'═' * 70}")

    params = TorusParameters(
        tau_imag=TAU_IMAG,
        delta=delta,
        s1=s1,
        s2=s2,
        u_res=res,
        v_res=res,
        v_periods=1,
        symmetry_fold=k,
    )

    try:
        torus = compute_torus(params)
    except Exception as e:
        print(f"  ✗ TORUS FAILED: {e}")
        return None

    # Retraction form
    R = compute_retraction_bonnet(torus, method='direct', verbose=True)

    # Bonnet pair (Eq. 49)
    try:
        pair = compute_bonnet_pair(torus, epsilon=epsilon)
        # Comparison
        comp = compare_retraction_vs_eq49(R, pair, verbose=True)
    except Exception as e:
        print(f"  ✗ BONNET PAIR FAILED: {e}")
        comp = None
        pair = None

    # Isometry check on F±
    u_grid = np.linspace(0, 2 * np.pi, res, endpoint=False)
    v = torus.frame_result.v_values
    du = u_grid[1] - u_grid[0]
    dv = (v[-1] - v[0]) / (len(v) - 1)
    iso = verify_retraction_isometry(R.F_plus, R.F_minus, du, dv)

    # Gate
    gate = retraction_validation_gate(
        R, bonnet_result=pair, verbose=True,
        tol_closure=0.5, tol_cross=1.0, tol_exactness=0.5,
        tol_procrustes=2.0,
    )

    # Summary
    print(f"\n  ── Summary ──")
    print(f"  F± isometry: max Δg_uu={iso['max_d_guu']:.2e}, "
          f"Δg_vv={iso['max_d_gvv']:.2e}")
    print(f"  Gate: {'PASS ✓' if gate['all_pass'] else 'FAIL ✗'}")

    return dict(
        label=label, k=k, retraction=R, gate=gate,
        iso=iso, comp=comp,
    )


def main():
    results = []

    # Canonical seeds k=3..7
    for k, seed in sorted(SEEDS.items()):
        label = seed["label"]
        r = run_seed(label, k, seed["delta"], seed["s1"], seed["s2"])
        if r:
            results.append(r)

    # Shallow variants (the "fire test")
    for label, v in sorted(VARIANTS.items()):
        r = run_seed(label, v["k"], v["delta"], v["s1"], v["s2"])
        if r:
            results.append(r)

    # ── Final report ──
    print(f"\n\n{'═' * 70}")
    print(f"  PHASE 12.8 — RETRACTION VALIDATION REPORT")
    print(f"{'═' * 70}")
    print(f"{'Label':<20} {'k':>2} {'dω=0':>8} {'ω̄∧dx':>8} {'exact':>8} "
          f"{'F±scal':>8} {'iso Δg':>8} {'gate':>6}")
    print(f"{'─' * 70}")

    n_pass = 0
    for r in results:
        R = r['retraction']
        gate_str = "PASS" if r['gate']['all_pass'] else "FAIL"
        if r['gate']['all_pass']:
            n_pass += 1
        print(f"{r['label']:<20} {r['k']:>2} "
              f"{R.closure_error:>8.2e} {R.cross_error:>8.2e} "
              f"{R.exactness_error:>8.2e} "
              f"{R.metrics['F_plus_max_scalar']:>8.2e} "
              f"{r['iso']['max_d_guu']:>8.2e} "
              f"{gate_str:>6}")

    print(f"{'─' * 70}")
    print(f"  Total: {n_pass}/{len(results)} PASS")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
