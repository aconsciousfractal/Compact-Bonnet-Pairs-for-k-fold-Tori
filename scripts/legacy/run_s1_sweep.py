"""
LEGACY / auxiliary runner.

Phase 11, Task 11.5 — Lemma 9 regime sweep.

Sweep s₁ across [-10, -3] for k=5,6,7 (fixed τ_imag) to map out the
solution landscape. At each s₁, solve for (δ, s₂) via Newton starting
from the nearest known seed.

The Lemma 9 regime (|s₁| large) guarantees asymptotic existence.
This sweep checks whether solutions persist across a wide s₁ range
and discovers additional families.

Strategy: continuation in s₁ from known seeds in both directions.
"""
import sys, math, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theorem7_periodicity import (
    solve_theorem7_local_fixed_tau_s1,
    theorem7_residuals,
    theorem7_lemma8_real_sqrt_expr,
    theorem7_lemma8_rationality_expr,
    theorem7_lemma8_vanishing_expr,
)
from src import theta_functions as TF

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAU_IMAG = 0.3205128205

# Known seeds to use as starting points for continuation
KNOWN_SEEDS = {
    5: {"delta": 1.678757500643, "s1": -3.80, "s2": 1.188432661226},
    6: {"delta": 1.749166739658, "s1": -4.50, "s2": 1.813323472665},
    7: [
        {"delta": 1.959258407446, "s1": -6.00, "s2": 2.979872632766},
        {"delta": 1.862353106277, "s1": -5.50, "s2": 2.647915844137},
    ],
}

# Precompute omega and tau
tau = 0.5 + 1j * TAU_IMAG
omega = TF.find_critical_omega(tau)

RESIDUAL_THRESHOLD = 1e-6  # accept solution if residual < this
S1_STEP = 0.25  # step size for s1 continuation


def try_solve(k, s1, delta_guess, s2_guess, label=""):
    """Try to solve at (k, s1) from a guess. Returns result dict or None."""
    try:
        sol = solve_theorem7_local_fixed_tau_s1(
            tau_imag=TAU_IMAG, s1=s1,
            initial_delta=delta_guess, initial_s2=s2_guess,
            symmetry_fold=k, target_ratio=1.0,
        )
        if sol.success and sol.residual_norm < RESIDUAL_THRESHOLD:
            # Check Lemma 8
            iii = theorem7_lemma8_real_sqrt_expr(omega, tau, s1, sol.s2)
            iv = theorem7_lemma8_rationality_expr(omega, tau, s1, sol.s2)
            v = theorem7_lemma8_vanishing_expr(omega, tau, s1, sol.s2)
            lemma8_ok = iii > 0 and abs(iv) > 1e-10 and abs(v) > 1e-10
            return {
                "s1": s1,
                "delta": sol.delta,
                "s2": sol.s2,
                "residual": sol.residual_norm,
                "ratio": sol.theorem7.ratio,
                "axial": sol.theorem7.axial_integral,
                "theta_deg": math.degrees(sol.theorem7.theta),
                "lemma8_ok": lemma8_ok,
                "lemma8_iii": iii,
                "lemma8_v": v,
                "nfev": sol.nfev,
            }
    except Exception:
        pass
    return None


def continue_in_s1(k, seed, s1_range, direction="forward"):
    """Continue a solution along s₁ from a known seed."""
    results = []
    current_delta = seed["delta"]
    current_s2 = seed["s2"]
    current_s1 = seed["s1"]

    if direction == "forward":
        s1_values = [s for s in s1_range if s > current_s1]
    else:
        s1_values = sorted([s for s in s1_range if s < current_s1], reverse=True)

    consecutive_failures = 0
    for s1_target in s1_values:
        result = try_solve(k, s1_target, current_delta, current_s2)
        if result is not None:
            results.append(result)
            current_delta = result["delta"]
            current_s2 = result["s2"]
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 4:
                break  # lost the branch

    return results


def sweep_k(k, seeds):
    """Full s₁ sweep for a given k-fold symmetry."""
    print(f"\n{'='*60}")
    print(f"  k={k}: sweeping s₁ ∈ [-10, -2]")
    print(f"{'='*60}")

    s1_grid = np.arange(-10.0, -1.75, S1_STEP)
    all_results = []

    if not isinstance(seeds, list):
        seeds = [seeds]

    for i, seed in enumerate(seeds):
        label = f"branch_{i}"
        print(f"\n  Branch {i}: starting from s₁={seed['s1']:.2f}")

        # Continue backward (toward more negative s₁)
        backward = continue_in_s1(k, seed, s1_grid, direction="backward")
        print(f"    backward: {len(backward)} solutions found", end="")
        if backward:
            print(f"  s₁ range: [{backward[-1]['s1']:.2f}, {backward[0]['s1']:.2f}]")
        else:
            print()

        # Continue forward (toward less negative s₁)
        forward = continue_in_s1(k, seed, s1_grid, direction="forward")
        print(f"    forward:  {len(forward)} solutions found", end="")
        if forward:
            print(f"  s₁ range: [{forward[0]['s1']:.2f}, {forward[-1]['s1']:.2f}]")
        else:
            print()

        # Include the seed itself
        seed_result = try_solve(k, seed["s1"], seed["delta"], seed["s2"])
        branch = list(reversed(backward)) + ([seed_result] if seed_result else []) + forward
        for r in branch:
            r["branch"] = i

        all_results.extend(branch)
        print(f"    total branch: {len(branch)} points")

    # Sort by s₁
    all_results.sort(key=lambda r: r["s1"])

    # Summary
    print(f"\n  --- k={k} summary ---")
    print(f"  Total solutions: {len(all_results)}")
    if all_results:
        s1_min = min(r["s1"] for r in all_results)
        s1_max = max(r["s1"] for r in all_results)
        print(f"  s₁ range: [{s1_min:.2f}, {s1_max:.2f}]")

        # Print table
        print(f"  {'s₁':>8s} {'δ':>12s} {'s₂':>12s} {'res':>10s} {'L8':>4s} {'br':>3s}")
        print(f"  {'-'*55}")
        for r in all_results:
            print(f"  {r['s1']:>8.2f} {r['delta']:>12.6f} {r['s2']:>12.6f} "
                  f"{r['residual']:>10.2e} {'OK' if r['lemma8_ok'] else 'NO':>4s} {r['branch']:>3d}")

    return all_results


def main():
    print("=" * 60)
    print("  PHASE 11.5: LEMMA 9 REGIME SWEEP")
    print(f"  τ_imag = {TAU_IMAG},  s₁ step = {S1_STEP}")
    print("=" * 60)

    t0 = time.time()
    all_data = {}

    for k in [5, 6, 7]:
        seeds = KNOWN_SEEDS[k]
        results = sweep_k(k, seeds)
        all_data[str(k)] = results

    # ── Cross-fold comparison ──
    print(f"\n{'='*60}")
    print("  CROSS-FOLD EXISTENCE MAP")
    print("="*60)
    for k_str, results in all_data.items():
        k = int(k_str)
        if results:
            s1s = [r["s1"] for r in results]
            n_lemma8 = sum(1 for r in results if r["lemma8_ok"])
            print(f"  k={k}: {len(results)} solutions, "
                  f"s₁ ∈ [{min(s1s):.2f}, {max(s1s):.2f}], "
                  f"Lemma 8 OK: {n_lemma8}/{len(results)}")
        else:
            print(f"  k={k}: no solutions found")

    # ── Parameter evolution plots (text-based) ──
    print(f"\n{'='*60}")
    print("  PARAMETER EVOLUTION: δ(s₁) and s₂(s₁)")
    print("="*60)
    for k_str, results in all_data.items():
        k = int(k_str)
        if not results:
            continue
        print(f"\n  k={k}:")
        print(f"  {'s₁':>8s} {'δ':>12s} {'s₂':>12s} {'δ/s₂':>10s} {'L8(v)':>10s}")
        print(f"  {'-'*55}")
        for r in results:
            ratio = r["delta"] / r["s2"] if abs(r["s2"]) > 1e-14 else float('inf')
            print(f"  {r['s1']:>8.2f} {r['delta']:>12.6f} {r['s2']:>12.6f} "
                  f"{ratio:>10.6f} {r['lemma8_v']:>10.4f}")

    # Save
    def jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    out = {k: [{kk: jsonable(vv) for kk, vv in r.items()} for r in rs]
           for k, rs in all_data.items()}

    out_path = RESULTS_DIR / "lemma9_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
