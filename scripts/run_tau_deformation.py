"""
Phase 19.1 — τ-deformation probe for Bonnet tori.

Objective: verify whether k-periodic solutions exist at τ ≠ τ₀ for k=5,
and if so, measure ∂(δ, s₂)/∂τ_imag and characterize the solution branch.

Strategy:
  1. Point probes at τ_imag ± 0.01, ± 0.02, ± 0.05 (single Newton solve)
  2. Adaptive continuation sweep τ_imag ∈ [0.20, 0.45] (fine branch)
  3. Repeat for k=10, 20 to test k-dependence
  4. Compute derivatives ∂δ/∂τ, ∂s₂/∂τ via finite differences + branch data
  5. Measure birapporto of spectral curve along the branch

Infrastructure: uses solve_theorem7_local_fixed_tau_s1() and
continue_theorem7_branch_adaptive_fixed_s1() from theorem7_periodicity.py.

Precedent: Phase 18.10 Atlas already showed convergence for k=3,4 over
τ_imag ∈ [0.2705, 0.3287] — 25 successful points.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Force unbuffered output for background terminal monitoring
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seed_catalog import SEEDS, TAU_IMAG
from src.theorem7_periodicity import (
    solve_theorem7_local_fixed_tau_s1,
    continue_theorem7_branch_adaptive_fixed_s1,
    theorem7_residuals,
)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "tau_deformation"

# Seeds to probe (different gauges, different k)
PROBE_SEEDS = {
    5: SEEDS[5],     # s₁=-3.80, paper gauge
    10: SEEDS[10],   # s₁=-8.50, Phase 15 gauge
    20: None,        # will load from full_series
}

# Point probes: τ_imag = TAU_IMAG + offset
# Note: ω bracket fails near τ_imag ≈ 0.37 for s₁=-3.8, so keep +0.05 as test
PROBE_OFFSETS = [-0.05, -0.02, -0.01, -0.005, +0.005, +0.01, +0.02, +0.05]

# Continuation sweep range — conservative upper limit (ω bracket issue above ~0.36)
SWEEP_TAU_MIN = 0.20
SWEEP_TAU_MAX = 0.42

# ============================================================================
# Load k=20 seed from full_series
# ============================================================================

def load_k20_seed():
    """Load k=20 spectral parameters from full_series JSON."""
    json_path = Path(__file__).resolve().parent.parent / "data" / "full_series_k3_1000.json"
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found, skipping k=20")
        return None
    with open(json_path) as f:
        data = json.load(f)
    lookup = {d["k"]: d for d in data}
    if 20 not in lookup:
        print(f"  WARNING: k=20 not found in JSON, skipping")
        return None
    d = lookup[20]
    return {
        "label": "full_series_20fold",
        "delta": d["delta"],
        "s1": d["s1"],
        "s2": d["s2"],
        "theta_deg": 360.0 / 20,
        "source": "phase15_full_series",
    }

# ============================================================================
# Part 1: Point probes
# ============================================================================

def run_point_probes(k: int, seed: dict) -> list[dict]:
    """Probe solver convergence at τ₀ ± offsets for a given k."""
    results = []
    s1 = seed["s1"]
    delta0 = seed["delta"]
    s2_0 = seed["s2"]

    print(f"\n{'='*70}")
    print(f"  POINT PROBES — k={k}  s₁={s1}  δ₀={delta0:.6f}  s₂₀={s2_0:.6f}")
    print(f"  τ₀ = 0.5 + {TAU_IMAG}i")
    print(f"{'='*70}")

    # First, confirm baseline at τ₀
    print(f"\n  [baseline] τ_imag = {TAU_IMAG:.10f} (τ₀)")
    t0 = time.time()
    sol0 = solve_theorem7_local_fixed_tau_s1(
        tau_imag=TAU_IMAG, s1=s1,
        initial_delta=delta0, initial_s2=s2_0,
        symmetry_fold=k,
    )
    dt = time.time() - t0
    print(f"    success={sol0.success}  residual={sol0.residual_norm:.2e}  "
          f"δ={sol0.delta:.10f}  s₂={sol0.s2:.10f}  nfev={sol0.nfev}  ({dt:.2f}s)")
    results.append({
        "k": k, "tau_imag": TAU_IMAG, "offset": 0.0,
        "success": sol0.success, "residual": sol0.residual_norm,
        "delta": sol0.delta, "s2": sol0.s2, "nfev": sol0.nfev, "time": dt,
        "type": "baseline",
    })

    # Probes at offsets
    for offset in PROBE_OFFSETS:
        tau_p = TAU_IMAG + offset
        if tau_p <= 0.01:  # sanity: τ_imag must be positive
            continue
        print(f"\n  [probe] τ_imag = {tau_p:.10f}  (Δτ = {offset:+.3f})")
        t0 = time.time()
        try:
            sol = solve_theorem7_local_fixed_tau_s1(
                tau_imag=tau_p, s1=s1,
                initial_delta=delta0, initial_s2=s2_0,
                symmetry_fold=k,
            )
            dt = time.time() - t0

            if sol.success and sol.residual_norm < 1e-6:
                status = "✓ CONVERGED"
                Δδ = sol.delta - sol0.delta
                Δs2 = sol.s2 - sol0.s2
                # Check for branch jump: if |Δδ| >> expected, flag it
                branch_jump = abs(Δδ) > 0.3 or sol.nfev > 40
                if branch_jump:
                    status = "⚠ BRANCH JUMP"
                print(f"    {status}  residual={sol.residual_norm:.2e}  "
                      f"δ={sol.delta:.10f} (Δδ={Δδ:+.8f})  "
                      f"s₂={sol.s2:.10f} (Δs₂={Δs2:+.8f})  nfev={sol.nfev}  ({dt:.2f}s)")
            else:
                status = "✗ FAILED"
                print(f"    {status}  residual={sol.residual_norm:.2e}  "
                      f"message={sol.message}  nfev={sol.nfev}  ({dt:.2f}s)")

            results.append({
                "k": k, "tau_imag": tau_p, "offset": offset,
                "success": sol.success and sol.residual_norm < 1e-6,
                "residual": sol.residual_norm,
                "delta": sol.delta, "s2": sol.s2, "nfev": sol.nfev, "time": dt,
                "type": "probe",
            })
        except Exception as exc:
            dt = time.time() - t0
            print(f"    ✗ EXCEPTION: {type(exc).__name__}: {exc}  ({dt:.2f}s)")
            results.append({
                "k": k, "tau_imag": tau_p, "offset": offset,
                "success": False, "residual": float('inf'),
                "delta": float('nan'), "s2": float('nan'), "nfev": 0, "time": dt,
                "type": "probe", "error": str(exc),
            })

    return results


# ============================================================================
# Part 2: Adaptive continuation sweep
# ============================================================================

def run_continuation_sweep(k: int, seed: dict) -> dict:
    """Run adaptive continuation in τ_imag for a given k."""
    s1 = seed["s1"]
    delta0 = seed["delta"]
    s2_0 = seed["s2"]

    print(f"\n{'='*70}")
    print(f"  CONTINUATION SWEEP — k={k}  τ_imag ∈ [{SWEEP_TAU_MIN}, {SWEEP_TAU_MAX}]")
    print(f"  s₁={s1}  starting from δ₀={delta0:.6f}  s₂₀={s2_0:.6f}")
    print(f"{'='*70}")

    sweep_results = {"k": k, "s1": s1, "forward": None, "backward": None}

    # Forward: τ₀ → SWEEP_TAU_MAX
    print(f"\n  [forward] τ₀={TAU_IMAG:.6f} → {SWEEP_TAU_MAX}")
    t0 = time.time()
    fwd = continue_theorem7_branch_adaptive_fixed_s1(
        start_tau_imag=TAU_IMAG,
        end_tau_imag=SWEEP_TAU_MAX,
        s1=s1,
        initial_delta=delta0, initial_s2=s2_0,
        symmetry_fold=k,
        initial_step=0.002, min_step=0.0003, max_step=0.005,
        max_steps=200,
    )
    dt_fwd = time.time() - t0
    n_ok = sum(1 for s in fwd.steps if s.corrector.success)
    n_fail = sum(1 for s in fwd.steps if not s.corrector.success)
    if fwd.steps:
        tau_reached = fwd.steps[-1].tau_imag if fwd.steps[-1].corrector.success else (
            max((s.tau_imag for s in fwd.steps if s.corrector.success), default=TAU_IMAG)
        )
    else:
        tau_reached = TAU_IMAG
    print(f"    {n_ok} converged, {n_fail} failed, τ reached: {tau_reached:.6f}  ({dt_fwd:.1f}s)")

    fwd_data = []
    for s in fwd.steps:
        if s.corrector.success:
            fwd_data.append({
                "tau_imag": s.tau_imag, "delta": s.corrector.delta, "s2": s.corrector.s2,
                "residual": s.corrector.residual_norm, "nfev": s.corrector.nfev,
            })
    sweep_results["forward"] = {
        "tau_start": TAU_IMAG, "tau_end": SWEEP_TAU_MAX, "tau_reached": tau_reached,
        "n_converged": n_ok, "n_failed": n_fail, "time": dt_fwd,
        "points": fwd_data,
    }

    # Backward: τ₀ → SWEEP_TAU_MIN
    print(f"\n  [backward] τ₀={TAU_IMAG:.6f} → {SWEEP_TAU_MIN}")
    t0 = time.time()
    bwd = continue_theorem7_branch_adaptive_fixed_s1(
        start_tau_imag=TAU_IMAG,
        end_tau_imag=SWEEP_TAU_MIN,
        s1=s1,
        initial_delta=delta0, initial_s2=s2_0,
        symmetry_fold=k,
        initial_step=0.002, min_step=0.0003, max_step=0.005,
        max_steps=200,
    )
    dt_bwd = time.time() - t0
    n_ok_b = sum(1 for s in bwd.steps if s.corrector.success)
    n_fail_b = sum(1 for s in bwd.steps if not s.corrector.success)
    if bwd.steps:
        tau_reached_b = bwd.steps[-1].tau_imag if bwd.steps[-1].corrector.success else (
            min((s.tau_imag for s in bwd.steps if s.corrector.success), default=TAU_IMAG)
        )
    else:
        tau_reached_b = TAU_IMAG
    print(f"    {n_ok_b} converged, {n_fail_b} failed, τ reached: {tau_reached_b:.6f}  ({dt_bwd:.1f}s)")

    bwd_data = []
    for s in bwd.steps:
        if s.corrector.success:
            bwd_data.append({
                "tau_imag": s.tau_imag, "delta": s.corrector.delta, "s2": s.corrector.s2,
                "residual": s.corrector.residual_norm, "nfev": s.corrector.nfev,
            })
    sweep_results["backward"] = {
        "tau_start": TAU_IMAG, "tau_end": SWEEP_TAU_MIN, "tau_reached": tau_reached_b,
        "n_converged": n_ok_b, "n_failed": n_fail_b, "time": dt_bwd,
        "points": bwd_data,
    }

    return sweep_results


# ============================================================================
# Part 3: Analysis — derivatives and branch characterization
# ============================================================================

def analyze_branch(probe_results: list[dict], sweep_results: dict) -> dict:
    """Compute derivatives and characterize the branch."""
    k = probe_results[0]["k"]
    analysis = {"k": k}

    # --- Finite difference derivatives from probes ---
    baseline = next(r for r in probe_results if r["type"] == "baseline")
    δ0 = baseline["delta"]
    s2_0 = baseline["s2"]
    τ0 = baseline["tau_imag"]

    converged_probes = [r for r in probe_results
                        if r["type"] == "probe" and r["success"]]

    if len(converged_probes) >= 2:
        # Use smallest symmetric pair for best derivative estimate
        offsets = sorted(set(abs(r["offset"]) for r in converged_probes))
        derivatives = []
        for h in offsets:
            plus = next((r for r in converged_probes if abs(r["offset"] - h) < 1e-10), None)
            minus = next((r for r in converged_probes if abs(r["offset"] + h) < 1e-10), None)
            if plus and minus:
                dδ_dτ = (plus["delta"] - minus["delta"]) / (2 * h)
                ds2_dτ = (plus["s2"] - minus["s2"]) / (2 * h)
                # Second derivative (central)
                d2δ_dτ2 = (plus["delta"] - 2 * δ0 + minus["delta"]) / (h ** 2)
                d2s2_dτ2 = (plus["s2"] - 2 * s2_0 + minus["s2"]) / (h ** 2)
                derivatives.append({
                    "h": h,
                    "dδ_dτ": dδ_dτ, "ds₂_dτ": ds2_dτ,
                    "d²δ_dτ²": d2δ_dτ2, "d²s₂_dτ²": d2s2_dτ2,
                })
        analysis["derivatives_fd"] = derivatives

    # --- Branch statistics from continuation ---
    all_pts = []
    if sweep_results["backward"] and sweep_results["backward"]["points"]:
        all_pts.extend(reversed(sweep_results["backward"]["points"]))
    all_pts.append({"tau_imag": τ0, "delta": δ0, "s2": s2_0, "residual": 0.0})
    if sweep_results["forward"] and sweep_results["forward"]["points"]:
        all_pts.extend(sweep_results["forward"]["points"])

    if len(all_pts) >= 3:
        taus = np.array([p["tau_imag"] for p in all_pts])
        deltas = np.array([p["delta"] for p in all_pts])
        s2s = np.array([p["s2"] for p in all_pts])

        # δ range
        analysis["tau_range"] = [float(taus.min()), float(taus.max())]
        analysis["delta_range"] = [float(deltas.min()), float(deltas.max())]
        analysis["s2_range"] = [float(s2s.min()), float(s2s.max())]
        analysis["n_branch_points"] = len(all_pts)

        # Find minimum δ
        idx_min = int(np.argmin(deltas))
        analysis["delta_min"] = {
            "tau_imag": float(taus[idx_min]),
            "delta": float(deltas[idx_min]),
            "s2": float(s2s[idx_min]),
            "is_at_tau0": abs(taus[idx_min] - TAU_IMAG) < 0.005,
        }

        # Parabolic fit δ(τ) near minimum
        if len(all_pts) >= 5:
            coeffs_δ = np.polyfit(taus - TAU_IMAG, deltas, 2)
            coeffs_s2 = np.polyfit(taus - TAU_IMAG, s2s, 2)
            analysis["parabolic_fit_delta"] = {
                "a2": float(coeffs_δ[0]),
                "a1": float(coeffs_δ[1]),
                "a0": float(coeffs_δ[2]),
                "tau_min_pred": float(-coeffs_δ[1] / (2 * coeffs_δ[0])) + TAU_IMAG if abs(coeffs_δ[0]) > 1e-12 else None,
            }
            analysis["parabolic_fit_s2"] = {
                "a2": float(coeffs_s2[0]),
                "a1": float(coeffs_s2[1]),
                "a0": float(coeffs_s2[2]),
            }

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 19.1 — τ-deformation Probe")
    print(f"  τ₀ = 0.5 + {TAU_IMAG}i")
    print(f"  Point probes: Δτ = {PROBE_OFFSETS}")
    print(f"  Continuation: τ_imag ∈ [{SWEEP_TAU_MIN}, {SWEEP_TAU_MAX}]")
    print(f"  k values: 5, 10, 20")
    print("=" * 70)

    # Load k=20 seed
    k20_seed = load_k20_seed()
    PROBE_SEEDS[20] = k20_seed

    all_results = {}
    t_global = time.time()

    for k in [5, 10, 20]:
        seed = PROBE_SEEDS.get(k)
        if seed is None:
            print(f"\n  *** Skipping k={k} — no seed available ***")
            continue

        print(f"\n{'#'*70}")
        print(f"  k = {k}")
        print(f"{'#'*70}")

        # Part 1: Point probes
        probe_res = run_point_probes(k, seed)

        # Part 2: Continuation sweep (only for k=5, main target)
        if k == 5:
            sweep_res = run_continuation_sweep(k, seed)
        else:
            # For k=10,20 do a shorter sweep ±0.05 only
            print(f"\n  [short sweep] τ_imag ∈ [{TAU_IMAG-0.05:.4f}, {TAU_IMAG+0.05:.4f}]")
            t0 = time.time()
            fwd = continue_theorem7_branch_adaptive_fixed_s1(
                start_tau_imag=TAU_IMAG, end_tau_imag=TAU_IMAG + 0.05,
                s1=seed["s1"],
                initial_delta=seed["delta"], initial_s2=seed["s2"],
                symmetry_fold=k,
                initial_step=0.002, min_step=0.0003, max_step=0.005,
                max_steps=50,
            )
            bwd = continue_theorem7_branch_adaptive_fixed_s1(
                start_tau_imag=TAU_IMAG, end_tau_imag=TAU_IMAG - 0.05,
                s1=seed["s1"],
                initial_delta=seed["delta"], initial_s2=seed["s2"],
                symmetry_fold=k,
                initial_step=0.002, min_step=0.0003, max_step=0.005,
                max_steps=50,
            )
            dt = time.time() - t0
            n_ok_f = sum(1 for s in fwd.steps if s.corrector.success)
            n_ok_b = sum(1 for s in bwd.steps if s.corrector.success)
            print(f"    fwd: {n_ok_f} pts, bwd: {n_ok_b} pts  ({dt:.1f}s)")
            sweep_res = {
                "k": k, "s1": seed["s1"],
                "forward": {
                    "tau_start": TAU_IMAG, "tau_end": TAU_IMAG + 0.05,
                    "n_converged": n_ok_f, "time": dt / 2,
                    "points": [{"tau_imag": s.tau_imag, "delta": s.corrector.delta, "s2": s.corrector.s2,
                                "residual": s.corrector.residual_norm, "nfev": s.corrector.nfev}
                               for s in fwd.steps if s.corrector.success],
                },
                "backward": {
                    "tau_start": TAU_IMAG, "tau_end": TAU_IMAG - 0.05,
                    "n_converged": n_ok_b, "time": dt / 2,
                    "points": [{"tau_imag": s.tau_imag, "delta": s.corrector.delta, "s2": s.corrector.s2,
                                "residual": s.corrector.residual_norm, "nfev": s.corrector.nfev}
                               for s in bwd.steps if s.corrector.success],
                },
            }

        # Part 3: Analysis
        analysis = analyze_branch(probe_res, sweep_res)

        all_results[k] = {
            "seed": {kk: vv for kk, vv in seed.items() if kk != "note"},
            "probes": probe_res,
            "sweep": sweep_res,
            "analysis": analysis,
        }

        # Print analysis summary
        print(f"\n  {'─'*60}")
        print(f"  ANALYSIS SUMMARY — k={k}")
        print(f"  {'─'*60}")
        n_conv = sum(1 for r in probe_res if r["success"])
        print(f"  Probes: {n_conv}/{len(probe_res)} converged")
        if "derivatives_fd" in analysis:
            for d in analysis["derivatives_fd"]:
                print(f"    h={d['h']:.3f}: ∂δ/∂τ={d['dδ_dτ']:+.6f}  ∂s₂/∂τ={d['ds₂_dτ']:+.6f}  "
                      f"∂²δ/∂τ²={d['d²δ_dτ²']:+.4f}  ∂²s₂/∂τ²={d['d²s₂_dτ²']:+.4f}")
        if "n_branch_points" in analysis:
            print(f"  Branch: {analysis['n_branch_points']} pts, "
                  f"τ ∈ [{analysis['tau_range'][0]:.4f}, {analysis['tau_range'][1]:.4f}]")
            print(f"  δ ∈ [{analysis['delta_range'][0]:.6f}, {analysis['delta_range'][1]:.6f}]")
            print(f"  s₂ ∈ [{analysis['s2_range'][0]:.6f}, {analysis['s2_range'][1]:.6f}]")
        if "delta_min" in analysis:
            dm = analysis["delta_min"]
            at_tau0 = "YES" if dm["is_at_tau0"] else "NO"
            print(f"  δ_min = {dm['delta']:.6f} at τ_imag = {dm['tau_imag']:.6f} (at τ₀? {at_tau0})")
        if "parabolic_fit_delta" in analysis:
            pf = analysis["parabolic_fit_delta"]
            print(f"  Parabolic fit: δ(τ) ≈ {pf['a2']:.2f}·(τ−τ₀)² + {pf['a1']:+.4f}·(τ−τ₀) + {pf['a0']:.6f}")
            if pf.get("tau_min_pred"):
                print(f"  Predicted minimum: τ_imag = {pf['tau_min_pred']:.6f}")

    # Save all results
    output_path = OUTPUT_DIR / "phase19_1_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    # Final summary table
    t_total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  Phase 19.1 COMPLETE — {t_total:.1f}s")
    print(f"{'='*70}")
    print(f"\n  {'k':>4s}  {'probes':>8s}  {'branch':>8s}  {'τ_range':>20s}  {'δ_min':>9s}  {'∂δ/∂τ':>10s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*20}  {'─'*9}  {'─'*10}")
    for k_val in [5, 10, 20]:
        if k_val not in all_results:
            continue
        a = all_results[k_val]["analysis"]
        probes = all_results[k_val]["probes"]
        n_conv = sum(1 for r in probes if r["success"])
        n_branch = a.get("n_branch_points", 0)
        τ_range = f"[{a['tau_range'][0]:.4f}, {a['tau_range'][1]:.4f}]" if "tau_range" in a else "—"
        δ_min = f"{a['delta_min']['delta']:.6f}" if "delta_min" in a else "—"
        dδ = "—"
        if "derivatives_fd" in a and a["derivatives_fd"]:
            dδ = f"{a['derivatives_fd'][0]['dδ_dτ']:+.6f}"
        print(f"  {k_val:>4d}  {n_conv:>4d}/{len(probes):<3d}  {n_branch:>8d}  {τ_range:>20s}  {δ_min:>9s}  {dδ:>10s}")

    # Key question answers
    print(f"\n  KEY QUESTIONS:")
    for k_val in [5, 10, 20]:
        if k_val not in all_results:
            continue
        probes = all_results[k_val]["probes"]
        non_baseline = [r for r in probes if r["type"] == "probe"]
        any_converged = any(r["success"] for r in non_baseline)
        print(f"    k={k_val}: solver converges at τ ≠ τ₀? {'YES ✓' if any_converged else 'NO ✗'}")
        if "delta_min" in all_results[k_val]["analysis"]:
            dm = all_results[k_val]["analysis"]["delta_min"]
            if dm["is_at_tau0"]:
                print(f"         δ minimum at τ₀? YES — τ₀ is special")
            else:
                tau_min_val = dm["tau_imag"]
                print(f"         δ minimum at τ₀? NO — minimum shifted to τ={tau_min_val:.6f}")


if __name__ == "__main__":
    main()
