#!/usr/bin/env python3
"""
LEGACY / auxiliary runner.

P2 — Universality of the high-fold 1/2 decay exponent across lattice parameters τ.

For each τ we:
  1. Find ω_crit(τ) = zero of ϑ₂'(·|τ)
  2. Solve the periodicity system for k = 3..200
  3. Compute running exponent α(k) = log(δ_k/δ_{k-1}) / log((k-1)/k)
  4. Check α → 0.500 and extract A(τ)

τ values chosen to span diverse geometry:
  - τ₀ = 0.5 + 0.3205i  (baseline, 25/78)
  - τ₁ = 0.5 + 0.20i    (narrow, Im(τ) small)
  - τ₂ = 0.5 + 0.40i    (wide, Im(τ) large)
  - τ₃ = 0.5 + 0.25i    (intermediate)
  - τ₄ = 0.5 + 0.50i    (very wide)
"""
import json, math, sys, time, numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT))

from src.theorem7_periodicity import (
    _paper_constants, _real_scalar,
    solve_theorem7_local_fixed_tau_s1,
)
from src import theta_functions as TF
from src.seed_catalog import SEEDS as KNOWN_SEEDS

# === Configuration ===
S1 = -8.5  # fixed gauge
K_MAX = 200  # solve up to this k
TAU_IMAG_VALUES = [0.3205128205, 0.25, 0.30, 0.20, 0.35]  # baseline first, then by distance
# NOTE: omega_crit ceases to exist for tau_imag >= ~0.37 (theta2' < 0 everywhere)

FULL_SERIES_CANDIDATES = [
    PROJECT / 'results' / 'phase15_asymptotic' / 'full_series_k3_1000.json',
    PROJECT / 'data' / 'full_series_k3_1000.json',
]


def load_baseline_seeds():
    for data_path in FULL_SERIES_CANDIDATES:
        if data_path.exists():
            with open(data_path) as f:
                return {int(e['k']): (float(e['delta']), float(e['s2'])) for e in json.load(f)}
    raise FileNotFoundError(
        'No baseline full_series_k3_1000.json found in canonical results path or compatibility mirror.'
    )


def find_seed_via_tau_continuation(target_tau_imag, s1, k_val=3, n_steps=40):
    """
    Find seed at a new tau by continuation from the baseline tau.
    Walk tau_imag from 0.3205128205 to target_tau_imag in small steps,
    using the previous solution as initial guess at each step.
    """
    BASELINE_TAU = 0.3205128205
    
    # Load baseline k=k_val solution
    seeds = load_baseline_seeds()
    
    if k_val not in seeds:
        print(f"    No baseline seed for k={k_val}")
        return None
    
    d_cur, s2_cur = seeds[k_val]
    
    if abs(target_tau_imag - BASELINE_TAU) < 1e-10:
        return d_cur, s2_cur
    
    # Walk in n_steps from baseline to target
    tau_steps = np.linspace(BASELINE_TAU, target_tau_imag, n_steps + 1)[1:]
    
    for i, tau_im in enumerate(tau_steps):
        try:
            res = solve_theorem7_local_fixed_tau_s1(
                tau_imag=float(tau_im), s1=s1,
                initial_delta=float(d_cur), initial_s2=float(s2_cur),
                symmetry_fold=k_val,
            )
            if res.success and res.residual_norm < 1e-6:
                d_cur, s2_cur = res.delta, res.s2
            else:
                # Halve step size and retry
                tau_prev = BASELINE_TAU if i == 0 else tau_steps[i-1]
                for sub_tau in np.linspace(tau_prev, tau_im, 6)[1:]:
                    res = solve_theorem7_local_fixed_tau_s1(
                        tau_imag=float(sub_tau), s1=s1,
                        initial_delta=float(d_cur), initial_s2=float(s2_cur),
                        symmetry_fold=k_val,
                    )
                    if res.success and res.residual_norm < 1e-6:
                        d_cur, s2_cur = res.delta, res.s2
                    else:
                        print(f"    τ-continuation failed at step {i+1}/{n_steps}, tau_imag={tau_im:.6f}")
                        return None
        except Exception as e:
            print(f"    τ-continuation error at step {i+1}: {e}")
            return None
    
    return d_cur, s2_cur


def find_seed_from_known(tau_imag, s1, k_val):
    """Try to find seed by continuing from known seeds (baseline τ)."""
    # For baseline τ, load from the full series
    if abs(tau_imag - 0.3205128205) < 1e-6:
        seeds = load_baseline_seeds()
        if k_val in seeds:
            return seeds[k_val]
    return None


def run_series_for_tau(tau_imag, s1=S1, k_max=K_MAX):
    """Solve k=3..k_max for a given tau_imag at fixed s1."""
    tau = 0.5 + tau_imag * 1j
    omega = _real_scalar(TF.find_critical_omega(tau))
    
    print(f"\n{'='*72}")
    print(f"  tau = 0.5 + {tau_imag}i,  omega_crit = {omega:.10f},  s1 = {s1}")
    print(f"{'='*72}")
    
    results = {}
    
    # Check if we already have baseline data
    known = find_seed_from_known(tau_imag, s1, 3)
    
    # --- Find k=3 seed ---
    if known:
        d_cur, s2_cur = known
        print(f"  k=3: loaded from database: delta={d_cur:.10f} s2={s2_cur:.8f}")
        results[3] = {'k': 3, 'delta': d_cur, 's2': s2_cur, 'residual': 0.0}
    else:
        print(f"  Finding k=3 seed via τ-continuation ...", end=" ", flush=True)
        seed_result = find_seed_via_tau_continuation(tau_imag, s1, k_val=3)
        if seed_result is None:
            print("FAILED — τ-continuation did not converge")
            return None
        d_cur, s2_cur = seed_result
        # Verify by solving
        res = solve_theorem7_local_fixed_tau_s1(
            tau_imag=tau_imag, s1=s1,
            initial_delta=float(d_cur), initial_s2=float(s2_cur),
            symmetry_fold=3,
        )
        if res.success and res.residual_norm < 1e-6:
            d_cur, s2_cur = res.delta, res.s2
            print(f"found: delta={d_cur:.10f} s2={s2_cur:.8f} |r|={res.residual_norm:.2e}")
            results[3] = {'k': 3, 'delta': d_cur, 's2': s2_cur, 'residual': res.residual_norm}
        else:
            print(f"FAILED — verification solve gave |r|={res.residual_norm:.2e}")
            return None
    
    # --- Continue k=4..k_max ---
    # For k=4,5 at non-baseline τ, use τ-continuation for safety
    is_baseline = abs(tau_imag - 0.3205128205) < 1e-6
    fail_count = 0
    for k in range(4, k_max + 1):
        # Check if we have a pre-loaded seed
        known_k = find_seed_from_known(tau_imag, s1, k)
        if known_k:
            d_cur, s2_cur = known_k
            results[k] = {'k': k, 'delta': d_cur, 's2': s2_cur, 'residual': 0.0}
            if k % 50 == 0:
                print(f"  k={k}: loaded delta={d_cur:.10f} s2={s2_cur:.8f}")
            continue
        
        # For first few k at non-baseline τ, use τ-continuation
        if not is_baseline and k <= 5:
            seed_k = find_seed_via_tau_continuation(tau_imag, s1, k_val=k)
            if seed_k:
                d_guess, s2_guess = seed_k
            else:
                d_guess = d_cur * 0.85
                s2_guess = s2_cur * 1.01
        elif k >= 6 and (k-1) in results and (k-2) in results:
            # Secant extrapolation
            d_prev = results[k-1]['delta']
            d_prev2 = results[k-2]['delta']
            s2_prev = results[k-1]['s2']
            s2_prev2 = results[k-2]['s2']
            d_guess = 2*d_prev - d_prev2
            s2_guess = 2*s2_prev - s2_prev2
        else:
            d_guess = d_cur * 0.95  # rough: delta decreases
            s2_guess = s2_cur * 1.01  # rough: s2 increases slowly
        
        try:
            res = solve_theorem7_local_fixed_tau_s1(
                tau_imag=tau_imag, s1=s1,
                initial_delta=float(d_guess), initial_s2=float(s2_guess),
                symmetry_fold=k,
            )
            if res.success and res.residual_norm < 1e-6:
                d_cur, s2_cur = res.delta, res.s2
                results[k] = {'k': k, 'delta': d_cur, 's2': s2_cur, 'residual': res.residual_norm}
                fail_count = 0
                if k % 20 == 0 or k <= 10:
                    print(f"  k={k}: delta={d_cur:.10f} s2={s2_cur:.8f} |r|={res.residual_norm:.2e}")
            else:
                print(f"  k={k}: FAILED (res_norm={res.residual_norm:.2e})")
                fail_count += 1
                if fail_count >= 3:
                    print(f"  Stopping after 3 consecutive failures at k={k}")
                    break
        except Exception as e:
            print(f"  k={k}: ERROR: {e}")
            fail_count += 1
            if fail_count >= 3:
                print(f"  Stopping after 3 consecutive failures at k={k}")
                break
    
    return results


def analyse_series(results, tau_imag):
    """Compute running exponent, Richardson A, and report."""
    ks = sorted(results.keys())
    if len(ks) < 10:
        print(f"  Too few points ({len(ks)}) for meaningful analysis")
        return None
    
    # Running exponent: α(k) from δ_k·√k = A_k
    A_k = {k: results[k]['delta'] * math.sqrt(k) for k in ks}
    
    # Running exponent from consecutive ratio
    alphas = {}
    for i in range(1, len(ks)):
        k1, k2 = ks[i-1], ks[i]
        if k2 == k1 + 1:
            d1 = results[k1]['delta']
            d2 = results[k2]['delta']
            if d1 > 0 and d2 > 0:
                alphas[k2] = math.log(d2/d1) / math.log(k1/k2)
    
    # Report running exponent convergence
    print(f"\n  Running exponent α(k) for tau_imag = {tau_imag}:")
    checkpoints = [10, 20, 50, 100, 150, 200]
    for kc in checkpoints:
        if kc in alphas:
            print(f"    α({kc}) = {alphas[kc]:.6f}  (deviation from 0.5: {alphas[kc] - 0.5:+.6f})")
    
    # A_k = δ_k·√k should converge
    k_max = max(ks)
    A_last = A_k[k_max]
    print(f"\n  A_k = δ·√k at k={k_max}: {A_last:.8f}")
    
    # Richardson extrapolation (if enough points)
    if k_max >= 100:
        # Use last 5 points for polynomial extrapolation
        kk = np.array([k for k in ks if k >= k_max - 10], dtype=float)
        AA = np.array([A_k[int(k)] for k in kk])
        # Fit A_k = A_∞ + c₁/k
        X = np.column_stack([np.ones(len(kk)), 1.0/kk])
        coeffs = np.linalg.lstsq(X, AA, rcond=None)[0]
        A_rich = coeffs[0]
        print(f"  Richardson A_∞ (linear in 1/k): {A_rich:.8f}")
        
        # Also try quadratic: A_k = A_∞ + c₁/k + c₂/k²
        if len(kk) >= 4:
            X2 = np.column_stack([np.ones(len(kk)), 1.0/kk, 1.0/kk**2])
            coeffs2 = np.linalg.lstsq(X2, AA, rcond=None)[0]
            A_rich2 = coeffs2[0]
            print(f"  Richardson A_∞ (quadratic in 1/k): {A_rich2:.8f}")
    else:
        A_rich = A_last
        A_rich2 = A_last
    
    # Final exponent from last 20 consecutive pairs
    last_alphas = [alphas[k] for k in sorted(alphas.keys()) if k >= max(ks) - 20]
    if last_alphas:
        mean_alpha = np.mean(last_alphas)
        print(f"\n  Mean α over last {len(last_alphas)} points: {mean_alpha:.6f}")
        print(f"  Deviation from 0.500: {mean_alpha - 0.5:+.6f}")
    else:
        mean_alpha = float('nan')
    
    return {
        'tau_imag': tau_imag,
        'k_max': k_max,
        'n_seeds': len(ks),
        'A_last': A_last,
        'A_richardson': float(A_rich),
        'mean_alpha_last20': float(mean_alpha),
    }


# === Main ===
if __name__ == '__main__':
    print("=" * 80)
    print("P2 — UNIVERSALITY OF EXPONENT -1/2 ACROSS LATTICE PARAMETERS")
    print("=" * 80)
    print(f"  s1 = {S1},  k_max = {K_MAX}")
    print(f"  τ values: {['0.5+'+str(t)+'i' for t in TAU_IMAG_VALUES]}")
    
    OUT = PROJECT / 'data' / 'tau_universality'
    OUT.mkdir(parents=True, exist_ok=True)
    
    all_analyses = []
    
    for tau_imag in TAU_IMAG_VALUES:
        # Skip if already completed
        series_file = OUT / f'series_tau_{tau_imag:.4f}.json'
        if series_file.exists():
            print(f"\n  Skipping tau_imag={tau_imag} — already completed ({series_file.name})")
            with open(series_file) as f:
                cached = json.load(f)
            results = {int(e['k']): e for e in cached}
            analysis = analyse_series(results, tau_imag)
            if analysis:
                analysis['elapsed_sec'] = 0.0
                all_analyses.append(analysis)
            continue

        t0 = time.time()
        try:
            results = run_series_for_tau(tau_imag, S1, K_MAX)
        except ValueError as e:
            print(f"\n  FAILED for tau_imag = {tau_imag}: {e}")
            continue
        elapsed = time.time() - t0
        
        if results is None:
            print(f"\n  FAILED for tau_imag = {tau_imag}")
            continue
        
        # Save raw series
        export = [{'k': int(k), **{kk: float(vv) for kk, vv in v.items() if kk != 'k'}}
                  for k, v in sorted(results.items())]
        with open(series_file, 'w') as f:
            json.dump(export, f, indent=2)
        
        analysis = analyse_series(results, tau_imag)
        if analysis:
            analysis['elapsed_sec'] = elapsed
            all_analyses.append(analysis)
        
        print(f"\n  Elapsed: {elapsed:.1f}s")
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY — ALL τ VALUES")
    print("=" * 80)
    
    if all_analyses:
        print(f"\n  {'tau_imag':>10s}  {'k_max':>6s}  {'seeds':>6s}  {'A':>12s}  {'α_last20':>10s}  {'α+0.5':>10s}  {'time(s)':>8s}")
        print("  " + "-" * 70)
        for a in all_analyses:
            dev = a['mean_alpha_last20'] - 0.5
            marker = " ✓" if abs(dev) < 0.01 else " ✗"
            print(f"  {a['tau_imag']:10.4f}  {a['k_max']:6d}  {a['n_seeds']:6d}  "
                  f"{a['A_richardson']:12.6f}  {a['mean_alpha_last20']:10.6f}  {dev:+10.6f}  "
                  f"{a['elapsed_sec']:8.1f}{marker}")
        
        # Universality check
        alphas_all = [a['mean_alpha_last20'] for a in all_analyses if not math.isnan(a['mean_alpha_last20'])]
        if alphas_all:
            spread = max(alphas_all) - min(alphas_all)
            mean_all = np.mean(alphas_all)
            print(f"\n  Exponent spread across τ: {spread:.6f}")
            print(f"  Mean exponent: {mean_all:.6f}")
            print(f"  UNIVERSAL? {'YES (spread < 0.02)' if spread < 0.02 else 'INCONCLUSIVE'}")
    
    # Save summary
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"\nResults saved to {OUT}")
    print("=" * 80)
    print("DONE")
