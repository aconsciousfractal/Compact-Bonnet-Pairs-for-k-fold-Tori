"""
Phase E — Uniform sweep for k=8..24, 25, 50, 100.

IDENTICAL methodology to run_phase11_lemma9_sweep.py:
  - s₁ grid: np.arange(-10.0, -1.75, 0.25)   ← same as k=5,6,7
  - S1_STEP = 0.25                              ← same
  - RESIDUAL_THRESHOLD = 1e-6                   ← same
  - try_solve with full Lemma 8 checks          ← same
  - continue_in_s1 with 4-failure cutoff        ← same
  - Output fields: s1, delta, s2, residual, ratio, axial,
    theta_deg, lemma8_ok, lemma8_iii, lemma8_v, nfev, branch

Seeds from Phase 15 (full_series_k3_1000.json) at s₁=-8.5.
"""
import sys, math, json, time, signal
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

sys.path.insert(0, '.')
from src.theorem7_periodicity import (
    solve_theorem7_local_fixed_tau_s1,
    theorem7_residuals,
    theorem7_lemma8_real_sqrt_expr,
    theorem7_lemma8_rationality_expr,
    theorem7_lemma8_vanishing_expr,
)
from src import theta_functions as TF

SOLVE_TIMEOUT = 30  # seconds per Newton solve

RESULTS_DIR = Path("results/phaseE_extended")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAU_IMAG = 0.3205128205
tau = 0.5 + 1j * TAU_IMAG
omega = TF.find_critical_omega(tau)

# ── IDENTICAL to run_phase11_lemma9_sweep.py ──
RESIDUAL_THRESHOLD = 1e-6
S1_STEP = 0.25


def _solve_worker(args):
    """Worker for subprocess timeout."""
    k, s1, delta_guess, s2_guess = args
    # Re-import inside worker (for ProcessPool)
    import sys; sys.path.insert(0, '.')
    from src.theorem7_periodicity import (
        solve_theorem7_local_fixed_tau_s1 as _solve,
        theorem7_lemma8_real_sqrt_expr as _iii,
        theorem7_lemma8_rationality_expr as _iv,
        theorem7_lemma8_vanishing_expr as _v,
    )
    from src import theta_functions as _TF
    import math as _m
    _TAU = 0.3205128205
    _tau = 0.5 + 1j * _TAU
    _omega = _TF.find_critical_omega(_tau)
    sol = _solve(tau_imag=_TAU, s1=s1,
                 initial_delta=delta_guess, initial_s2=s2_guess,
                 symmetry_fold=k, target_ratio=1.0)
    if sol.success and sol.residual_norm < 1e-6:
        iii = _iii(_omega, _tau, s1, sol.s2)
        iv = _iv(_omega, _tau, s1, sol.s2)
        v = _v(_omega, _tau, s1, sol.s2)
        lemma8_ok = iii > 0 and abs(iv) > 1e-10 and abs(v) > 1e-10
        return {
            "s1": s1, "delta": sol.delta, "s2": sol.s2,
            "residual": sol.residual_norm,
            "ratio": sol.theorem7.ratio,
            "axial": sol.theorem7.axial_integral,
            "theta_deg": _m.degrees(sol.theorem7.theta),
            "lemma8_ok": lemma8_ok, "lemma8_iii": iii, "lemma8_v": v,
            "nfev": sol.nfev,
        }
    return None


def try_solve(k, s1, delta_guess, s2_guess):
    """Solve with timeout to avoid hanging near fold boundary."""
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_solve_worker, (k, s1, delta_guess, s2_guess))
            return future.result(timeout=SOLVE_TIMEOUT)
    except (FuturesTimeout, Exception):
        return None


def continue_in_s1(k, seed, s1_range, direction="forward"):
    """IDENTICAL to run_phase11_lemma9_sweep.continue_in_s1."""
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
                break
    return results


def sweep_k(k, seed):
    """Sweep s₁ for a single k. Same grid as original."""
    # ── IDENTICAL grid to run_phase11_lemma9_sweep.py line 117 ──
    s1_grid = np.arange(-10.0, -1.75, S1_STEP)

    print(f"  k={k}: ", end="", flush=True)

    backward = continue_in_s1(k, seed, s1_grid, direction="backward")
    print(f"←{len(backward)} ", end="", flush=True)

    forward = continue_in_s1(k, seed, s1_grid, direction="forward")
    print(f"→{len(forward)} ", end="", flush=True)

    seed_result = try_solve(k, seed["s1"], seed["delta"], seed["s2"])

    branch = list(reversed(backward))
    if seed_result:
        branch.append(seed_result)
    branch.extend(forward)
    for r in branch:
        r["branch"] = 0
    branch.sort(key=lambda r: r["s1"])

    if branch:
        s1s = [r["s1"] for r in branch]
        n_l8 = sum(1 for r in branch if r["lemma8_ok"])
        print(f"= {len(branch)} pts, s₁∈[{min(s1s):.2f},{max(s1s):.2f}], L8:{n_l8}/{len(branch)}",
              flush=True)
    else:
        print("= 0 pts", flush=True)

    return branch


def main():
    print("=" * 70)
    print("  PHASE E — UNIFORM SWEEP k = 8..24, 25, 50, 100")
    print(f"  Grid: s₁ ∈ [-10.0, -1.75], step = {S1_STEP}")
    print(f"  (IDENTICAL to run_phase11_lemma9_sweep.py)")
    print("=" * 70)

    # Load seeds from Phase 15
    phase15 = json.load(open("results/phase15_asymptotic/full_series_k3_1000.json"))
    by_k = {d["k"]: d for d in phase15}

    target_ks = list(range(8, 25)) + [25, 50, 100]
    n_total = len(target_ks)

    t0 = time.time()
    all_data = {}

    for idx, k in enumerate(target_ks, 1):
        elapsed = time.time() - t0
        eta = (elapsed / idx * (n_total - idx)) if idx > 1 else 0
        bar_len = 30
        filled = int(bar_len * idx / n_total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\n  [{bar}] {idx}/{n_total}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)
        s = by_k[k]
        seed = {"delta": s["delta"], "s1": s["s1"], "s2": s["s2"]}
        results = sweep_k(k, seed)
        all_data[str(k)] = results

    # ── Cross-fold comparison (same format as original) ──
    print(f"\n{'='*70}")
    print("  CROSS-FOLD EXISTENCE MAP")
    print("=" * 70)
    for k_str in sorted(all_data.keys(), key=lambda x: int(x)):
        k = int(k_str)
        results = all_data[k_str]
        if results:
            s1s = [r["s1"] for r in results]
            n_lemma8 = sum(1 for r in results if r["lemma8_ok"])
            print(f"  k={k:>3d}: {len(results):>3d} solutions, "
                  f"s₁ ∈ [{min(s1s):.2f}, {max(s1s):.2f}], "
                  f"Lemma 8 OK: {n_lemma8}/{len(results)}")
        else:
            print(f"  k={k:>3d}: no solutions found")

    # ── Transition analysis ──
    print(f"\n{'='*70}")
    print("  MONOTONICITY / SIMPLE-ARC ANALYSIS")
    print("=" * 70)
    print(f"  {'k':>4s} {'pts':>5s} {'s1_min':>8s} {'s1_max':>8s} "
          f"{'d_mono':>8s} {'s2_mono':>8s} {'arc(d,s2)':>10s} {'bump':>10s}")
    print("  " + "-" * 67)

    for k_str in sorted(all_data.keys(), key=lambda x: int(x)):
        k = int(k_str)
        pts = all_data[k_str]
        if len(pts) < 3:
            print(f"  {k:>4d} {len(pts):>5d}  — too few points —")
            continue

        s1 = np.array([p["s1"] for p in pts])
        delta = np.array([p["delta"] for p in pts])
        s2 = np.array([p["s2"] for p in pts])

        dd = np.diff(delta)
        ds2 = np.diff(s2)
        d_mono = bool(np.all(dd < 0) or np.all(dd > 0))
        s2_mono = bool(np.all(ds2 < 0) or np.all(ds2 > 0))

        # Self-intersection test in (delta, s2) plane
        coords = np.column_stack([delta, s2])
        n = len(coords)
        has_crossing = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                p1, p2 = coords[i], coords[i + 1]
                p3, p4 = coords[j], coords[j + 1]
                d1 = p2 - p1
                d2 = p4 - p3
                cross_val = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(cross_val) < 1e-15:
                    continue
                t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross_val
                u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross_val
                if 0 < t < 1 and 0 < u < 1:
                    has_crossing = True
                    break
            if has_crossing:
                break

        simple_arc = not has_crossing

        # Bump amplitude (for non-monotone delta)
        bump = 0.0
        if not d_mono:
            i_min = np.argmin(delta)
            if i_min < len(delta) - 1:
                bump = delta[-1] - delta[i_min]

        label_d = "YES" if d_mono else "NO"
        label_s2 = "YES" if s2_mono else "NO"
        label_arc = "YES" if simple_arc else "NO"
        bump_str = f"{bump:.6f}" if bump > 0 else "—"

        print(f"  {k:>4d} {len(pts):>5d} {s1[0]:>8.2f} {s1[-1]:>8.2f} "
              f"{label_d:>8s} {label_s2:>8s} {label_arc:>10s} {bump_str:>10s}")

    # ── Detailed turning-point report ──
    print(f"\n{'='*70}")
    print("  TURNING POINT DETAILS (non-monotone delta only)")
    print("=" * 70)
    for k_str in sorted(all_data.keys(), key=lambda x: int(x)):
        k = int(k_str)
        pts = all_data[k_str]
        if len(pts) < 3:
            continue
        s1 = np.array([p["s1"] for p in pts])
        delta = np.array([p["delta"] for p in pts])
        s2 = np.array([p["s2"] for p in pts])
        dd = np.diff(delta)
        d_mono = bool(np.all(dd < 0) or np.all(dd > 0))
        if d_mono:
            continue
        i_min = np.argmin(delta)
        print(f"\n  k={k}: delta_min={delta[i_min]:.6f} at s₁={s1[i_min]:.2f}, "
              f"delta_end={delta[-1]:.6f} at s₁={s1[-1]:.2f}, "
              f"bump={delta[-1]-delta[i_min]:.6f}")
        print(f"    Near-fold points:")
        print(f"    {'s1':>8s} {'delta':>12s} {'s2':>12s} {'L8':>4s}")
        start = max(0, len(pts) - 8)
        for i in range(start, len(pts)):
            marker = " <<<" if i == i_min else ""
            l8 = "OK" if pts[i]["lemma8_ok"] else "NO"
            print(f"    {s1[i]:>8.2f} {delta[i]:>12.6f} {s2[i]:>12.6f} {l8:>4s}{marker}")

    # ── Save (same format as lemma9_sweep.json) ──
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

    out_path = RESULTS_DIR / "uniform_sweep_k8_100.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
