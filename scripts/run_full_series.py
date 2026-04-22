#!/usr/bin/env python
"""Reproduce the full Bonnet-pair dataset (k=3..1000) from scratch.

Usage (from repo root):
    python scripts/run_full_series.py

Output:
    results/phase15_asymptotic/full_series_k3_1000.json

Compatibility mirror:
    data/full_series_k3_1000.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theorem7_periodicity import solve_theorem7_local_fixed_tau_s1

TAU_IMAG = 0.3205128205
S1 = -8.5

# Bootstrap guess for k=3 (hand-tuned from original exploration)
DELTA_INIT = 3.20
S2_INIT = 2.50


def main():
    project = Path(__file__).resolve().parent.parent
    out_dir = project / "results" / "phase15_asymptotic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_series_k3_1000.json"
    mirror_dir = project / "data"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_path = mirror_dir / "full_series_k3_1000.json"

    results = []
    delta_guess = DELTA_INIT
    s2_guess = S2_INIT
    failures = []

    t0 = time.time()
    for k in range(3, 1001):
        sol = solve_theorem7_local_fixed_tau_s1(
            tau_imag=TAU_IMAG,
            s1=S1,
            initial_delta=delta_guess,
            initial_s2=s2_guess,
            symmetry_fold=k,
            target_ratio=1.0,
        )
        if sol.success and sol.residual_norm < 1e-6:
            entry = {
                "k": k,
                "s1": S1,
                "delta": sol.delta,
                "s2": sol.s2,
                "residual": sol.residual_norm,
                "theta_deg": math.degrees(sol.theorem7.theta),
                "ratio": sol.theorem7.ratio,
                "nfev": sol.nfev,
            }
            results.append(entry)
            # continuation: use current solution as next guess
            delta_guess = sol.delta
            s2_guess = sol.s2
        else:
            failures.append(k)
            print(f"  WARNING: k={k} failed (residual={sol.residual_norm:.2e})")

        if k % 100 == 0:
            elapsed = time.time() - t0
            print(f"  k={k:4d}  seeds={len(results):4d}  "
                  f"failures={len(failures)}  elapsed={elapsed:.1f}s")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    with open(mirror_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} seeds written to {out_path}")
    print(f"Mirror: {mirror_path}")
    print(f"Failures: {len(failures)} — {failures}")
    print(f"Total time: {elapsed:.1f}s")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
