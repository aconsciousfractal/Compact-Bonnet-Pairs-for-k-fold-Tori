#!/usr/bin/env python3
"""Re-solve the critical rows of the 998-seed phase-15 branch file.

Critical rows are those for which the checked-in archive either:
- stores placeholder metadata (`residual = 0`, `nfev = 0`), or
- stores a residual norm >= 1e-10.

The script re-solves those rows from their archived `(delta, s2)` values with
the current local Theorem 7 solver and exports a per-row CSV together with a
summary JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.theorem7_periodicity import solve_theorem7_local_fixed_tau_s1


TAU_IMAG = 0.3205128205
S1 = -8.5
THRESHOLD = 1e-10

PHASE15_PATH = PROJECT / "results" / "phase15_asymptotic" / "full_series_k3_1000.json"
OUT_DIR = PROJECT / "results" / "phase15_asymptotic" / "critical_recheck"
OUT_CSV = OUT_DIR / "phase15_critical_recheck.csv"
OUT_JSON = OUT_DIR / "phase15_critical_recheck_summary.json"


def load_phase15() -> list[dict]:
    return json.loads(PHASE15_PATH.read_text(encoding="utf-8"))


def main() -> None:
    rows = load_phase15()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stored_zero = [row for row in rows if float(row["residual"]) == 0.0]
    stored_ge_threshold = [row for row in rows if float(row["residual"]) >= THRESHOLD]
    stored_good = [row for row in rows if 0.0 < float(row["residual"]) < THRESHOLD]
    critical_rows = [
        row for row in rows if float(row["residual"]) == 0.0 or float(row["residual"]) >= THRESHOLD
    ]

    records: list[dict] = []
    max_inf = (-1.0, None)
    max_l2 = (-1.0, None)
    max_nfev = (-1, None)
    inf_gt_threshold = 0
    l2_gt_threshold = 0
    failures = 0

    for row in critical_rows:
        k = int(row["k"])
        sol = solve_theorem7_local_fixed_tau_s1(
            tau_imag=TAU_IMAG,
            s1=S1,
            initial_delta=float(row["delta"]),
            initial_s2=float(row["s2"]),
            symmetry_fold=k,
            target_ratio=1.0,
        )
        residual_inf = float(np.max(np.abs(sol.residual_vector)))
        residual_l2 = float(sol.residual_norm)
        delta_shift = abs(float(sol.delta) - float(row["delta"]))
        s2_shift = abs(float(sol.s2) - float(row["s2"]))

        if not sol.success:
            failures += 1
        if residual_inf > THRESHOLD:
            inf_gt_threshold += 1
        if residual_l2 > THRESHOLD:
            l2_gt_threshold += 1
        if residual_inf > max_inf[0]:
            max_inf = (residual_inf, k)
        if residual_l2 > max_l2[0]:
            max_l2 = (residual_l2, k)
        if int(sol.nfev) > max_nfev[0]:
            max_nfev = (int(sol.nfev), k)

        records.append(
            {
                "k": k,
                "archive_delta": float(row["delta"]),
                "archive_s2": float(row["s2"]),
                "archive_residual_l2": float(row["residual"]),
                "archive_nfev": int(row["nfev"]),
                "resolved_success": bool(sol.success),
                "resolved_delta": float(sol.delta),
                "resolved_s2": float(sol.s2),
                "resolved_residual_inf": residual_inf,
                "resolved_residual_l2": residual_l2,
                "resolved_nfev": int(sol.nfev),
                "delta_shift": delta_shift,
                "s2_shift": s2_shift,
            }
        )

    records.sort(key=lambda item: item["k"])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "k",
                "archive_delta",
                "archive_s2",
                "archive_residual_l2",
                "archive_nfev",
                "resolved_success",
                "resolved_delta",
                "resolved_s2",
                "resolved_residual_inf",
                "resolved_residual_l2",
                "resolved_nfev",
                "delta_shift",
                "s2_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    worst_inf = sorted(records, key=lambda item: item["resolved_residual_inf"], reverse=True)[:20]
    worst_l2 = sorted(records, key=lambda item: item["resolved_residual_l2"], reverse=True)[:20]

    summary = {
        "source": "results/phase15_asymptotic/full_series_k3_1000.json",
        "tau_imag": TAU_IMAG,
        "s1": S1,
        "threshold": THRESHOLD,
        "total_rows": len(rows),
        "stored_nonzero_lt_threshold": len(stored_good),
        "stored_zero_rows": len(stored_zero),
        "stored_ge_threshold_rows": len(stored_ge_threshold),
        "critical_row_count": len(critical_rows),
        "critical_resolve_failures": failures,
        "critical_resolved_inf_gt_threshold": inf_gt_threshold,
        "critical_resolved_l2_gt_threshold": l2_gt_threshold,
        "critical_max_residual_inf": {
            "value": max_inf[0],
            "k": max_inf[1],
        },
        "critical_max_residual_l2": {
            "value": max_l2[0],
            "k": max_l2[1],
        },
        "critical_max_nfev": {
            "value": max_nfev[0],
            "k": max_nfev[1],
        },
        "worst_resolved_inf_rows": worst_inf,
        "worst_resolved_l2_rows": worst_l2,
        "csv": str(OUT_CSV.relative_to(PROJECT)),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
