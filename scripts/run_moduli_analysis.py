#!/usr/bin/env python3
"""Phase 17.1–17.2–17.6: Moduli space analysis from lemma9_sweep data.

Tasks:
  17.1 — Empirical gradients dδ/ds₁, ds₂/ds₁ for k=5,6,7
  17.2 — Polynomial/spline fits of solution curves δ(s₁), s₂(s₁)
  17.6 — Branch convergence analysis for k=7

Reads: results/phase11_higher_fold/lemma9_sweep.json
Writes: results/phase17_moduli/moduli_analysis.json
        results/phase17_moduli/moduli_curves.png
        results/phase17_moduli/branch_convergence_k7.png
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SWEEP_PATH = PROJECT / "data" / "s1_sweep.json"
OUT_DIR = PROJECT / "data"


def load_sweep():
    with open(SWEEP_PATH) as f:
        raw = json.load(f)
    data = {}
    for k_str, entries in raw.items():
        k = int(k_str)
        data[k] = sorted(entries, key=lambda e: e["s1"])
    return data


def extract_branch(entries, branch_id):
    return [e for e in entries if e["branch"] == branch_id]


def compute_gradients(s1, y):
    """Central FD gradients, forward/backward at endpoints."""
    n = len(s1)
    dy = np.zeros(n)
    for i in range(n):
        if i == 0:
            dy[i] = (y[1] - y[0]) / (s1[1] - s1[0])
        elif i == n - 1:
            dy[i] = (y[-1] - y[-2]) / (s1[-1] - s1[-2])
        else:
            dy[i] = (y[i + 1] - y[i - 1]) / (s1[i + 1] - s1[i - 1])
    return dy


def fit_polynomial(s1, y, max_deg=5):
    """Fit polynomial, select degree by BIC."""
    best_bic = np.inf
    best_deg = 1
    best_coeffs = None
    n = len(s1)
    for deg in range(1, min(max_deg + 1, n)):
        coeffs = np.polyfit(s1, y, deg)
        resid = y - np.polyval(coeffs, s1)
        sse = np.sum(resid**2)
        if sse < 1e-30:
            mse = 1e-30
        else:
            mse = sse / n
        bic = n * np.log(mse) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_deg = deg
            best_coeffs = coeffs
    r2 = 1.0 - np.sum((y - np.polyval(best_coeffs, s1))**2) / np.sum((y - np.mean(y))**2)
    return best_deg, best_coeffs.tolist(), float(r2)


def analyze_single_branch(s1_arr, delta_arr, s2_arr, label):
    """Full analysis for one solution branch."""
    s1 = np.array(s1_arr)
    delta = np.array(delta_arr)
    s2 = np.array(s2_arr)

    # 17.1: gradients
    dd_ds1 = compute_gradients(s1, delta)
    ds2_ds1 = compute_gradients(s1, s2)

    # 17.2: polynomial fits
    deg_d, coeffs_d, r2_d = fit_polynomial(s1, delta)
    deg_s2, coeffs_s2, r2_s2 = fit_polynomial(s1, s2)

    # statistics
    result = {
        "label": label,
        "n_points": len(s1),
        "s1_range": [float(s1[0]), float(s1[-1])],
        "delta_range": [float(delta.min()), float(delta.max())],
        "s2_range": [float(s2.min()), float(s2.max())],
        "gradient_dd_ds1": {
            "mean": float(np.mean(dd_ds1)),
            "std": float(np.std(dd_ds1)),
            "min": float(dd_ds1.min()),
            "max": float(dd_ds1.max()),
            "at_s1_min": float(dd_ds1[0]),
            "at_s1_max": float(dd_ds1[-1]),
        },
        "gradient_ds2_ds1": {
            "mean": float(np.mean(ds2_ds1)),
            "std": float(np.std(ds2_ds1)),
            "min": float(ds2_ds1.min()),
            "max": float(ds2_ds1.max()),
            "at_s1_min": float(ds2_ds1[0]),
            "at_s1_max": float(ds2_ds1[-1]),
        },
        "fit_delta": {"degree": deg_d, "coefficients": coeffs_d, "R2": r2_d},
        "fit_s2": {"degree": deg_s2, "coefficients": coeffs_s2, "R2": r2_s2},
    }

    # critical points: where gradient changes sign
    sign_dd = np.sign(dd_ds1)
    sign_changes = np.where(np.diff(sign_dd) != 0)[0]
    if len(sign_changes) > 0:
        result["delta_critical_s1"] = [float(s1[i]) for i in sign_changes]
    else:
        result["delta_critical_s1"] = []

    sign_ds2 = np.sign(ds2_ds1)
    sign_changes_s2 = np.where(np.diff(sign_ds2) != 0)[0]
    if len(sign_changes_s2) > 0:
        result["s2_critical_s1"] = [float(s1[i]) for i in sign_changes_s2]
    else:
        result["s2_critical_s1"] = []

    return result, s1, delta, s2, dd_ds1, ds2_ds1


def analyze_branch_convergence(entries_k7):
    """17.6: Measure where the two k=7 branches converge."""
    b0 = extract_branch(entries_k7, 0)
    b1 = extract_branch(entries_k7, 1)

    # match by s1
    s1_b0 = {round(e["s1"], 4): e for e in b0}
    s1_b1 = {round(e["s1"], 4): e for e in b1}
    common = sorted(set(s1_b0.keys()) & set(s1_b1.keys()))

    s1_vals = []
    delta_diff = []
    s2_diff = []
    for s1 in common:
        e0, e1 = s1_b0[s1], s1_b1[s1]
        s1_vals.append(s1)
        delta_diff.append(abs(e0["delta"] - e1["delta"]))
        s2_diff.append(abs(e0["s2"] - e1["s2"]))

    s1_arr = np.array(s1_vals)
    dd = np.array(delta_diff)
    ds = np.array(s2_diff)

    # find crossing: where difference drops below threshold
    THRESHOLD = 1e-6
    merged_mask = (dd < THRESHOLD) & (ds < THRESHOLD)
    crossing_idx = np.where(merged_mask)[0]

    if len(crossing_idx) > 0:
        cross_s1 = float(s1_arr[crossing_idx[0]])
        cross_region = [float(s1_arr[crossing_idx[0]]), float(s1_arr[crossing_idx[-1]])]
    else:
        # find minimum separation
        total_sep = np.sqrt(dd**2 + ds**2)
        min_idx = np.argmin(total_sep)
        cross_s1 = float(s1_arr[min_idx])
        cross_region = None

    result = {
        "n_common_s1": len(common),
        "s1_range": [float(s1_arr[0]), float(s1_arr[-1])],
        "max_delta_diff": float(dd.max()),
        "max_s2_diff": float(ds.max()),
        "min_delta_diff": float(dd.min()),
        "min_s2_diff": float(ds.min()),
        "min_separation_s1": cross_s1,
        "min_separation_dist": float(np.sqrt(dd[np.argmin(dd**2 + ds**2)]**2 + ds[np.argmin(dd**2 + ds**2)]**2)),
        "branches_merged": bool(np.any(merged_mask)),
        "merged_region": cross_region,
        "separation_vs_s1": {
            "s1": [float(x) for x in s1_arr],
            "delta_diff": [float(x) for x in dd],
            "s2_diff": [float(x) for x in ds],
        },
    }
    return result, s1_arr, dd, ds


def plot_moduli_curves(all_curves, out_path):
    """Plot δ(s₁) and s₂(s₁) for all k."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {5: "#2196F3", 6: "#4CAF50", 7: "#FF9800"}
    markers = {5: "o", 6: "s", 7: "^"}

    # Top-left: δ(s₁)
    ax = axes[0, 0]
    for k, (_, s1, delta, s2, dd, ds2) in all_curves.items():
        ax.plot(s1, delta, f"{markers[k]}-", color=colors[k], label=f"k={k}", markersize=3, linewidth=1)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$\\delta$")
    ax.set_title("$\\delta(s_1)$ — Solution curves in moduli space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: s₂(s₁)
    ax = axes[0, 1]
    for k, (_, s1, delta, s2, dd, ds2) in all_curves.items():
        ax.plot(s1, s2, f"{markers[k]}-", color=colors[k], label=f"k={k}", markersize=3, linewidth=1)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$s_2$")
    ax.set_title("$s_2(s_1)$ — Solution curves in moduli space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: dδ/ds₁
    ax = axes[1, 0]
    for k, (_, s1, delta, s2, dd, ds2) in all_curves.items():
        ax.plot(s1, dd, f"{markers[k]}-", color=colors[k], label=f"k={k}", markersize=3, linewidth=1)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$d\\delta/ds_1$")
    ax.set_title("Gradient $d\\delta/ds_1$")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: ds₂/ds₁
    ax = axes[1, 1]
    for k, (_, s1, delta, s2, dd, ds2) in all_curves.items():
        ax.plot(s1, ds2, f"{markers[k]}-", color=colors[k], label=f"k={k}", markersize=3, linewidth=1)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$ds_2/ds_1$")
    ax.set_title("Gradient $ds_2/ds_1$")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_branch_convergence(s1, dd, ds, out_path):
    """Plot k=7 branch separation vs s₁."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.semilogy(s1, dd, "o-", color="#FF5722", label="|Δδ| between branches", markersize=4)
    ax.semilogy(s1, ds, "s-", color="#9C27B0", label="|Δs₂| between branches", markersize=4)
    ax.semilogy(s1, np.sqrt(dd**2 + ds**2), "^--", color="#607D8B", label="Total separation", markersize=4)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("Branch separation")
    ax.set_title("k=7 Branch Convergence Analysis (Task 17.6)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main(plot=True):
    print("=" * 70)
    print("Phase 17.1–17.2–17.6: Moduli Space Analysis")
    print("=" * 70)

    data = load_sweep()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis = {"version": "17.1", "date": "2026-04-04", "source": str(SWEEP_PATH)}
    all_curves = {}

    # ── 17.1 & 17.2: Per-fold analysis ──
    for k in sorted(data.keys()):
        entries = data[k]
        branches = sorted(set(e["branch"] for e in entries))
        print(f"\n{'─' * 50}")
        print(f"  k = {k}  ({len(entries)} points, {len(branches)} branch(es))")
        print(f"{'─' * 50}")

        if k == 7:
            # Use branch 0 as primary (they converge anyway)
            branch_entries = extract_branch(entries, 0)
        else:
            branch_entries = entries

        s1 = [e["s1"] for e in branch_entries]
        delta = [e["delta"] for e in branch_entries]
        s2 = [e["s2"] for e in branch_entries]

        result, s1_arr, d_arr, s2_arr, dd, ds2 = analyze_single_branch(s1, delta, s2, f"k={k}")
        analysis[f"k{k}"] = result
        all_curves[k] = (result, s1_arr, d_arr, s2_arr, dd, ds2)

        # Print summary
        g = result["gradient_dd_ds1"]
        print(f"  δ  range: [{result['delta_range'][0]:.4f}, {result['delta_range'][1]:.4f}]")
        print(f"  s₂ range: [{result['s2_range'][0]:.4f}, {result['s2_range'][1]:.4f}]")
        print(f"  dδ/ds₁:  mean={g['mean']:.6f}, range=[{g['min']:.6f}, {g['max']:.6f}]")
        g2 = result["gradient_ds2_ds1"]
        print(f"  ds₂/ds₁: mean={g2['mean']:.6f}, range=[{g2['min']:.6f}, {g2['max']:.6f}]")
        print(f"  Fit δ(s₁):  deg={result['fit_delta']['degree']}, R²={result['fit_delta']['R2']:.6f}")
        print(f"  Fit s₂(s₁): deg={result['fit_s2']['degree']}, R²={result['fit_s2']['R2']:.6f}")
        if result["delta_critical_s1"]:
            print(f"  δ critical points at s₁ = {result['delta_critical_s1']}")
        if result["s2_critical_s1"]:
            print(f"  s₂ critical points at s₁ = {result['s2_critical_s1']}")

    # ── 17.6: Branch convergence k=7 ──
    s1_conv, dd_conv, ds_conv = None, None, None
    if 7 in data:
        print(f"\n{'=' * 50}")
        print("  17.6: k=7 Branch Convergence Analysis")
        print(f"{'=' * 50}")

        conv, s1_conv, dd_conv, ds_conv = analyze_branch_convergence(data[7])
        analysis["k7_branch_convergence"] = conv

        print(f"  Common s₁ points: {conv['n_common_s1']}")
        print(f"  Max |Δδ|:  {conv['max_delta_diff']:.2e}")
        print(f"  Max |Δs₂|: {conv['max_s2_diff']:.2e}")
        print(f"  Min |Δδ|:  {conv['min_delta_diff']:.2e}")
        print(f"  Min |Δs₂|: {conv['min_s2_diff']:.2e}")
        print(f"  Min separation at s₁ = {conv['min_separation_s1']:.2f}")
        print(f"  Branches fully merged: {conv['branches_merged']}")

    # ── Cross-fold comparison ──
    print(f"\n{'=' * 50}")
    print("  Cross-fold gradient comparison")
    print(f"{'=' * 50}")
    print(f"  {'k':>3}  {'mean dδ/ds₁':>14}  {'mean ds₂/ds₁':>14}  {'δ slope ratio':>14}")
    prev_dd_mean = None
    for k in sorted(all_curves.keys()):
        r = all_curves[k][0]
        dd_mean = r["gradient_dd_ds1"]["mean"]
        ds2_mean = r["gradient_ds2_ds1"]["mean"]
        ratio = dd_mean / prev_dd_mean if prev_dd_mean else float("nan")
        print(f"  {k:3d}  {dd_mean:14.6f}  {ds2_mean:14.6f}  {ratio:14.4f}")
        prev_dd_mean = dd_mean

    # ── Save ──
    if plot:
        plot_moduli_curves(all_curves, OUT_DIR / "moduli_curves.png")
        if s1_conv is not None:
            plot_branch_convergence(s1_conv, dd_conv, ds_conv,
                                    OUT_DIR / "branch_convergence_k7.png")

    out_json = OUT_DIR / "moduli_analysis.json"
    # Strip non-serializable numpy from analysis
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"\n  Saved: {out_json}")

    print(f"\n{'=' * 70}")
    print("  Phase 17.1–17.2–17.6 COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main(plot="--no-plot" not in sys.argv)
