#!/usr/bin/env python3
"""LEGACY / auxiliary runner.

Phase 17.3–17.4–17.5: Curvature, inter-fold section, full Jacobian.

Tasks:
  17.3 — Curvature κ(s₁) of solution curves in M; inflection points; inter-fold comparison
  17.4 — Inter-fold section at fixed s₁: (δ, s₂) vs k from sweep + full_series data
  17.5 — Full numerical Jacobian dF/d(δ,s₂) on selected points: eigenvalues, conditioning, soft directions

Reads: data/s1_sweep.json
       data/full_series_k3_1000.json
Writes: data/phase17_345_results.json
        data/curvature_curves.png
        data/interfold_section.png
        data/jacobian_analysis.png
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SWEEP_PATH = PROJECT / "data" / "s1_sweep.json"
FULL_SERIES_PATH = PROJECT / "data" / "full_series_k3_1000.json"
OUT_DIR = PROJECT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.seed_catalog import SEEDS, TAU_IMAG
from src.theorem7_periodicity import theorem7_local_jacobian, theorem7_local_residual_vector


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def central_diff(x, y):
    """Central FD gradients, one-sided at endpoints."""
    n = len(x)
    dy = np.zeros(n)
    for i in range(n):
        if i == 0:
            dy[i] = (y[1] - y[0]) / (x[1] - x[0])
        elif i == n - 1:
            dy[i] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dy


def curvature_2d(x, y, t):
    """Curvature κ of a plane curve (x(t), y(t)) parametrized by t.
    
    κ = |x'y'' - y'x''| / (x'² + y'²)^{3/2}
    Signed curvature: κ_s = (x'y'' - y'x'') / (x'² + y'²)^{3/2}
    """
    dx = central_diff(t, x)
    dy = central_diff(t, y)
    d2x = central_diff(t, dx)
    d2y = central_diff(t, dy)
    
    numer = dx * d2y - dy * d2x
    denom = (dx**2 + dy**2)**1.5
    # Avoid division by zero
    safe = denom > 1e-30
    kappa_signed = np.where(safe, numer / denom, 0.0)
    kappa_unsigned = np.abs(kappa_signed)
    return kappa_signed, kappa_unsigned


def curvature_3d(x, y, z, t):
    """Curvature κ of a space curve (x(t), y(t), z(t)) in R³.
    
    κ = |r' × r''| / |r'|³
    """
    dx = central_diff(t, x)
    dy = central_diff(t, y)
    dz = central_diff(t, z)
    d2x = central_diff(t, dx)
    d2y = central_diff(t, dy)
    d2z = central_diff(t, dz)
    
    # Cross product r' × r''
    cx = dy * d2z - dz * d2y
    cy = dz * d2x - dx * d2z
    cz = dx * d2y - dy * d2x
    
    cross_mag = np.sqrt(cx**2 + cy**2 + cz**2)
    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    
    safe = speed > 1e-30
    kappa = np.where(safe, cross_mag / speed**3, 0.0)
    return kappa


def load_sweep():
    with open(SWEEP_PATH) as f:
        raw = json.load(f)
    data = {}
    for k_str, entries in raw.items():
        k = int(k_str)
        arr = sorted(entries, key=lambda e: e["s1"])
        # For k=7, take only branch 0
        if k == 7:
            arr = [e for e in arr if e["branch"] == 0]
        data[k] = arr
    return data


def load_full_series():
    with open(FULL_SERIES_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Task 17.3: Curvature of solution curves in M
# ═══════════════════════════════════════════════════════════════════

def task_17_3(sweep_data):
    """Compute curvature of solution curves in moduli space."""
    print("\n" + "=" * 70)
    print("  TASK 17.3 — Curvature of solution curves in M")
    print("=" * 70)
    
    results = {}
    
    for k in [5, 6, 7]:
        entries = sweep_data[k]
        s1 = np.array([e["s1"] for e in entries])
        delta = np.array([e["delta"] for e in entries])
        s2 = np.array([e["s2"] for e in entries])
        n = len(s1)
        
        # 2D curvature in (δ, s₂) plane (parametrized by s₁)
        kappa_s, kappa = curvature_2d(delta, s2, s1)
        
        # 3D curvature in (δ, s₁, s₂) space
        kappa_3d = curvature_3d(delta, s1, s2, s1)
        
        # Inflection points: where signed curvature changes sign
        sign_changes = np.where(np.diff(np.sign(kappa_s)))[0]
        inflections = []
        for idx in sign_changes:
            # Linear interpolation to find exact s₁ of inflection
            s1_infl = s1[idx] - kappa_s[idx] * (s1[idx+1] - s1[idx]) / (kappa_s[idx+1] - kappa_s[idx])
            inflections.append(float(s1_infl))
        
        # Interior statistics (skip 2 boundary points on each side)
        margin = 3
        interior = slice(margin, -margin) if n > 2 * margin + 1 else slice(None)
        
        print(f"\n  k = {k}  ({n} points)")
        print(f"  2D curvature κ in (δ, s₂) plane:")
        print(f"    mean |κ|    = {np.mean(kappa[interior]):.6f}")
        print(f"    max  |κ|    = {np.max(kappa[interior]):.6f}  at s₁ = {s1[interior][np.argmax(kappa[interior])]:.2f}")
        print(f"    min  |κ|    = {np.min(kappa[interior]):.6f}  at s₁ = {s1[interior][np.argmin(kappa[interior])]:.2f}")
        print(f"    inflections = {len(inflections)} at s₁ ≈ {[f'{x:.2f}' for x in inflections]}")
        
        print(f"  3D curvature κ₃D in (δ, s₁, s₂) space:")
        print(f"    mean κ₃D    = {np.mean(kappa_3d[interior]):.6f}")
        print(f"    max  κ₃D    = {np.max(kappa_3d[interior]):.6f}")
        print(f"    range ratio = {np.max(kappa_3d[interior]) / np.max(np.abs(kappa_3d[interior].min()), initial=1e-30):.2f}")
        
        # Curvature radius ρ = 1/κ
        rho = np.where(kappa > 1e-10, 1.0 / kappa, np.inf)
        print(f"  Curvature radius ρ = 1/κ:")
        print(f"    min ρ = {np.min(rho[interior]):.4f}  (tightest bend)")
        print(f"    max ρ = {np.max(rho[interior]):.4f}" if np.max(rho[interior]) < 1e10 else f"    max ρ = ∞")
        
        results[str(k)] = {
            "n_points": n,
            "kappa_2d": {
                "values": kappa.tolist(),
                "signed": kappa_s.tolist(),
                "mean": float(np.mean(kappa[interior])),
                "max": float(np.max(kappa[interior])),
                "min": float(np.min(kappa[interior])),
                "max_at_s1": float(s1[interior][np.argmax(kappa[interior])]),
            },
            "kappa_3d": {
                "values": kappa_3d.tolist(),
                "mean": float(np.mean(kappa_3d[interior])),
                "max": float(np.max(kappa_3d[interior])),
            },
            "inflections": inflections,
            "n_inflections": len(inflections),
            "s1": s1.tolist(),
            "delta": delta.tolist(),
            "s2": s2.tolist(),
        }
    
    # Cross-fold curvature comparison
    print(f"\n  --- CROSS-FOLD CURVATURE COMPARISON ---")
    print(f"  {'k':>3}  {'mean κ₂D':>10}  {'max κ₂D':>10}  {'mean κ₃D':>10}  {'inflections':>12}")
    for k in [5, 6, 7]:
        r = results[str(k)]
        print(f"  {k:>3}  {r['kappa_2d']['mean']:10.6f}  {r['kappa_2d']['max']:10.6f}  "
              f"{r['kappa_3d']['mean']:10.6f}  {r['n_inflections']:>12}")
    
    # Curvature scaling with k
    means_2d = [results[str(k)]["kappa_2d"]["mean"] for k in [5, 6, 7]]
    ks = np.array([5, 6, 7], dtype=float)
    if all(m > 0 for m in means_2d):
        log_k = np.log(ks)
        log_kappa = np.log(means_2d)
        slope, intercept = np.polyfit(log_k, log_kappa, 1)
        print(f"\n  Scaling: mean κ₂D ∝ k^{slope:.3f}  (log-log fit)")
        results["scaling_2d"] = {"exponent": float(slope), "prefactor": float(np.exp(intercept))}
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  Task 17.4: Inter-fold section at fixed s₁
# ═══════════════════════════════════════════════════════════════════

def task_17_4(sweep_data, full_series):
    """Cross-sections at fixed s₁: how (δ, s₂) move with k."""
    print("\n" + "=" * 70)
    print("  TASK 17.4 — Inter-fold section at fixed s₁")
    print("=" * 70)
    
    results = {}
    
    # Strategy: for each s₁ in the overlap range of sweep data,
    # interpolate (δ, s₂) at that s₁ for each k=5,6,7 from the sweep.
    # Then add k=10..50 from full_series at s₁=-8.5 (fixed gauge).
    # Also use SEEDS for k=3..12 at their native s₁.

    # Part A: Seeds k=3..12 (SEEDS catalog, different s₁)
    print("\n  Part A: Seeds catalog (native s₁)")
    seed_ks = []
    seed_deltas = []
    seed_s2s = []
    seed_s1s = []
    for k in sorted(SEEDS.keys()):
        s = SEEDS[k]
        seed_ks.append(k)
        seed_deltas.append(s["delta"])
        seed_s2s.append(s["s2"])
        seed_s1s.append(s["s1"])
    
    seed_ks = np.array(seed_ks, dtype=float)
    seed_deltas = np.array(seed_deltas)
    seed_s2s = np.array(seed_s2s)
    
    print(f"  {'k':>3}  {'s₁':>8}  {'δ':>10}  {'s₂':>10}")
    for i, k in enumerate(seed_ks):
        print(f"  {int(k):>3}  {seed_s1s[i]:>8.2f}  {seed_deltas[i]:10.6f}  {seed_s2s[i]:10.6f}")
    
    results["seeds"] = {
        "k": [int(x) for x in seed_ks],
        "s1": [float(x) for x in seed_s1s],
        "delta": seed_deltas.tolist(),
        "s2": seed_s2s.tolist(),
        "note": "Seeds at native s₁ (gauge-mixed)"
    }
    
    # Part B: Gauge-fixed s₁=-8.5 from full_series (k=10..50)
    print(f"\n  Part B: Gauge-fixed s₁ = -8.5 (full_series k=10..50)")
    fs_k = np.array([e["k"] for e in full_series], dtype=float)
    fs_delta = np.array([e["delta"] for e in full_series])
    fs_s2 = np.array([e["s2"] for e in full_series])
    
    print(f"  {len(full_series)} points, k ∈ [{int(fs_k.min())}, {int(fs_k.max())}]")
    print(f"  δ range:  [{fs_delta.min():.6f}, {fs_delta.max():.6f}]")
    print(f"  s₂ range: [{fs_s2.min():.6f}, {fs_s2.max():.6f}]")
    
    # Fit δ(k) and s₂(k) for gauge-fixed data
    # Power law: δ = a * k^b
    log_k = np.log(fs_k)
    log_d = np.log(fs_delta)
    b_d, a_d = np.polyfit(log_k, log_d, 1)
    r2_d = 1 - np.sum((log_d - (b_d * log_k + a_d))**2) / np.sum((log_d - np.mean(log_d))**2)
    print(f"\n  δ(k) power law: δ = {np.exp(a_d):.4f} · k^({b_d:.4f})  R² = {r2_d:.6f}")
    
    # s₂ asymptote: s₂ = c + a/k^b
    # Simple: fit s₂ vs 1/k
    inv_k = 1.0 / fs_k
    c_s2 = np.polyfit(inv_k, fs_s2, 2)  # quadratic in 1/k
    s2_pred = np.polyval(c_s2, inv_k)
    ss_res = np.sum((fs_s2 - s2_pred)**2)
    ss_tot = np.sum((fs_s2 - np.mean(fs_s2))**2)
    r2_s2 = 1 - ss_res / ss_tot
    s2_asymptote = np.polyval(c_s2, 0)
    print(f"  s₂(k) quadratic in 1/k: s₂∞ = {s2_asymptote:.4f}  R² = {r2_s2:.6f}")
    
    results["gauge_fixed"] = {
        "s1": -8.5,
        "k": [int(x) for x in fs_k],
        "delta": fs_delta.tolist(),
        "s2": fs_s2.tolist(),
        "delta_power_law": {"a": float(np.exp(a_d)), "b": float(b_d), "R2": float(r2_d)},
        "s2_asymptote": float(s2_asymptote),
        "s2_fit_R2": float(r2_s2),
    }
    
    # Part C: Interpolation from sweep at common s₁ values
    print(f"\n  Part C: Interpolated cross-sections from sweep data")
    
    # Find s₁ range common to all k=5,6,7
    s1_ranges = {}
    for k in [5, 6, 7]:
        entries = sweep_data[k]
        s1s = [e["s1"] for e in entries]
        s1_ranges[k] = (min(s1s), max(s1s))
    
    s1_common_min = max(r[0] for r in s1_ranges.values())
    s1_common_max = min(r[1] for r in s1_ranges.values())
    print(f"  Common s₁ range: [{s1_common_min}, {s1_common_max}]")
    
    # Sample at specific s₁ values
    sample_s1 = np.arange(s1_common_min, s1_common_max + 0.01, 0.5)
    
    cross_sections = {}
    for s1_target in sample_s1:
        section = {}
        for k in [5, 6, 7]:
            entries = sweep_data[k]
            s1_arr = np.array([e["s1"] for e in entries])
            d_arr = np.array([e["delta"] for e in entries])
            s2_arr = np.array([e["s2"] for e in entries])
            # Interpolate
            d_interp = np.interp(s1_target, s1_arr, d_arr)
            s2_interp = np.interp(s1_target, s1_arr, s2_arr)
            section[k] = {"delta": float(d_interp), "s2": float(s2_interp)}
        cross_sections[f"{s1_target:.1f}"] = section
    
    # Print cross-sections at a few representative s₁'s
    sample_print = [s1_common_min, (s1_common_min + s1_common_max) / 2, s1_common_max]
    for s1_val in sample_print:
        s1_key = f"{s1_val:.1f}"
        if s1_key in cross_sections:
            cs = cross_sections[s1_key]
            print(f"\n  At s₁ = {s1_val:.1f}:")
            print(f"    {'k':>3}  {'δ':>10}  {'s₂':>10}  {'Δδ/Δk':>10}  {'Δs₂/Δk':>10}")
            ks_here = sorted(cs.keys())
            for i, k in enumerate(ks_here):
                dd = ""
                ds = ""
                if i > 0:
                    k_prev = ks_here[i-1]
                    dd_val = (cs[k]["delta"] - cs[k_prev]["delta"]) / (k - k_prev)
                    ds_val = (cs[k]["s2"] - cs[k_prev]["s2"]) / (k - k_prev)
                    dd = f"{dd_val:10.6f}"
                    ds = f"{ds_val:10.6f}"
                print(f"    {k:>3}  {cs[k]['delta']:10.6f}  {cs[k]['s2']:10.6f}  {dd:>10}  {ds:>10}")
    
    # Inter-fold gradients: dδ/dk and ds₂/dk at each cross-section
    print(f"\n  --- INTER-FOLD GRADIENTS ---")
    dd_dk_list = []
    ds2_dk_list = []
    for s1_key, cs in cross_sections.items():
        ks = np.array(sorted(cs.keys()), dtype=float)
        ds = np.array([cs[int(k)]["delta"] for k in ks])
        s2s = np.array([cs[int(k)]["s2"] for k in ks])
        dd_dk = (ds[-1] - ds[0]) / (ks[-1] - ks[0])
        ds2_dk = (s2s[-1] - s2s[0]) / (ks[-1] - ks[0])
        dd_dk_list.append(dd_dk)
        ds2_dk_list.append(ds2_dk)
    
    dd_dk_arr = np.array(dd_dk_list)
    ds2_dk_arr = np.array(ds2_dk_list)
    print(f"  dδ/dk  (k=5→7): mean = {np.mean(dd_dk_arr):.6f}, std = {np.std(dd_dk_arr):.6f}")
    print(f"  ds₂/dk (k=5→7): mean = {np.mean(ds2_dk_arr):.6f}, std = {np.std(ds2_dk_arr):.6f}")
    print(f"  Ratio (ds₂/dk)/(dδ/dk): {np.mean(ds2_dk_arr)/np.mean(dd_dk_arr):.4f}")
    
    # Verify against known scaling law δ²·k → 50.13
    print(f"\n  --- SCALING LAW VERIFICATION ---")
    print(f"  δ²·k for gauge-fixed (s₁=-8.5) series:")
    d2k = fs_delta**2 * fs_k
    print(f"    k=10: δ²·k = {d2k[0]:.4f}")
    print(f"    k=30: {d2k[20]:.4f}")
    print(f"    k=50: {d2k[-1]:.4f}")
    print(f"    asymptote: {np.mean(d2k[-10:]):.4f}")
    
    results["cross_sections"] = {
        "s1_values": [float(x) for x in sample_s1],
        "data": cross_sections,
        "interfold_gradients": {
            "dd_dk_mean": float(np.mean(dd_dk_arr)),
            "dd_dk_std": float(np.std(dd_dk_arr)),
            "ds2_dk_mean": float(np.mean(ds2_dk_arr)),
            "ds2_dk_std": float(np.std(ds2_dk_arr)),
        }
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  Task 17.5: Full numerical Jacobian
# ═══════════════════════════════════════════════════════════════════

def task_17_5(sweep_data):
    """Compute full Jacobian dF/d(δ,s₂) at selected points."""
    print("\n" + "=" * 70)
    print("  TASK 17.5 — Full numerical Jacobian dF/d(δ,s₂)")
    print("=" * 70)
    
    results = {}
    
    for k in [5, 6, 7]:
        entries = sweep_data[k]
        n = len(entries)
        
        # Select points: first, middle, last, plus 2 intermediate
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        indices = sorted(set(indices))
        
        print(f"\n  k = {k}  ({n} points, sampling {len(indices)})")
        print(f"  {'s₁':>8}  {'δ':>10}  {'s₂':>10}  {'det(J)':>12}  {'cond(J)':>10}  {'λ₁':>14}  {'λ₂':>14}  {'θ_soft':>8}")
        print(f"  " + "-" * 108)
        
        k_results = []
        for idx in indices:
            e = entries[idx]
            s1 = e["s1"]
            delta = e["delta"]
            s2 = e["s2"]
            
            try:
                J = theorem7_local_jacobian(
                    tau_imag=TAU_IMAG,
                    delta=delta,
                    s1=s1,
                    s2=s2,
                    symmetry_fold=k,
                    target_ratio=1.0
                )
                
                det_J = np.linalg.det(J)
                cond_J = np.linalg.cond(J)
                eigenvalues = np.linalg.eigvals(J)
                
                # SVD for soft directions
                U, sigma, Vt = np.linalg.svd(J)
                # Soft direction = column of V corresponding to smallest singular value
                soft_dir = Vt[-1, :]  # last row of Vt = last column of V
                # Angle of soft direction in (δ, s₂) space
                theta_soft = np.degrees(np.arctan2(soft_dir[1], soft_dir[0]))
                
                ev0 = eigenvalues[0].real if abs(eigenvalues[0].imag) < 1e-12 else eigenvalues[0]
                ev1 = eigenvalues[1].real if abs(eigenvalues[1].imag) < 1e-12 else eigenvalues[1]
                ev0_s = f"{ev0:14.6e}" if isinstance(ev0, float) else f"{ev0.real:+.3e}{ev0.imag:+.3e}j"
                ev1_s = f"{ev1:14.6e}" if isinstance(ev1, float) else f"{ev1.real:+.3e}{ev1.imag:+.3e}j"
                print(f"  {s1:8.2f}  {delta:10.6f}  {s2:10.6f}  {det_J:12.6e}  {cond_J:10.2f}  "
                      f"{ev0_s}  {ev1_s}  {theta_soft:8.2f} deg")
                
                k_results.append({
                    "s1": float(s1),
                    "delta": float(delta),
                    "s2": float(s2),
                    "jacobian": J.tolist(),
                    "det": float(det_J),
                    "condition": float(cond_J),
                    "eigenvalues": [complex(ev).real if abs(complex(ev).imag) < 1e-15 else
                                    {"re": float(complex(ev).real), "im": float(complex(ev).imag)}
                                    for ev in eigenvalues],
                    "singular_values": sigma.tolist(),
                    "soft_direction": soft_dir.tolist(),
                    "soft_angle_deg": float(theta_soft),
                })
            except Exception as ex:
                print(f"  {s1:8.2f}  {delta:10.6f}  {s2:10.6f}  ERROR: {ex}")
                k_results.append({"s1": float(s1), "error": str(ex)})
        
        # Summary for this k
        valid = [r for r in k_results if "det" in r]
        if valid:
            dets = [r["det"] for r in valid]
            conds = [r["condition"] for r in valid]
            soft_angles = [r["soft_angle_deg"] for r in valid]
            
            print(f"\n  Summary k={k}:")
            print(f"    |det(J)| range: [{min(abs(d) for d in dets):.4e}, {max(abs(d) for d in dets):.4e}]")
            print(f"    cond(J) range:  [{min(conds):.2f}, {max(conds):.2f}]")
            print(f"    soft angle:     [{min(soft_angles):.1f}°, {max(soft_angles):.1f}°]")
            
            # Check sign of det(J) — consistent?
            signs = [np.sign(d) for d in dets]
            if all(s == signs[0] for s in signs):
                print(f"    det(J) sign:    CONSISTENT ({'+' if signs[0] > 0 else '-'})")
            else:
                print(f"    det(J) sign:    CHANGES — fold in the solution family!")
        
        results[str(k)] = k_results
    
    # Cross-fold Jacobian comparison
    print(f"\n  --- CROSS-FOLD JACOBIAN COMPARISON ---")
    print(f"  (at midpoint of each sweep)")
    print(f"  {'k':>3}  {'det(J)':>12}  {'cond(J)':>10}  {'σ_min':>12}  {'σ_max':>12}  {'θ_soft':>8}")
    for k in [5, 6, 7]:
        k_res = results[str(k)]
        mid = k_res[len(k_res) // 2]
        if "det" in mid:
            sigma = mid["singular_values"]
            print(f"  {k:>3}  {mid['det']:12.6e}  {mid['condition']:10.2f}  "
                  f"{min(sigma):12.6e}  {max(sigma):12.6e}  {mid['soft_angle_deg']:8.2f}°")
    
    # Conditioning vs k: does the problem become better or worse conditioned?
    mid_conds = []
    for k in [5, 6, 7]:
        k_res = results[str(k)]
        mid = k_res[len(k_res) // 2]
        if "condition" in mid:
            mid_conds.append((k, mid["condition"]))
    if len(mid_conds) >= 2:
        ks_c = np.array([x[0] for x in mid_conds], dtype=float)
        cs_c = np.array([x[1] for x in mid_conds])
        if cs_c[-1] > cs_c[0]:
            print(f"\n  Conditioning WORSENS with k: cond increases from {cs_c[0]:.1f} to {cs_c[-1]:.1f}")
        else:
            print(f"\n  Conditioning IMPROVES with k: cond decreases from {cs_c[0]:.1f} to {cs_c[-1]:.1f}")
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_curvature(curv_results, out_path):
    """Plot curvature κ(s₁) for k=5,6,7."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {5: "#2196F3", 6: "#4CAF50", 7: "#FF9800"}
    
    # Top-left: 2D unsigned curvature
    ax = axes[0, 0]
    for k in [5, 6, 7]:
        r = curv_results[str(k)]
        s1 = np.array(r["s1"])
        kappa = np.array(r["kappa_2d"]["values"])
        ax.plot(s1[2:-2], kappa[2:-2], "-", color=colors[k], label=f"k={k}", linewidth=1.5)
        # Mark inflection points
        for s1_infl in r["inflections"]:
            ax.axvline(s1_infl, color=colors[k], linestyle=":", alpha=0.5)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$|\\kappa|$")
    ax.set_title("2D curvature $|\\kappa|$ in $(\\delta, s_2)$ plane")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: 2D signed curvature
    ax = axes[0, 1]
    for k in [5, 6, 7]:
        r = curv_results[str(k)]
        s1 = np.array(r["s1"])
        kappa_s = np.array(r["kappa_2d"]["signed"])
        ax.plot(s1[2:-2], kappa_s[2:-2], "-", color=colors[k], label=f"k={k}", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$\\kappa_s$")
    ax.set_title("Signed curvature $\\kappa_s$ (inflections at zero-crossings)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: 3D curvature
    ax = axes[1, 0]
    for k in [5, 6, 7]:
        r = curv_results[str(k)]
        s1 = np.array(r["s1"])
        kappa_3d = np.array(r["kappa_3d"]["values"])
        ax.plot(s1[2:-2], kappa_3d[2:-2], "-", color=colors[k], label=f"k={k}", linewidth=1.5)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$\\kappa_{3D}$")
    ax.set_title("3D curvature in $(\\delta, s_1, s_2)$ space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: curvature radius
    ax = axes[1, 1]
    for k in [5, 6, 7]:
        r = curv_results[str(k)]
        s1 = np.array(r["s1"])
        kappa = np.array(r["kappa_2d"]["values"])
        rho = np.where(kappa > 1e-10, 1.0 / kappa, np.nan)
        ax.plot(s1[2:-2], rho[2:-2], "-", color=colors[k], label=f"k={k}", linewidth=1.5)
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$\\rho = 1/|\\kappa|$")
    ax.set_title("Curvature radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_interfold(interfold_results, out_path):
    """Plot inter-fold sections."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: δ(k) from gauge-fixed
    ax = axes[0]
    gf = interfold_results["gauge_fixed"]
    k_arr = np.array(gf["k"])
    d_arr = np.array(gf["delta"])
    s2_arr = np.array(gf["s2"])
    ax.plot(k_arr, d_arr, "ko-", markersize=3, linewidth=1)
    # Overlay power law fit
    k_fit = np.linspace(10, 50, 100)
    pl = gf["delta_power_law"]
    d_fit = pl["a"] * k_fit**pl["b"]
    ax.plot(k_fit, d_fit, "r--", linewidth=1, label=f"δ = {pl['a']:.2f}·k^({pl['b']:.3f})")
    ax.set_xlabel("k")
    ax.set_ylabel("δ")
    ax.set_title("δ(k) at s₁ = -8.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Middle: s₂(k) from gauge-fixed
    ax = axes[1]
    ax.plot(k_arr, s2_arr, "ko-", markersize=3, linewidth=1)
    ax.axhline(gf["s2_asymptote"], color="r", linestyle="--", label=f"s₂∞ = {gf['s2_asymptote']:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel("s₂")
    ax.set_title("s₂(k) at s₁ = -8.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: (δ, s₂) trajectories for k=5,6,7 plus seeds
    ax = axes[2]
    colors = {5: "#2196F3", 6: "#4CAF50", 7: "#FF9800"}
    
    # Seeds catalog
    seeds = interfold_results["seeds"]
    ax.scatter(seeds["delta"], seeds["s2"], c="red", s=60, zorder=5, label="Seeds (catalog)")
    for i in range(len(seeds["k"])):
        ax.annotate(f"k={seeds['k'][i]}", (seeds["delta"][i], seeds["s2"][i]),
                   fontsize=7, xytext=(4, 4), textcoords="offset points")
    
    # Gauge-fixed series
    ax.plot(d_arr, s2_arr, "k--", linewidth=1, alpha=0.5, label="k=10..50 at s₁=-8.5")
    
    ax.set_xlabel("δ")
    ax.set_ylabel("s₂")
    ax.set_title("(δ, s₂) — seeds + asymptotic series")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_jacobian(jac_results, sweep_data, out_path):
    """Plot Jacobian analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {5: "#2196F3", 6: "#4CAF50", 7: "#FF9800"}
    
    for k in [5, 6, 7]:
        k_res = jac_results[str(k)]
        valid = [r for r in k_res if "det" in r]
        s1 = [r["s1"] for r in valid]
        dets = [abs(r["det"]) for r in valid]
        conds = [r["condition"] for r in valid]
        sigma_min = [min(r["singular_values"]) for r in valid]
        angles = [r["soft_angle_deg"] for r in valid]
        
        # Top-left: |det(J)|
        axes[0, 0].plot(s1, dets, "o-", color=colors[k], label=f"k={k}", markersize=5)
        # Top-right: cond(J)
        axes[0, 1].plot(s1, conds, "o-", color=colors[k], label=f"k={k}", markersize=5)
        # Bottom-left: σ_min
        axes[1, 0].plot(s1, sigma_min, "o-", color=colors[k], label=f"k={k}", markersize=5)
        # Bottom-right: soft direction angle
        axes[1, 1].plot(s1, angles, "o-", color=colors[k], label=f"k={k}", markersize=5)
    
    axes[0, 0].set_xlabel("$s_1$"); axes[0, 0].set_ylabel("$|\\det(J)|$")
    axes[0, 0].set_title("Jacobian determinant"); axes[0, 0].legend()
    axes[0, 0].set_yscale("log"); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("$s_1$"); axes[0, 1].set_ylabel("cond$(J)$")
    axes[0, 1].set_title("Condition number"); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel("$s_1$"); axes[1, 0].set_ylabel("$\\sigma_{\\min}$")
    axes[1, 0].set_title("Smallest singular value"); axes[1, 0].legend()
    axes[1, 0].set_yscale("log"); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel("$s_1$"); axes[1, 1].set_ylabel("$\\theta_{soft}$ (deg)")
    axes[1, 1].set_title("Soft direction angle in $(\\delta, s_2)$"); axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Phase 17.3–17.4–17.5: Curvature, Inter-fold, Jacobian")
    print("=" * 70)
    
    sweep_data = load_sweep()
    full_series = load_full_series()
    
    # 17.3
    curv_results = task_17_3(sweep_data)
    try:
        plot_curvature(curv_results, OUT_DIR / "curvature_curves.png")
    except Exception as ex:
        print(f"  [WARN] plot_curvature failed: {ex}")
    
    # 17.4
    interfold_results = task_17_4(sweep_data, full_series)
    try:
        plot_interfold(interfold_results, OUT_DIR / "interfold_section.png")
    except Exception as ex:
        print(f"  [WARN] plot_interfold failed: {ex}")
    
    # 17.5
    jac_results = task_17_5(sweep_data)
    try:
        plot_jacobian(jac_results, sweep_data, OUT_DIR / "jacobian_analysis.png")
    except Exception as ex:
        print(f"  [WARN] plot_jacobian failed: {ex}")
    
    # Save all results
    all_results = {
        "version": "17.3-17.4-17.5",
        "date": "2026-04-03",
        "task_17_3_curvature": curv_results,
        "task_17_4_interfold": interfold_results,
        "task_17_5_jacobian": jac_results,
    }
    
    out_path = OUT_DIR / "phase17_345_results.json"
    
    # Custom serializer for complex eigenvalues
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {out_path}")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  PHASE 17.3-17.4-17.5 SUMMARY")
    print(f"{'=' * 70}")
    print(f"  17.3 Curvature:")
    for k in [5, 6, 7]:
        r = curv_results[str(k)]
        print(f"    k={k}: mean κ₂D = {r['kappa_2d']['mean']:.6f}, "
              f"{r['n_inflections']} inflection(s), "
              f"max κ₂D = {r['kappa_2d']['max']:.6f}")
    if "scaling_2d" in curv_results:
        print(f"    Scaling: κ₂D ∝ k^{curv_results['scaling_2d']['exponent']:.3f}")
    print(f"  17.4 Inter-fold:")
    gf = interfold_results["gauge_fixed"]
    print(f"    δ(k) = {gf['delta_power_law']['a']:.3f}·k^({gf['delta_power_law']['b']:.3f}), R²={gf['delta_power_law']['R2']:.5f}")
    print(f"    s₂ → {gf['s2_asymptote']:.3f} (R² = {gf['s2_fit_R2']:.5f})")
    print(f"  17.5 Jacobian:")
    for k in [5, 6, 7]:
        k_res = jac_results[str(k)]
        valid = [r for r in k_res if "det" in r]
        if valid:
            mid = valid[len(valid) // 2]
            print(f"    k={k}: |det(J)| = {abs(mid['det']):.4e}, cond = {mid['condition']:.1f}, "
                  f"θ_soft = {mid['soft_angle_deg']:.1f}°")


if __name__ == "__main__":
    main()
