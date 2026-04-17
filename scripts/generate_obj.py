"""
Phase 18.10+ — High-k Bonnet pair OBJ export with publication-quality materials.

Generates Bonnet pairs f⁺/f⁻ for k = 50, 100, 250, 500, 750, 1000
using exact spectral parameters from the 991-seed campaign (Phase 18.9).

All seeds at gauge-fixed s₁ = -8.5, τ₀ = 0.3205128205.

Output per k:
  - f_plus.obj  + f_minus.obj  (solid mesh with per-vertex colors + normals)
  - material.mtl                (two PBR-style materials: plus + minus)
  - flux.obj                    (flux wireframe)
  - metadata.json               (spectral params, timing, metrics)

Color palette evolves with k via δ→0 gradient:
  k=50:   Magma (deep red-orange)    — high δ
  k=100:  Amber (warm gold)          — mid-high δ
  k=250:  Jade  (teal-green)         — mid δ
  k=500:  Sapphire (deep blue)       — mid-low δ
  k=750:  Amethyst (violet-purple)   — low δ
  k=1000: Obsidian (silver-charcoal) — lowest δ, near-limit surface
"""
from __future__ import annotations

import colorsys
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.isothermic_torus import TorusParameters, compute_torus
from src.bonnet_pair import compute_bonnet_pair
from src.bonnet_flux_utils import (
    compute_curvature_proxy,
    color_map_surface,
    compute_flux_edges,
    write_flux_wireframe_obj,
)
from src.bonnet_flux_utils import triangulate_quads
from src.obj_writer import write_obj, write_mtl

# ============================================================================
# Configuration
# ============================================================================

TAU_IMAG = 0.3205128205
S1_GAUGE = -8.5
EPSILON = 0.3
RESOLUTION = 120          # Higher than phase 11 (100) for high-k detail
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "obj"

# Spectral parameters from full_series_k10_1000.json (Phase 18.9)
SEEDS = [
    {"k":   50, "s1": -8.5, "delta": 0.984585, "s2": 6.864665},
    {"k":  100, "s1": -8.5, "delta": 0.702122, "s2": 7.090180},
    {"k":  250, "s1": -8.5, "delta": 0.446337, "s2": 7.229097},
    {"k":  500, "s1": -8.5, "delta": 0.316148, "s2": 7.276017},
    {"k":  750, "s1": -8.5, "delta": 0.258281, "s2": 7.291726},
    {"k": 1000, "s1": -8.5, "delta": 0.223742, "s2": 7.299593},
]


@dataclass
class HighKPalette:
    """Color specification for one k value."""
    name: str              # short label
    hue_range: tuple       # (h_start, h_end) for HSV sweep
    sat_base: float        # saturation base for f⁺
    val_base: float        # value base for f⁺
    mtl_kd_plus: tuple     # (R, G, B) diffuse for f⁺ MTL
    mtl_kd_minus: tuple    # (R, G, B) diffuse for f⁻ MTL
    mtl_ks: tuple          # specular highlight
    mtl_ns: float          # specular shininess


# δ-gradient palette: warm → cool → neutral as δ→0
PALETTES = {
    50: HighKPalette(
        name="Magma",
        hue_range=(0.00, 0.08),
        sat_base=0.90, val_base=0.78,
        mtl_kd_plus= (0.92, 0.28, 0.08),   # deep vermilion
        mtl_kd_minus=(0.75, 0.18, 0.06),
        mtl_ks=(0.50, 0.30, 0.15), mtl_ns=90.0,
    ),
    100: HighKPalette(
        name="Amber",
        hue_range=(0.06, 0.14),
        sat_base=0.88, val_base=0.80,
        mtl_kd_plus= (0.95, 0.68, 0.12),   # rich gold
        mtl_kd_minus=(0.78, 0.52, 0.08),
        mtl_ks=(0.55, 0.45, 0.20), mtl_ns=85.0,
    ),
    250: HighKPalette(
        name="Jade",
        hue_range=(0.38, 0.48),
        sat_base=0.85, val_base=0.76,
        mtl_kd_plus= (0.15, 0.78, 0.52),   # emerald jade
        mtl_kd_minus=(0.10, 0.58, 0.38),
        mtl_ks=(0.35, 0.55, 0.40), mtl_ns=75.0,
    ),
    500: HighKPalette(
        name="Sapphire",
        hue_range=(0.56, 0.66),
        sat_base=0.86, val_base=0.74,
        mtl_kd_plus= (0.10, 0.38, 0.88),   # deep sapphire
        mtl_kd_minus=(0.06, 0.25, 0.68),
        mtl_ks=(0.30, 0.40, 0.60), mtl_ns=95.0,
    ),
    750: HighKPalette(
        name="Amethyst",
        hue_range=(0.74, 0.84),
        sat_base=0.82, val_base=0.72,
        mtl_kd_plus= (0.58, 0.18, 0.82),   # rich amethyst
        mtl_kd_minus=(0.42, 0.12, 0.62),
        mtl_ks=(0.45, 0.30, 0.55), mtl_ns=80.0,
    ),
    1000: HighKPalette(
        name="Obsidian",
        hue_range=(0.00, 1.00),            # full spectrum (aurora)
        sat_base=0.35, val_base=0.82,
        mtl_kd_plus= (0.72, 0.72, 0.78),   # polished silver
        mtl_kd_minus=(0.50, 0.50, 0.56),
        mtl_ks=(0.65, 0.65, 0.70), mtl_ns=120.0,
    ),
}


def compute_vertex_normals(vertices: np.ndarray, triangles) -> np.ndarray:
    """Compute per-vertex normals from triangle faces via area-weighted average."""
    N = len(vertices)
    normals = np.zeros((N, 3))
    for tri in triangles:
        i0, i1, i2 = tri
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        face_n = np.cross(v1 - v0, v2 - v0)
        normals[i0] += face_n
        normals[i1] += face_n
        normals[i2] += face_n
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens < 1e-12, 1.0, lens)
    return normals / lens


def generate_one_k(seed: dict) -> dict:
    """Generate full OBJ suite for one k value."""
    k = seed["k"]
    pal = PALETTES[k]
    theta_deg = 360.0 / k

    print(f"\n{'='*65}")
    print(f"  k={k:>4d}  [{pal.name}]  δ={seed['delta']:.6f}  "
          f"s₂={seed['s2']:.6f}  θ={theta_deg:.2f}°")
    print(f"{'='*65}")

    t0 = time.time()
    out_dir = OUTPUT_DIR / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Isothermic torus ---
    print(f"  [1/6] Computing isothermic torus (Theorem 7)...")
    params = TorusParameters(
        tau_imag=TAU_IMAG,
        delta=seed["delta"],
        s1=seed["s1"],
        s2=seed["s2"],
        u_res=RESOLUTION,
        v_res=RESOLUTION,
        symmetry_fold=k,
    )
    torus = compute_torus(params)
    dt = time.time() - t0
    print(f"         {len(torus.vertices)} vertices, ω={torus.omega:.6f} ({dt:.1f}s)")

    # --- 2. Bonnet pair ---
    print(f"  [2/6] Computing Bonnet pair (Eq. 49, ε={EPSILON})...")
    pair = compute_bonnet_pair(torus, EPSILON)
    dt2 = time.time() - t0
    print(f"         Done ({dt2 - dt:.1f}s)")

    n_u, n_v = RESOLUTION, RESOLUTION

    # --- 3. Curvature coloring ---
    print(f"  [3/6] Computing curvature-mapped colors [{pal.name}]...")
    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)

    colors_plus = color_map_surface(
        n_u, n_v, curv_plus,
        hue_range=pal.hue_range,
        sat_base=pal.sat_base,
        val_base=pal.val_base,
    )
    colors_minus = color_map_surface(
        n_u, n_v, curv_minus,
        hue_range=pal.hue_range,
        sat_base=pal.sat_base * 0.85,
        val_base=pal.val_base * 0.90,
        hue_shift=0.06,
    )

    # --- 4. Triangulate + normals ---
    print(f"  [4/6] Triangulating + normals...")
    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)
    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    # --- 5. Export OBJ + MTL ---
    print(f"  [5/6] Writing OBJ + MTL...")

    header = [
        f"Bonnet Torus — High-k Visualization",
        f"k-fold symmetry: {k}",
        f"Spectral: δ={seed['delta']:.6f}, s₁={seed['s1']}, s₂={seed['s2']:.6f}",
        f"τ₀ = 0.5 + {TAU_IMAG}i  (gauge-fixed s₁={S1_GAUGE})",
        f"Resolution: {RESOLUTION}×{RESOLUTION} = {RESOLUTION**2} vertices",
        f"ε = {EPSILON}",
        f"Asymptotic formula: δ(k) = 7.0814/√k · (1 − 0.858/k + 0.75/k²)",
        f"Palette: {pal.name}",
        f"Generated by: Bonnet Pipeline",
    ]

    mtl_name = f"k{k}_material.mtl"
    mat_plus = f"k{k}_plus"
    mat_minus = f"k{k}_minus"

    # f⁺
    write_obj(
        out_dir / f"k{k}_f_plus.obj",
        pair.f_plus.vertices, tris_plus,
        normals=nrm_plus, colors=colors_plus,
        object_name=f"bonnet_k{k}_f_plus",
        header=header + [f"Surface: f⁺ (positive Bonnet branch)"],
        mtl_file=mtl_name, material_name=mat_plus,
    )

    # f⁻
    write_obj(
        out_dir / f"k{k}_f_minus.obj",
        pair.f_minus.vertices, tris_minus,
        normals=nrm_minus, colors=colors_minus,
        object_name=f"bonnet_k{k}_f_minus",
        header=header + [f"Surface: f⁻ (negative Bonnet branch)"],
        mtl_file=mtl_name, material_name=mat_minus,
    )

    # MTL (two materials)
    write_mtl(
        out_dir / mtl_name,
        mat_plus,
        ka=(0.02, 0.02, 0.02),
        kd=pal.mtl_kd_plus,
        ks=pal.mtl_ks,
        ns=pal.mtl_ns,
        comment=f"Bonnet k={k} f⁺ — {pal.name} palette",
    )
    # Append f⁻ material
    mtl_path = out_dir / mtl_name
    with mtl_path.open("a", encoding="utf-8") as f:
        f.write(f"\n# f⁻ material — darker {pal.name}\n")
        f.write(f"newmtl {mat_minus}\n")
        f.write(f"Ka 0.0200 0.0200 0.0200\n")
        kd = pal.mtl_kd_minus
        f.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
        ks = tuple(c * 0.85 for c in pal.mtl_ks)
        f.write(f"Ks {ks[0]:.4f} {ks[1]:.4f} {ks[2]:.4f}\n")
        f.write(f"Ns {pal.mtl_ns * 0.8:.1f}\n")
        f.write(f"d 1.0\n")
        f.write(f"illum 2\n")

    # --- 6. Flux wireframe ---
    print(f"  [6/6] Flux wireframe...")
    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        out_dir / f"k{k}_flux.obj",
        pair.f_plus.vertices, flux_edges, colors_plus,
        object_name=f"bonnet_k{k}_flux",
        header=header + ["Flux wireframe: metric-resonance edges"],
    )

    dt_total = time.time() - t0

    # --- Metadata ---
    bb_plus = pair.f_plus.vertices
    bb_minus = pair.f_minus.vertices
    metadata = {
        "k": k,
        "palette": pal.name,
        "tau_imag": TAU_IMAG,
        "s1": seed["s1"],
        "delta": seed["delta"],
        "s2": seed["s2"],
        "theta_deg": theta_deg,
        "epsilon": EPSILON,
        "resolution": RESOLUTION,
        "n_vertices": int(len(pair.f_plus.vertices)),
        "n_triangles": int(len(tris_plus)),
        "n_flux_edges": int(len(flux_edges)),
        "time_seconds": round(dt_total, 1),
        "omega": float(torus.omega),
        "asymptotic_formula": "delta(k) = 7.0814165/sqrt(k) * (1 - 0.85766/k + 0.75/k^2)",
        "asymptotic_prediction": round(7.0814165 / np.sqrt(k) * (1 - 0.85766/k + 0.75/k**2), 6),
        "actual_delta": seed["delta"],
        "bbox_plus": {
            "min": bb_plus.min(axis=0).tolist(),
            "max": bb_plus.max(axis=0).tolist(),
        },
        "bbox_minus": {
            "min": bb_minus.min(axis=0).tolist(),
            "max": bb_minus.max(axis=0).tolist(),
        },
        "color_scheme": {
            "name": pal.name,
            "hue_range": list(pal.hue_range),
            "mtl_kd_plus": list(pal.mtl_kd_plus),
            "mtl_kd_minus": list(pal.mtl_kd_minus),
        },
        "files": [
            f"k{k}_f_plus.obj",
            f"k{k}_f_minus.obj",
            mtl_name,
            f"k{k}_flux.obj",
        ],
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ k={k} complete — {dt_total:.1f}s")
    for fn in metadata["files"]:
        print(f"    {out_dir / fn}")

    return metadata


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 65)
    print("  Bonnet Torus — High-k OBJ Generation")
    print(f"  k = {', '.join(str(s['k']) for s in SEEDS)}")
    print(f"  Gauge: s₁ = {S1_GAUGE}")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"  Formula: δ(k) = 7.0814165/√k · (1 − 0.858/k + 0.75/k²)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = []
    t_start = time.time()

    for seed in SEEDS:
        meta = generate_one_k(seed)
        all_meta.append(meta)

    # Combined metadata
    combined = {
        "project": "Bonnet's Problem — Phase 18 High-k Visualization",
        "pipeline": "Bonnet (Theorem 7 + Eq. 49)",
        "gauge": f"s1={S1_GAUGE}",
        "tau_imag": TAU_IMAG,
        "total_time_seconds": round(time.time() - t_start, 1),
        "entries": all_meta,
        "delta_evolution": {
            "formula": "delta = A/sqrt(k) * (1 - c1/k + c3/k^2)",
            "A": 7.0814165,
            "c1": 0.85766,
            "c3": 0.75,
            "delta_50": SEEDS[0]["delta"],
            "delta_1000": SEEDS[-1]["delta"],
            "ratio_50_1000": round(SEEDS[0]["delta"] / SEEDS[-1]["delta"], 3),
        },
    }
    combined_path = OUTPUT_DIR / "high_k_metadata.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  COMPLETE — {len(all_meta)} fold values generated")
    print(f"  Total: {combined['total_time_seconds']:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")

    # Summary table
    print(f"\n  {'k':>5s}  {'δ':>9s}  {'δ_asym':>9s}  {'err%':>7s}  {'Palette':>10s}  {'Time':>6s}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*10}  {'─'*6}")
    for m in all_meta:
        err_pct = abs(m["actual_delta"] - m["asymptotic_prediction"]) / m["actual_delta"] * 100
        print(f"  {m['k']:>5d}  {m['actual_delta']:>9.6f}  "
              f"{m['asymptotic_prediction']:>9.6f}  {err_pct:>6.3f}%  "
              f"{m['palette']:>10s}  {m['time_seconds']:>5.1f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
