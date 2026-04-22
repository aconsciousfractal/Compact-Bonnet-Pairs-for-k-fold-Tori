"""
LEGACY / auxiliary runner.

Phase 18.10+ — Base isothermic torus OBJ (before Bonnet) for k=50..1000.

Generates the BASE isothermic torus (no Bonnet rotation) for each high-k seed,
so the user can compare:
  k{N}_base.obj  ← isothermic torus (this script)
  k{N}_f_plus.obj / k{N}_f_minus.obj  ← Bonnet pair (generate_high_k_obj.py)

Uses same spectral parameters as the Bonnet pair script.
Saves ~60s/k by skipping the Bonnet pair computation.

Output per k (in same directory as Bonnet OBJ):
  - k{N}_base.obj      (solid mesh with per-vertex colors + normals)
  - k{N}_base.mtl      (single material, silver-blue tint)
  - k{N}_base_flux.obj (flux wireframe)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.isothermic_torus import TorusParameters, compute_torus
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
RESOLUTION = 120
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "obj"

# Same seeds as generate_high_k_obj.py — exact from full_series_k10_1000.json
SEEDS = [
    {"k":   50, "s1": -8.5, "delta": 0.984585, "s2": 6.864665},
    {"k":  100, "s1": -8.5, "delta": 0.702122, "s2": 7.090180},
    {"k":  250, "s1": -8.5, "delta": 0.446337, "s2": 7.229097},
    {"k":  500, "s1": -8.5, "delta": 0.316148, "s2": 7.276017},
    {"k":  750, "s1": -8.5, "delta": 0.258281, "s2": 7.291726},
    {"k": 1000, "s1": -8.5, "delta": 0.223742, "s2": 7.299593},
]


@dataclass
class BasePalette:
    """Color specification for base torus — cool metallic tones."""
    name: str
    hue_range: tuple
    sat_base: float
    val_base: float
    mtl_kd: tuple
    mtl_ks: tuple
    mtl_ns: float


# Cool/metallic palette for base torus — visually distinct from warm Bonnet palettes
PALETTES = {
    50: BasePalette(
        name="Titanium",
        hue_range=(0.55, 0.62),
        sat_base=0.45, val_base=0.88,
        mtl_kd=(0.62, 0.68, 0.78),   # light steel blue
        mtl_ks=(0.50, 0.55, 0.60), mtl_ns=100.0,
    ),
    100: BasePalette(
        name="Platinum",
        hue_range=(0.53, 0.60),
        sat_base=0.40, val_base=0.90,
        mtl_kd=(0.70, 0.72, 0.80),   # platinum
        mtl_ks=(0.55, 0.55, 0.62), mtl_ns=110.0,
    ),
    250: BasePalette(
        name="Chrome",
        hue_range=(0.50, 0.58),
        sat_base=0.35, val_base=0.92,
        mtl_kd=(0.75, 0.78, 0.82),   # chrome silver
        mtl_ks=(0.60, 0.60, 0.65), mtl_ns=120.0,
    ),
    500: BasePalette(
        name="Pewter",
        hue_range=(0.48, 0.56),
        sat_base=0.32, val_base=0.85,
        mtl_kd=(0.65, 0.68, 0.72),   # pewter
        mtl_ks=(0.48, 0.50, 0.55), mtl_ns=95.0,
    ),
    750: BasePalette(
        name="Graphite",
        hue_range=(0.52, 0.60),
        sat_base=0.28, val_base=0.78,
        mtl_kd=(0.55, 0.58, 0.65),   # graphite
        mtl_ks=(0.42, 0.45, 0.50), mtl_ns=90.0,
    ),
    1000: BasePalette(
        name="Mercury",
        hue_range=(0.00, 1.00),       # full spectrum shimmer
        sat_base=0.20, val_base=0.92,
        mtl_kd=(0.80, 0.80, 0.85),   # liquid mercury
        mtl_ks=(0.70, 0.70, 0.75), mtl_ns=140.0,
    ),
}


def compute_vertex_normals(vertices: np.ndarray, triangles) -> np.ndarray:
    """Compute per-vertex normals from triangle faces via area-weighted average."""
    N = len(vertices)
    normals = np.zeros((N, 3))
    for tri in triangles:
        i0, i1, i2 = tri
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        face_n = np.cross(v1 - v0, v2 - v0)
        normals[i0] += face_n
        normals[i1] += face_n
        normals[i2] += face_n
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens < 1e-12, 1.0, lens)
    return normals / lens


def generate_one_k(seed: dict) -> dict:
    """Generate base torus OBJ for one k value (no Bonnet)."""
    k = seed["k"]
    pal = PALETTES[k]
    theta_deg = 360.0 / k

    print(f"\n{'='*65}")
    print(f"  k={k:>4d}  [{pal.name}]  δ={seed['delta']:.6f}  "
          f"s₂={seed['s2']:.6f}  θ={theta_deg:.2f}° — BASE TORUS")
    print(f"{'='*65}")

    t0 = time.time()
    out_dir = OUTPUT_DIR / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Isothermic torus (only — no Bonnet pair) ---
    print(f"  [1/4] Computing isothermic torus (Theorem 7)...")
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

    n_u, n_v = RESOLUTION, RESOLUTION

    # --- 2. Curvature coloring ---
    print(f"  [2/4] Computing curvature-mapped colors [{pal.name}]...")
    curv = compute_curvature_proxy(torus.vertices, n_u, n_v)
    colors = color_map_surface(
        n_u, n_v, curv,
        hue_range=pal.hue_range,
        sat_base=pal.sat_base,
        val_base=pal.val_base,
    )

    # --- 3. Triangulate + normals + export ---
    print(f"  [3/4] Triangulating + normals + writing OBJ...")
    tris = triangulate_quads(torus.faces)
    nrm = compute_vertex_normals(torus.vertices, tris)

    header = [
        f"Bonnet Torus — BASE isothermic torus (k={k})",
        f"k-fold symmetry: {k}",
        f"Spectral: δ={seed['delta']:.6f}, s₁={seed['s1']}, s₂={seed['s2']:.6f}",
        f"τ₀ = 0.5 + {TAU_IMAG}i  (gauge s₁={S1_GAUGE})",
        f"Resolution: {RESOLUTION}×{RESOLUTION} = {RESOLUTION**2} vertices",
        f"Surface type: ISOTHERMIC TORUS (before Bonnet rotation)",
        f"Compare with: k{k}_f_plus.obj / k{k}_f_minus.obj (Bonnet pair)",
        f"Palette: {pal.name}",
        f"Generated by: Bonnet Pipeline",
    ]

    mtl_name = f"k{k}_base.mtl"
    mat_name = f"k{k}_base"

    write_obj(
        out_dir / f"k{k}_base.obj",
        torus.vertices, tris,
        normals=nrm, colors=colors,
        object_name=f"isothermic_k{k}_base",
        header=header,
        mtl_file=mtl_name, material_name=mat_name,
    )

    write_mtl(
        out_dir / mtl_name,
        mat_name,
        ka=(0.03, 0.03, 0.03),
        kd=pal.mtl_kd,
        ks=pal.mtl_ks,
        ns=pal.mtl_ns,
        comment=f"isothermic torus k={k} — {pal.name} (base, pre-Bonnet)",
    )

    # --- 4. Flux wireframe ---
    print(f"  [4/4] Flux wireframe...")
    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        out_dir / f"k{k}_base_flux.obj",
        torus.vertices, flux_edges, colors,
        object_name=f"isothermic_k{k}_base_flux",
        header=header + ["Flux wireframe: base torus metric-resonance edges"],
    )

    dt_total = time.time() - t0

    bb = torus.vertices
    metadata = {
        "k": k,
        "palette": pal.name,
        "category": "base_torus",
        "surface_type": "isothermic_torus_before_bonnet",
        "tau_imag": TAU_IMAG,
        "s1": seed["s1"],
        "delta": seed["delta"],
        "s2": seed["s2"],
        "theta_deg": theta_deg,
        "resolution": RESOLUTION,
        "n_vertices": int(len(torus.vertices)),
        "n_triangles": int(len(tris)),
        "n_flux_edges": int(len(flux_edges)),
        "time_seconds": round(dt_total, 1),
        "omega": float(torus.omega),
        "bbox": {
            "min": bb.min(axis=0).tolist(),
            "max": bb.max(axis=0).tolist(),
        },
        "color_scheme": {
            "name": pal.name,
            "hue_range": list(pal.hue_range),
            "mtl_kd": list(pal.mtl_kd),
        },
        "files": [f"k{k}_base.obj", mtl_name, f"k{k}_base_flux.obj"],
        "companion_bonnet_files": [f"k{k}_f_plus.obj", f"k{k}_f_minus.obj", f"k{k}_material.mtl"],
    }
    meta_path = out_dir / "base_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ k={k} base torus — {dt_total:.1f}s")
    for fn in metadata["files"]:
        print(f"    {out_dir / fn}")

    return metadata


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 65)
    print("  Base Isothermic Torus OBJ (pre-Bonnet)")
    print(f"  k = {', '.join(str(s['k']) for s in SEEDS)}")
    print(f"  Gauge: s₁ = {S1_GAUGE}")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  NOTE: No Bonnet pair — torus only (~70s/k instead of ~135s)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = []
    t_start = time.time()

    for seed in SEEDS:
        meta = generate_one_k(seed)
        all_meta.append(meta)

    # Combined metadata
    combined_path = OUTPUT_DIR / "base_torus_metadata.json"
    combined = {
        "project": "Bonnet's Problem — Base Isothermic Torus OBJ",
        "pipeline": "Theorem 7 only — no Bonnet rotation",
        "gauge": f"s1={S1_GAUGE}",
        "tau_imag": TAU_IMAG,
        "total_time_seconds": round(time.time() - t_start, 1),
        "note": "Base torus for comparison with Bonnet pair (f⁺/f⁻)",
        "entries": all_meta,
    }
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    t_total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  COMPLETE — {len(all_meta)} base tori generated")
    print(f"  Total: {t_total:.1f}s  (no Bonnet pair → ~50% faster)")
    print(f"  Output: {OUTPUT_DIR}")

    # Summary table
    print(f"\n  {'k':>5s}  {'δ':>9s}  {'s₂':>9s}  {'ω':>9s}  {'Palette':>10s}  {'Time':>6s}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*6}")
    for m in all_meta:
        s = next(s for s in SEEDS if s["k"] == m["k"])
        print(f"  {m['k']:>5d}  {s['delta']:>9.6f}  {s['s2']:>9.6f}  "
              f"{m['omega']:>9.6f}  {m['palette']:>10s}  {m['time_seconds']:>5.1f}s")

    print(f"\n  Per-k file comparison:")
    print(f"    k{{N}}_base.obj       ← isothermic torus (THIS script)")
    print(f"    k{{N}}_f_plus.obj     ← Bonnet f⁺ (generate_high_k_obj.py)")
    print(f"    k{{N}}_f_minus.obj    ← Bonnet f⁻ (generate_high_k_obj.py)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
