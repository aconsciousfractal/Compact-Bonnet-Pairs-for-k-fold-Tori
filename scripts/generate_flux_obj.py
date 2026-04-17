"""
Bonnet Torus — Flux High-Quality OBJ Export

Generates 3 Bonnet pairs at exact closure τ* values from Phase 7,
rendered with per-vertex curvature-based coloring,
smooth normals, and Wavefront MTL materials.

3 pairs selected for visual diversity:
  1. ratio 4/5, τ*=0.17601 — "Solar" (warm fire gradient)
  2. ratio 1/2, τ*=0.20818 — "Ocean" (cool deep-sea gradient)
  3. ratio 1/1, τ*=0.16006 — "Aurora" (spectral plasma gradient)

Each pair produces: f_plus.obj, f_minus.obj, material.mtl

Output: results/ folder

Usage:
    cd P:\\...\\Bonnet's Problem
    python generate_flux_bonnet.py
"""
from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# --- Path setup ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.isothermic_torus import (
    TorusParameters, TorusResult, compute_torus,
    build_torus_faces, compute_vertex_normals,
)
from src.bonnet_pair import compute_bonnet_pair, BonnetPairResult

# Import OBJ writer
from src.obj_writer import write_obj, write_mtl

# Import shared flux utilities
from src.bonnet_flux_utils import (
    compute_curvature_proxy,
    color_map_surface,
    triangulate_quads,
    compute_flux_edges,
    write_flux_wireframe_obj,
)


# ============================================================================
# Configuration
# ============================================================================

RESOLUTION = 80  # 80×80 grid — 6400 vertices (matches Flux Calabi-Yau standard)
EPSILON = 0.3    # Bonnet parameter
OUTPUT_DIR = REPO_ROOT / "results"

@dataclass
class BonnetFluxSpec:
    """Specification for one Bonnet torus flux visualization."""
    name: str           # e.g. "bonnet_4_5"
    ratio_label: str    # e.g. "4/5"
    n: int              # numerator
    m: int              # denominator
    tau_star: float     # exact τ* from Phase 7
    hue_range: tuple    # (hue_start, hue_end) in [0, 1]
    sat_base: float     # base saturation
    val_base: float     # base value
    mtl_kd: tuple       # diffuse color for material

SPECS = [
    BonnetFluxSpec(
        name="bonnet_4_5",
        ratio_label="4/5",
        n=4, m=5,
        tau_star=0.17601476,
        hue_range=(0.0, 0.12),       # red → orange
        sat_base=0.85,
        val_base=0.75,
        mtl_kd=(0.85, 0.35, 0.10),   # warm orange
    ),
    BonnetFluxSpec(
        name="bonnet_1_2",
        ratio_label="1/2",
        n=1, m=2,
        tau_star=0.20818,
        hue_range=(0.52, 0.68),       # cyan → blue
        sat_base=0.80,
        val_base=0.70,
        mtl_kd=(0.10, 0.35, 0.80),   # deep blue
    ),
    BonnetFluxSpec(
        name="bonnet_1_1",
        ratio_label="1/1",
        n=1, m=1,
        tau_star=0.16006,
        hue_range=(0.28, 0.82),       # green → magenta (aurora)
        sat_base=0.75,
        val_base=0.70,
        mtl_kd=(0.20, 0.70, 0.45),   # emerald
    ),
]


# ============================================================================
# Main generation pipeline
# ============================================================================

def generate_one_pair(spec: BonnetFluxSpec, output_dir: Path) -> dict:
    """Generate a single Bonnet pair at high quality and export all artifacts."""

    print(f"\n{'='*60}")
    print(f"  Generating: {spec.name} (ratio {spec.ratio_label}, τ*={spec.tau_star:.5f})")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION} = {RESOLUTION**2} vertices")
    print(f"{'='*60}")

    t0 = time.time()

    # --- Step 1: Compute isothermic torus ---
    print(f"  [1/5] Computing isothermic torus...")
    params = TorusParameters(
        tau_imag=spec.tau_star,
        u_res=RESOLUTION,
        v_res=RESOLUTION,
        symmetry_fold=spec.m,
        w0=0.2,
    )
    torus = compute_torus(params)
    t_torus = time.time() - t0
    print(f"         Done ({t_torus:.1f}s). Vertices: {len(torus.vertices)}")

    # --- Step 2: Compute Bonnet pair ---
    print(f"  [2/5] Computing Bonnet pair (ε={EPSILON})...")
    pair = compute_bonnet_pair(torus, EPSILON)
    t_pair = time.time() - t0
    print(f"         Done ({t_pair - t_torus:.1f}s).")

    n_u, n_v = RESOLUTION, RESOLUTION

    # --- Step 3: Compute curvature and colors ---
    print(f"  [3/5] Computing curvature coloring...")

    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)

    colors_plus = color_map_surface(
        n_u, n_v, curv_plus,
        hue_range=spec.hue_range,
        sat_base=spec.sat_base,
        val_base=spec.val_base,
        hue_shift=0.0,
    )
    colors_minus = color_map_surface(
        n_u, n_v, curv_minus,
        hue_range=spec.hue_range,
        sat_base=spec.sat_base * 0.85,
        val_base=spec.val_base * 0.90,
        hue_shift=0.06,  # slight hue shift for f⁻
    )

    # --- Step 4: Triangulate and compute normals ---
    print(f"  [4/5] Triangulating and computing normals...")

    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)

    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    # --- Step 5: Export ---
    print(f"  [5/5] Exporting OBJ + MTL + flux wireframe...")

    pair_dir = output_dir
    pair_dir.mkdir(parents=True, exist_ok=True)

    header_lines = [
        f"Bonnet Torus Flux Visualization",
        f"Ratio: {spec.ratio_label} (n={spec.n}, m={spec.m})",
        f"tau*: {spec.tau_star:.8f}",
        f"Resolution: {RESOLUTION}x{RESOLUTION}",
        f"Epsilon: {EPSILON}",
        f"Generated by: Bonnet Pipeline",
    ]

    mtl_name = f"{spec.name}_material.mtl"

    # f⁺ solid mesh
    obj_plus_path = write_obj(
        pair_dir / f"{spec.name}_f_plus.obj",
        pair.f_plus.vertices,
        tris_plus,
        normals=nrm_plus,
        colors=colors_plus,
        object_name=f"{spec.name}_f_plus",
        header=header_lines + [f"Surface: f⁺ (positive Bonnet branch)"],
        mtl_file=mtl_name,
        material_name=f"{spec.name}_plus",
    )

    # f⁻ solid mesh
    obj_minus_path = write_obj(
        pair_dir / f"{spec.name}_f_minus.obj",
        pair.f_minus.vertices,
        tris_minus,
        normals=nrm_minus,
        colors=colors_minus,
        object_name=f"{spec.name}_f_minus",
        header=header_lines + [f"Surface: f⁻ (negative Bonnet branch)"],
        mtl_file=mtl_name,
        material_name=f"{spec.name}_minus",
    )

    # Material file (two materials: plus + minus)
    mtl_path = pair_dir / mtl_name

    kd_plus = spec.mtl_kd
    kd_minus = tuple(c * 0.7 for c in spec.mtl_kd)  # darker for f⁻

    write_mtl(
        mtl_path,
        f"{spec.name}_plus",
        ka=(0.03, 0.03, 0.03),
        kd=kd_plus,
        ks=(0.35, 0.30, 0.25),
        ns=80.0,
    )
    # Append second material
    with mtl_path.open("a", encoding="utf-8") as f:
        f.write(f"\nnewmtl {spec.name}_minus\n")
        f.write(f"Ka {0.03:.4f} {0.03:.4f} {0.03:.4f}\n")
        f.write(f"Kd {kd_minus[0]:.4f} {kd_minus[1]:.4f} {kd_minus[2]:.4f}\n")
        f.write(f"Ks {0.30:.4f} {0.25:.4f} {0.20:.4f}\n")
        f.write(f"Ns 60.0\n")
        f.write(f"d 1.0\n")

    # Flux wireframe
    flux_edges_plus = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        pair_dir / f"{spec.name}_flux.obj",
        pair.f_plus.vertices,
        flux_edges_plus,
        colors_plus,
        object_name=f"{spec.name}_flux",
        header=header_lines + ["Flux wireframe: metric-resonance edges + diagonal reinforcement"],
    )

    t_total = time.time() - t0

    # Metrics
    metrics = {
        "name": spec.name,
        "ratio": spec.ratio_label,
        "n": spec.n,
        "m": spec.m,
        "tau_star": spec.tau_star,
        "epsilon": EPSILON,
        "resolution": RESOLUTION,
        "n_vertices": int(len(pair.f_plus.vertices)),
        "n_triangles": int(len(tris_plus)),
        "n_flux_edges": int(len(flux_edges_plus)),
        "time_seconds": round(t_total, 1),
        "files": {
            "f_plus_obj": str(obj_plus_path.name),
            "f_minus_obj": str(obj_minus_path.name),
            "material": mtl_name,
            "flux_wireframe": f"{spec.name}_flux.obj",
        },
        "bbox_plus": {
            "min": pair.f_plus.vertices.min(axis=0).tolist(),
            "max": pair.f_plus.vertices.max(axis=0).tolist(),
        },
        "bbox_minus": {
            "min": pair.f_minus.vertices.min(axis=0).tolist(),
            "max": pair.f_minus.vertices.max(axis=0).tolist(),
        },
        "omega": float(torus.omega),
        "V_period": float(torus.metrics.get("V_period", 0)),
        "isometry_max_scalar": float(torus.metrics.get("max_scalar_part", 0)),
        "color_scheme": {
            "hue_range": list(spec.hue_range),
            "sat_base": spec.sat_base,
            "val_base": spec.val_base,
        },
    }

    print(f"\n  Output files:")
    print(f"    {obj_plus_path}")
    print(f"    {obj_minus_path}")
    print(f"    {mtl_path}")
    print(f"    {pair_dir / f'{spec.name}_flux.obj'}")
    print(f"  Total time: {t_total:.1f}s")

    return metrics


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 60)
    print("  Bonnet Torus — Flux High-Quality Generation")
    print("  3 pairs × 2 surfaces = 6 solid OBJ + 3 flux wireframes")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION} ({RESOLUTION**2} vertices each)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    t_start = time.time()

    for spec in SPECS:
        metrics = generate_one_pair(spec, OUTPUT_DIR)
        all_metrics.append(metrics)

    # Write combined metadata
    meta_path = OUTPUT_DIR / "bonnet_flux_metadata.json"
    metadata = {
        "project": "Bonnet's Problem — Flux Visualization",
        "phase": "7.5 — High-Quality Flux Export",
        "system": "bonnet-pipeline",
        "flux_standard": "2D Parametric Surface Mesh + Diagonal Reinforcement",
        "total_time_seconds": round(time.time() - t_start, 1),
        "pairs": all_metrics,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {len(all_metrics)} pairs generated")
    print(f"  Total time: {metadata['total_time_seconds']:.1f}s")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
