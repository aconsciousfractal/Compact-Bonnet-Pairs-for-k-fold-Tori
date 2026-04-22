"""
LEGACY / auxiliary runner.

Phase 11 — Visual OBJ export for higher-fold Bonnet pairs k=5, 6, 7.

Generates 3 objects per fold at different s₁ values from the Lemma 9 sweep,
producing both f⁺ and f⁻ surfaces for each.

Total output: 3 folds × 3 variants × 2 surfaces = 18 OBJ files.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.seed_catalog import TAU_IMAG
from src.isothermic_torus import (
    TorusParameters,
    compute_torus,
    export_torus_obj,
)
from src.bonnet_pair import compute_bonnet_pair, export_bonnet_pair_obj
from src.bonnet_flux_utils import (
    compute_curvature_proxy,
    color_map_surface,
    compute_flux_edges,
    write_flux_wireframe_obj,
)

OUTPUT_DIR = Path("results/obj/higher_fold/visual_obj")

# 3 samples per fold: seed value, one at large |s₁|, one at small |s₁|
# Parameters from the Lemma 9 sweep (lemma9_sweep.json)
SAMPLES = {
    5: [
        {"label": "5fold_seed",    "s1": -3.80, "delta": 1.678758, "s2": 1.188433},
        {"label": "5fold_deep",    "s1": -8.00, "delta": 2.616968, "s2": 3.572107},
        {"label": "5fold_shallow", "s1": -3.00, "delta": 1.464690, "s2": 0.628483},
    ],
    6: [
        {"label": "6fold_seed",    "s1": -4.50, "delta": 1.749167, "s2": 1.813323},
        {"label": "6fold_deep",    "s1": -8.00, "delta": 2.447783, "s2": 3.977120},
        {"label": "6fold_shallow", "s1": -3.00, "delta": 1.370707, "s2": 0.752906},
    ],
    7: [
        {"label": "7fold_seed",    "s1": -6.00, "delta": 1.959258, "s2": 2.979873},
        {"label": "7fold_deep",    "s1": -9.00, "delta": 2.462744, "s2": 4.948258},
        {"label": "7fold_shallow", "s1": -3.50, "delta": 1.415894, "s2": 1.261727},
    ],
}

# Visual resolution — good for viewing without being too heavy
U_RES = 100
V_RES = 100
EPSILON = 0.3  # Bonnet parameter for pair construction

# Color schemes per fold (distinct from k=3 blue-teal and k=4 amber-gold)
# Each: (hue_range, sat_base, val_base) for f⁺, then hue_shift for f⁻
FLUX_COLORS = {
    5: {  # Emerald / green
        "hue_range": (0.28, 0.42),
        "sat_base_plus": 0.84, "val_base_plus": 0.74,
        "sat_base_minus": 0.74, "val_base_minus": 0.68,
        "hue_shift_minus": 0.04,
    },
    6: {  # Violet / purple
        "hue_range": (0.72, 0.86),
        "sat_base_plus": 0.86, "val_base_plus": 0.76,
        "sat_base_minus": 0.76, "val_base_minus": 0.70,
        "hue_shift_minus": 0.05,
    },
    7: {  # Coral / red-orange
        "hue_range": (0.95, 1.09),
        "sat_base_plus": 0.88, "val_base_plus": 0.75,
        "sat_base_minus": 0.78, "val_base_minus": 0.69,
        "hue_shift_minus": 0.04,
    },
}


def generate_pair(k: int, sample: dict) -> None:
    """Generate torus + Bonnet pair OBJ for one sample."""
    label = sample["label"]
    print(f"  {label}: δ={sample['delta']:.6f}, s₁={sample['s1']:.2f}, s₂={sample['s2']:.6f}")

    # Build torus parameters
    params = TorusParameters(
        tau_imag=TAU_IMAG,
        delta=sample["delta"],
        s1=sample["s1"],
        s2=sample["s2"],
        u_res=U_RES,
        v_res=V_RES,
        symmetry_fold=k,
    )

    # Compute base torus (auto-w(v) via Theorem 7)
    t0 = time.time()
    torus = compute_torus(params)
    dt_torus = time.time() - t0

    # Compute Bonnet pair
    t1 = time.time()
    pair = compute_bonnet_pair(torus, EPSILON)
    dt_pair = time.time() - t1

    # Export
    out_dir = OUTPUT_DIR / f"k{k}"
    p_plus, p_minus = export_bonnet_pair_obj(pair, out_dir, prefix=label)

    # Also export the base torus
    p_base = export_torus_obj(torus, out_dir / f"{label}_base.obj",
                              object_name=f"{label}_base")

    # --- Flux wireframe OBJs ---
    fc = FLUX_COLORS[k]
    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, U_RES, V_RES)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, U_RES, V_RES)
    colors_plus = color_map_surface(
        U_RES, V_RES, curv_plus,
        hue_range=fc["hue_range"],
        sat_base=fc["sat_base_plus"],
        val_base=fc["val_base_plus"],
    )
    colors_minus = color_map_surface(
        U_RES, V_RES, curv_minus,
        hue_range=fc["hue_range"],
        sat_base=fc["sat_base_minus"],
        val_base=fc["val_base_minus"],
        hue_shift=fc["hue_shift_minus"],
    )
    flux_edges = compute_flux_edges(U_RES, V_RES)
    flux_header = [
        f"PAPP Phase 11 Flux Wireframe — k={k}",
        f"Label: {label}",
        f"s1={sample['s1']}, delta={sample['delta']}, s2={sample['s2']}",
        f"Resolution: {U_RES}x{V_RES}, epsilon={EPSILON}",
    ]
    p_flux_plus = out_dir / f"{label}_f_plus_flux.obj"
    p_flux_minus = out_dir / f"{label}_f_minus_flux.obj"
    write_flux_wireframe_obj(
        p_flux_plus, pair.f_plus.vertices, flux_edges, colors_plus,
        object_name=f"{label}_f_plus_flux", header=flux_header,
    )
    write_flux_wireframe_obj(
        p_flux_minus, pair.f_minus.vertices, flux_edges, colors_minus,
        object_name=f"{label}_f_minus_flux", header=flux_header,
    )

    nv = len(torus.vertices)
    nf = len(torus.faces)
    print(f"    torus: {dt_torus:.1f}s, pair: {dt_pair:.1f}s")
    print(f"    mesh: {nv} verts, {nf} faces")
    print(f"    → {p_base.name}")
    print(f"    → {p_plus.name}")
    print(f"    → {p_minus.name}")
    print(f"    → {p_flux_plus.name}  (flux)")
    print(f"    → {p_flux_minus.name}  (flux)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  PHASE 11 — VISUAL OBJ EXPORT: k=5, 6, 7 Bonnet Pairs")
    print(f"  Resolution: {U_RES}×{V_RES}, ε = {EPSILON}")
    print("=" * 70)

    total_files = 0
    for k in [5, 6, 7]:
        print(f"\n{'─' * 70}")
        print(f"  k={k} ({len(SAMPLES[k])} variants)")
        print(f"{'─' * 70}")
        for sample in SAMPLES[k]:
            try:
                generate_pair(k, sample)
                total_files += 5  # base + f_plus + f_minus + 2 flux
            except Exception as e:
                print(f"    ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done: {total_files} OBJ files in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
