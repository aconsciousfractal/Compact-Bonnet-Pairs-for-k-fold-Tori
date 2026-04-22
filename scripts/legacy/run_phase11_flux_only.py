"""
LEGACY / auxiliary runner.

Phase 11 — Flux wireframe OBJ generation for existing k=5,6,7 surfaces.

Reads the already-generated surface OBJs (f_plus / f_minus) and produces
flux wireframe versions with per-vertex HSV color mapping.

No torus recomputation needed — vertices are parsed directly from existing OBJs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bonnet_flux_utils import (
    compute_curvature_proxy,
    color_map_surface,
    compute_flux_edges,
    write_flux_wireframe_obj,
)

BASE_DIR = Path("results/obj/higher_fold/visual_obj")
U_RES = 100
V_RES = 100

# Color schemes per fold (distinct from k=3 blue-teal / k=4 amber-gold)
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

LABELS = {
    5: ["5fold_seed", "5fold_deep", "5fold_shallow"],
    6: ["6fold_seed", "6fold_deep", "6fold_shallow"],
    7: ["7fold_seed", "7fold_deep", "7fold_shallow"],
}


def parse_obj_vertices(path: Path) -> np.ndarray:
    """Read vertex positions from an OBJ file (lines starting with 'v ')."""
    verts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)


def generate_flux(k: int, label: str) -> int:
    """Generate flux wireframe OBJs for one variant. Returns count of files."""
    fc = FLUX_COLORS[k]
    out_dir = BASE_DIR / f"k{k}"
    count = 0

    for surf_type in ("f_plus", "f_minus"):
        src = out_dir / f"{label}_{surf_type}.obj"
        if not src.exists():
            print(f"    SKIP {src.name} (not found)")
            continue

        vertices = parse_obj_vertices(src)
        expected = U_RES * V_RES
        if len(vertices) != expected:
            print(f"    WARN {src.name}: {len(vertices)} verts (expected {expected})")
            continue

        curv = compute_curvature_proxy(vertices, U_RES, V_RES)

        if surf_type == "f_plus":
            colors = color_map_surface(
                U_RES, V_RES, curv,
                hue_range=fc["hue_range"],
                sat_base=fc["sat_base_plus"],
                val_base=fc["val_base_plus"],
            )
        else:
            colors = color_map_surface(
                U_RES, V_RES, curv,
                hue_range=fc["hue_range"],
                sat_base=fc["sat_base_minus"],
                val_base=fc["val_base_minus"],
                hue_shift=fc["hue_shift_minus"],
            )

        flux_edges = compute_flux_edges(U_RES, V_RES)
        header = [
            f"PAPP Phase 11 Flux Wireframe — k={k}",
            f"Label: {label}  Surface: {surf_type}",
            f"Resolution: {U_RES}x{V_RES}",
            f"Source: {src.name}",
        ]

        dst = out_dir / f"{label}_{surf_type}_flux.obj"
        write_flux_wireframe_obj(
            dst, vertices, flux_edges, colors,
            object_name=f"{label}_{surf_type}_flux",
            header=header,
        )
        size_kb = dst.stat().st_size / 1024
        print(f"    → {dst.name}  ({size_kb:.0f} KB)")
        count += 1

    return count


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  PHASE 11 — FLUX WIREFRAME OBJ EXPORT (from existing surfaces)")
    print(f"  Grid: {U_RES}×{V_RES}")
    print("=" * 70)

    total = 0
    for k in [5, 6, 7]:
        print(f"\n{'─' * 70}")
        print(f"  k={k}  —  {FLUX_COLORS[k]['hue_range']}")
        print(f"{'─' * 70}")
        for label in LABELS[k]:
            print(f"  {label}:")
            total += generate_flux(k, label)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done: {total} flux OBJ files in {elapsed:.1f}s")
    print(f"  Output: {BASE_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
