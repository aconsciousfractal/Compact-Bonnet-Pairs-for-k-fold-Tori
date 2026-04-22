"""
LEGACY / auxiliary runner.

Generate a visual companion for the perturbative 4-fold Bonnet example.

This is not the best validation case.
It intentionally uses a larger Bonnet epsilon to increase visual separation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.obj_writer import write_mtl, write_obj

from src.bonnet_flux_utils import (
    color_map_surface,
    compute_curvature_proxy,
    compute_flux_edges,
    triangulate_quads,
    write_flux_wireframe_obj,
)
from src.bonnet_pair import closure_gate
from src.isothermic_torus import compute_vertex_normals
from src.theorem9_perturbation import (
    build_default_theorem9_basis,
    build_theorem9_seed_from_theorem7,
    solve_theorem9_perturbation,
    verify_theorem9_bonnet_pipeline,
)


TAU_IMAG = 0.3205128205
DELTA = 1.61245155
S1 = -3.13060628
S2 = 0.5655771591
SYMMETRY_FOLD = 4

EPSILON_PERT = 5e-4
EPSILON_GEOM = 0.30

SEED_HALF_SAMPLES = 61
SOLVE_N_POINTS = 80
SOLVE_LINEARIZATION_N_POINTS = 60

EXPORT_RESOLUTION = 160

NAME = "bonnet_theorem9_perturbed_4fold_visual"
OUTPUT_DIR = REPO_ROOT / "results" / "obj" / "theorem9" / NAME


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    print("=" * 74)
    print("  BONNET THEOREM 9 VISUAL COMPANION (4-FOLD)")
    print("=" * 74)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Perturbation epsilon: {EPSILON_PERT}")
    print(f"  Bonnet epsilon: {EPSILON_GEOM}")
    print(f"  Export resolution: {EXPORT_RESOLUTION}x{EXPORT_RESOLUTION}")

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  [1/3] Building and solving perturbative seed...")
    seed = build_theorem9_seed_from_theorem7(
        tau_imag=TAU_IMAG,
        delta=DELTA,
        s1=S1,
        s2=S2,
        symmetry_fold=SYMMETRY_FOLD,
        n_half_samples=SEED_HALF_SAMPLES,
    )
    basis = build_default_theorem9_basis(seed)
    solve = solve_theorem9_perturbation(
        seed=seed,
        epsilon=EPSILON_PERT,
        basis=basis,
        n_points=SOLVE_N_POINTS,
        linearization_n_points=SOLVE_LINEARIZATION_N_POINTS,
    )
    if not solve.success:
        raise RuntimeError(f"Theorem 9 solve failed: {solve.message}")

    print("  [2/3] Running Bonnet pipeline...")
    verification = verify_theorem9_bonnet_pipeline(
        seed=seed,
        solve=solve,
        epsilon_geom=EPSILON_GEOM,
        u_res=EXPORT_RESOLUTION,
        v_res=EXPORT_RESOLUTION,
    )
    pair = verification.pair
    closure = closure_gate(pair)
    n_u = EXPORT_RESOLUTION
    n_v = EXPORT_RESOLUTION

    print(f"         ratio = {seed.symmetry_fold * verification.torus.frame_result.rotation_angle / (2*np.pi):.9f}")
    print(f"         non_congruence = {verification.non_congruence['procrustes_disparity']:.6f}")
    print(f"         interior_metric_mean_err = {verification.isometry['interior_metric_mean_err']:.6f}")

    print("  [3/3] Exporting visual companion...")
    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)
    colors_plus = color_map_surface(
        n_u, n_v, curv_plus,
        hue_range=(0.80, 0.96),
        sat_base=0.90,
        val_base=0.76,
        hue_shift=0.0,
    )
    colors_minus = color_map_surface(
        n_u, n_v, curv_minus,
        hue_range=(0.02, 0.14),
        sat_base=0.88,
        val_base=0.74,
        hue_shift=0.0,
    )

    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)
    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    header = [
        "PAPP Bonnet Theorem 9 Visual Companion",
        f"Name: {NAME}",
        f"tau_imag = {TAU_IMAG}",
        f"delta = {DELTA}",
        f"s1 = {S1}",
        f"s2 = {S2}",
        f"symmetry_fold = {SYMMETRY_FOLD}",
        f"perturbation_epsilon = {EPSILON_PERT}",
        f"bonnet_epsilon = {EPSILON_GEOM}",
        f"alpha = {solve.alpha.tolist()}",
        f"residual_norm = {solve.residual_norm:.6e}",
    ]

    mtl_name = f"{NAME}_material.mtl"
    write_obj(
        OUTPUT_DIR / f"{NAME}_f_plus.obj",
        pair.f_plus.vertices,
        tris_plus,
        normals=nrm_plus,
        colors=colors_plus,
        object_name=f"{NAME}_f_plus",
        header=header + ["Visual companion: f_plus"],
        mtl_file=mtl_name,
        material_name=f"{NAME}_plus",
    )
    write_obj(
        OUTPUT_DIR / f"{NAME}_f_minus.obj",
        pair.f_minus.vertices,
        tris_minus,
        normals=nrm_minus,
        colors=colors_minus,
        object_name=f"{NAME}_f_minus",
        header=header + ["Visual companion: f_minus"],
        mtl_file=mtl_name,
        material_name=f"{NAME}_minus",
    )

    mtl_path = OUTPUT_DIR / mtl_name
    write_mtl(
        mtl_path,
        f"{NAME}_plus",
        ka=(0.03, 0.03, 0.03),
        kd=(0.68, 0.22, 0.88),
        ks=(0.35, 0.30, 0.25),
        ns=80.0,
        comment="Theorem 9 perturbative visual companion",
    )
    with mtl_path.open("a", encoding="utf-8") as f:
        f.write(f"\nnewmtl {NAME}_minus\n")
        f.write("Ka 0.0300 0.0300 0.0300\n")
        f.write("Kd 0.9000 0.4200 0.1200\n")
        f.write("Ks 0.3000 0.2500 0.2000\n")
        f.write("Ns 60.0\n")
        f.write("d 1.0\n")

    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        OUTPUT_DIR / f"{NAME}_flux.obj",
        pair.f_plus.vertices,
        flux_edges,
        colors_plus,
        object_name=f"{NAME}_flux",
        header=header + ["Flux wireframe: visual companion"],
    )

    metadata = {
        "name": NAME,
        "kind": "theorem9_perturbative_bonnet_visual_companion",
        "tau_imag": TAU_IMAG,
        "delta": DELTA,
        "s1": S1,
        "s2": S2,
        "symmetry_fold": SYMMETRY_FOLD,
        "epsilon_perturbation": EPSILON_PERT,
        "epsilon_bonnet": EPSILON_GEOM,
        "alpha": solve.alpha.tolist(),
        "residual_norm": float(solve.residual_norm),
        "ratio": float(seed.symmetry_fold * verification.torus.frame_result.rotation_angle / (2*np.pi)),
        "closure": closure,
        "isometry": verification.isometry,
        "mean_curvature": verification.mean_curvature,
        "non_congruence": verification.non_congruence,
        "n_vertices": int(len(pair.f_plus.vertices)),
        "n_triangles": int(len(tris_plus)),
        "n_flux_edges": int(len(flux_edges)),
        "output_files": [
            f"{NAME}_f_plus.obj",
            f"{NAME}_f_minus.obj",
            f"{NAME}_flux.obj",
            mtl_name,
        ],
        "time_seconds": round(time.time() - t0, 2),
    }
    with (OUTPUT_DIR / f"{NAME}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2)

    print("\nDone.")
    print(f"  Files written to: {OUTPUT_DIR}")
    print(f"  Total time: {metadata['time_seconds']:.2f}s")


if __name__ == "__main__":
    main()
