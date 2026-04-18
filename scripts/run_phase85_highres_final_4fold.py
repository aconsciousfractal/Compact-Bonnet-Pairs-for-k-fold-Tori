"""
Phase 8.5 - High-resolution finalization for the 4-fold perturbative Bonnet example.

Workflow:
1. Solve the local Theorem 9 perturbative corrector once
2. Sweep epsilon_bonnet at a reference resolution
3. Pick the best epsilon according to interior metric / curvature criteria
4. Run a resolution ladder
5. Export the highest-resolution successful case as OBJ + flux wireframe
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
SEED_HALF_SAMPLES = 61
SOLVE_N_POINTS = 80
SOLVE_LINEARIZATION_N_POINTS = 60

EPSILON_GEOM_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_RESOLUTION = 80
RESOLUTION_LADDER = [80, 120, 160]

NAME = "bonnet_theorem9_perturbed_4fold_highres"
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


def _score(metrics: dict) -> float:
    ratio_penalty = 10.0 * abs(metrics['ratio'] - 1.0)
    interior_metric = metrics['isometry']['interior_metric_mean_err']
    interior_curvature = metrics['mean_curvature']['interior_mean_diff']
    nc_bonus = 0.05 * metrics['non_congruence']['procrustes_disparity']
    return interior_curvature + 0.25 * interior_metric + ratio_penalty - nc_bonus


def _evaluate(seed, solve, epsilon_geom: float, resolution: int) -> dict:
    t0 = time.time()
    verification = verify_theorem9_bonnet_pipeline(
        seed=seed,
        solve=solve,
        epsilon_geom=epsilon_geom,
        u_res=resolution,
        v_res=resolution,
    )
    ratio = seed.symmetry_fold * verification.torus.frame_result.rotation_angle / (2 * np.pi)
    closure = closure_gate(verification.pair)
    result = {
        'resolution': resolution,
        'epsilon_geom': epsilon_geom,
        'ratio': float(ratio),
        'closure': closure,
        'isometry': verification.isometry,
        'mean_curvature': verification.mean_curvature,
        'non_congruence': verification.non_congruence,
        'time_seconds': round(time.time() - t0, 2),
        'verification': verification,
    }
    result['score'] = float(_score(result))
    return result


def _export_case(name: str, metrics: dict) -> list[str]:
    verification = metrics['verification']
    pair = verification.pair
    n_u = verification.torus.params.u_res
    n_v = verification.torus.params.v_res

    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)
    colors_plus = color_map_surface(n_u, n_v, curv_plus, hue_range=(0.10, 0.24), sat_base=0.84, val_base=0.74)
    colors_minus = color_map_surface(n_u, n_v, curv_minus, hue_range=(0.10, 0.24), sat_base=0.74, val_base=0.68, hue_shift=0.06)

    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)
    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    header = [
        "PAPP Bonnet Theorem 9 High-Res Final Export",
        f"Name: {name}",
        f"tau_imag = {TAU_IMAG}",
        f"delta = {DELTA}",
        f"s1 = {S1}",
        f"s2 = {S2}",
        f"symmetry_fold = {SYMMETRY_FOLD}",
        f"epsilon_perturbation = {EPSILON_PERT}",
        f"epsilon_bonnet = {metrics['epsilon_geom']}",
        f"resolution = {metrics['resolution']}x{metrics['resolution']}",
        f"ratio = {metrics['ratio']:.9f}",
        f"score = {metrics['score']:.6f}",
    ]

    mtl_name = f"{name}_material.mtl"
    write_obj(
        OUTPUT_DIR / f"{name}_f_plus.obj",
        pair.f_plus.vertices,
        tris_plus,
        normals=nrm_plus,
        colors=colors_plus,
        object_name=f"{name}_f_plus",
        header=header + ["Surface: f_plus perturbative Bonnet branch"],
        mtl_file=mtl_name,
        material_name=f"{name}_plus",
    )
    write_obj(
        OUTPUT_DIR / f"{name}_f_minus.obj",
        pair.f_minus.vertices,
        tris_minus,
        normals=nrm_minus,
        colors=colors_minus,
        object_name=f"{name}_f_minus",
        header=header + ["Surface: f_minus perturbative Bonnet branch"],
        mtl_file=mtl_name,
        material_name=f"{name}_minus",
    )

    mtl_path = OUTPUT_DIR / mtl_name
    write_mtl(
        mtl_path,
        f"{name}_plus",
        ka=(0.03, 0.03, 0.03),
        kd=(0.82, 0.54, 0.22),
        ks=(0.35, 0.30, 0.25),
        ns=80.0,
        comment="High-res Theorem 9 perturbative Bonnet example",
    )
    with mtl_path.open("a", encoding="utf-8") as f:
        f.write(f"\nnewmtl {name}_minus\n")
        f.write("Ka 0.0300 0.0300 0.0300\n")
        f.write("Kd 0.7200 0.3000 0.1800\n")
        f.write("Ks 0.3000 0.2500 0.2000\n")
        f.write("Ns 60.0\n")
        f.write("d 1.0\n")

    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        OUTPUT_DIR / f"{name}_flux.obj",
        pair.f_plus.vertices,
        flux_edges,
        colors_plus,
        object_name=f"{name}_flux",
        header=header + ["Flux wireframe: high-res perturbative example"],
    )

    return [
        f"{name}_f_plus.obj",
        f"{name}_f_minus.obj",
        f"{name}_flux.obj",
        mtl_name,
    ]


def main() -> None:
    print("=" * 78)
    print("  PHASE 8.5 - HIGH-RES FINALIZATION (4-FOLD PERTURBATIVE BONNET)")
    print("=" * 78)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n[1/5] Solving perturbative seed...")
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
        raise RuntimeError(f"Perturbative solve failed: {solve.message}")
    print(f"       residual_norm = {solve.residual_norm:.3e}")
    print(f"       alpha = {solve.alpha.tolist()}")

    print("\n[2/5] Sweeping epsilon_bonnet at reference resolution...")
    eps_results = []
    for eps_geom in EPSILON_GEOM_VALUES:
        result = _evaluate(seed, solve, eps_geom, SWEEP_RESOLUTION)
        eps_results.append(result)
        print(
            f"       eps={eps_geom:.2f}  ratio={result['ratio']:.9f}  "
            f"iso_int={result['isometry']['interior_metric_mean_err']:.4f}  "
            f"H_int={result['mean_curvature']['interior_mean_diff']:.4f}  "
            f"nc={result['non_congruence']['procrustes_disparity']:.4f}  "
            f"score={result['score']:.5f}"
        )

    best_eps = min(eps_results, key=lambda item: item['score'])
    print(f"       -> best epsilon_bonnet at ref res: {best_eps['epsilon_geom']:.2f}")

    print("\n[3/5] Running resolution ladder on chosen epsilon...")
    ladder = []
    for res in RESOLUTION_LADDER:
        result = _evaluate(seed, solve, best_eps['epsilon_geom'], res)
        ladder.append(result)
        print(
            f"       res={res:3d}  ratio={result['ratio']:.9f}  "
            f"iso_int={result['isometry']['interior_metric_mean_err']:.4f}  "
            f"H_int={result['mean_curvature']['interior_mean_diff']:.4f}  "
            f"nc={result['non_congruence']['procrustes_disparity']:.4f}"
        )

    final_result = ladder[-1]
    final_name = f"{NAME}_res{final_result['resolution']}_eps{int(round(100 * final_result['epsilon_geom'])):02d}"

    print("\n[4/5] Exporting final high-res case...")
    output_files = _export_case(final_name, final_result)

    print("\n[5/5] Writing report...")
    report = {
        'name': NAME,
        'kind': 'phase85_highres_finalization',
        'base_seed': {
            'tau_imag': TAU_IMAG,
            'delta': DELTA,
            's1': S1,
            's2': S2,
            'symmetry_fold': SYMMETRY_FOLD,
            'epsilon_perturbation': EPSILON_PERT,
            'alpha': solve.alpha.tolist(),
            'residual_norm': float(solve.residual_norm),
        },
        'epsilon_sweep': [{k: v for k, v in r.items() if k != 'verification'} for r in eps_results],
        'resolution_ladder': [{k: v for k, v in r.items() if k != 'verification'} for r in ladder],
        'chosen_epsilon_geom': float(best_eps['epsilon_geom']),
        'final_resolution': int(final_result['resolution']),
        'final_score': float(final_result['score']),
        'final_metrics': {k: v for k, v in final_result.items() if k != 'verification'},
        'output_files': output_files,
        'time_seconds': round(time.time() - t0, 2),
    }
    with (OUTPUT_DIR / f"{NAME}_report.json").open('w', encoding='utf-8') as f:
        json.dump(_jsonable(report), f, indent=2)

    print("\nDone.")
    print(f"  Chosen epsilon_bonnet: {best_eps['epsilon_geom']:.2f}")
    print(f"  Final resolution: {final_result['resolution']}x{final_result['resolution']}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Total time: {report['time_seconds']:.2f}s")


if __name__ == '__main__':
    main()
