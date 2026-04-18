"""
Select and export 2-3 canonical perturbative Bonnet candidates from the current 8.5 atlas.

Selection policy:
1. strongest paper_3fold continuation candidate
2. strongest paper_4fold continuation candidate
3. strongest secondary-seed candidate from the mini-atlas

Each selected candidate is:
- reconstructed from its Theorem 7 seed + forcing + epsilon_perturbation
- solved again with the stored alpha as initial guess
- verified through the Bonnet pipeline
- exported as f_plus / f_minus / flux OBJ + MTL + metadata JSON
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
    Theorem9ForcingSpec,
    build_theorem9_basis_from_forcing,
    build_theorem9_seed_from_theorem7,
    solve_theorem9_perturbation,
    verify_theorem9_bonnet_pipeline,
)


RESULTS_DIR = REPO_ROOT / "results"
FORCING_PATH = RESULTS_DIR / "phase85_forcing_continuation" / "phase85_forcing_continuation.json"
SECONDARY_PATH = RESULTS_DIR / "phase85_secondary_seed_atlas" / "phase85_secondary_seed_atlas.json"
OUTPUT_DIR = RESULTS_DIR / "obj" / "paper_figures"


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


def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _pick_best_step(steps):
    return min(steps, key=lambda s: (s['residual_norm'], abs(s['ratio'] - 1.0), abs(s['b_scalar']), abs(s['c_scalar'])))


def _validation_score(residual_norm: float, ratio: float, paper_axial_scalar: float, c_scalar: float) -> float:
    return float(abs(residual_norm) + abs(ratio - 1.0) + abs(paper_axial_scalar) + abs(c_scalar))


def _export_case(name: str, output_dir: Path, verification, solve, forcing_label: str, mode: str) -> list[str]:
    pair = verification.pair
    n_u = verification.torus.params.u_res
    n_v = verification.torus.params.v_res
    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)

    if mode == 'paper_like':
        colors_plus = color_map_surface(n_u, n_v, curv_plus, hue_range=(0.58, 0.72), sat_base=0.82, val_base=0.72)
        colors_minus = color_map_surface(n_u, n_v, curv_minus, hue_range=(0.58, 0.72), sat_base=0.72, val_base=0.66, hue_shift=0.05)
        kd_plus, kd_minus = (0.24, 0.46, 0.82), (0.46, 0.28, 0.74)
    else:
        colors_plus = color_map_surface(n_u, n_v, curv_plus, hue_range=(0.80, 0.96), sat_base=0.90, val_base=0.76)
        colors_minus = color_map_surface(n_u, n_v, curv_minus, hue_range=(0.02, 0.14), sat_base=0.88, val_base=0.74)
        kd_plus, kd_minus = (0.68, 0.22, 0.88), (0.90, 0.42, 0.12)

    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)
    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    header = [
        "PAPP Bonnet Theorem 9 Canonical Candidate Export",
        f"Name: {name}",
        f"forcing = {forcing_label}",
        f"mode = {mode}",
        f"ratio = {verification.torus.params.symmetry_fold * verification.torus.frame_result.rotation_angle / (2*np.pi):.9f}",
        f"residual_norm = {solve.residual_norm:.6e}",
    ]

    mtl_name = f"{name}_material.mtl"
    write_obj(
        output_dir / f"{name}_f_plus.obj",
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
        output_dir / f"{name}_f_minus.obj",
        pair.f_minus.vertices,
        tris_minus,
        normals=nrm_minus,
        colors=colors_minus,
        object_name=f"{name}_f_minus",
        header=header + ["Surface: f_minus perturbative Bonnet branch"],
        mtl_file=mtl_name,
        material_name=f"{name}_minus",
    )

    mtl_path = output_dir / mtl_name
    write_mtl(
        mtl_path,
        f"{name}_plus",
        ka=(0.03, 0.03, 0.03),
        kd=kd_plus,
        ks=(0.35, 0.30, 0.25),
        ns=80.0,
        comment="Phase 8.5 canonical perturbative candidate",
    )
    with mtl_path.open('a', encoding='utf-8') as f:
        f.write(f"\nnewmtl {name}_minus\n")
        f.write("Ka 0.0300 0.0300 0.0300\n")
        f.write(f"Kd {kd_minus[0]:.4f} {kd_minus[1]:.4f} {kd_minus[2]:.4f}\n")
        f.write("Ks 0.3000 0.2500 0.2000\n")
        f.write("Ns 60.0\n")
        f.write("d 1.0\n")

    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        output_dir / f"{name}_flux.obj",
        pair.f_plus.vertices,
        flux_edges,
        colors_plus,
        object_name=f"{name}_flux",
        header=header + ["Flux wireframe"],
    )

    return [
        f"{name}_f_plus.obj",
        f"{name}_f_minus.obj",
        f"{name}_flux.obj",
        mtl_name,
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    forcing_data = _load_json(FORCING_PATH)
    secondary_data = _load_json(SECONDARY_PATH)
    forcing_label = forcing_data['forcing']['label']
    forcing = Theorem9ForcingSpec(**forcing_data['forcing'])

    candidates = []

    for seed_entry in forcing_data['seed_results']:
        seed = seed_entry['seed']
        best_step = _pick_best_step(seed_entry['continuation']['steps'])
        candidates.append({
            'label': seed['label'],
            'kind': 'paper_seed',
            'seed': seed,
            'step': best_step,
        })

    secondary_best = None
    for primary in secondary_data['primary_seed_entries']:
        for sec in primary['secondary_seeds']:
            step = _pick_best_step(sec['continuation']['steps'])
            cand = {
                'label': sec['label'],
                'kind': 'secondary_seed',
                'seed': {
                    'label': sec['label'],
                    'tau_imag': sec['tau_imag'],
                    'delta': sec['delta'],
                    's1': primary['primary_seed']['s1'],
                    's2': sec['s2'],
                    'symmetry_fold': primary['primary_seed']['symmetry_fold'],
                },
                'step': step,
            }
            if secondary_best is None or step['residual_norm'] < secondary_best['step']['residual_norm']:
                secondary_best = cand

    if secondary_best is not None:
        candidates.append(secondary_best)

    exported = []
    print("=" * 76)
    print("  PHASE 8.5 - EXPORT CANONICAL PERTURBATIVE CANDIDATES")
    print("=" * 76)
    print(f"  forcing = {forcing_label}")

    for cand in candidates:
        seed_spec = cand['seed']
        step = cand['step']
        print(f"\nCandidate: {cand['label']}")
        print(f"  epsilon = {step['epsilon']:.1e}, residual = {step['residual_norm']:.3e}")
        seed = build_theorem9_seed_from_theorem7(
            tau_imag=seed_spec['tau_imag'],
            delta=seed_spec['delta'],
            s1=seed_spec['s1'],
            s2=seed_spec['s2'],
            symmetry_fold=seed_spec['symmetry_fold'],
            n_half_samples=61,
        )
        basis = build_theorem9_basis_from_forcing(seed, forcing)
        solve = solve_theorem9_perturbation(
            seed=seed,
            epsilon=float(step['epsilon']),
            basis=basis,
            initial_alpha=np.asarray(step['alpha'], dtype=float),
            n_points=80,
            linearization_n_points=60,
        )
        verification = verify_theorem9_bonnet_pipeline(
            seed=seed,
            solve=solve,
            epsilon_geom=0.3,
            u_res=120,
            v_res=120,
        )
        ratio = seed['symmetry_fold'] * verification.torus.frame_result.rotation_angle / (2*np.pi) if isinstance(seed, dict) else seed.symmetry_fold * verification.torus.frame_result.rotation_angle / (2*np.pi)
        closure = closure_gate(verification.pair)
        paper_axial_scalar = float(solve.evaluation.b_scalar)
        paper_axial_abs = float(abs(paper_axial_scalar))
        closure['paper_axial_scalar'] = paper_axial_scalar
        closure['paper_axial_scalar_abs'] = paper_axial_abs
        mode = 'paper_like' if cand['kind'] == 'paper_seed' else 'visual'
        case_dir = OUTPUT_DIR / cand['label']
        case_dir.mkdir(parents=True, exist_ok=True)
        files = _export_case(cand['label'], case_dir, verification, solve, forcing_label, mode)
        validation_score = _validation_score(
            residual_norm=float(solve.residual_norm),
            ratio=float(ratio),
            paper_axial_scalar=paper_axial_scalar,
            c_scalar=float(solve.evaluation.c_scalar),
        )
        exported.append({
            'label': cand['label'],
            'kind': cand['kind'],
            'seed': seed_spec,
            'forcing': forcing.__dict__,
            'epsilon_perturbation': float(step['epsilon']),
            'alpha': solve.alpha.tolist(),
            'residual_norm': float(solve.residual_norm),
            'ratio': float(ratio),
            'paper_axial_scalar': paper_axial_scalar,
            'paper_axial_scalar_abs': paper_axial_abs,
            'c_scalar': float(solve.evaluation.c_scalar),
            'validation_score': validation_score,
            'closure': _jsonable(closure),
            'isometry': _jsonable(verification.isometry),
            'mean_curvature': _jsonable(verification.mean_curvature),
            'non_congruence': _jsonable(verification.non_congruence),
            'output_dir': str(case_dir),
            'files': files,
        })
        print(f"  exported to: {case_dir}")

    exported.sort(key=lambda item: (item['validation_score'], -item['non_congruence']['procrustes_disparity'], item['label']))
    for rank, item in enumerate(exported, start=1):
        item['rank'] = rank

    summary = {
        'name': 'phase85_canonical_candidates',
        'forcing': forcing.__dict__,
        'candidates': exported,
        'time_seconds': round(time.time() - t0, 2),
    }
    with (OUTPUT_DIR / 'phase85_canonical_candidates.json').open('w', encoding='utf-8') as f:
        json.dump(_jsonable(summary), f, indent=2)

    lines = [
        '# Phase 8.5 Canonical Perturbative Candidates',
        '',
        f"Forcing: `{forcing.label}` ({forcing.family})",
        '',
        '| rank | candidate | kind | epsilon | residual | ratio | axial_Lemma3 | nc | validation_score | output |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in exported:
        lines.append(
            f"| {item['rank']} | {item['label']} | {item['kind']} | {item['epsilon_perturbation']:.1e} | {item['residual_norm']:.3e} | "
            f"{item['ratio']:.9f} | {item['paper_axial_scalar_abs']:.3e} | {item['non_congruence']['procrustes_disparity']:.4f} | "
            f"{item['validation_score']:.3e} | `{item['output_dir']}` |"
        )
    (OUTPUT_DIR / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nWrote: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
