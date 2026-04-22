#!/usr/bin/env python3
"""Export raw-grid C2 audit data and rank candidate normalization objects.

This target-repo version is self-contained:

- it computes the raw second-fundamental-form grid data from `src/`,
- it exports the per-grid CSV used in the paper's C2 audit layer,
- it ranks candidate unit/constant objects directly, without depending on the
  old `paper/math work with agent.md/...` verifier path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src import quaternion_ops as Q
from src.isothermic_torus import TorusParameters, compute_torus
from src.retraction_form import compute_retraction_bonnet
from src.seed_catalog import get_seed

RESULTS = PROJECT / "results"
OUT_DIR = RESULTS / "C2" / "raw_grid"

N0_U = 40
N0_V = 80
SCALE_FACTORS = [1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.5, -0.5, 0.25, -0.25]


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def qmul_grid(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    shape = P.shape
    n = int(np.prod(shape[:-1]))
    result = Q.qmul_batch(P.reshape(n, 4), R.reshape(n, 4))
    return result.reshape(shape)


def qconj_grid(A: np.ndarray) -> np.ndarray:
    result = A.copy()
    result[..., 1:] *= -1
    return result


def compute_exact_first_derivatives(x_grid: np.ndarray, omega_u: np.ndarray, omega_v: np.ndarray):
    """Exact first derivatives from the retraction integrands."""
    ob_u = qconj_grid(omega_u)
    ob_v = qconj_grid(omega_v)
    Fp_u = 0.5 * qmul_grid(x_grid, ob_u)
    Fp_v = 0.5 * qmul_grid(x_grid, ob_v)
    Fm_u = -0.5 * qmul_grid(ob_u, x_grid)
    Fm_v = -0.5 * qmul_grid(ob_v, x_grid)
    return Fp_u, Fp_v, Fm_u, Fm_v


def fd_second_from_exact_first(f_u: np.ndarray, f_v: np.ndarray, du: float, dv: float):
    """Single-level central finite differences on exact first derivatives."""
    f_uu = (np.roll(f_u, -1, axis=0) - np.roll(f_u, 1, axis=0)) / (2 * du)
    f_uv = np.full_like(f_u, np.nan)
    f_uv[:, 1:-1, :] = (f_u[:, 2:, :] - f_u[:, :-2, :]) / (2 * dv)
    f_vv = np.full_like(f_v, np.nan)
    f_vv[:, 1:-1, :] = (f_v[:, 2:, :] - f_v[:, :-2, :]) / (2 * dv)
    return f_uu, f_uv, f_vv


def second_fundamental_form(f_u: np.ndarray, f_v: np.ndarray, f_uu: np.ndarray, f_uv: np.ndarray, f_vv: np.ndarray):
    """Compute first and second fundamental-form coefficients."""
    E = np.sum(f_u**2, axis=-1)
    F_I = np.sum(f_u * f_v, axis=-1)
    G = np.sum(f_v**2, axis=-1)

    n_vec = np.cross(f_u, f_v)
    n_norm = np.linalg.norm(n_vec, axis=-1, keepdims=True)
    n_vec = n_vec / np.maximum(n_norm, 1e-30)

    e = np.sum(f_uu * n_vec, axis=-1)
    f_II = np.sum(f_uv * n_vec, axis=-1)
    g = np.sum(f_vv * n_vec, axis=-1)

    det_I = E * G - F_I**2
    det_I_safe = np.where(np.abs(det_I) < 1e-30, 1e-30, det_I)

    H = (e * G - 2 * f_II * F_I + g * E) / (2 * det_I_safe)
    K = (e * g - f_II**2) / det_I_safe

    valid = np.isfinite(e) & np.isfinite(f_II) & np.isfinite(g) & (np.abs(det_I) > 1e-20)

    return {
        "E": E,
        "F_I": F_I,
        "G": G,
        "e": e,
        "f_II": f_II,
        "g": g,
        "H": H,
        "K": K,
        "normal": n_vec,
        "det_I": det_I,
        "valid": valid,
    }


@dataclass
class CandidateScore:
    name: str
    scale: float
    n: int
    mean_scaled: float
    std_scaled: float
    max_abs_dev_from_mean: float
    max_abs_dev_from_one: float
    constancy_score: float
    unit_score: float
    combined_score: float


def best_candidate_kind(name: str) -> str:
    if "over_e2h" in name or "b_shape" in name:
        return "normalized"
    if "raw" in name or "qdiff" in name:
        return "raw_or_distortion"
    return "other"


def build_candidates(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    e2h = data["e2h"]
    m_plus = data["M_plus_raw"]
    m_minus = data["M_minus_raw"]
    qdiff = data["qdiff_imag"]
    candidates: dict[str, np.ndarray] = {
        "M_minus_raw": m_minus,
        "M_plus_raw": m_plus,
        "M_minus_raw_minus_M_plus_raw": m_minus - m_plus,
        "M_minus_raw_over_e2h": m_minus / e2h,
        "raw_distortion_over_e2h": (m_minus - m_plus) / e2h,
        "qdiff_imag_supplied": qdiff,
        "qdiff_imag_over_e2h": qdiff / e2h,
        "b_shape_supplied": m_minus / e2h,
    }
    return candidates


def score_candidate(name: str, values: np.ndarray, scale: float) -> CandidateScore:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        raise ValueError(f"candidate {name!r} contains no finite values")
    scaled = scale * finite
    mean = float(np.mean(scaled))
    std = float(np.std(scaled))
    max_dev_mean = float(np.max(np.abs(scaled - mean)))
    max_dev_one = float(np.max(np.abs(scaled - 1.0)))
    denom = max(1.0, abs(mean))
    constancy = max_dev_mean / denom
    unit = abs(float(max_dev_one)) if math.isfinite(max_dev_one) else float("inf")
    combined = max(constancy, unit)
    return CandidateScore(
        name=name,
        scale=scale,
        n=int(finite.size),
        mean_scaled=mean,
        std_scaled=std,
        max_abs_dev_from_mean=max_dev_mean,
        max_abs_dev_from_one=max_dev_one,
        constancy_score=constancy,
        unit_score=unit,
        combined_score=combined,
    )


def rank_candidates(candidates: dict[str, np.ndarray]) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for name, values in candidates.items():
        for scale in SCALE_FACTORS:
            scores.append(score_candidate(name, values, scale))
    scores.sort(key=lambda s: (s.combined_score, s.constancy_score, s.unit_score, s.name))
    return scores


def analyze_grid(k: int, u_res: int, v_res: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed = get_seed(k)
    params = TorusParameters(
        tau_imag=seed["tau_imag"],
        delta=seed["delta"],
        s1=seed["s1"],
        s2=seed["s2"],
        u_res=u_res,
        v_res=v_res,
        symmetry_fold=k,
    )
    torus = compute_torus(params)
    ret = compute_retraction_bonnet(torus, method="analytic", verbose=False)

    du = float(torus.u_grid[1] - torus.u_grid[0])
    dv = float(torus.v_grid[1] - torus.v_grid[0])

    Fp_u, Fp_v, Fm_u, Fm_v = compute_exact_first_derivatives(ret.x_grid, ret.omega_u, ret.omega_v)

    fpu3 = Fp_u[:, :, 1:4]
    fpv3 = Fp_v[:, :, 1:4]
    fmu3 = Fm_u[:, :, 1:4]
    fmv3 = Fm_v[:, :, 1:4]

    fpuu3, fpuv3, fpvv3 = fd_second_from_exact_first(fpu3, fpv3, du, dv)
    fmuu3, fmuv3, fmvv3 = fd_second_from_exact_first(fmu3, fmv3, du, dv)

    sff_p = second_fundamental_form(fpu3, fpv3, fpuu3, fpuv3, fpvv3)
    sff_m = second_fundamental_form(fmu3, fmv3, fmuu3, fmuv3, fmvv3)

    hp_mean = np.nanmean(sff_p["H"])
    hm_mean = np.nanmean(sff_m["H"])
    orient_flip = bool(hp_mean * hm_mean < 0)
    if orient_flip:
        sff_m["e"] = -sff_m["e"]
        sff_m["f_II"] = -sff_m["f_II"]
        sff_m["g"] = -sff_m["g"]
        sff_m["H"] = -sff_m["H"]
        sff_m["normal"] = -sff_m["normal"]

    valid = sff_p["valid"] & sff_m["valid"]
    grid_level = f"{u_res}x{v_res}"

    E_avg = 0.5 * (sff_p["E"] + sff_m["E"])
    G_avg = 0.5 * (sff_p["G"] + sff_m["G"])
    e2h = 0.5 * (E_avg + G_avg)
    M_plus_raw = sff_p["f_II"]
    M_minus_raw = sff_m["f_II"]
    qdiff_imag = M_minus_raw - M_plus_raw

    rows: list[dict[str, object]] = []
    for iu in range(u_res):
        for iv in range(v_res):
            if not valid[iu, iv]:
                continue
            rows.append(
                {
                    "k": k,
                    "seed_id": f"k{k}",
                    "grid_level": grid_level,
                    "u_res": u_res,
                    "v_res": v_res,
                    "du": du,
                    "dv": dv,
                    "u_index": iu,
                    "v_index": iv,
                    "u": float(torus.u_grid[iu]),
                    "v": float(torus.v_grid[iv]),
                    "e2h": float(e2h[iu, iv]),
                    "E_plus": float(sff_p["E"][iu, iv]),
                    "G_plus": float(sff_p["G"][iu, iv]),
                    "E_minus": float(sff_m["E"][iu, iv]),
                    "G_minus": float(sff_m["G"][iu, iv]),
                    "L_plus_raw": float(sff_p["e"][iu, iv]),
                    "M_plus_raw": float(M_plus_raw[iu, iv]),
                    "N_plus_raw": float(sff_p["g"][iu, iv]),
                    "L_minus_raw": float(sff_m["e"][iu, iv]),
                    "M_minus_raw": float(M_minus_raw[iu, iv]),
                    "N_minus_raw": float(sff_m["g"][iu, iv]),
                    "qdiff_imag": float(qdiff_imag[iu, iv]),
                    "b_shape": float(M_minus_raw[iu, iv] / e2h[iu, iv]),
                }
            )

    data = {
        "e2h": e2h[valid],
        "M_plus_raw": M_plus_raw[valid],
        "M_minus_raw": M_minus_raw[valid],
        "qdiff_imag": qdiff_imag[valid],
    }
    candidates = build_candidates(data)
    scores = rank_candidates(candidates)
    best = scores[0]
    top3 = scores[:3]

    summary = {
        "k": k,
        "seed_id": f"k{k}",
        "grid_level": grid_level,
        "u_res": u_res,
        "v_res": v_res,
        "n_valid": int(np.sum(valid)),
        "du": du,
        "dv": dv,
        "orient_flip": orient_flip,
        "best_candidate": {
            "name": best.name,
            "scale": best.scale,
            "kind": best_candidate_kind(best.name),
            "mean_scaled": best.mean_scaled,
            "std_scaled": best.std_scaled,
            "max_abs_dev_from_one": best.max_abs_dev_from_one,
            "max_abs_dev_from_mean": best.max_abs_dev_from_mean,
            "combined_score": best.combined_score,
        },
        "top3": [
            {
                "name": s.name,
                "scale": s.scale,
                "kind": best_candidate_kind(s.name),
                "mean_scaled": s.mean_scaled,
                "std_scaled": s.std_scaled,
                "max_abs_dev_from_one": s.max_abs_dev_from_one,
                "combined_score": s.combined_score,
            }
            for s in top3
        ],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="3,5,7,10,12")
    parser.add_argument("--levels", default="1")
    parser.add_argument("--out-prefix", default="c2_raw_grid_audit")
    args = parser.parse_args()

    ks = parse_int_list(args.ks)
    levels = parse_int_list(args.levels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for k in ks:
        for level in levels:
            u_res = N0_U * level
            v_res = N0_V * level
            print(f"Computing raw-grid C2 data for k={k}, grid={u_res}x{v_res}...")
            rows, summary = analyze_grid(k, u_res, v_res)
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"  best = {summary['best_candidate']['scale']} * {summary['best_candidate']['name']}"
                f"  (score={summary['best_candidate']['combined_score']:.3e}, n={summary['n_valid']})"
            )

    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    json_path = OUT_DIR / f"{args.out_prefix}_summary.json"
    write_csv(csv_path, all_rows)
    json_path.write_text(json.dumps({"groups": summaries}, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path.relative_to(PROJECT)}")
    print(f"Wrote {json_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
