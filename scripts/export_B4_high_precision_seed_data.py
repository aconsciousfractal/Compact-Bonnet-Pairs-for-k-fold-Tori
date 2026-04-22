#!/usr/bin/env python3
"""Export a high-precision B4 seed dataset for lambda/CR verification.

This script rebuilds ``tau_spec`` from spectral periods using mpmath and writes
the seed-level CSV expected by ``verify_b4_cr_lambda_dataset_hp.py``.

It uses the same seed selection as ``results/P10/P10_2_spectral_periods.json``.
Whenever available, it takes ``delta`` and ``s2`` from the 40-digit checkpoint
``results/P5/P5v3_checkpoint.json``; otherwise it falls back to the archived
phase-15 seed series.
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import mpmath as mp


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

P10_PATH = RESULTS / "P10" / "P10_2_spectral_periods.json"
PHASE15_PATH = RESULTS / "phase15_asymptotic" / "full_series_k3_1000.json"
P5_PATH = RESULTS / "P5" / "P5v3_checkpoint.json"

OUT_DIR = RESULTS / "B4" / "high_precision"
DEFAULT_OUT_CSV = OUT_DIR / "B4_full_precision_seed_data.csv"
DEFAULT_OUT_JSON = OUT_DIR / "B4_full_precision_seed_data_summary.json"

TAU_RE = "0.5"
TAU_IM = "0.3205128205"
S1_DEFAULT = "-8.5"


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh, parse_float=Decimal)


def mpc_to_pair(z: mp.mpc, digits: int) -> tuple[str, str]:
    return (mp.nstr(mp.re(z), digits), mp.nstr(mp.im(z), digits))


def init_constants_mp(dps: int) -> dict[str, mp.mpf | mp.mpc]:
    mp.mp.dps = dps
    tau = mp.mpc(mp.mpf(TAU_RE), mp.mpf(TAU_IM))
    nome = mp.exp(1j * mp.pi * tau)

    def f(omega: mp.mpf) -> mp.mpf:
        return mp.re(mp.jtheta(2, omega, nome, derivative=1))

    omega = mp.findroot(f, (mp.mpf("0.38"), mp.mpf("0.40")))

    th1p_0 = mp.jtheta(1, 0, nome, derivative=1)
    th1_2om = mp.jtheta(1, 2 * omega, nome)
    th1p_2om = mp.jtheta(1, 2 * omega, nome, derivative=1)
    th2_om = mp.jtheta(2, omega, nome)
    th1_om = mp.jtheta(1, omega, nome)
    th2_0 = mp.jtheta(2, 0, nome)
    th3_0 = mp.jtheta(3, 0, nome)
    th4_0 = mp.jtheta(4, 0, nome)
    th3_om = mp.jtheta(3, omega, nome)
    th4_om = mp.jtheta(4, omega, nome)

    U = mp.re(-mp.mpf("0.5") * th1p_0 * th1_2om / th2_om**2)
    Up = mp.re(-mp.mpf("0.5") * th1p_0 * th1p_2om / th2_om**2)
    U1p = mp.re(mp.mpf("0.5") * th1p_0**2 / th2_om**2)
    U2 = mp.re((th1p_0**2 / th2_om**2) * ((th1_om / th2_0) ** 2 + (th4_om / th3_0) ** 2 + (th3_om / th4_0) ** 2))

    return {
        "tau": tau,
        "nome": nome,
        "omega": omega,
        "U": U,
        "Up": Up,
        "U1p": U1p,
        "U2": U2,
    }


def q_coefficients_mp(delta: mp.mpf, s1: mp.mpf, s2: mp.mpf, const: dict[str, mp.mpf | mp.mpc]) -> list[mp.mpf]:
    U = const["U"]
    Up = const["Up"]
    U1p = const["U1p"]
    U2 = const["U2"]
    sig = s1 + s2
    pro = s1 * s2
    a4 = mp.mpf("-1")
    a3 = 2 * sig + 2 * delta**2 * U1p
    a2 = -(sig**2 + 2 * pro) - delta**2 * U2
    a1 = 2 * pro * sig - 2 * delta**2 * Up
    a0 = -(pro**2) - delta**2 * U**2
    return [a4, a3, a2, a1, a0]


def poly_eval(coeffs: list[mp.mpf], z: mp.mpf | mp.mpc) -> mp.mpf | mp.mpc:
    total = mp.mpc(0)
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        total += coeff * z ** (degree - i)
    return total


def poly_derivative_eval(coeffs: list[mp.mpf], z: mp.mpf | mp.mpc) -> mp.mpf | mp.mpc:
    total = mp.mpc(0)
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs[:-1]):
        power = degree - i
        total += power * coeff * z ** (power - 1)
    return total


_GL_CACHE: dict[tuple[int, int], tuple[list[mp.mpf], list[mp.mpf], list[mp.mpf], list[mp.mpf]]] = {}


def compute_gl_nodes_weights(n: int, dps: int) -> tuple[list[mp.mpf], list[mp.mpf], list[mp.mpf], list[mp.mpf]]:
    key = (n, dps)
    if key in _GL_CACHE:
        return _GL_CACHE[key]

    original_dps = mp.mp.dps
    mp.mp.dps = dps + 15
    nodes_half: list[mp.mpf] = []
    weights_half: list[mp.mpf] = []
    m = (n + 1) // 2
    for i in range(m):
        theta = mp.pi * (4 * i + 3) / (4 * n + 2)
        x = mp.cos(theta)
        for _ in range(100):
            p0 = mp.mpf(1)
            p1 = x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = n * (x * p1 - p0) / (x**2 - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mp.power(10, -(dps + 10)):
                break
        w = 2 / ((1 - x**2) * dp**2)
        nodes_half.append(x)
        weights_half.append(w)

    full_nodes: list[mp.mpf] = []
    full_weights: list[mp.mpf] = []
    for i in range(m):
        full_nodes.append(-nodes_half[i])
        full_weights.append(weights_half[i])
    for i in range(m - 1, -1, -1):
        if n % 2 == 1 and i == 0:
            continue
        full_nodes.append(nodes_half[i])
        full_weights.append(weights_half[i])

    half_pi = mp.pi / 2
    nodes = [half_pi * t for t in full_nodes]
    weights = [half_pi * w for w in full_weights]
    sins = [mp.sin(phi) for phi in nodes]
    coss = [mp.cos(phi) for phi in nodes]
    mp.mp.dps = original_dps

    _GL_CACHE[key] = (nodes, weights, sins, coss)
    return _GL_CACHE[key]


def classify_roots(coeffs: list[mp.mpf], dps: int) -> tuple[mp.mpf, mp.mpf, mp.mpc, mp.mpc]:
    roots = [mp.mpc(r) for r in mp.polyroots(coeffs, maxsteps=200, error=False)]
    root_tol = mp.power(10, -max(20, dps // 2))
    real_roots = sorted([mp.re(r) for r in roots if abs(mp.im(r)) <= root_tol])
    complex_roots = sorted([r for r in roots if abs(mp.im(r)) > root_tol], key=lambda z: mp.im(z), reverse=True)
    if len(real_roots) != 2 or len(complex_roots) != 2:
        raise ValueError(f"unexpected root topology: real={real_roots}, complex={complex_roots}")
    return real_roots[0], real_roots[1], complex_roots[0], complex_roots[1]


def a_half_period(e1: mp.mpf, e2: mp.mpf, coeffs: list[mp.mpf], n_gauss: int, dps: int) -> mp.mpf:
    mid = (e1 + e2) / 2
    half = (e2 - e1) / 2
    _, weights, sins, coss = compute_gl_nodes_weights(n_gauss, dps)
    total = mp.mpf(0)
    for weight, sphi, cphi in zip(weights, sins, coss):
        s = mid + half * sphi
        q = mp.re(poly_eval(coeffs, s))
        if q <= 0:
            continue
        total += weight * half * cphi / mp.sqrt(q)
    return total


def b_half_period(e_start: mp.mpf, e_end: mp.mpc, coeffs: list[mp.mpf], n_gauss: int, dps: int) -> mp.mpc:
    diff = e_end - e_start
    _, weights, sins, coss = compute_gl_nodes_weights(n_gauss, dps)
    total = mp.mpc(0)
    prev_sq = None
    qp_start = poly_derivative_eval(coeffs, e_start)
    expected_phase = mp.arg(qp_start) / 2 + mp.arg(diff) / 2

    for weight, sphi, cphi in zip(weights, sins, coss):
        t = (1 + sphi) / 2
        s = e_start + t * diff
        q = poly_eval(coeffs, s)
        sq = mp.sqrt(q)
        if prev_sq is None:
            phase_diff = mp.arg(sq) - expected_phase
            while phase_diff <= -mp.pi:
                phase_diff += 2 * mp.pi
            while phase_diff > mp.pi:
                phase_diff -= 2 * mp.pi
            if abs(phase_diff) > mp.pi / 2:
                sq = -sq
        elif abs(sq + prev_sq) < abs(sq - prev_sq):
            sq = -sq
        prev_sq = sq
        total += weight * diff * cphi / (2 * sq)
    return total


def cross_ratio_mixed(c: mp.mpc, a: mp.mpf, b: mp.mpf) -> mp.mpc:
    cb = mp.conj(c)
    return (c - a) * (cb - b) / ((c - b) * (cb - a))


def load_seed_sources() -> tuple[dict[int, dict[str, str]], list[int]]:
    phase15 = load_json(PHASE15_PATH)
    p5 = json.loads(P5_PATH.read_text(encoding="utf-8"))
    p10 = load_json(P10_PATH)

    seeds: dict[int, dict[str, str]] = {}
    for row in phase15:
        k = int(row["k"])
        seeds[k] = {
            "k": str(k),
            "s1": str(row["s1"]),
            "delta": str(row["delta"]),
            "s2": str(row["s2"]),
            "source": "phase15_asymptotic/full_series_k3_1000.json",
        }
    for row in p5["seeds"]:
        k = int(row["k"])
        seeds[k] = {
            "k": str(k),
            "s1": S1_DEFAULT,
            "delta": row["delta"],
            "s2": row["s2"],
            "source": "P5/P5v3_checkpoint.json",
        }
    analysis_ks = [int(row["k"]) for row in p10["results"]]
    return seeds, analysis_ks


def parse_k_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--gauss-a", type=int, default=320)
    parser.add_argument("--gauss-b", type=int, default=400)
    parser.add_argument("--digits-out", type=int, default=40)
    parser.add_argument("--ks", type=str, default=None, help="comma-separated subset of k values")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    dps = args.dps
    n_gauss_a = args.gauss_a
    n_gauss_b = args.gauss_b
    digits_out = args.digits_out
    selected_ks = parse_k_list(args.ks)

    out_csv = args.out_csv.resolve()
    out_json = args.out_json.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    seeds, analysis_ks = load_seed_sources()
    if selected_ks is not None:
        analysis_ks = [k for k in analysis_ks if k in selected_ks]
    const = init_constants_mp(dps)

    rows: list[dict[str, str]] = []
    for k in analysis_ks:
        seed = seeds[k]
        s1 = mp.mpf(seed["s1"])
        delta = mp.mpf(seed["delta"])
        s2 = mp.mpf(seed["s2"])
        coeffs = q_coefficients_mp(delta, s1, s2, const)
        e1, e2, e3, e4 = classify_roots(coeffs, dps)
        w1_half = a_half_period(e1, e2, coeffs, n_gauss_a, dps)
        w2_half = b_half_period(e2, e3, coeffs, n_gauss_b, dps)
        tau_spec = w2_half / w1_half
        if mp.im(tau_spec) < 0:
            tau_spec = -tau_spec
        # choose the cusp representative used in the paper
        if mp.re(tau_spec) > 0:
            tau_spec -= 1

        cr = cross_ratio_mixed(e3, e1, e2)
        tau_re, tau_im = mpc_to_pair(tau_spec, digits_out)
        cr_re, cr_im = mpc_to_pair(cr, digits_out)

        rows.append(
            {
                "k": str(k),
                "s1": seed["s1"],
                "delta": mp.nstr(delta, digits_out),
                "s2": mp.nstr(s2, digits_out),
                "tau_real": tau_re,
                "tau_imag": tau_im,
                "cr_real": cr_re,
                "cr_imag": cr_im,
                "source_seed": seed["source"],
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["k", "s1", "delta", "s2", "tau_real", "tau_imag", "cr_real", "cr_imag", "source_seed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dps": dps,
        "n_gauss_a": n_gauss_a,
        "n_gauss_b": n_gauss_b,
        "n_rows": len(rows),
        "analysis_ks": analysis_ks,
        "omega_crit": mp.nstr(const["omega"], 30),
        "constants": {
            "U": mp.nstr(const["U"], 30),
            "Up": mp.nstr(const["Up"], 30),
            "U1p": mp.nstr(const["U1p"], 30),
            "U2": mp.nstr(const["U2"], 30),
        },
        "output_csv": str(out_csv.relative_to(PROJECT)),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_csv.relative_to(PROJECT)}")
    print(f"Rows: {len(rows)}")
    print(f"omega_crit = {summary['omega_crit']}")


if __name__ == "__main__":
    main()
