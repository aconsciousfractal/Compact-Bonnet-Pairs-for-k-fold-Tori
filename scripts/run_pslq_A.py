#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  P9 — A to 19+ digits from existing P5v3 data  (Colab / Local)
═══════════════════════════════════════════════════════════════════════════

Zero-cost extraction: the P5v3 checkpoint already stores δ_k at 40 dps
for 120 k-values (100..5000).  We compute A_k = δ_k·√k at full precision,
then Richardson extrapolation deg 3–6 and PSLQ on 36 bases.

Asymptotic: δ_k = A/√k · (1 − c₁/k + c₃/k² + O(k⁻³))
    ⟹     A_k = δ_k·√k = A · (1 − c₁/k + c₃/k² + ...)

Same polynomial structure as s₂ → C₂, so same Richardson works.

Requirements: mpmath, numpy (standard on Colab)
Input:  P5v3_checkpoint.json  (120 seeds, each with k, delta, s2, residual)
Output: results/P9_A_identification/

Usage:
    python scripts/P9_A_identification.py
    (or upload to Colab with the checkpoint file)
═══════════════════════════════════════════════════════════════════════════
"""

import subprocess, sys

def ensure_packages():
    for pkg in ['mpmath', 'numpy']:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

ensure_packages()

import json, math, time
from pathlib import Path
import numpy as np
import mpmath

# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════

DPS = 40
mpmath.mp.dps = DPS

TAU_IMAG = '0.3205128205'

# ── Checkpoint locations (searched in order) ──
_this = Path(__file__).resolve().parent.parent  # project root
CHECKPOINT_CANDIDATES = [
    _this / "data" / "high_precision_checkpoints.json",
    Path("/content/drive/MyDrive/Bonnet_P5/P5v3_checkpoint.json"),
    Path("P5v3_checkpoint.json"),
]

# ── Output ──
OUT_DIR = _this / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "P9_log.txt"
RESULTS_FILE = OUT_DIR / "P9_final_results.json"


def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Load checkpoint and compute A_k = δ_k · √k
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint():
    """Load P5v3 checkpoint and extract δ_k at mpmath precision."""
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            log_msg(f"Loading checkpoint: {path}")
            with open(path) as f:
                raw = json.load(f)
            seeds = raw if isinstance(raw, list) else raw.get("seeds", raw.get("data", []))
            results = {}
            for entry in seeds:
                k = int(entry["k"])
                delta_str = str(entry["delta"])
                s2_str = str(entry["s2"])
                results[k] = {
                    "delta": mpmath.mpf(delta_str),
                    "s2": mpmath.mpf(s2_str),
                }
            log_msg(f"  Loaded {len(results)} seeds, k ∈ [{min(results)}..{max(results)}]")
            return results
    raise FileNotFoundError(
        "No P5v3 checkpoint found. Searched:\n" +
        "\n".join(f"  {p}" for p in CHECKPOINT_CANDIDATES)
    )


def compute_Ak(results_by_k):
    """Compute A_k = δ_k · √k at full mpmath precision."""
    Ak_by_k = {}
    for k, data in sorted(results_by_k.items()):
        k_mp = mpmath.mpf(k)
        Ak = data["delta"] * mpmath.sqrt(k_mp)
        Ak_by_k[k] = Ak
    return Ak_by_k


# ═══════════════════════════════════════════════════════════════════════════
# 3. Richardson extrapolation (mpmath precision)
# ═══════════════════════════════════════════════════════════════════════════

def mp_polyfit(x_list, y_list, deg):
    """Least-squares polynomial in x at full mpmath precision."""
    n = len(x_list)
    V = mpmath.matrix(n, deg + 1)
    yv = mpmath.matrix(n, 1)
    for i in range(n):
        xi = x_list[i]
        for j in range(deg + 1):
            V[i, j] = xi ** j
        yv[i, 0] = y_list[i]
    VtV = V.T * V
    Vty = V.T * yv
    c = mpmath.lu_solve(VtV, Vty)
    return [c[j, 0] for j in range(deg + 1)]


def richardson_A(Ak_by_k):
    """Multi-window Richardson on A_k = δ_k·√k → A."""
    ks_sorted = sorted(Ak_by_k.keys())
    k_max = ks_sorted[-1]

    windows = []
    for k_lo, k_hi in [
        (100, 500), (100, 1000), (100, 2000), (100, 3000), (100, 5000),
        (200, 1000), (200, 2000), (200, 3000), (200, 5000),
        (300, 1000), (300, 2000), (300, 3000), (300, 5000),
        (500, 2000), (500, 3000), (500, 5000),
        (1000, 3000), (1000, 5000),
        (2000, 5000),
    ]:
        if k_hi <= k_max:
            windows.append((k_lo, k_hi))

    degs = [3, 4, 5, 6]

    print(f"\n{'═'*100}")
    print(f"   RICHARDSON EXTRAPOLATION: A = lim δ_k·√k  ({DPS} dps)")
    print(f"{'═'*100}")
    header = f"   {'Window':16s}"
    for d in degs:
        header += f" {'deg '+str(d):>22s}"
    print(header)
    print(f"   {'─'*16}" + " " + " ".join(f"{'─'*22}" for _ in degs))

    table = {}
    for k_lo, k_hi in windows:
        mask_k = [k for k in ks_sorted if k_lo <= k <= k_hi]
        xw = [1 / mpmath.mpf(k) for k in mask_k]
        yw = [Ak_by_k[k] for k in mask_k]
        label = f"k={k_lo}..{k_hi}"
        parts = [label]
        for deg in degs:
            if len(xw) < deg + 3:
                parts.append("—")
                continue
            try:
                c = mp_polyfit(xw, yw, deg)
                A_est = c[0]  # constant term = A
                table[(k_lo, k_hi, deg)] = A_est
                parts.append(mpmath.nstr(A_est, 20))
            except Exception:
                parts.append("SING")
        print(f"   {parts[0]:16s}" + " ".join(f"{p:>22s}" for p in parts[1:]))

    if not table:
        return {'A_best': '0', 'stable_digits': 0}

    # ── Per-degree consensus (k_lo ≥ 200) ──
    print(f"\n   ── PER-DEGREE CONSENSUS (k_lo ≥ 200) ──")
    best_deg_vals = {}
    for deg in degs:
        vals = [table[k] for k in table if k[2] == deg and k[0] >= 200]
        if not vals:
            continue
        farr = [float(v) for v in vals]
        spread = max(farr) - min(farr)
        stable = max(0, -int(math.floor(math.log10(max(spread, 1e-40)))))
        mean = sum(vals) / len(vals)
        best_deg_vals[deg] = (vals, mean, spread, stable)
        print(f"   deg {deg}: mean = {mpmath.nstr(mean, 30)}  "
              f"spread = {spread:.2e}  ~{stable} digits  ({len(vals)} windows)")

    # ── Best estimate: highest degree with ≥2 windows ──
    A_best = None
    best_spread = 1.0
    best_stable = 0
    for deg in reversed(degs):
        if deg in best_deg_vals and len(best_deg_vals[deg][0]) >= 2:
            vals, mean, spread, stable = best_deg_vals[deg]
            A_best = mean
            best_spread = spread
            best_stable = stable
            break

    if A_best is None:
        all_est = list(table.values())
        A_best = sum(all_est) / len(all_est)
        best_spread = max(float(v) for v in all_est) - min(float(v) for v in all_est)
        best_stable = max(0, -int(math.floor(math.log10(max(best_spread, 1e-40)))))

    print(f"\n   ★ BEST: A = {mpmath.nstr(A_best, 35)}")
    print(f"     Spread: {best_spread:.2e}, ~{best_stable} stable digits")
    print(f"{'═'*100}\n")

    # Also extract c₁ (the first correction coefficient)
    # From deg 6 best window: c[1] = -c₁·A, so c₁ = -c[1]/A
    c1_est = None
    for deg in [6, 5]:
        best_window_vals = [(k, table[k]) for k in table if k[2] == deg and k[0] >= 500]
        if best_window_vals:
            # Refit to extract c[1]
            k_lo, k_hi = best_window_vals[-1][0][0], best_window_vals[-1][0][1]
            mask_k = [k for k in ks_sorted if k_lo <= k <= k_hi]
            xw = [1 / mpmath.mpf(k) for k in mask_k]
            yw = [Ak_by_k[k] for k in mask_k]
            c = mp_polyfit(xw, yw, deg)
            c1_est = -c[1] / c[0]
            print(f"   Sub-leading: c₁ ≈ {mpmath.nstr(c1_est, 15)} (from deg {deg}, k={k_lo}..{k_hi})")
            break

    return {
        'A_best': mpmath.nstr(A_best, DPS - 2),
        'A_best_mpf': A_best,
        'spread': best_spread,
        'stable_digits': best_stable,
        'c1_estimate': mpmath.nstr(c1_est, 15) if c1_est else None,
        'table': {f"{k[0]}-{k[1]}_d{k[2]}": mpmath.nstr(v, DPS - 2)
                  for k, v in table.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Constants for PSLQ (same as P5 v3)
# ═══════════════════════════════════════════════════════════════════════════

def init_constants():
    """Compute all paper constants at mpmath precision."""
    from scipy.optimize import brentq

    tau_mp = mpmath.mpf('0.5') + 1j * mpmath.mpf(TAU_IMAG)
    nome_mp = mpmath.exp(1j * mpmath.pi * tau_mp)

    nome_f64 = complex(nome_mp)
    def f_f64(w):
        return complex(mpmath.jtheta(2, w, nome_f64, derivative=1)).real
    omega_f64 = brentq(f_f64, 0.01, 0.78)

    def f_mp(omega):
        return mpmath.re(mpmath.jtheta(2, omega, nome_mp, derivative=1))
    omega_mp = mpmath.findroot(f_mp, mpmath.mpf(str(omega_f64)))

    th1p_0   = mpmath.jtheta(1, 0,          nome_mp, derivative=1)
    th1_2om  = mpmath.jtheta(1, 2*omega_mp, nome_mp)
    th1p_2om = mpmath.jtheta(1, 2*omega_mp, nome_mp, derivative=1)
    th2_om   = mpmath.jtheta(2, omega_mp,   nome_mp)
    th1_om   = mpmath.jtheta(1, omega_mp,   nome_mp)
    th2_0    = mpmath.jtheta(2, 0,          nome_mp)
    th3_0    = mpmath.jtheta(3, 0,          nome_mp)
    th4_0    = mpmath.jtheta(4, 0,          nome_mp)
    th3_om   = mpmath.jtheta(3, omega_mp,   nome_mp)
    th4_om   = mpmath.jtheta(4, omega_mp,   nome_mp)

    U   = mpmath.re(-mpmath.mpf('0.5') * th1p_0 * th1_2om / th2_om**2)
    Up  = mpmath.re(-mpmath.mpf('0.5') * th1p_0 * th1p_2om / th2_om**2)
    U1p = mpmath.re(mpmath.mpf('0.5') * th1p_0**2 / th2_om**2)
    U2  = mpmath.re((th1p_0**2 / th2_om**2) * (
        (th1_om / th2_0)**2 + (th4_om / th3_0)**2 + (th3_om / th4_0)**2
    ))

    return {
        'omega': omega_mp, 'nome': nome_mp, 'tau': tau_mp,
        'U': U, 'Up': Up, 'U1p': U1p, 'U2': U2,
        'th1p_0': th1p_0, 'th2_0': th2_0, 'th3_0': th3_0, 'th4_0': th4_0,
        'th2_om': th2_om, 'th3_om': th3_om, 'th4_om': th4_om,
        'th1_om': th1_om,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. PSLQ for A (36 bases, maxcoeff=10000, tol=1e-25)
# ═══════════════════════════════════════════════════════════════════════════

def run_pslq_A(A_mpf, const):
    """PSLQ identification of A against 36 bases."""
    A = A_mpf if isinstance(A_mpf, mpmath.mpf) else mpmath.mpf(A_mpf)
    omega = const['omega']
    tau_mp = const['tau']
    nome_mp = const['nome']

    th1p_0   = const['th1p_0']
    th2_0    = const['th2_0']
    th3_0    = const['th3_0']
    th4_0    = const['th4_0']
    th1_om   = const['th1_om']
    th2_om   = const['th2_om']
    th3_om   = const['th3_om']
    th4_om   = const['th4_om']
    U_mp     = const['U']
    U1p_mp   = const['U1p']
    s0_mp    = mpmath.re(th1_om**2 / th2_0**2)

    # ── Elliptic modulus and elliptic integrals ──
    m_mod = (mpmath.re(th2_0) / mpmath.re(th3_0))**4
    K_ell = mpmath.ellipk(m_mod)
    E_ell = mpmath.ellipe(m_mod)
    Kp_ell = mpmath.ellipk(1 - m_mod)

    # ── Dedekind eta ──
    eta_val = mpmath.exp(mpmath.pi * 1j * tau_mp / 12)
    for n in range(1, 200):
        eta_val *= (1 - mpmath.exp(2j * mpmath.pi * n * tau_mp))
    eta_re = mpmath.re(eta_val)

    # ── Eisenstein series E4, E6 ──
    def _eisenstein(tau, k_eis, terms=100):
        q_loc = mpmath.exp(2j * mpmath.pi * tau)
        B_map = {2: mpmath.mpf('1')/30, 3: mpmath.mpf('-1')/42}
        B2k = B_map[k_eis]
        coeff = (-1)**k_eis * 2 * k_eis / B2k
        sigma_sum = mpmath.mpf(0)
        for n in range(1, terms):
            d_sum = mpmath.mpf(0)
            for d in range(1, n+1):
                if n % d == 0:
                    d_sum += mpmath.power(d, 2*k_eis - 1)
            sigma_sum += d_sum * q_loc**n
        return 1 + coeff * sigma_sum

    E4_val = _eisenstein(tau_mp, 2)
    E6_val = _eisenstein(tau_mp, 3)
    j_inv = 1728 * E4_val**3 / (E4_val**3 - E6_val**2)

    # ── 36 bases (same as P5, but with A instead of C₂) ──
    # Also add A² since the amplitude formula involves A²
    bases = {
        # ── Basic ──
        'simple':        [1, A, A**2],
        'cubic':         [1, A, A**2, A**3],
        # ── omega ──
        'omega':         [1, A, omega, A*omega],
        'omega_sq':      [1, A, omega, omega**2],
        'omega_cubic':   [1, A, omega, omega**2, omega**3],
        # ── pi ──
        'pi':            [1, A, mpmath.pi, A*mpmath.pi],
        'pi2':           [1, A, mpmath.pi, mpmath.pi**2],
        # ── sqrt ──
        'sqrt':          [1, A, mpmath.sqrt(2), mpmath.sqrt(3), mpmath.sqrt(5)],
        # ── theta null ──
        'theta_null':    [1, A, mpmath.re(th2_0), mpmath.re(th3_0), mpmath.re(th4_0)],
        'theta_sq':      [1, A, mpmath.re(th2_0**2), mpmath.re(th3_0**2), mpmath.re(th4_0**2)],
        'theta_mixed':   [1, A, mpmath.re(th3_0), omega, mpmath.re(th3_0)*omega],
        # ── s₀ ──
        's0':            [1, A, s0_mp],
        's0_ext':        [1, A, s0_mp, s0_mp**2],
        's0_omega':      [1, A, s0_mp, omega],
        # ── U constants ──
        'U_const':       [1, A, U_mp, U1p_mp],
        # ── cross ──
        'omega_pi':      [1, A, omega, mpmath.pi, omega*mpmath.pi],
        'pi_theta':      [1, A, mpmath.pi, mpmath.re(th3_0), mpmath.re(th3_0)*mpmath.pi],
        'euler_log':     [1, A, mpmath.euler, mpmath.log(2), mpmath.log(3)],
        # ── deep ──
        'deep':          [1, A, mpmath.re(th1p_0), mpmath.re(th2_0*th3_0),
                          mpmath.re(th4_0)**2, omega**2],
        # ── special functions ──
        'zeta':          [1, A, mpmath.zeta(3), mpmath.zeta(5)],
        'catalan':       [1, A, mpmath.catalan, mpmath.pi**2/6],
        # ── theta at omega ──
        'theta_omega':   [1, A, mpmath.re(th2_om), mpmath.re(th3_om), mpmath.re(th4_om)],
        'th1p':          [1, A, mpmath.re(th1p_0), mpmath.re(th1p_0)**2],
        # ── ELLIPTIC INTEGRALS ──
        'K_E':           [1, A, K_ell, E_ell],
        'K_Kp':          [1, A, K_ell, Kp_ell],
        'K_pi':          [1, A, K_ell, mpmath.pi, K_ell/mpmath.pi],
        'KE_ext':        [1, A, K_ell, E_ell, K_ell*E_ell],
        'K_omega':       [1, A, K_ell, omega, K_ell*omega],
        # ── EISENSTEIN / MODULAR ──
        'E4E6':          [1, A, mpmath.re(E4_val), mpmath.re(E6_val)],
        'j_inv':         [1, A, mpmath.re(j_inv)],
        # ── ETA ──
        'eta':           [1, A, eta_re],
        'eta_ext':       [1, A, eta_re, eta_re**2],
        'eta_omega':     [1, A, eta_re, omega],
        'eta_K':         [1, A, eta_re, K_ell],
        'eta_pi':        [1, A, eta_re, mpmath.pi],
        # ── A-specific: amplitude formula ingredients ──
        'A2_formula':    [1, A, A**2, mpmath.re(th3_0)**2 * omega],
        'A_C2_cross':    [1, A, mpmath.mpf('7.3232473689982989680')],  # C₂ from P5
    }

    tol = mpmath.mpf(10) ** (-25)
    maxcoeff = 10000

    print(f"\n{'═'*100}")
    print(f"   PSLQ IDENTIFICATION OF A ({DPS} dps, tol=1e-25, maxcoeff={maxcoeff})")
    print(f"   A = {mpmath.nstr(A, 30)}")
    print(f"   {len(bases)} bases")
    print(f"{'═'*100}")
    print(f"\n   {'Basis':20s} {'Relation':>50s}  {'N':>5s} {'Err':>12s}")
    print(f"   {'─'*20} {'─'*50}  {'─'*5} {'─'*12}")

    results = {}
    hits = []

    for name, basis in bases.items():
        bvec = [mpmath.re(b) if isinstance(b, mpmath.mpc) else
                mpmath.mpf(b) if isinstance(b, (int, float)) else b
                for b in basis]
        try:
            rel = mpmath.pslq(bvec, tol=tol, maxcoeff=maxcoeff)
        except Exception:
            rel = None

        if rel is not None:
            norm = sum(abs(c) for c in rel)
            dot = sum(c * b for c, b in zip(rel, bvec))
            err = abs(dot)
            involves_A = len(rel) > 1 and rel[1] != 0

            if not involves_A:
                tag = "SPURI"
                results[name] = {'relation': [int(c) for c in rel], 'norm': int(norm),
                                 'error': float(err), 'spurious': True}
            elif norm <= 100 and float(err) < float(tol) * 10:
                tag = "HIT!!"
                hits.append((name, [int(c) for c in rel], int(norm), float(err)))
                results[name] = {'relation': [int(c) for c in rel], 'norm': int(norm),
                                 'error': float(err)}
            elif norm <= 500 and float(err) < float(tol) * 100:
                tag = " hit "
                hits.append((name, [int(c) for c in rel], int(norm), float(err)))
                results[name] = {'relation': [int(c) for c in rel], 'norm': int(norm),
                                 'error': float(err)}
            else:
                tag = "weak "
                results[name] = {'relation': [int(c) for c in rel], 'norm': int(norm),
                                 'error': float(err), 'weak': True}

            rel_str = str([int(c) for c in rel])
            print(f"   [{tag}] {name:20s} {rel_str:>50s}  {norm:5d} {float(err):.2e}")
        else:
            results[name] = None
            print(f"   [  —  ] {name:20s} {'no relation':>50s}")

    print(f"{'═'*100}")
    if hits:
        print(f"\n   ★ PSLQ HITS ({len(hits)}):")
        for name, rel, norm, err in hits:
            print(f"     {name}: {rel}  (norm={norm}, err={err:.2e})")
    else:
        print(f"\n   ★ PSLQ: DEFINITIVELY NO IDENTIFICATION FOR A")
        print(f"     {len(bases)} bases × maxcoeff={maxcoeff} × tol=1e-25")
        print(f"     A is a transcendental τ-specific constant (WKB residue).")
    print()
    return results, hits


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  P9: A to 19+ digits — from existing P5v3 data              ║")
    print("║  Bonnet's Problem — Zero-cost extraction                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    t0 = time.time()
    log_msg("P9 started")

    # ── [1/4] Load checkpoint ──
    print("[1/4] Loading P5v3 checkpoint...")
    raw_data = load_checkpoint()

    # ── [2/4] Compute A_k ──
    print(f"\n[2/4] Computing A_k = δ_k·√k at {DPS} dps...")
    Ak_by_k = compute_Ak(raw_data)

    # Quick sanity check
    for k in [100, 500, 1000, 5000]:
        if k in Ak_by_k:
            print(f"  k={k:>5d}: A_k = {mpmath.nstr(Ak_by_k[k], 20)}")

    # Compare with paper value A = 7.0814165 ± 3×10⁻⁷
    if 500 in Ak_by_k:
        A_500 = float(Ak_by_k[500])
        print(f"\n  A_500 = {A_500:.10f}  (paper: 7.0814165 ± 3e-7)")
        print(f"  |A_500 - 7.0814165| ≈ {abs(A_500 - 7.0814165):.2e}")

    # ── [3/4] Richardson extrapolation ──
    print(f"\n[3/4] Richardson extrapolation...")
    rich_results = richardson_A(Ak_by_k)

    A_best = rich_results.get('A_best_mpf')
    if A_best is None:
        log_msg("ERROR: Richardson failed — no estimate")
        return

    log_msg(f"A_best = {mpmath.nstr(A_best, 30)}")
    log_msg(f"Stable digits: {rich_results['stable_digits']}")
    log_msg(f"Spread: {rich_results['spread']:.2e}")

    # ── [4/4] PSLQ identification ──
    print(f"\n[4/4] PSLQ identification ({DPS} dps, 36+ bases)...")
    const = init_constants()
    print(f"  ω = {mpmath.nstr(const['omega'], 20)}")

    pslq_results, pslq_hits = run_pslq_A(A_best, const)

    elapsed = time.time() - t0

    # ── Save results ──
    conclusion = "identified" if pslq_hits else "transcendental"
    output = {
        "A_best": rich_results['A_best'],
        "stable_digits": rich_results['stable_digits'],
        "spread": rich_results['spread'],
        "c1_estimate": rich_results.get('c1_estimate'),
        "richardson_table": rich_results['table'],
        "pslq_conclusion": conclusion,
        "pslq_hits": [{"basis": h[0], "relation": h[1], "norm": h[2], "error": h[3]}
                       for h in pslq_hits],
        "n_bases_tested": len(pslq_results),
        "elapsed_sec": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # ── Final summary ──
    print(f"\n{'═'*72}")
    print(f"  P9 COMPLETE — {elapsed:.0f}s")
    print(f"{'═'*72}")
    print(f"  A = {rich_results['A_best']}")
    print(f"  Stable digits: {rich_results['stable_digits']}")
    print(f"  PSLQ hits: {len(pslq_hits)}")
    print(f"  Conclusion: {conclusion}")
    if rich_results.get('c1_estimate'):
        print(f"  c₁ ≈ {rich_results['c1_estimate']}")
    print(f"  Saved to {RESULTS_FILE}")
    print(f"{'═'*72}")

    log_msg(f"P9 complete — A = {rich_results['A_best']}, "
            f"{rich_results['stable_digits']} digits, "
            f"PSLQ: {conclusion}, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
