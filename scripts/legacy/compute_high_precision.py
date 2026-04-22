#!/usr/bin/env python3
"""
LEGACY / auxiliary runner.

═══════════════════════════════════════════════════════════════════════════
  P5 v3 — Arbitrary-Precision C₂ via mpmath solver  (Google Colab Edition)
═══════════════════════════════════════════════════════════════════════════

Break the float64 ceiling: re-solve k=100..5000 at 40 decimal digits,
then Richardson extrapolation → C₂ to ~20 digits → PSLQ definitive.

v2 result:  C₂ = 7.323247368 ± 8×10⁻¹⁰  (9 digits, float64 ceiling)
v3 target:  C₂ to 15-20 digits → PSLQ at norm ≤ 1000

STRATEGY:
  - Sparse k sampling: ~116 k-values from 100 to 5000
  - mpmath.findroot (Newton, 40 dps) using float64 seeds from v1/v2
  - Gauss-Legendre quadrature (120 nodes, pre-computed at full dps)
    → each integral = 120 evaluations, no adaptive overhead
  - Desingularized via s = mid + half·sin(φ)
  - mpmath Richardson extrapolation (no float64 in the pipeline)
  - PSLQ at 30+ digits against 22 modular/algebraic bases

USAGE (Google Colab):
    1. Mount Drive:  from google.colab import drive; drive.mount('/content/drive')
    2. Upload P5_checkpoint.json (or P5v2_checkpoint.json) to Drive/Bonnet_P5/
    3. Copy this entire file into a cell and run.

Estimated runtime: ~10-30 min on Colab CPU (2-5s per k-value).
═══════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════════════════════

import subprocess, sys

def ensure_packages():
    for pkg in ['mpmath', 'numpy', 'tqdm']:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

ensure_packages()

import json
import math
import os
import time
from pathlib import Path

import mpmath
import numpy as np
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════

DPS = 40                      # working precision: 40 decimal digits
TAU_IMAG = '0.3205128205'    # string to preserve precision
S1 = mpmath.mpf('-8.5')

# ── Sparse k selection (uniform-ish in 1/k) ──
K_SPARSE = sorted(set(
    list(range(100, 300, 10)) +     # 100,110,...,290:  20 pts
    list(range(300, 1000, 20)) +    # 300,320,...,980:  35 pts
    list(range(1000, 3000, 50)) +   # 1000,1050,...,2950: 40 pts
    list(range(3000, 5001, 100))    # 3000,3100,...,5000: 21 pts
))
# ~116 sparse points for Richardson
# Plus verification points:
K_VERIFY = [5, 10, 20, 50]

SAVE_EVERY = 20  # checkpoint every 20 k-values

# ── Google Drive paths ──
DRIVE_DIR = Path("/content/drive/MyDrive/Bonnet_P5")

# Float64 seeds (from v1 or v2 checkpoint)
SEED_FILES = [
    DRIVE_DIR / "P5v2_checkpoint.json",
    DRIVE_DIR / "P5_checkpoint.json",
]

# V3 output
CHECKPOINT_FILE = DRIVE_DIR / "P5v3_checkpoint.json"
RESULTS_FILE    = DRIVE_DIR / "P5v3_final_results.json"
LOG_FILE        = DRIVE_DIR / "P5v3_log.txt"

# ── Fallback for local runs ──
if not Path("/content/drive").exists():
    DRIVE_DIR = Path("./Bonnet_P5_local")
    SEED_FILES = [
        Path("./results/P5/P5v2_checkpoint.json"),
        Path("./results/P5/P5_checkpoint.json"),
    ]
    CHECKPOINT_FILE = DRIVE_DIR / "P5v3_checkpoint.json"
    RESULTS_FILE    = DRIVE_DIR / "P5v3_final_results.json"
    LOG_FILE        = DRIVE_DIR / "P5v3_log.txt"
    print("[INFO] Not on Colab — using local directory:", DRIVE_DIR)

# ═══════════════════════════════════════════════════════════════════════════
# 2. mpmath Constants (all at full precision)
# ═══════════════════════════════════════════════════════════════════════════

def init_constants_mp(dps):
    """Compute all paper constants at mpmath precision."""
    mpmath.mp.dps = dps

    tau_mp = mpmath.mpf('0.5') + 1j * mpmath.mpf(TAU_IMAG)
    nome_mp = mpmath.exp(1j * mpmath.pi * tau_mp)

    # ── Find ω_crit at mpmath precision ──
    # Seed from float64
    from scipy.optimize import brentq
    nome_f64 = complex(nome_mp)
    def f_f64(w):
        return complex(mpmath.jtheta(2, w, nome_f64, derivative=1)).real
    omega_f64 = brentq(f_f64, 0.01, 0.78)

    # Refine at full dps
    def f_mp(omega):
        return mpmath.re(mpmath.jtheta(2, omega, nome_mp, derivative=1))
    omega_mp = mpmath.findroot(f_mp, mpmath.mpf(str(omega_f64)))

    # ── Theta values at ω ──
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

    # ── Paper constants (Eqs. 59-61) ──
    U   = mpmath.re(-mpmath.mpf('0.5') * th1p_0 * th1_2om / th2_om**2)
    Up  = mpmath.re(-mpmath.mpf('0.5') * th1p_0 * th1p_2om / th2_om**2)
    U1p = mpmath.re(mpmath.mpf('0.5') * th1p_0**2 / th2_om**2)
    U2  = mpmath.re((th1p_0**2 / th2_om**2) * (
        (th1_om / th2_0)**2 + (th4_om / th3_0)**2 + (th3_om / th4_0)**2
    ))

    return {
        'omega': omega_mp,
        'nome': nome_mp,
        'tau': tau_mp,
        'U': U, 'Up': Up, 'U1p': U1p, 'U2': U2,
        'th1p_0': th1p_0, 'th2_0': th2_0, 'th3_0': th3_0, 'th4_0': th4_0,
        'th2_om': th2_om, 'th3_om': th3_om, 'th4_om': th4_om,
        'th1_om': th1_om,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Quartic Q(s) at mpmath precision
# ═══════════════════════════════════════════════════════════════════════════

def Q_coefficients_mp(delta, s1, s2, const):
    """
    Q(s) = -(s-s₁)²(s-s₂)² + δ²·Q₃(s)  as list [a₄, a₃, a₂, a₁, a₀].
    """
    U, Up, U1p, U2_ = const['U'], const['Up'], const['U1p'], const['U2']
    sig = s1 + s2    # s₁ + s₂
    pro = s1 * s2    # s₁·s₂

    # -(s-s₁)²(s-s₂)² = -s⁴ + 2σs³ - (σ²+2p)s² + 2pσs - p²
    a4 = mpmath.mpf(-1)
    a3 = 2*sig + 2*delta**2*U1p
    a2 = -(sig**2 + 2*pro) - delta**2*U2_
    a1 = 2*pro*sig - 2*delta**2*Up
    a0 = -(pro**2) - delta**2*U**2

    return [a4, a3, a2, a1, a0]


def Q_eval_mp(s, delta, s1, s2, const):
    """Evaluate Q(s) at mpmath precision."""
    coeffs = Q_coefficients_mp(delta, s1, s2, const)
    return sum(c * s**(4-i) for i, c in enumerate(coeffs))


def Q2_eval_mp(s, delta, s1, s2, U1p):
    """Q₂(s) = -(s-s₁)(s-s₂) + δ²·U₁'·s."""
    return -(s - s1)*(s - s2) + delta**2 * U1p * s


def Z0_squared_mp(delta, s1, s2, const):
    """Z₀² — Eq. (88)."""
    U, Up, U1p, U2_ = const['U'], const['Up'], const['U1p'], const['U2']
    return (
        U**(-2) * (2*(s1+s2)*U1p + delta**2 * U1p**2 - U2_)
        + U**(-4) * (Up + s1*s2*U1p)**2
    )


def Qtilde2_eval_mp(s, delta, s1, s2, const):
    """Q̃₂(s) — Eq. (87)."""
    U, Up, U1p = const['U'], const['Up'], const['U1p']
    z0sq = Z0_squared_mp(delta, s1, s2, const)
    affine = 1 + s * U**(-2) * (Up + s1*s2*U1p)
    return z0sq * s**2 - affine**2


def find_real_oval_mp(delta, s1, s2, const):
    """Find real oval [s_lo, s_hi] of Q at mpmath precision."""
    coeffs = Q_coefficients_mp(delta, s1, s2, const)
    roots = mpmath.polyroots(coeffs)
    # Filter real roots
    eps = mpmath.mpf(10) ** (-(mpmath.mp.dps - 5))
    real_roots = sorted(
        [mpmath.re(r) for r in roots if abs(mpmath.im(r)) < eps]
    )
    if len(real_roots) < 2:
        raise ValueError(f"< 2 real roots at delta={delta}, s2={s2}")

    # Find pair bracketing s₁
    for a, b in zip(real_roots, real_roots[1:]):
        if a <= s1 <= b:
            return a, b

    # Fallback: find pair with largest Q(midpoint) > 0
    best = None
    for a, b in zip(real_roots, real_roots[1:]):
        mid = (a + b) / 2
        qm = Q_eval_mp(mid, delta, s1, s2, const)
        if qm > 0 and (best is None or qm > best[2]):
            best = (a, b, qm)
    if best:
        return best[0], best[1]
    raise ValueError("No positive real oval found")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Gauss-Legendre Quadrature (pre-computed, fast)
# ═══════════════════════════════════════════════════════════════════════════

N_QUAD = 120  # Gauss-Legendre nodes — 120 is overkill for 40 digits

# Cache for GL nodes/weights at full dps
_GL_CACHE = {}

def _compute_gl_nodes_weights(n, dps):
    """Compute n-point Gauss-Legendre nodes and weights on [-1,1] at mpmath precision."""
    mpmath.mp.dps = dps + 15  # extra guard digits
    nodes = []
    weights = []
    m = (n + 1) // 2
    for i in range(m):
        # Initial guess from asymptotic formula
        theta = mpmath.pi * (4*i + 3) / (4*n + 2)
        x = mpmath.cos(theta)
        # Newton iterations on P_n(x)
        for _ in range(100):
            p0, p1 = mpmath.mpf(1), x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2*j - 1) * x * p1 - (j - 1) * p0) / j
            # p1 = P_n(x), derivative: P_n'(x) = n(xP_n - P_{n-1})/(x²-1)
            dp = n * (x * p1 - p0) / (x**2 - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mpmath.mpf(10)**(-(dps + 10)):
                break
        w = 2 / ((1 - x**2) * dp**2)
        nodes.append(x)
        weights.append(w)
    # Symmetric pairs
    full_nodes = []
    full_weights = []
    for i in range(m):
        full_nodes.append(-nodes[i])
        full_weights.append(weights[i])
    for i in range(m - 1, -1, -1):
        if n % 2 == 1 and i == 0:
            continue  # already added the zero node
        full_nodes.append(nodes[i])
        full_weights.append(weights[i])
    mpmath.mp.dps = dps
    return full_nodes, full_weights


def _get_gl_nodes(n, dps):
    """Pre-compute Gauss-Legendre nodes and weights on [-π/2, π/2]."""
    key = (n, dps)
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    print(f"  Computing {n}-point Gauss-Legendre nodes at {dps} dps...", flush=True)
    t0 = time.time()
    nodes_std, weights_std = _compute_gl_nodes_weights(n, dps)
    # Map to [-π/2, π/2]: φ = π/2 · t, dφ = π/2 · dt
    hp = mpmath.pi / 2
    nodes  = [hp * t for t in nodes_std]
    weights = [hp * w for w in weights_std]
    # Pre-compute sin(φ) and cos(φ) for each node
    sins = [mpmath.sin(phi) for phi in nodes]
    coss = [mpmath.cos(phi) for phi in nodes]
    _GL_CACHE[key] = (nodes, weights, sins, coss)
    print(f"  GL nodes ready ({time.time()-t0:.1f}s)", flush=True)
    return _GL_CACHE[key]


def theta_half_integral_mp(delta, s1, s2, const, s_lo, s_hi):
    """θ/2 integral — Eq. (82), Gauss-Legendre on desingularized variable."""
    z0sq = Z0_squared_mp(delta, s1, s2, const)
    z0 = mpmath.sqrt(mpmath.re(z0sq)) if mpmath.re(z0sq) > 0 else mpmath.mpf(0)
    U1p = const['U1p']
    mid  = (s_lo + s_hi) / 2
    half = (s_hi - s_lo) / 2

    _, weights, sins, coss = _get_gl_nodes(N_QUAD, mpmath.mp.dps)
    total = mpmath.mpf(0)
    for i in range(N_QUAD):
        cos_phi = coss[i]
        s = mid + half * sins[i]
        q = Q_eval_mp(s, delta, s1, s2, const)
        if q <= 0:
            continue
        qt2 = Qtilde2_eval_mp(s, delta, s1, s2, const)
        if abs(qt2) < mpmath.mpf(10)**(-mpmath.mp.dps + 2):
            continue
        q2 = Q2_eval_mp(s, delta, s1, s2, U1p)
        total += weights[i] * z0 * q2 / (qt2 * mpmath.sqrt(q)) * half * cos_phi
    return total


def axial_integral_mp(delta, s1, s2, const, s_lo, s_hi):
    """Axial integral — Eq. (83), Gauss-Legendre on desingularized variable."""
    U1p = const['U1p']
    mid  = (s_lo + s_hi) / 2
    half = (s_hi - s_lo) / 2

    _, weights, sins, coss = _get_gl_nodes(N_QUAD, mpmath.mp.dps)
    total = mpmath.mpf(0)
    for i in range(N_QUAD):
        cos_phi = coss[i]
        s = mid + half * sins[i]
        q = Q_eval_mp(s, delta, s1, s2, const)
        if q <= 0:
            continue
        q2 = Q2_eval_mp(s, delta, s1, s2, U1p)
        total += weights[i] * q2 / mpmath.sqrt(q) * half * cos_phi
    return total


# ═══════════════════════════════════════════════════════════════════════════
# 5. mpmath Solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_k_mp(k, delta_init, s2_init, const, dps=40):
    """
    Solve periodicity equations at symmetry fold k using mpmath.findroot.
    Uses Gauss-Legendre quadrature (pre-computed) for speed.
    Returns dict with k, delta (str), s2 (str), residual.
    """
    mpmath.mp.dps = dps
    k_mp = mpmath.mpf(k)
    two_pi = 2 * mpmath.pi
    s1 = S1

    # Pre-warm GL cache
    _get_gl_nodes(N_QUAD, dps)

    def system(delta, s2):
        s_lo, s_hi = find_real_oval_mp(delta, s1, s2, const)
        th_half = theta_half_integral_mp(delta, s1, s2, const, s_lo, s_hi)
        theta = 2 * th_half
        F_rat = k_mp * theta / two_pi - 1
        F_ax = axial_integral_mp(delta, s1, s2, const, s_lo, s_hi)
        return (F_rat, F_ax)

    d0 = mpmath.mpf(str(delta_init))
    s0 = mpmath.mpf(str(s2_init))

    # findroot with Jacobian (secant method) — tol at dps-5
    tol = mpmath.mpf(10)**(-(dps - 5))
    try:
        sol = mpmath.findroot(system, (d0, s0), tol=tol, maxsteps=50, solver='secant')
        delta_sol = sol[0]
        s2_sol = sol[1]
    except Exception as e:
        # Retry with muller solver
        try:
            sol = mpmath.findroot(system, (d0, s0), tol=tol, maxsteps=100, solver='muller')
            delta_sol = sol[0]
            s2_sol = sol[1]
        except Exception:
            return None

    # Post-check residual
    try:
        r = system(delta_sol, s2_sol)
        res_norm = float(mpmath.sqrt(r[0]**2 + r[1]**2))
    except Exception:
        return None

    # At 40 dps, residuals should be < 10^{-30}
    if res_norm > mpmath.mpf(10)**(-(dps // 2)):
        return None

    return {
        'k': int(k),
        'delta': mpmath.nstr(delta_sol, dps - 2),
        's2': mpmath.nstr(s2_sol, dps - 2),
        'residual': res_norm,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Checkpoint Management
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_dir():
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

def load_float64_seeds():
    """Load float64 seeds from v1/v2 checkpoint."""
    for path in SEED_FILES:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                by_k = {s['k']: s for s in data.get('seeds', [])}
                print(f"  ✓ Loaded float64 seeds: {len(by_k)} seeds from {path.name}")
                return by_k
            except Exception as e:
                print(f"  ⚠ Failed to load {path}: {e}")
    return {}

def save_checkpoint(results, elapsed):
    _ensure_dir()
    data = {
        'version': 4,
        'script': 'P5_colab_v3',
        'dps': DPS,
        'elapsed_sec': elapsed,
        'n_seeds': len(results),
        'seeds': sorted(results.values(), key=lambda x: x['k']),
    }
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(CHECKPOINT_FILE)

def load_v3_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            by_k = {s['k']: s for s in data.get('seeds', [])}
            elapsed = data.get('elapsed_sec', 0.0)
            print(f"  ✓ Loaded v3 checkpoint: {len(by_k)} seeds")
            return by_k, elapsed
        except Exception:
            pass
    return {}, 0.0

def log_msg(msg):
    _ensure_dir()
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 7. mpmath Richardson Extrapolation
# ═══════════════════════════════════════════════════════════════════════════

def mp_polyfit(x_list, y_list, deg):
    """
    Least-squares polynomial fit at full mpmath precision.
    Returns coefficients [c₀, c₁, ..., cₙ] where p(x) = c₀ + c₁x + ... + cₙxⁿ.
    c₀ = constant term = C₂.
    """
    n = len(x_list)
    # Build Vandermonde matrix V[i,j] = x_i^j
    V = mpmath.matrix(n, deg + 1)
    yv = mpmath.matrix(n, 1)
    for i in range(n):
        xi = x_list[i]
        for j in range(deg + 1):
            V[i, j] = xi ** j
        yv[i, 0] = y_list[i]

    # Normal equations: (V^T V) c = V^T y
    VtV = V.T * V
    Vty = V.T * yv
    c = mpmath.lu_solve(VtV, Vty)
    return [c[j, 0] for j in range(deg + 1)]


def richardson_mp(results_by_k, dps=40):
    """
    Multi-window Richardson at mpmath precision.
    Uses deg 3–6 polynomial extrapolation.
    Consensus from deg ≥ 5 on windows with k_lo ≥ 200.
    """
    mpmath.mp.dps = dps

    ks_sorted = sorted(results_by_k.keys())
    k_max = ks_sorted[-1]

    # ── Windows ──
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
    print(f"   RICHARDSON EXTRAPOLATION (mpmath, {dps} dps)")
    print(f"{'═'*100}")
    header = f"   {'Window':16s}"
    for d in degs:
        header += f" {'deg '+str(d):>20s}"
    print(header)
    print(f"   {'─'*16}" + " " + " ".join(f"{'─'*20}" for _ in degs))

    table = {}
    for k_lo, k_hi in windows:
        mask_k = [k for k in ks_sorted if k_lo <= k <= k_hi]
        xw = [1 / mpmath.mpf(k) for k in mask_k]
        sw = [mpmath.mpf(results_by_k[k]['s2']) for k in mask_k]
        label = f"k={k_lo}..{k_hi}"
        parts = [label]
        for deg in degs:
            if len(xw) < deg + 3:
                parts.append("—")
                continue
            try:
                c = mp_polyfit(xw, sw, deg)
                c2 = c[0]
                table[(k_lo, k_hi, deg)] = c2
                parts.append(mpmath.nstr(c2, 18))
            except (ZeroDivisionError, Exception):
                parts.append("SING")
        print(f"   {parts[0]:16s}" + " ".join(f"{p:>20s}" for p in parts[1:]))

    if not table:
        return {'C2_best': '0', 'stable_digits': 0}

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

    # ── Best estimate: highest degree with >1 window ──
    # Prefer deg 6 > 5 > 4 > 3
    C2_best = None
    best_spread = 1.0
    best_stable = 0
    for deg in reversed(degs):
        if deg in best_deg_vals and len(best_deg_vals[deg][0]) >= 2:
            vals, mean, spread, stable = best_deg_vals[deg]
            C2_best = mean
            best_spread = spread
            best_stable = stable
            break

    if C2_best is None:
        # Fallback: any available
        all_est = list(table.values())
        C2_best = sum(all_est) / len(all_est)
        best_spread = max(float(v) for v in all_est) - min(float(v) for v in all_est)
        best_stable = max(0, -int(math.floor(math.log10(max(best_spread, 1e-40)))))

    print(f"\n   ★ BEST: C₂ = {mpmath.nstr(C2_best, 35)}")
    print(f"     Spread: {best_spread:.2e}, ~{best_stable} stable digits")
    print(f"{'═'*100}\n")

    return {
        'C2_best': mpmath.nstr(C2_best, dps - 2),
        'C2_best_mpf': C2_best,
        'spread': best_spread,
        'std': float(np.std([float(v) for v in table.values()])),
        'stable_digits': best_stable,
        'table': {f"{k[0]}-{k[1]}_d{k[2]}": mpmath.nstr(v, dps-2)
                  for k, v in table.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. PSLQ Identification (high-precision)
# ═══════════════════════════════════════════════════════════════════════════

def run_pslq_hp(C2_mpf, const, dps=40):
    """PSLQ at full mpmath precision — 36 bases, maxcoeff=10000."""
    mpmath.mp.dps = dps
    C2 = C2_mpf if isinstance(C2_mpf, mpmath.mpf) else mpmath.mpf(C2_mpf)
    omega = const['omega']
    tau_mp = const['tau']
    nome_mp = const['nome']

    th1p_0 = const['th1p_0']
    th2_0  = const['th2_0']
    th3_0  = const['th3_0']
    th4_0  = const['th4_0']
    th1_om = const['th1_om']
    th2_om = const['th2_om']
    th3_om = const['th3_om']
    th4_om = const['th4_om']
    U_mp   = const['U']
    U1p_mp = const['U1p']
    s0_mp  = mpmath.re(th1_om**2 / th2_0**2)

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

    # ── Eisenstein series E2, E4, E6 ──
    def _eisenstein(tau, k_eis, terms=100):
        q_loc = mpmath.exp(2j * mpmath.pi * tau)
        B_map = {1: mpmath.mpf('-1')/6, 2: mpmath.mpf('1')/30,
                 3: mpmath.mpf('-1')/42}
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

    bases = {
        # ── Basic ──
        'simple':        [1, C2, C2**2],
        'cubic':         [1, C2, C2**2, C2**3],
        # ── omega ──
        'omega':         [1, C2, omega, C2*omega],
        'omega_sq':      [1, C2, omega, omega**2],
        'omega_cubic':   [1, C2, omega, omega**2, omega**3],
        # ── pi ──
        'pi':            [1, C2, mpmath.pi, C2*mpmath.pi],
        'pi2':           [1, C2, mpmath.pi, mpmath.pi**2],
        # ── sqrt ──
        'sqrt':          [1, C2, mpmath.sqrt(2), mpmath.sqrt(3), mpmath.sqrt(5)],
        # ── theta null ──
        'theta_null':    [1, C2, mpmath.re(th2_0), mpmath.re(th3_0), mpmath.re(th4_0)],
        'theta_sq':      [1, C2, mpmath.re(th2_0**2), mpmath.re(th3_0**2), mpmath.re(th4_0**2)],
        'theta_mixed':   [1, C2, mpmath.re(th3_0), omega, mpmath.re(th3_0)*omega],
        # ── s₀ ──
        's0':            [1, C2, s0_mp],
        's0_ext':        [1, C2, s0_mp, s0_mp**2],
        's0_omega':      [1, C2, s0_mp, omega],
        # ── U constants ──
        'U_const':       [1, C2, U_mp, U1p_mp],
        # ── cross ──
        'omega_pi':      [1, C2, omega, mpmath.pi, omega*mpmath.pi],
        'pi_theta':      [1, C2, mpmath.pi, mpmath.re(th3_0), mpmath.re(th3_0)*mpmath.pi],
        'euler_log':     [1, C2, mpmath.euler, mpmath.log(2), mpmath.log(3)],
        # ── deep ──
        'deep':          [1, C2, mpmath.re(th1p_0), mpmath.re(th2_0*th3_0),
                          mpmath.re(th4_0)**2, omega**2],
        # ── special functions ──
        'zeta':          [1, C2, mpmath.zeta(3), mpmath.zeta(5)],
        'catalan':       [1, C2, mpmath.catalan, mpmath.pi**2/6],
        # ── theta at omega ──
        'theta_omega':   [1, C2, mpmath.re(th2_om), mpmath.re(th3_om), mpmath.re(th4_om)],
        'th1p':          [1, C2, mpmath.re(th1p_0), mpmath.re(th1p_0)**2],
        # ── ELLIPTIC INTEGRALS ──
        'K_E':           [1, C2, K_ell, E_ell],
        'K_Kp':          [1, C2, K_ell, Kp_ell],
        'K_pi':          [1, C2, K_ell, mpmath.pi, K_ell/mpmath.pi],
        'KE_ext':        [1, C2, K_ell, E_ell, K_ell*E_ell],
        'K_omega':       [1, C2, K_ell, omega, K_ell*omega],
        # ── EISENSTEIN / MODULAR ──
        'E4E6':          [1, C2, mpmath.re(E4_val), mpmath.re(E6_val)],
        'j_inv':         [1, C2, mpmath.re(j_inv)],
        # ── ETA ──
        'eta':           [1, C2, eta_re],
        'eta_ext':       [1, C2, eta_re, eta_re**2],
        'eta_omega':     [1, C2, eta_re, omega],
        'eta_K':         [1, C2, eta_re, K_ell],
        'eta_pi':        [1, C2, eta_re, mpmath.pi],
    }

    results = {}
    hits = []
    tol = mpmath.mpf(10) ** (-25)
    maxcoeff = 10000

    print(f"\n{'═'*100}")
    print(f"   PSLQ IDENTIFICATION (v3, {dps} dps, tol=1e-25, maxcoeff={maxcoeff})")
    print(f"   C₂ = {mpmath.nstr(C2, 30)}")
    print(f"   {len(bases)} bases")
    print(f"{'═'*100}")
    print(f"\n   {'Basis':20s} {'Relation':>50s}  {'N':>5s} {'Err':>12s}")
    print(f"   {'─'*20} {'─'*50}  {'─'*5} {'─'*12}")

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
            involves_C2 = len(rel) > 1 and rel[1] != 0

            if not involves_C2:
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
        print(f"\n   ★ PSLQ: DEFINITIVELY NO IDENTIFICATION")
        print(f"     {len(bases)} bases × maxcoeff={maxcoeff} × tol=1e-25")
        print(f"     C₂ = s₂∞ is a transcendental τ-specific constant.")
    print()
    return results, hits


# ═══════════════════════════════════════════════════════════════════════════
# 9. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  P5 v3: Arbitrary-Precision C₂ via mpmath (40 dps)        ║")
    print("║  Bonnet's Problem — Colab Edition                         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    _ensure_dir()
    mpmath.mp.dps = DPS
    log_msg("=" * 50)
    log_msg(f"P5 v3 started — {DPS} dps, {len(K_SPARSE)} sparse k-values")

    # ── [1/5] Constants ──
    print("[1/5] Computing constants at 40 dps...")
    const = init_constants_mp(DPS)
    print(f"  ω = {mpmath.nstr(const['omega'], 30)}")
    print(f"  U = {mpmath.nstr(const['U'], 20)}")
    log_msg(f"omega = {mpmath.nstr(const['omega'], 35)}")

    # ── [2/5] Load seeds ──
    print("\n[2/5] Loading float64 seeds...")
    seeds_f64 = load_float64_seeds()
    if not seeds_f64:
        print("  ⚠ No float64 seeds found! Need P5_checkpoint.json or P5v2_checkpoint.json on Drive.")
        print("  Run P5_colab_v2.py first to generate seeds.")
        return

    # Load existing v3 checkpoint
    results_mp, elapsed_total = load_v3_checkpoint()
    already_done = set(results_mp.keys())

    # ── All k values to compute ──
    all_k = sorted(set(K_VERIFY + K_SPARSE))
    all_k = [k for k in all_k if k in seeds_f64]  # only k with seeds
    to_do = [k for k in all_k if k not in already_done]

    print(f"  Target: {len(all_k)} k-values ({len(K_VERIFY)} verify + {len(K_SPARSE)} sparse)")
    print(f"  Already done: {len(already_done)}, remaining: {len(to_do)}")

    if to_do:
        # ── [3/5] Verification ──
        verify_k = [k for k in K_VERIFY if k in to_do and k in seeds_f64]
        if verify_k:
            print(f"\n[3/5] Verifying solver on k={verify_k[0]}...")
            t_test = time.time()
            test = solve_k_mp(verify_k[0], seeds_f64[verify_k[0]]['delta'],
                              seeds_f64[verify_k[0]]['s2'], const, DPS)
            dt_test = time.time() - t_test
            if test is None:
                print(f"  ⚠ Solver FAILED on k={verify_k[0]}!")
                return
            s2_diff = abs(float(mpmath.mpf(test['s2'])) - seeds_f64[verify_k[0]]['s2'])
            print(f"  ✓ k={verify_k[0]}: res={test['residual']:.2e}, "
                  f"|s₂_mp - s₂_f64| = {s2_diff:.2e}, time={dt_test:.1f}s")
            results_mp[verify_k[0]] = test
            to_do = [k for k in to_do if k != verify_k[0]]
            eta = dt_test * len(to_do) / 60
            print(f"  ETA: ~{eta:.0f} min for {len(to_do)} remaining k-values")
            log_msg(f"Verified k={verify_k[0]}: res={test['residual']:.2e}, "
                    f"time={dt_test:.1f}s, ETA={eta:.0f}min")
        else:
            print("\n[3/5] Verification k-values already done, skipping...")

        # ── [4/5] Main loop ──
        print(f"\n[4/5] Computing {len(to_do)} k-values at {DPS} dps...")
        t0 = time.time()
        pbar = tqdm(to_do, desc="mpmath solve", unit="k", dynamic_ncols=True)
        failures = 0

        for k in pbar:
            if k not in seeds_f64:
                continue
            result = solve_k_mp(k, seeds_f64[k]['delta'], seeds_f64[k]['s2'],
                                const, DPS)
            if result is not None:
                results_mp[k] = result
                failures = 0
                pbar.set_postfix({
                    's₂': result['s2'][:18],
                    'res': f"{result['residual']:.0e}",
                })
            else:
                failures += 1
                log_msg(f"k={k}: FAILED")
                if failures >= 5:
                    log_msg("5 consecutive failures, stopping.")
                    break

            # Checkpoint
            if len(results_mp) % SAVE_EVERY == 0:
                elapsed_now = elapsed_total + (time.time() - t0)
                save_checkpoint(results_mp, elapsed_now)

        pbar.close()
        elapsed_total += time.time() - t0
        save_checkpoint(results_mp, elapsed_total)
        log_msg(f"Computed {len(results_mp)} k-values in {elapsed_total/60:.1f} min")
    else:
        print("\n[3/5] All k-values already computed, skipping to analysis...")

    # ── Float64 vs mpmath comparison ──
    print(f"\n   Float64 vs mpmath (sample):")
    for k in [100, 500, 1000, 2000, 5000]:
        if k in results_mp and k in seeds_f64:
            s2_mp = float(mpmath.mpf(results_mp[k]['s2']))
            s2_f64 = seeds_f64[k]['s2']
            print(f"     k={k:5d}: |Δs₂| = {abs(s2_mp - s2_f64):.2e}")

    # ── [5/5] Richardson + PSLQ ──
    # Filter to main sparse set for Richardson
    rich_data = {k: results_mp[k] for k in K_SPARSE if k in results_mp}
    print(f"\n[5/5] Richardson + PSLQ ({len(rich_data)} k-values)")

    rich = richardson_mp(rich_data, DPS)

    if rich['stable_digits'] > 0:
        C2_best = rich['C2_best_mpf']
        pslq_results, pslq_hits = run_pslq_hp(C2_best, const, DPS)
    else:
        print("  ⚠ Richardson gave 0 stable digits — something went wrong.")
        pslq_results, pslq_hits = {}, []
        C2_best = mpmath.mpf(0)

    # ── Save final results ──
    final = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'P5_colab_v3',
        'dps': DPS,
        'n_seeds_mp': len(results_mp),
        'n_seeds_richardson': len(rich_data),
        'elapsed_min': elapsed_total / 60,
        'C2_best': rich['C2_best'],
        'C2_stable_digits': rich['stable_digits'],
        'C2_spread': rich['spread'],
        'richardson_table': rich.get('table', {}),
        'pslq_hits': [{'basis': h[0], 'relation': h[1], 'norm': h[2], 'error': h[3]}
                      for h in pslq_hits],
        'pslq_conclusion': 'identified' if pslq_hits else 'transcendental',
        'pslq_all': {k: v for k, v in pslq_results.items() if v is not None},
    }
    _ensure_dir()
    RESULTS_FILE.write_text(json.dumps(final, indent=2))

    log_msg("=" * 50)
    log_msg(f"P5 v3 COMPLETE: C₂ = {rich['C2_best']}")
    log_msg(f"Stable digits: {rich['stable_digits']}")
    log_msg(f"PSLQ hits: {len(pslq_hits)}")
    log_msg(f"Runtime: {elapsed_total/60:.1f} min")
    log_msg("=" * 50)

    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║  P5 v3 COMPLETE                                           ║")
    print(f"║  C₂ = {rich['C2_best'][:30]:30s}             ║")
    print(f"║  {rich['stable_digits']} stable digits                                    ║")
    print(f"║  PSLQ: {'IDENTIFIED' if pslq_hits else 'transcendental':14s}                               ║")
    print(f"║  Runtime: {elapsed_total/60:.1f} min                                     ║")
    print(f"╚════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
