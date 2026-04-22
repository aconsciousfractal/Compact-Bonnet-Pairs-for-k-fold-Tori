"""
LEGACY / auxiliary runner.

Phase 18.1 — PSLQ Algebraic Identification of Empirical Constants

Goal: identify the 6 empirical constants of the project as algebraic
expressions in terms of modular invariants of τ = 0.5 + 0.3205i.

Constants to identify:
  C1: δ²·k  → 48.30       (asymptotic collapse coefficient)
  C2: s₂∞   → 7.319       (moduli space asymptote)
  C3: s₁*·k → −11.5       (boundary singularity invariant)
  C4: ds₂/dδ slope → 0.478  (inter-fold linear law)
  C5: arg(x)·k → −4 rad   (spectral curve cross-ratio phase)
  C6: Procrustes floor → 0.04688  (F± vs f± geometric gap)

Method: mpmath high-precision evaluation of modular invariants,
then PSLQ / identify() search over basis functions.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Check if mpmath is available
try:
    import mpmath
    mpmath.mp.dps = 50  # 50 decimal digits
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    print("ERROR: mpmath required. Install with: pip install mpmath")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
#  Step 1: Compute modular invariants at τ = 0.5 + 0.3205i
# ═══════════════════════════════════════════════════════════════════

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical parameters
omega = mpmath.mpf("0.3890")
tau = mpmath.mpc("0.5", "0.3205")

print("=" * 70)
print("  PHASE 18.1 — PSLQ Algebraic Identification")
print("=" * 70)
print(f"\n  tau = {tau}")
print(f"  omega = {omega}")
print(f"  Im(tau) = {mpmath.im(tau)}")

# Jacobi theta functions at z=0 (theta constants)
# mpmath uses: jtheta(n, z, q)  where q = exp(i*pi*tau)
q = mpmath.exp(mpmath.j * mpmath.pi * tau)
print(f"\n  q = exp(i*pi*tau) = {q}")
print(f"  |q| = {abs(q)}")

# Theta constants theta_j(0|tau) for j=1,2,3,4
# Note: theta_1(0|tau) = 0 always, so we use theta_1'(0|tau)
theta2_0 = mpmath.jtheta(2, 0, q)
theta3_0 = mpmath.jtheta(3, 0, q)
theta4_0 = mpmath.jtheta(4, 0, q)

# theta_1'(0|tau) = pi * theta_2(0) * theta_3(0) * theta_4(0)
theta1_prime_0 = mpmath.pi * theta2_0 * theta3_0 * theta4_0

print(f"\n  Theta constants at z=0:")
print(f"    theta_2(0|tau) = {theta2_0}")
print(f"    theta_3(0|tau) = {theta3_0}")
print(f"    theta_4(0|tau) = {theta4_0}")
print(f"    theta_1'(0|tau) = {theta1_prime_0}")

# Dedekind eta function: eta(tau) = q^(1/24) * prod(1-q^(2n))
eta_tau = mpmath.eta(tau)
print(f"    eta(tau)        = {eta_tau}")

# Weierstrass invariants
# eta1 = -pi^2/(6*omega) * (theta_3(0)^4 + theta_4(0)^4) / theta_3(0)^2 / theta_4(0)^2 ... 
# Actually, let's compute directly:
# In the lattice Lambda = Z*2*omega + Z*2*omega*tau:
omega1 = 2 * omega  # real period
omega2 = 2 * omega * tau  # complex period

# Weierstrass zeta quasi-period: eta1 = zeta(omega1/2) -- use Eisenstein series
# eta_1 = pi^2 / (6 * omega1) * E_2(tau)  -- Eisenstein E_2
# Simpler: use the relation eta1 = -pi^2/(6*omega1) * (theta_3^4 + theta_4^4 - theta_2^4) ... hmm
# Let mpmath do it:
g2 = mpmath.mpf(60) * mpmath.nsum(lambda m, n: (m*omega1 + n*omega2)**(-4),
                                    [[-2, 2], [-2, 2]], 
                                    ignore=True) if False else None

# Use standard modular forms instead
# Klein j-invariant
j_tau = mpmath.kleinj(tau)
print(f"    j(tau)          = {j_tau}")

# Modular lambda
lam = mpmath.ellipk(mpmath.mpf(0))  # not quite... 
# Actually modular lambda = (theta_2/theta_3)^4
modular_lambda = (theta2_0 / theta3_0)**4
print(f"    lambda(tau)     = {modular_lambda}")

# Key ratios and products
print(f"\n  Derived quantities:")
im_tau = mpmath.im(tau)
re_tau = mpmath.re(tau)

# Build a dictionary of candidate basis elements
basis = {}
basis["1"] = mpmath.mpf(1)
basis["pi"] = mpmath.pi
basis["pi^2"] = mpmath.pi**2
basis["omega"] = omega
basis["omega^2"] = omega**2
basis["Im_tau"] = im_tau
basis["Im_tau^2"] = im_tau**2
basis["1/Im_tau"] = 1/im_tau
basis["1/Im_tau^2"] = 1/im_tau**2
basis["theta2"] = mpmath.re(theta2_0)  # theta2 at 0 is real for our tau
basis["theta3"] = mpmath.re(theta3_0)
basis["theta4"] = mpmath.re(theta4_0)
basis["theta1p"] = mpmath.re(theta1_prime_0)
basis["eta"] = abs(eta_tau)
basis["pi*Im_tau"] = mpmath.pi * im_tau
basis["omega*Im_tau"] = omega * im_tau
basis["pi/omega"] = mpmath.pi / omega
basis["pi^2/omega^2"] = mpmath.pi**2 / omega**2
basis["theta3^2"] = mpmath.re(theta3_0)**2
basis["theta4^2"] = mpmath.re(theta4_0)**2
basis["theta2^2"] = mpmath.re(theta2_0)**2
basis["theta3^4"] = mpmath.re(theta3_0)**4
basis["j_tau_re"] = mpmath.re(j_tau)
basis["mod_lambda_re"] = mpmath.re(modular_lambda)

print(f"\n  Basis elements:")
for name, val in basis.items():
    print(f"    {name:20s} = {float(val):.10f}")

# ═══════════════════════════════════════════════════════════════════
#  Step 2: Define empirical constants
# ═══════════════════════════════════════════════════════════════════

# Load precise values from results
with open(PROJECT / "data" / "spectral_invariants.json") as f:
    p17 = json.load(f)

with open(PROJECT / "data" / "full_series_k3_1000.json") as f:
    fs_list = json.load(f)
# Build arrays from list of records
fs = {"k": [r["k"] for r in fs_list], "delta": [r["delta"] for r in fs_list], "s2": [r["s2"] for r in fs_list]}

# C1: delta^2 * k asymptote — compute from full_series data
# Take last few k values where delta^2*k is converged
k_arr = np.array(fs["k"], dtype=float)
d_arr = np.array(fs["delta"], dtype=float)
d2k = d_arr**2 * k_arr
# Use k>=40 average as best estimate of asymptote
mask = k_arr >= 40
C1 = mpmath.mpf(str(float(np.mean(d2k[mask]))))

# C2: s2_infinity
gf = p17["task_17_4_interfold"]["gauge_fixed"]
C2 = mpmath.mpf(str(gf["s2_asymptote"]))

# C3: s1* * k ~ -11.5 (from Phase 17.2 — load from earlier results)
# Use the boundary singularity: s1* = 0.32k - 3.87, so s1*·k ≈ 0.32k² - 3.87k
# At large k, that diverges — so the "near-constant" is approximate for k=5,6,7
# Better: use the actual values. s1*·k values: k=5: -11.35, k=6: -11.1, k=7: -11.41
# Average ~ -11.3
C3 = mpmath.mpf("-11.3")

# C4: ds2/ddelta linear slope = 0.478
C4 = mpmath.mpf("0.478")

# C5: arg(x)*k → -4 rad  (from Phase 13B)
C5 = mpmath.mpf("-4.0")

# C6: Procrustes floor
C6 = mpmath.mpf("0.04688")

constants = {
    "C1_delta2k": (C1, "delta^2 * k asymptote"),
    "C2_s2_inf": (C2, "s2 infinity"),
    "C3_s1k": (C3, "s1* * k invariant"),
    "C4_slope": (C4, "ds2/ddelta slope"),
    "C5_argxk": (C5, "arg(x)*k spectral phase"),
    "C6_floor": (C6, "Procrustes floor"),
}

print(f"\n  Empirical constants:")
for name, (val, desc) in constants.items():
    print(f"    {name:15s} = {float(val):12.6f}  ({desc})")

# ═══════════════════════════════════════════════════════════════════
#  Step 3: PSLQ / identify() search
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PSLQ IDENTIFICATION")
print(f"{'='*70}")

results = {}

for cname, (cval, cdesc) in constants.items():
    print(f"\n  --- {cname}: {float(cval):.6f} ({cdesc}) ---")
    
    # Method 1: mpmath.identify()
    try:
        ident = mpmath.identify(float(cval), tol=1e-4)
        if ident:
            print(f"    identify(tol=1e-4): {ident}")
        else:
            print(f"    identify(tol=1e-4): no match")
    except Exception as e:
        print(f"    identify error: {e}")
        ident = None
    
    # Method 2: mpmath.identify() with stricter tolerance
    try:
        ident2 = mpmath.identify(float(cval), tol=1e-6)
        if ident2:
            print(f"    identify(tol=1e-6): {ident2}")
    except:
        ident2 = None
    
    # Method 3: PSLQ against modular basis
    # Try: c = a0 + a1*pi + a2*omega + a3*Im_tau + a4*theta3 + a5*pi^2
    # Using small integer coefficients
    print(f"    PSLQ against modular basis:")
    
    # Select relevant basis subsets for PSLQ (max ~6 elements for stability)
    pslq_bases = [
        ("simple", [cval, mpmath.mpf(1), mpmath.pi, mpmath.pi**2]),
        ("omega", [cval, mpmath.mpf(1), omega, omega**2, mpmath.pi]),
        ("Im_tau", [cval, mpmath.mpf(1), im_tau, 1/im_tau, mpmath.pi]),
        ("theta", [cval, mpmath.mpf(1), mpmath.re(theta3_0), mpmath.re(theta4_0), mpmath.pi]),
        ("theta_sq", [cval, mpmath.mpf(1), mpmath.re(theta3_0)**2, mpmath.re(theta4_0)**2, mpmath.pi]),
        ("mixed1", [cval, mpmath.mpf(1), mpmath.pi/omega, im_tau, mpmath.re(theta3_0)]),
        ("mixed2", [cval, mpmath.mpf(1), mpmath.pi**2/omega**2, im_tau**2, mpmath.re(theta3_0)**2]),
        ("pi_omega", [cval, mpmath.mpf(1), mpmath.pi, omega, mpmath.pi*omega, mpmath.pi/omega]),
    ]
    
    best_pslq = None
    best_pslq_norm = 1e20
    
    for bname, bvec in pslq_bases:
        try:
            rel = mpmath.pslq(bvec, tol=mpmath.mpf(10)**(-8), maxcoeff=100)
            if rel is not None:
                # rel[0]*cval + rel[1]*b1 + rel[2]*b2 + ... = 0
                # So cval = -(rel[1]*b1 + rel[2]*b2 + ...) / rel[0]
                norm = sum(abs(r) for r in rel)
                if rel[0] != 0 and norm < 50:  # small integer relation
                    print(f"      [{bname}] coeffs={rel} (norm={norm})")
                    if norm < best_pslq_norm:
                        best_pslq_norm = norm
                        best_pslq = (bname, rel, bvec)
        except Exception as e:
            pass
    
    if best_pslq:
        bname, rel, bvec = best_pslq
        # Reconstruct value from relation
        reconstructed = -sum(rel[i] * bvec[i] for i in range(1, len(rel))) / rel[0]
        error = abs(float(cval) - float(reconstructed))
        print(f"    Best PSLQ [{bname}]: error = {error:.2e}")
        results[cname] = {
            "value": float(cval),
            "basis": bname,
            "coefficients": [int(r) for r in rel],
            "reconstructed": float(reconstructed),
            "error": error,
            "description": cdesc,
        }
    else:
        print(f"    No PSLQ relation found with small coefficients")
        results[cname] = {
            "value": float(cval),
            "basis": None,
            "description": cdesc,
        }

# ═══════════════════════════════════════════════════════════════════
#  Step 4: Specific targeted identifications
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  TARGETED IDENTIFICATIONS")
print(f"{'='*70}")

# C1 = 48.30: is it related to pi^2/omega^2 * Im_tau or similar?
print(f"\n  C1 = {float(C1):.4f} (delta^2*k asymptote)")
candidates_C1 = {
    "pi^2 / Im_tau^2": float(mpmath.pi**2 / im_tau**2),
    "4*pi^2 / (omega * Im_tau)": float(4*mpmath.pi**2 / (omega * im_tau)),
    "pi^2 / (omega^2 * Im_tau)": float(mpmath.pi**2 / (omega**2 * im_tau)),
    "2*pi^2*omega / Im_tau": float(2*mpmath.pi**2*omega / im_tau),
    "8*pi*omega / Im_tau": float(8*mpmath.pi*omega / im_tau),
    "pi^2 * theta3^4": float(mpmath.pi**2 * mpmath.re(theta3_0)**4),
    "theta1'^2 / pi^2": float(mpmath.re(theta1_prime_0)**2 / mpmath.pi**2),
    "16*pi^2*omega^2": float(16*mpmath.pi**2*omega**2),
    "(2*pi*omega)^2 / Im_tau": float((2*mpmath.pi*omega)**2 / im_tau),
    "pi/(omega*Im_tau^2)": float(mpmath.pi / (omega*im_tau**2)),
    "4*pi / (omega*Im_tau)": float(4*mpmath.pi / (omega*im_tau)),
}
for name, val in sorted(candidates_C1.items(), key=lambda x: abs(x[1] - float(C1))):
    err = abs(val - float(C1))
    pct = err / float(C1) * 100
    if pct < 20:
        print(f"    {name:35s} = {val:12.6f}  (err={pct:.2f}%)")

# C2 = 7.319: moduli space asymptote
print(f"\n  C2 = {float(C2):.4f} (s2 infinity)")
candidates_C2 = {
    "pi / Im_tau": float(mpmath.pi / im_tau),
    "2*pi*omega": float(2*mpmath.pi*omega),
    "pi / omega": float(mpmath.pi / omega),
    "pi * theta3^2": float(mpmath.pi * mpmath.re(theta3_0)**2),
    "theta1' / theta3": float(mpmath.re(theta1_prime_0) / mpmath.re(theta3_0)),
    "theta1' / pi": float(mpmath.re(theta1_prime_0) / mpmath.pi),
    "pi^2 * omega": float(mpmath.pi**2 * omega),
    "pi*omega/Im_tau": float(mpmath.pi*omega/im_tau),
    "4*omega/Im_tau": float(4*omega/im_tau),
    "2*pi*Im_tau": float(2*mpmath.pi*im_tau),
    "pi + pi*omega": float(mpmath.pi + mpmath.pi*omega),
    "theta3^2 + theta4^2": float(mpmath.re(theta3_0)**2 + mpmath.re(theta4_0)**2),
    "3*pi*omega/Im_tau": float(3*mpmath.pi*omega/im_tau),
    "2/Im_tau + pi": float(2/im_tau + mpmath.pi),
}
for name, val in sorted(candidates_C2.items(), key=lambda x: abs(x[1] - float(C2))):
    err = abs(val - float(C2))
    pct = err / float(C2) * 100
    if pct < 20:
        print(f"    {name:35s} = {val:12.6f}  (err={pct:.2f}%)")

# C5 = -4: this is already an integer! But is it exactly -4?
print(f"\n  C5 = {float(C5):.4f} (arg(x)*k spectral phase)")
print(f"    Note: arg(x)*k = -4 rad is already integer-valued")
print(f"    This could be -4 = -2*2 or -4*1, related to ramification degree")
# Load precise 13B data to check
try:
    with open(PROJECT / "data" / "cross_ratio_analysis.json") as f:
        p13b = json.load(f)
    a13 = p13b["analysis"]
    for k in [5, 6, 7]:
        kd = a13[f"k{k}"]
        x_re = kd["x_at_s1_min"]["x_re"]
        x_im = kd["x_at_s1_min"]["x_im"]
        import math
        arg_rad = math.atan2(x_im, x_re)
        print(f"    k={k}: arg(x) = {arg_rad:.6f} rad, arg*k = {arg_rad*k:.6f}")
except Exception as e:
    print(f"    [Could not load 13B data: {e}]")
candidates_C3 = {
    "-4*pi*omega/Im_tau": float(-4*mpmath.pi*omega/im_tau),
    "-pi^2/omega": float(-mpmath.pi**2/omega),
    "-pi^2*Im_tau": float(-mpmath.pi**2*im_tau),
    "-2*pi*omega*Im_tau*10": float(-20*mpmath.pi*omega*im_tau),
    "-pi/Im_tau^2": float(-mpmath.pi/im_tau**2),
    "-8/Im_tau": float(-8/im_tau),
    "-4/Im_tau": float(-4/im_tau),
    "-pi^2/(omega*Im_tau)": float(-mpmath.pi**2/(omega*im_tau)),
    "-3/Im_tau - pi": float(-3/im_tau - mpmath.pi),
    "-2*pi*Im_tau - pi": float(-2*mpmath.pi*im_tau - mpmath.pi),
}
for name, val in sorted(candidates_C3.items(), key=lambda x: abs(x[1] - float(C3))):
    err = abs(val - float(C3))
    pct = abs(err / float(C3) * 100)
    if pct < 20:
        print(f"    {name:35s} = {val:12.6f}  (err={pct:.2f}%)")

# ═══════════════════════════════════════════════════════════════════
#  Step 5: Continued fraction analysis
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  CONTINUED FRACTION ANALYSIS")
print(f"{'='*70}")

for cname, (cval, cdesc) in constants.items():
    cf = mpmath.identify(float(cval), tol=1e-3, full=True)
    # Simple continued fraction
    try:
        cf_list = list(mpmath.cf(mpmath.mpf(str(float(cval)))))[:8]
        print(f"  {cname} = {float(cval):.6f}: CF = {cf_list}")
    except:
        pass

# ═══════════════════════════════════════════════════════════════════
#  Step 6: Ratios between constants
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RATIOS BETWEEN CONSTANTS")
print(f"{'='*70}")

cvals = {n: float(v) for n, (v, _) in constants.items()}

print(f"  C1/C2 = {cvals['C1_delta2k']/cvals['C2_s2_inf']:.6f} (delta2k / s2_inf)")
print(f"  C2/pi = {cvals['C2_s2_inf']/np.pi:.6f}")
print(f"  C1/pi^2 = {cvals['C1_delta2k']/np.pi**2:.6f}")
print(f"  C3/pi = {cvals['C3_s1k']/np.pi:.6f}")
print(f"  C4*C2 = {cvals['C4_slope']*cvals['C2_s2_inf']:.6f}")
print(f"  C1*C6 = {cvals['C1_delta2k']*cvals['C6_floor']:.6f}")
print(f"  C6*pi = {cvals['C6_floor']*np.pi:.6f}")
print(f"  sqrt(C1) = {np.sqrt(cvals['C1_delta2k']):.6f}")
print(f"  sqrt(C1)/pi = {np.sqrt(cvals['C1_delta2k'])/np.pi:.6f}")
print(f"  C2*omega = {cvals['C2_s2_inf']*0.389:.6f}")
print(f"  C2*Im_tau = {cvals['C2_s2_inf']*0.3205:.6f}")

# ═══════════════════════════════════════════════════════════════════
#  Save results
# ═══════════════════════════════════════════════════════════════════

out_path = RESULTS_DIR / "phase18_1_pslq_results.json"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, mpmath.mpf)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

save_data = {
    "tau": {"re": 0.5, "im": 0.3205},
    "omega": 0.389,
    "modular_invariants": {
        "theta2_0": float(mpmath.re(theta2_0)),
        "theta3_0": float(mpmath.re(theta3_0)),
        "theta4_0": float(mpmath.re(theta4_0)),
        "theta1_prime_0": float(mpmath.re(theta1_prime_0)),
        "eta_tau_abs": float(abs(eta_tau)),
        "j_tau_re": float(mpmath.re(j_tau)),
        "modular_lambda_re": float(mpmath.re(modular_lambda)),
    },
    "constants": {n: {"value": float(v), "desc": d} for n, (v, d) in constants.items()},
    "pslq_results": results,
}
with open(out_path, "w") as f:
    json.dump(save_data, f, indent=2, cls=NpEncoder)
print(f"\n  Saved: {out_path}")

print(f"\n{'='*70}")
print(f"  PHASE 18.1 COMPLETE")
print(f"{'='*70}")
