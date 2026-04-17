"""
Weierstrass Elliptic Function for Bonnet pair computation.

Implements ℘(z; g₂, g₃) and related functions.
"""
from __future__ import annotations

import numpy as np
import mpmath


# ---------------------------------------------------------------------------
# ℘-function from invariants
# ---------------------------------------------------------------------------

def weierstrass_p(z: complex, g2: complex, g3: complex, N: int = 25) -> complex:
    omega1, omega3 = _half_periods(g2, g3)
    return _wp_from_periods(z, omega1, omega3, N)

def weierstrass_p_prime(z: complex, g2: complex, g3: complex, N: int = 25) -> complex:
    h = 1e-7
    return (weierstrass_p(z + h, g2, g3, N) -
            weierstrass_p(z - h, g2, g3, N)) / (2 * h)

def weierstrass_p_from_periods(z: complex, omega1: complex,
                               omega3: complex, N: int = 25) -> complex:
    return _wp_from_periods(z, omega1, omega3, N)

def _wp_from_periods(z: complex, omega1: complex, omega3: complex,
                     N: int = 25) -> complex:
    result = 1.0 / z**2
    for m in range(-N, N + 1):
        for n in range(-N, N + 1):
            if m == 0 and n == 0:
                continue
            omega_mn = 2 * m * omega1 + 2 * n * omega3
            result += 1.0 / (z - omega_mn)**2 - 1.0 / omega_mn**2
    return complex(result)

def _half_periods(g2: complex, g3: complex) -> tuple[complex, complex]:
    g2_m = mpmath.mpc(g2)
    g3_m = mpmath.mpc(g3)
    coeffs = [mpmath.mpf(4), mpmath.mpf(0), -g2_m, -g3_m]
    roots = mpmath.polyroots(coeffs)
    roots_c = sorted([complex(r) for r in roots], key=lambda x: -x.real)
    e1, e2, e3 = roots_c
    omega1 = complex(
        mpmath.ellipk((e2 - e3) / (e1 - e3)) / mpmath.sqrt(e1 - e3)
    )
    omega3 = complex(
        1j * mpmath.ellipk((e1 - e2) / (e1 - e3)) / mpmath.sqrt(e1 - e3)
    )
    return omega1, omega3


# ---------------------------------------------------------------------------
# Invariants from periods (q-expansion, faster than lattice sum)
# ---------------------------------------------------------------------------

def invariants_from_periods(omega1: complex, omega3: complex) -> tuple[complex, complex]:
    """
    Compute invariants (g₂, g₃) from half-periods (ω₁, ω₃).
    g₂ = 60 Σ' 1/(mω₁ + nω₃)⁴
    g₃ = 140 Σ' 1/(mω₁ + nω₃)⁶

    Uses Eisenstein q-expansion for fast convergence.
    """
    tau = omega3 / omega1
    q = mpmath.exp(2j * mpmath.pi * tau)

    sum_g2 = mpmath.mpf(0)
    sum_g3 = mpmath.mpf(0)
    for n in range(1, 100):
        qn = q**n
        factor = qn / (1 - qn)
        sum_g2 += n**3 * factor
        sum_g3 += n**5 * factor

    prefactor2 = (2 * mpmath.pi / omega1)**4
    prefactor3 = (2 * mpmath.pi / omega1)**6

    g2 = complex(prefactor2 * (mpmath.mpf(1) / 12 + 20 * sum_g2))
    g3 = complex(prefactor3 * (mpmath.mpf(1) / 216 - mpmath.mpf(7) / 3 * sum_g3))

    return g2, g3


# ---------------------------------------------------------------------------
# Weierstrass inversion (for Lemma 4)
# ---------------------------------------------------------------------------

def weierstrass_p_inverse(target: complex, g2: complex, g3: complex,
                          search_domain: tuple[float, float] = (0.01, 5.0),
                          n_samples: int = 200) -> complex:
    """
    Find z such that ℘(z; g₂, g₃) = target.

    Uses a grid search + refinement approach on the real axis for real targets,
    or Newton's method for complex targets.

    Note: ℘ is doubly periodic — returns the principal value.
    """
    omega1, omega3 = _half_periods(g2, g3)

    if abs(complex(target).imag) < 1e-10:
        # Real target: search along real axis (fundamental domain)
        from scipy.optimize import brentq

        t_real = complex(target).real
        a, b = search_domain

        def residual(x):
            return complex(mpmath.ellipfun('wp', x, omega1, omega3)).real - t_real

        # Find bracketing interval via sampling
        xs = np.linspace(a, b, n_samples)
        vals = [residual(x) for x in xs]

        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                return complex(brentq(residual, xs[i], xs[i + 1]))

        raise ValueError(f"Could not bracket ℘⁻¹({target}) in [{a}, {b}]")

    else:
        # Complex target: Newton's method
        # ℘(z) = target → iterate z_{n+1} = z_n - (℘(z_n)-target)/℘'(z_n)
        z = complex(omega1) * 0.5  # initial guess
        for _ in range(50):
            wp_val = complex(mpmath.ellipfun('wp', z, omega1, omega3))
            residual = wp_val - complex(target)
            if abs(residual) < 1e-14:
                return z
            # ℘'² = 4℘³ - g₂℘ - g₃
            wp_prime_sq = 4 * wp_val**3 - complex(g2) * wp_val - complex(g3)
            wp_prime = np.sqrt(wp_prime_sq)
            if abs(wp_prime) < 1e-30:
                raise ValueError("℘' ≈ 0 at iteration point")
            z = z - residual / wp_prime

        raise ValueError(f"Newton's method did not converge for ℘⁻¹({target})")
