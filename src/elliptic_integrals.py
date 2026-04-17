"""
Elliptic Integrals for Bonnet pair computation.

Wrappers around scipy.special with paper conventions:
  scipy uses parameter m = k²; our API accepts modulus k.
"""
from __future__ import annotations

import numpy as np
from scipy import special
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Complete elliptic integrals
# ---------------------------------------------------------------------------

def K(k: float) -> float:
    return float(special.ellipk(k**2))

def E(k: float) -> float:
    return float(special.ellipe(k**2))

def Kp(k: float) -> float:
    kp = np.sqrt(1 - k**2)
    return float(special.ellipk(kp**2))

def Ep(k: float) -> float:
    kp = np.sqrt(1 - k**2)
    return float(special.ellipe(kp**2))

def F_incomplete(phi: float, k: float) -> float:
    return float(special.ellipkinc(phi, k**2))

def E_incomplete(phi: float, k: float) -> float:
    return float(special.ellipeinc(phi, k**2))

def jacobi_elliptic(u: float, k: float) -> tuple[float, float, float]:
    sn, cn, dn, _ph = special.ellipj(u, k**2)
    return float(sn), float(cn), float(dn)

def legendre_relation(k: float) -> float:
    return E(k) * Kp(k) + Ep(k) * K(k) - K(k) * Kp(k) - np.pi / 2

def nome_from_modulus(k: float) -> float:
    return float(np.exp(-np.pi * Kp(k) / K(k)))

def tau_from_modulus(k: float) -> complex:
    return 1j * Kp(k) / K(k)


# ---------------------------------------------------------------------------
# Generalized elliptic integrals (for periodicity conditions)
# ---------------------------------------------------------------------------

def elliptic_integral_general(
    integrand,
    a: float,
    b: float,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
    limit: int = 200,
) -> tuple[float, float]:
    """
    Numerical quadrature for generalized elliptic integrals of the form
    ∫_a^b f(s) ds, possibly with square-root singularities at endpoints.

    Uses scipy.integrate.quad with weight='alg' not assumed — caller must
    handle singularity extraction if needed.

    Returns (value, error_estimate).
    """
    val, err = quad(integrand, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return val, err


def periodicity_integral_theta(
    Q2_func,
    Q_func,
    Q2_tilde_func,
    Z0: float,
    s_lower: float,
    s_upper: float,
) -> tuple[float, float]:
    """
    Rationality condition integral — paper Eq. (82):
    θ/2 = ∫_{s₁⁻}^{s₁⁺} Z₀·Q₂(s) / [Q̃₂(s)·√Q(s)] ds

    Returns (theta_half, error_estimate).
    The caller must verify kθ ∈ 2πℕ.
    """
    def integrand(s):
        Q_val = Q_func(s)
        if Q_val <= 0:
            return 0.0  # at boundary
        Q2_tilde_val = Q2_tilde_func(s)
        if abs(Q2_tilde_val) < 1e-14:
            return 0.0
        return Z0 * Q2_func(s) / (Q2_tilde_val * np.sqrt(Q_val))

    val, err = elliptic_integral_general(integrand, s_lower, s_upper)
    return val, err


def axial_vanishing_integral(
    Q2_func,
    Q_func,
    s_lower: float,
    s_upper: float,
) -> tuple[float, float]:
    """
    Axial vanishing condition — paper Eq. (83):
    ∫_{s₁⁻}^{s₁⁺} Q₂(s)/√Q(s) ds = 0

    Returns (value, error_estimate). Value should be ≈ 0.
    """
    def integrand(s):
        Q_val = Q_func(s)
        if Q_val <= 0:
            return 0.0
        return Q2_func(s) / np.sqrt(Q_val)

    return elliptic_integral_general(integrand, s_lower, s_upper)
