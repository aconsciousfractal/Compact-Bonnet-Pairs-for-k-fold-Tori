"""
Jacobi Theta Functions for Bonnet pair computation.

Convention: Whittaker & Watson, Section 21.11
  ϑ_n(z | τ)  with nome q = exp(iπτ), period lattice {π, τπ}

Backend: mpmath.jtheta for arbitrary-precision computation.
"""
from __future__ import annotations

import numpy as np
import mpmath


# ---------------------------------------------------------------------------
# Nome / modular parameter
# ---------------------------------------------------------------------------

def nome_from_tau(tau: complex) -> complex:
    """Compute the nome  q = exp(iπτ).  Requires Im(τ) > 0."""
    if tau.imag <= 0:
        raise ValueError(f"Im(τ) must be > 0, got {tau.imag}")
    return complex(mpmath.exp(1j * mpmath.pi * tau))


# ---------------------------------------------------------------------------
# Scalar theta functions  (ϑ₁ … ϑ₄)
# ---------------------------------------------------------------------------

def theta1(z: complex, tau: complex, derivative: int = 0) -> complex:
    q = nome_from_tau(tau)
    return complex(mpmath.jtheta(1, z, q, derivative=derivative))

def theta2(z: complex, tau: complex, derivative: int = 0) -> complex:
    q = nome_from_tau(tau)
    return complex(mpmath.jtheta(2, z, q, derivative=derivative))

def theta3(z: complex, tau: complex, derivative: int = 0) -> complex:
    q = nome_from_tau(tau)
    return complex(mpmath.jtheta(3, z, q, derivative=derivative))

def theta4(z: complex, tau: complex, derivative: int = 0) -> complex:
    q = nome_from_tau(tau)
    return complex(mpmath.jtheta(4, z, q, derivative=derivative))


# ---------------------------------------------------------------------------
# Vectorised wrappers
# ---------------------------------------------------------------------------

def theta1_vec(z_array: np.ndarray, tau: complex,
               derivative: int = 0) -> np.ndarray:
    q = nome_from_tau(tau)
    return np.array([complex(mpmath.jtheta(1, z, q, derivative=derivative))
                     for z in z_array.flat]).reshape(z_array.shape)

def theta2_vec(z_array: np.ndarray, tau: complex,
               derivative: int = 0) -> np.ndarray:
    q = nome_from_tau(tau)
    return np.array([complex(mpmath.jtheta(2, z, q, derivative=derivative))
                     for z in z_array.flat]).reshape(z_array.shape)


# ---------------------------------------------------------------------------
# Special values
# ---------------------------------------------------------------------------

def theta1_prime_zero(tau: complex) -> complex:
    """ϑ₁'(0|τ)."""
    return theta1(0, tau, derivative=1)


def find_critical_omega(tau: complex, bracket: tuple[float, float] = (0.01, np.pi / 4 - 0.01)) -> float:
    """
    Find ω ∈ (0, π/4) such that ϑ₂'(ω | τ) = 0.
    This is the critical parameter from the paper Section 5.

    Returns real ω.
    """
    from scipy.optimize import brentq

    def f(omega_real):
        val = theta2(omega_real, tau, derivative=1)
        return val.real  # ϑ₂'(ω) is real for real ω when τ = 1/2 + iR

    return brentq(f, bracket[0], bracket[1])


def R_omega(omega: float, tau: complex) -> complex:
    """
    R(ω) = 2ϑ₂(ω)² / (ϑ₁'(0) · ϑ₁(2ω))   — paper Eq. (42)
    Spherical curve radius. Also = -U(ω)⁻¹.
    """
    th2_om = theta2(omega, tau)
    th1p_0 = theta1_prime_zero(tau)
    th1_2om = theta1(2 * omega, tau)
    return 2 * th2_om**2 / (th1p_0 * th1_2om)


def U_omega(omega: float, tau: complex) -> complex:
    """U(ω) = -(1/2) · ϑ₁'(0) · ϑ₁(2ω) / ϑ₂(ω)²  — paper Eq. (59)"""
    th1p_0 = theta1_prime_zero(tau)
    th1_2om = theta1(2 * omega, tau)
    th2_om = theta2(omega, tau)
    return -0.5 * th1p_0 * th1_2om / th2_om**2


def U_prime_omega(omega: float, tau: complex) -> complex:
    """U'(ω) = -(1/2) · ϑ₁'(0) · ϑ₁'(2ω) / ϑ₂(ω)²  — paper Eq. (59)"""
    th1p_0 = theta1_prime_zero(tau)
    th1p_2om = theta1(2 * omega, tau, derivative=1)
    th2_om = theta2(omega, tau)
    return -0.5 * th1p_0 * th1p_2om / th2_om**2


def U1_prime_omega(omega: float, tau: complex) -> complex:
    """U₁'(ω) = (1/2) · ϑ₁'(0)² / ϑ₂(ω)²  — paper Eq. (60)"""
    th1p_0 = theta1_prime_zero(tau)
    th2_om = theta2(omega, tau)
    return 0.5 * th1p_0**2 / th2_om**2


def U2_omega(omega: float, tau: complex) -> complex:
    """
    U₂(ω) = ϑ₁'(0)²/ϑ₂(ω)² · [ϑ₁(ω)²/ϑ₂(0)² + ϑ₄(ω)²/ϑ₃(0)² + ϑ₃(ω)²/ϑ₄(0)²]
    — paper Eq. (61)
    """
    th1p_0 = theta1_prime_zero(tau)
    th2_om = theta2(omega, tau)
    prefactor = th1p_0**2 / th2_om**2

    th1_om = theta1(omega, tau)
    th2_0 = theta2(0, tau)
    th3_0 = theta3(0, tau)
    th4_0 = theta4(0, tau)
    th3_om = theta3(omega, tau)
    th4_om = theta4(omega, tau)

    bracket = (th1_om / th2_0)**2 + (th4_om / th3_0)**2 + (th3_om / th4_0)**2
    return prefactor * bracket


def Q3_polynomial(s: complex, omega: float, tau: complex) -> complex:
    """
    Q₃(s) = 2U₁'·s³ - U₂·s² - 2U'·s - U²  — paper Eq. (58)
    First elliptic curve in the Bonnet construction.
    """
    U = U_omega(omega, tau)
    Up = U_prime_omega(omega, tau)
    U1p = U1_prime_omega(omega, tau)
    U2 = U2_omega(omega, tau)
    return 2 * U1p * s**3 - U2 * s**2 - 2 * Up * s - U**2


def Q_polynomial(s: complex, s1: float, s2: float, delta: float,
                 omega: float, tau: complex) -> complex:
    """
    Q(s) = -(s-s₁)²(s-s₂)² + δ²Q₃(s)  — paper Eq. (63)
    Second elliptic curve for spherical v-curves.
    """
    return -(s - s1)**2 * (s - s2)**2 + delta**2 * Q3_polynomial(s, omega, tau)


# ---------------------------------------------------------------------------
# Gamma and W1 — the core immersion building blocks (Phase 2 prep)
# ---------------------------------------------------------------------------

def gamma_curve(u: float, w: float, omega: float, tau: complex) -> complex:
    """
    γ(u, w) — family of planar curves, paper Eq. (33).

    γ(u,w) = -i · 2ϑ₂(ω)² · ϑ₁((u+iw-3ω)/2) / [ϑ₁'(0)·ϑ₁(2ω)·ϑ₁((u+iw+ω)/2)]
    """
    th2_om = theta2(omega, tau)
    th1p_0 = theta1_prime_zero(tau)
    th1_2om = theta1(2 * omega, tau)

    z_num = 0.5 * (u + 1j * w - 3 * omega)
    z_den = 0.5 * (u + 1j * w + omega)

    th1_num = theta1(z_num, tau)
    th1_den = theta1(z_den, tau)

    return -1j * 2 * th2_om**2 * th1_num / (th1p_0 * th1_2om * th1_den)


def W1_function(w: float, omega: float, tau: complex) -> complex:
    """
    W₁(w) — cone tangent line direction, paper Eq. (35).

    W₁(w) = i · ϑ₁'(0) · ϑ₂(ω - iw) / [2ϑ₂(ω)·ϑ₁(iw)]
    """
    th1p_0 = theta1_prime_zero(tau)
    th2_om = theta2(omega, tau)
    th2_shifted = theta2(omega - 1j * w, tau)
    th1_iw = theta1(1j * w, tau)

    return 1j * th1p_0 * th2_shifted / (2 * th2_om * th1_iw)


def s_of_w(w: float, omega: float, tau: complex) -> complex:
    """
    s(w) = e^{-h(ω,w)} — Gauss map weight, paper Eq. (65).

    s(w) = ϑ₂(ω)²/ϑ₂(0)² · [ϑ₁(ω)²/ϑ₂(ω)² - ϑ₁(iw/2)²/ϑ₂(iw/2)²]

    Equivalently:
      s(w) = ϑ₁(ω)²/ϑ₂(0)² - ϑ₂(ω)²/ϑ₂(0)² · ϑ₁(iw/2)²/ϑ₂(iw/2)²
    """
    th2_om = theta2(omega, tau)
    th2_0 = theta2(0, tau)
    th1_om = theta1(omega, tau)
    th1_iw2 = theta1(1j * w / 2, tau)
    th2_iw2 = theta2(1j * w / 2, tau)

    return (th2_om / th2_0)**2 * (
        (th1_om / th2_om)**2 - (th1_iw2 / th2_iw2)**2
    )


def gamma_curve_vec(u_array: np.ndarray, w: float, omega: float,
                    tau: complex) -> np.ndarray:
    """
    Vectorized γ(u, w) for an array of u values at fixed w.

    Pre-computes constants once, then batches theta1 calls.
    Returns complex array of same shape as u_array.
    """
    th2_om = theta2(omega, tau)
    th1p_0 = theta1_prime_zero(tau)
    th1_2om = theta1(2 * omega, tau)
    prefactor = -1j * 2 * th2_om**2 / (th1p_0 * th1_2om)

    z_num = 0.5 * (u_array + 1j * w - 3 * omega)
    z_den = 0.5 * (u_array + 1j * w + omega)

    th1_num = theta1_vec(z_num, tau)
    th1_den = theta1_vec(z_den, tau)

    return prefactor * th1_num / th1_den
