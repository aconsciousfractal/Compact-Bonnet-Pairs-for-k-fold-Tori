"""
Frame Integrator — Bonnet's Problem Phase 1.5

ODE integrator for the quaternionic frame Φ(v) ∈ S³ ⊂ ℍ₁.

Paper Eq. (34):
  Φ'(v)·Φ⁻¹(v) = √(1 - w'(v)²) · W₁(w(v)) · k

This gives the ODE system:
  Φ'(v) = √(1 - w'(v)²) · W₁(w(v)) · k · Φ(v)

where Φ(v) ∈ ℍ₁ (unit quaternion), W₁ is from theta_functions.py,
and k = [0,0,0,1] is the unit quaternion k.

Key properties to preserve:
  1. |Φ(v)| = 1 for all v  (unitarity)
  2. Monodromy: Φ(V) = Φ(0)·exp(θ·k/2)  for some angle θ
  3. Closure: kθ ∈ 2πℕ for k-fold symmetry

Uses scipy.integrate.solve_ivp with DOP853 (8th-order Runge-Kutta)
and periodic renormalization to maintain |Φ| = 1.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable

from . import quaternion_ops as Q
from . import theta_functions as TF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Result of frame integration."""
    v_values: np.ndarray          # shape (N,) — independent variable
    phi_values: np.ndarray        # shape (N, 4) — quaternion frame at each v
    unitarity_errors: np.ndarray  # shape (N,) — ||Φ(v)| - 1|
    monodromy: np.ndarray         # shape (4,) — Φ(V)·Φ(0)⁻¹
    rotation_angle: float         # θ extracted from monodromy
    n_renormalizations: int       # how many times we renormalized


# ---------------------------------------------------------------------------
# Core ODE system
# ---------------------------------------------------------------------------

def frame_ode_rhs(
    v: float,
    phi_flat: np.ndarray,
    w_func: Callable[[float], float],
    w_prime_func: Callable[[float], float],
    speed_func: Callable[[float], float] | None,
    omega: float,
    tau: complex,
) -> np.ndarray:
    """
    Right-hand side of the frame ODE: Φ' = √(1 - w'²) · W₁(w) · k · Φ

    Parameters
    ----------
    v : float
        Independent variable
    phi_flat : ndarray shape (4,)
        Current quaternion frame [q₀, q₁, q₂, q₃]
    w_func : callable
        Reparametrization w(v)
    w_prime_func : callable
        Derivative w'(v)
    omega : float
        Critical parameter
    tau : complex
        Period ratio
    """
    phi = phi_flat

    w_val = w_func(v)
    wp_val = w_prime_func(v)

    # Signed branch when available; otherwise use the principal square root.
    if speed_func is not None:
        speed = speed_func(v)
    else:
        speed = np.sqrt(max(1.0 - wp_val**2, 0.0))

    # W₁(w) is a complex number from theta functions
    W1_val = TF.W1_function(w_val, omega, tau)

    # Convert W₁ · k to a quaternion
    # W₁ = a + bi (complex) → quaternion: a + b·i (unit imaginary i)
    # W₁ · k: quaternion multiplication of (a + b·i) · k
    # (a + bi)·k = a·k + b·(ik) = a·k + b·(-j) ... wait
    # In quaternion: i·k = -j, so (a + b·i)·k = a·k - b·j
    # As quaternion array: [0, 0, -b, a] ... let me be precise.
    #
    # W₁·k where W₁ = W1_re + W1_im·i (as quaternion: [W1_re, W1_im, 0, 0])
    # and k = [0, 0, 0, 1]
    # Product: see qmul
    W1_quat = np.array([W1_val.real, W1_val.imag, 0.0, 0.0])
    k_quat = Q.quat_k()
    W1k = Q.qmul(W1_quat, k_quat)

    # Φ' = speed · W1k · Φ
    dphi = speed * Q.qmul(W1k, phi)

    return dphi


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_frame(
    w_func: Callable[[float], float],
    w_prime_func: Callable[[float], float],
    omega: float,
    tau: complex,
    v_span: tuple[float, float],
    phi0: np.ndarray | None = None,
    speed_func: Callable[[float], float] | None = None,
    n_points: int = 500,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    renorm_interval: int = 50,
) -> FrameResult:
    """
    Integrate the frame ODE Φ'(v) = √(1-w'²)·W₁(w)·k·Φ(v).

    Parameters
    ----------
    w_func : callable
        Reparametrization function w(v)
    w_prime_func : callable
        Derivative w'(v)
    omega : float
        Critical parameter (ϑ₂'(ω) = 0)
    tau : complex
        Period ratio (τ = 1/2 + iR)
    v_span : (v_start, v_end)
        Integration domain
    phi0 : ndarray (4,), optional
        Initial condition. Default: [1, 0, 0, 0] (identity)
    n_points : int
        Number of output points
    rtol, atol : float
        Integrator tolerances
    renorm_interval : int
        Renormalize to S³ every this many steps

    Returns
    -------
    FrameResult
        Contains v_values, phi_values, unitarity errors, monodromy
    """
    if phi0 is None:
        phi0 = np.array([1.0, 0.0, 0.0, 0.0])

    v_eval = np.linspace(v_span[0], v_span[1], n_points)

    # Event function for renormalization: we handle it post-hoc
    sol = solve_ivp(
        fun=lambda v, y: frame_ode_rhs(v, y, w_func, w_prime_func, speed_func, omega, tau),
        t_span=v_span,
        y0=phi0,
        method='DOP853',
        t_eval=v_eval,
        rtol=rtol,
        atol=atol,
        max_step=(v_span[1] - v_span[0]) / n_points,
    )

    if not sol.success:
        raise RuntimeError(f"Frame ODE integration failed: {sol.message}")

    phi_values = sol.y.T  # shape (n_points, 4)

    # Post-hoc renormalization
    n_renorm = 0
    for i in range(len(phi_values)):
        norm_i = np.linalg.norm(phi_values[i])
        if abs(norm_i - 1.0) > 1e-10:
            phi_values[i] /= norm_i
            n_renorm += 1

    # Compute unitarity errors
    norms = np.linalg.norm(phi_values, axis=1)
    unitarity_errors = np.abs(norms - 1.0)

    # Compute monodromy: Φ(V)·Φ(0)⁻¹
    phi_end = phi_values[-1]
    phi_start = phi_values[0]
    monodromy = Q.qmul(phi_end, Q.qinv(phi_start))

    # Extract rotation angle from monodromy
    # monodromy = cos(θ/2) + sin(θ/2)·n̂
    cos_half = np.clip(monodromy[0], -1.0, 1.0)
    rotation_angle = 2.0 * np.arccos(abs(cos_half))

    return FrameResult(
        v_values=sol.t,
        phi_values=phi_values,
        unitarity_errors=unitarity_errors,
        monodromy=monodromy,
        rotation_angle=rotation_angle,
        n_renormalizations=n_renorm,
    )


# ---------------------------------------------------------------------------
# Auxiliary: B̃(v) integrator (Appendix A, Eq. 122-123)
# ---------------------------------------------------------------------------

@dataclass
class AxialBResult:
    """Result of B̃(v) integration."""
    v_values: np.ndarray       # shape (N,)
    B_tilde: np.ndarray        # shape (N, 4) — quaternion values


def b_tilde_ode_rhs(
    v: float,
    B_flat: np.ndarray,
    phi_interp: Callable[[float], np.ndarray],
    w_func: Callable[[float], float],
    w_prime_func: Callable[[float], float],
    speed_func: Callable[[float], float] | None,
    b_tilde_func: Callable[[float], complex],
    omega: float,
    tau: complex,
) -> np.ndarray:
    """
    RHS of B̃'(v) = Φ⁻¹(v) · R(ω) · √(1-w'²) · b̃(w) · k · Φ(v)

    Parameters
    ----------
    B_flat : quaternion [4,]
    phi_interp : v → Φ(v) quaternion
    b_tilde_func : w → b̃(w) complex scalar (from theta functions)
    """
    phi_v = phi_interp(v)
    w_val = w_func(v)
    wp_val = w_prime_func(v)

    if speed_func is not None:
        speed = speed_func(v)
    else:
        speed = np.sqrt(max(1.0 - wp_val**2, 0.0))
    R_om = TF.R_omega(omega, tau)

    b_val = b_tilde_func(w_val)

    # Construct: R(ω) · b̃(w) · k as quaternion
    # R(ω) is complex, b̃(w) is complex, their product is complex
    scalar = complex(R_om) * b_val
    rbk_quat = Q.qmul(
        np.array([scalar.real, scalar.imag, 0.0, 0.0]),
        Q.quat_k()
    )

    # Φ⁻¹ · (speed · rbk) · Φ
    phi_inv = Q.qinv(phi_v)
    inner = Q.qmul(Q.qmul(phi_inv, speed * rbk_quat), phi_v)

    return inner


def integrate_B_tilde(
    phi_interp: Callable[[float], np.ndarray],
    w_func: Callable[[float], float],
    w_prime_func: Callable[[float], float],
    b_tilde_func: Callable[[float], complex],
    omega: float,
    tau: complex,
    v_span: tuple[float, float],
    speed_func: Callable[[float], float] | None = None,
    n_points: int = 500,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> AxialBResult:
    """
    Integrate the axial B̃(v) ODE — paper Eq. (122-123).

    B̃(0) = 0 (zero quaternion initial condition).
    """
    B0 = np.zeros(4)
    v_eval = np.linspace(v_span[0], v_span[1], n_points)

    sol = solve_ivp(
        fun=lambda v, y: b_tilde_ode_rhs(
            v, y, phi_interp, w_func, w_prime_func, speed_func, b_tilde_func, omega, tau
        ),
        t_span=v_span,
        y0=B0,
        method='DOP853',
        t_eval=v_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"B̃ ODE integration failed: {sol.message}")

    return AxialBResult(
        v_values=sol.t,
        B_tilde=sol.y.T,
    )


# ---------------------------------------------------------------------------
# Simple reparametrizations for testing
# ---------------------------------------------------------------------------

def constant_w(w0: float) -> tuple[Callable, Callable]:
    """Constant reparametrization w(v) = w₀ (simplest case for testing)."""
    return lambda v: w0, lambda v: 0.0


def linear_w(w0: float, slope: float) -> tuple[Callable, Callable]:
    """Linear reparametrization w(v) = w₀ + slope·v."""
    return lambda v: w0 + slope * v, lambda v: slope


def sinusoidal_w(A: float, B: float, C: float) -> tuple[Callable, Callable]:
    """
    Paper 2-fold example reparametrization (simplified):
    w(v) = C + A·sin(v) + B·sin(2v)/2

    Full paper version: w(v) = C + A sin(v)/π - A cos(v)/π² + B sin(2v)/(2π) - B cos(2v)/(4π²)
    """
    def w(v):
        return C + A * np.sin(v) + B * np.sin(2 * v) / 2

    def wp(v):
        return A * np.cos(v) + B * np.cos(2 * v)

    return w, wp
