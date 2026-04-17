"""
Analytic Derivatives — Phase 16

Exact derivatives of the isothermic torus surface f(u,v) from theta function
formulas, replacing finite differences.

Surface formula (Paper Eq. 32):
    f(u,v) = Φ(v)⁻¹ · γ(u, w(v)) · j · Φ(v)

Gamma curve (Paper Eq. 33):
    γ(u,w) = -i · 2ϑ₂(ω)² · ϑ₁(z₁) / [ϑ₁'(0) · ϑ₁(2ω) · ϑ₁(z₂)]
    z₁ = (u + iw - 3ω)/2,  z₂ = (u + iw + ω)/2

Key identity: ∂γ/∂w = i · ∂γ/∂u  (follows from z₁, z₂ both linear in u + iw).
This gives: γ_w = i·γ_u,  γ_uw = i·γ_uu,  γ_ww = -γ_uu.

Frame ODE (Paper Eq. 34):
    Φ'(v) = A(v) · Φ(v),   A(v) = c(v) · W₁(w(v)) · k
    c(v) = speed(v) = √(1 - w'(v)²)

Surface derivatives via adjoint representation:
    f = Φ⁻¹ · G · Φ           where G = γ · j
    f_u = Φ⁻¹ · G_u · Φ       where G_u = γ_u · j
    f_v = Φ⁻¹ · (G_v + [G, A]) · Φ

Dependencies: theta_functions, quaternion_ops, isothermic_torus, frame_integrator.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

from . import theta_functions as TF
from . import quaternion_ops as Q


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AnalyticDerivatives:
    """Result of analytic derivative computation on the surface grid."""
    f_u: np.ndarray       # (n_u, n_v, 4) — ∂f/∂u  (quaternion)
    f_v: np.ndarray       # (n_u, n_v, 4) — ∂f/∂v  (quaternion)
    f_uu: np.ndarray      # (n_u, n_v, 4) — ∂²f/∂u²
    f_uv: np.ndarray      # (n_u, n_v, 4) — ∂²f/∂u∂v
    f_vv: np.ndarray      # (n_u, n_v, 4) — ∂²f/∂v²
    H: np.ndarray         # (n_u, n_v)    — mean curvature


# ═══════════════════════════════════════════════════════════════════
# Part 1: Gamma curve analytic derivatives
# ═══════════════════════════════════════════════════════════════════

def _psi_psi_prime(z: complex, tau: complex):
    """
    Logarithmic derivative of θ₁ and its derivative.

    ψ(z) = θ₁'(z)/θ₁(z)
    ψ'(z) = θ₁''(z)/θ₁(z) - ψ(z)²

    Returns (ψ, ψ') as complex numbers.
    """
    th1 = TF.theta1(z, tau)
    th1p = TF.theta1(z, tau, derivative=1)
    th1pp = TF.theta1(z, tau, derivative=2)
    psi = th1p / th1
    psi_prime = th1pp / th1 - psi ** 2
    return psi, psi_prime


def gamma_derivatives_vec(u_array: np.ndarray, w: float,
                          omega: float, tau: complex):
    """
    Vectorized analytic derivatives of γ(u, w).

    Returns (γ, γ_u, γ_uu) as complex arrays of same shape as u_array.

    γ_w  = i · γ_u    (use after call if needed)
    γ_uw = i · γ_uu
    γ_ww = -γ_uu
    """
    # Precompute constants
    th2_om = TF.theta2(omega, tau)
    th1p_0 = TF.theta1_prime_zero(tau)
    th1_2om = TF.theta1(2 * omega, tau)
    prefactor = -1j * 2 * th2_om ** 2 / (th1p_0 * th1_2om)

    z_num = 0.5 * (u_array + 1j * w - 3 * omega)
    z_den = 0.5 * (u_array + 1j * w + omega)

    n = u_array.size
    gamma_vals = np.empty(n, dtype=complex)
    gamma_u_vals = np.empty(n, dtype=complex)
    gamma_uu_vals = np.empty(n, dtype=complex)

    q = TF.nome_from_tau(tau)

    for idx in range(n):
        z1, z2 = complex(z_num.flat[idx]), complex(z_den.flat[idx])

        import mpmath
        th1_z1 = complex(mpmath.jtheta(1, z1, q))
        th1p_z1 = complex(mpmath.jtheta(1, z1, q, derivative=1))
        th1pp_z1 = complex(mpmath.jtheta(1, z1, q, derivative=2))
        th1_z2 = complex(mpmath.jtheta(1, z2, q))
        th1p_z2 = complex(mpmath.jtheta(1, z2, q, derivative=1))
        th1pp_z2 = complex(mpmath.jtheta(1, z2, q, derivative=2))

        g = complex(prefactor) * th1_z1 / th1_z2

        psi1 = th1p_z1 / th1_z1
        psi2 = th1p_z2 / th1_z2
        psip1 = th1pp_z1 / th1_z1 - psi1 ** 2
        psip2 = th1pp_z2 / th1_z2 - psi2 ** 2

        L_u = 0.5 * (psi1 - psi2)
        L_uu = 0.25 * (psip1 - psip2)

        gamma_vals[idx] = g
        gamma_u_vals[idx] = g * L_u
        gamma_uu_vals[idx] = g * (L_u ** 2 + L_uu)

    return (gamma_vals.reshape(u_array.shape),
            gamma_u_vals.reshape(u_array.shape),
            gamma_uu_vals.reshape(u_array.shape))


# ═══════════════════════════════════════════════════════════════════
# Part 2: W₁ derivative
# ═══════════════════════════════════════════════════════════════════

def W1_and_dW1(w: float, omega: float, tau: complex):
    """
    W₁(w) and dW₁/dw.

    W₁(w) = i · θ₁'(0) · θ₂(ω-iw) / [2θ₂(ω) · θ₁(iw)]

    dW₁/dw = W₁ · (-i) · [θ₂'(ω-iw)/θ₂(ω-iw) + θ₁'(iw)/θ₁(iw)]

    Returns (W1, dW1_dw) as complex numbers.
    """
    W1 = TF.W1_function(w, omega, tau)

    th2_shifted = TF.theta2(omega - 1j * w, tau)
    th2p_shifted = TF.theta2(omega - 1j * w, tau, derivative=1)
    th1_iw = TF.theta1(1j * w, tau)
    th1p_iw = TF.theta1(1j * w, tau, derivative=1)

    log_deriv = th2p_shifted / th2_shifted + th1p_iw / th1_iw
    dW1_dw = W1 * (-1j) * log_deriv

    return W1, dW1_dw


# ═══════════════════════════════════════════════════════════════════
# Part 3: Quaternion helpers
# ═══════════════════════════════════════════════════════════════════

def _complex_to_j_quat(c: complex) -> np.ndarray:
    """Map complex c to quaternion c·j = [0, 0, Re(c), Im(c)]."""
    return np.array([0.0, 0.0, c.real, c.imag])


def _adjoint(X: np.ndarray, phi: np.ndarray, phi_inv: np.ndarray) -> np.ndarray:
    """Φ⁻¹ · X · Φ (adjoint action)."""
    return Q.qmul(Q.qmul(phi_inv, X), phi)


def _commutator(G: np.ndarray, A: np.ndarray) -> np.ndarray:
    """[G, A] = G·A - A·G."""
    return Q.qmul(G, A) - Q.qmul(A, G)


# ═══════════════════════════════════════════════════════════════════
# Part 4: Full surface derivative computation
# ═══════════════════════════════════════════════════════════════════

def compute_analytic_derivatives(torus_result, verbose: bool = False):
    """
    Compute all analytic derivatives of f(u,v) and mean curvature H.

    Parameters
    ----------
    torus_result : TorusResult
        Output of compute_torus().
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnalyticDerivatives
    """
    params = torus_result.params
    omega = torus_result.omega
    tau = params.tau
    u_grid = torus_result.u_grid
    v_grid = torus_result.v_grid
    n_u = len(u_grid)
    n_v = len(v_grid)

    # Recover callable functions
    w_func = params.w_func
    w_prime_func = params.w_prime_func
    speed_func = params.speed_func

    if w_func is None:
        w_func = lambda v: params.w0
        w_prime_func = lambda v: 0.0

    # Build Φ(v) interpolator from frame result
    fr = torus_result.frame_result
    phi_interp = CubicSpline(fr.v_values, fr.phi_values, axis=0)

    # Precompute j and k quaternions
    j_quat = Q.quat_j()
    k_quat = Q.quat_k()

    # Allocate output
    f_u = np.zeros((n_u, n_v, 4))
    f_v = np.zeros((n_u, n_v, 4))
    f_uu = np.zeros((n_u, n_v, 4))
    f_uv = np.zeros((n_u, n_v, 4))
    f_vv = np.zeros((n_u, n_v, 4))

    # ── Loop over v slices ──
    for iv in range(n_v):
        v = v_grid[iv]
        w_val = w_func(v)
        wp_val = w_prime_func(v)

        if speed_func is not None:
            c_val = speed_func(v)
        else:
            c_val = np.sqrt(max(1.0 - wp_val ** 2, 0.0))

        # Frame at this v
        phi_v = phi_interp(v)
        phi_v = phi_v / np.linalg.norm(phi_v)
        phi_inv = Q.qconj(phi_v)

        # W₁ and dW₁/dw
        W1_val, dW1_dw_val = W1_and_dW1(w_val, omega, tau)
        W1_quat = np.array([W1_val.real, W1_val.imag, 0.0, 0.0])
        W1k = Q.qmul(W1_quat, k_quat)

        dW1_quat = np.array([dW1_dw_val.real, dW1_dw_val.imag, 0.0, 0.0])
        dW1k = Q.qmul(dW1_quat, k_quat)

        # A = c · W₁k (frame ODE coefficient)
        A_val = c_val * W1k

        # w''(v) via central difference of w'
        dv_num = 1e-6
        wpp_val = (w_prime_func(v + dv_num) - w_prime_func(v - dv_num)) / (2 * dv_num)

        # speed'(v) = dc/dv
        if speed_func is not None:
            dc_dv = (speed_func(v + dv_num) - speed_func(v - dv_num)) / (2 * dv_num)
        else:
            dc_dv = -wp_val * wpp_val / max(c_val, 1e-30)

        # dA/dv = dc/dv · W₁k + c · w' · dW₁k
        dA_dv = dc_dv * W1k + c_val * wp_val * dW1k

        # ── Vectorized gamma derivatives over u ──
        gamma_vals, gamma_u_vals, gamma_uu_vals = gamma_derivatives_vec(
            u_grid, w_val, omega, tau)

        if verbose and iv % max(1, n_v // 10) == 0:
            print(f"  v[{iv}/{n_v}] w={w_val:.4f} |γ₀|={abs(gamma_vals[0]):.6f}")

        # ── Inner loop over u ──
        for iu in range(n_u):
            g = gamma_vals[iu]
            g_u = gamma_u_vals[iu]
            g_uu = gamma_uu_vals[iu]

            # Quaternion forms: X · j for complex X
            G = _complex_to_j_quat(g)          # γ · j
            G_u = _complex_to_j_quat(g_u)      # γ_u · j
            G_uu = _complex_to_j_quat(g_uu)    # γ_uu · j

            # γ_w = i·γ_u → (i·γ_u)·j
            g_w = 1j * g_u
            G_w = _complex_to_j_quat(g_w)

            # γ_uw = i·γ_uu → (i·γ_uu)·j
            g_uw = 1j * g_uu
            G_uw = _complex_to_j_quat(g_uw)

            # γ_ww = -γ_uu → (-γ_uu)·j
            G_ww = _complex_to_j_quat(-g_uu)

            # ────────────────────────────────────
            # f_u = Φ⁻¹ · G_u · Φ
            # ────────────────────────────────────
            f_u[iu, iv] = _adjoint(G_u, phi_v, phi_inv)

            # ────────────────────────────────────
            # f_v = Φ⁻¹ · M_v · Φ
            # M_v = w' · G_w + c · [G, W₁k]
            # ────────────────────────────────────
            M_v = wp_val * G_w + c_val * _commutator(G, W1k)
            f_v[iu, iv] = _adjoint(M_v, phi_v, phi_inv)

            # ────────────────────────────────────
            # f_uu = Φ⁻¹ · G_uu · Φ
            # ────────────────────────────────────
            f_uu[iu, iv] = _adjoint(G_uu, phi_v, phi_inv)

            # ────────────────────────────────────
            # f_uv = Φ⁻¹ · (w'·G_uw + c·[G_u, W₁k]) · Φ
            # ────────────────────────────────────
            M_uv = wp_val * G_uw + c_val * _commutator(G_u, W1k)
            f_uv[iu, iv] = _adjoint(M_uv, phi_v, phi_inv)

            # ────────────────────────────────────
            # f_vv = Φ⁻¹ · (dM_v/dv + [M_v, A]) · Φ
            # ────────────────────────────────────
            # dM_v/dv has contributions from:
            # (a) d/dv[w' · G_w] = w'' · G_w + w' · w' · G_ww
            #     (since dG_w/dv = w' · (∂/∂w)(γ_w·j) = w' · γ_ww · j = w'·G_ww)
            # (b) d/dv[c · (G·W₁k - W₁k·G)]
            #     = dc/dv · (G·W₁k - W₁k·G)
            #     + c · [dG/dv · W₁k + G · d(W₁k)/dv - d(W₁k)/dv · G - W₁k · dG/dv]
            #     = dc/dv · [G, W₁k]
            #     + c · [w'·G_w·W₁k + w'·G·dW₁k - w'·dW₁k·G - W₁k·w'·G_w]
            #     = dc/dv · [G, W₁k]
            #     + c·w' · ([G_w, W₁k] + [G, dW₁k])

            dW1k_scaled = wp_val * dW1k  # w' · dW₁/dw · k

            dMv_dv = (
                wpp_val * G_w
                + wp_val * wp_val * G_ww
                + dc_dv * _commutator(G, W1k)
                + c_val * wp_val * (_commutator(G_w, W1k) + _commutator(G, dW1k))
            )

            M_vv = dMv_dv + _commutator(M_v, A_val)
            f_vv[iu, iv] = _adjoint(M_vv, phi_v, phi_inv)

    # ── Mean curvature ──
    H = _compute_mean_curvature(f_u, f_v, f_uu, f_uv, f_vv)

    return AnalyticDerivatives(
        f_u=f_u, f_v=f_v,
        f_uu=f_uu, f_uv=f_uv, f_vv=f_vv,
        H=H,
    )


# ═══════════════════════════════════════════════════════════════════
# Part 5: Mean curvature from analytic derivatives
# ═══════════════════════════════════════════════════════════════════

def _compute_mean_curvature(f_u, f_v, f_uu, f_uv, f_vv):
    """
    H = (eG − 2fF + gE) / (2(EG − F²))

    f_u, f_v, etc. are quaternion arrays (n_u, n_v, 4).
    The ℝ³ coordinates are in slots 1:4.
    """
    n_u, n_v = f_u.shape[:2]

    # Extract ℝ³ parts
    fu3 = f_u[:, :, 1:4]
    fv3 = f_v[:, :, 1:4]
    fuu3 = f_uu[:, :, 1:4]
    fuv3 = f_uv[:, :, 1:4]
    fvv3 = f_vv[:, :, 1:4]

    # First fundamental form
    E = np.sum(fu3 ** 2, axis=2)
    F = np.sum(fu3 * fv3, axis=2)
    G = np.sum(fv3 ** 2, axis=2)

    # Unit normal n = f_u × f_v / |f_u × f_v|
    n_vec = np.cross(fu3, fv3)
    n_norm = np.linalg.norm(n_vec, axis=2, keepdims=True)
    n_vec = n_vec / np.maximum(n_norm, 1e-30)

    # Second fundamental form
    e = np.sum(fuu3 * n_vec, axis=2)
    f_coeff = np.sum(fuv3 * n_vec, axis=2)
    g = np.sum(fvv3 * n_vec, axis=2)

    # Mean curvature
    denom = 2.0 * (E * G - F ** 2)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    H = (e * G - 2.0 * f_coeff * F + g * E) / denom

    return H


# ═══════════════════════════════════════════════════════════════════
# Part 6: Stereographic derivatives  f → x ∈ S³
# ═══════════════════════════════════════════════════════════════════

def inverse_stereographic_with_derivatives(f_grid, f_u_grid, f_v_grid):
    """
    Lift f ∈ Im(ℍ) to x ∈ S³ and compute analytic x_u, x_v.

    σ⁻¹(f): x₀ = (r²−1)/(r²+1),  x_im = 2f/(r²+1),  r² = |f|²

    Chain rule:
        ∂x₀/∂p = 4(f·f_p) / (r²+1)²
        ∂x_im/∂p = 2[(r²+1)f_p − 2(f·f_p)f] / (r²+1)²

    Parameters
    ----------
    f_grid : (n_u, n_v, 4) — quaternion surface, slot 0 ≈ 0
    f_u_grid, f_v_grid : (n_u, n_v, 4) — analytic derivatives

    Returns
    -------
    x_grid, x_u, x_v : (n_u, n_v, 4) — surface and derivatives on S³
    """
    f_im = f_grid[..., 1:4]                     # (n_u, n_v, 3)
    fu_im = f_u_grid[..., 1:4]
    fv_im = f_v_grid[..., 1:4]

    r_sq = np.sum(f_im ** 2, axis=-1)            # (n_u, n_v)
    denom = r_sq + 1.0
    denom_sq = denom ** 2

    # x on S³
    x_grid = np.empty_like(f_grid)
    x_grid[..., 0] = (r_sq - 1.0) / denom
    x_grid[..., 1:4] = 2.0 * f_im / denom[..., np.newaxis]

    # f · f_u  (ℝ³ dot product)
    f_dot_fu = np.sum(f_im * fu_im, axis=-1)     # (n_u, n_v)
    f_dot_fv = np.sum(f_im * fv_im, axis=-1)

    # x_u
    x_u = np.empty_like(f_grid)
    x_u[..., 0] = 4.0 * f_dot_fu / denom_sq
    x_u[..., 1:4] = 2.0 * (
        denom[..., np.newaxis] * fu_im
        - 2.0 * f_dot_fu[..., np.newaxis] * f_im
    ) / denom_sq[..., np.newaxis]

    # x_v
    x_v = np.empty_like(f_grid)
    x_v[..., 0] = 4.0 * f_dot_fv / denom_sq
    x_v[..., 1:4] = 2.0 * (
        denom[..., np.newaxis] * fv_im
        - 2.0 * f_dot_fv[..., np.newaxis] * f_im
    ) / denom_sq[..., np.newaxis]

    return x_grid, x_u, x_v


# ═══════════════════════════════════════════════════════════════════
# Part 7: Retraction form with analytic derivatives
# ═══════════════════════════════════════════════════════════════════

def compute_analytic_retraction_form(torus_result, ad=None, verbose=True):
    """
    Full retraction form pipeline using analytic derivatives.

    f → analytic f_u, f_v → stereographic chain rule → x, x_u, x_v → ω → F±

    Parameters
    ----------
    torus_result : TorusResult
    ad : AnalyticDerivatives (optional, computed if None)
    verbose : print diagnostics

    Returns
    -------
    dict with x_grid, x_u, x_v, omega_u, omega_v, closure_error,
    cross_error, conformality_error, F_plus, F_minus, exactness_error
    """
    from .retraction_form import (
        compute_retraction_omega,
        verify_closure,
        verify_cross_condition,
        integrate_bonnet_pair,
    )

    if ad is None:
        if verbose:
            print("[Phase 16] Computing analytic derivatives...")
        ad = compute_analytic_derivatives(torus_result, verbose=verbose)

    f_grid = torus_result.f_grid
    n_u, n_v = f_grid.shape[:2]
    u_grid = torus_result.u_grid
    v_values = torus_result.frame_result.v_values
    du = u_grid[1] - u_grid[0]
    dv = (v_values[-1] - v_values[0]) / (len(v_values) - 1) if len(v_values) > 1 else 1.0

    if verbose:
        print(f"[Phase 16] Retraction form (analytic): grid {n_u}×{n_v}")

    # Step 1: analytic stereographic lift + derivatives
    x_grid, x_u, x_v = inverse_stereographic_with_derivatives(
        f_grid, ad.f_u, ad.f_v)

    unit_err = float(np.max(np.abs(
        np.sqrt(np.sum(x_grid ** 2, axis=-1)) - 1.0)))

    # Step 2: x ⊥ x_u, x ⊥ x_v check (tangency on S³)
    x_dot_xu = np.abs(np.sum(x_grid * x_u, axis=-1))
    x_dot_xv = np.abs(np.sum(x_grid * x_v, axis=-1))
    tangent_err = max(float(np.max(x_dot_xu)), float(np.max(x_dot_xv)))

    # Step 3: conformality
    e2u_u = np.sum(x_u ** 2, axis=-1)
    e2u_v = np.sum(x_v ** 2, axis=-1)
    conf_err = float(np.max(np.abs(e2u_u - e2u_v))
                     / max(float(np.max(np.maximum(e2u_u, e2u_v))), 1e-30))

    if verbose:
        print(f"  S³ lift: max ||x|−1| = {unit_err:.2e}")
        print(f"  Tangency: max |⟨x,x_p⟩| = {tangent_err:.2e}")
        print(f"  Conformality: max ||x_u|²−|x_v|²|/max = {conf_err:.2e}")

    # Step 4: retraction form
    omega_u, omega_v, e2u = compute_retraction_omega(x_u, x_v)

    # Step 5: x ⊥ ω check
    N = n_u * n_v
    flat_x = x_grid.reshape(N, 4)
    flat_ou = omega_u.reshape(N, 4)
    flat_ov = omega_v.reshape(N, 4)
    dot_u = np.abs(np.sum(flat_x * flat_ou, axis=1))
    dot_v = np.abs(np.sum(flat_x * flat_ov, axis=1))
    orth_err = max(float(np.max(dot_u)), float(np.max(dot_v)))

    # Step 6: closure dω = 0  (still uses FD on ω — this is inherent
    #          to discrete verification, not the derivative computation)
    closure_err = verify_closure(omega_u, omega_v, du, dv)

    # Step 7: cross condition
    cross_err = verify_cross_condition(omega_u, omega_v, x_u, x_v)

    if verbose:
        print(f"  x⊥ω: max |⟨x,ω⟩| = {orth_err:.2e}")
        print(f"  Closure dω=0: {closure_err:.2e}")
        print(f"  Cross ω̄∧dx=0: {cross_err:.2e}")

    # Step 8: integrate F±
    F_plus, F_minus, exact_err = integrate_bonnet_pair(
        x_grid, omega_u, omega_v, du, dv)

    if verbose:
        print(f"  Exactness: {exact_err:.2e}")
        print(f"  F⁺ max |scalar|: {float(np.max(np.abs(F_plus[..., 0]))):.2e}")
        print(f"  F⁻ max |scalar|: {float(np.max(np.abs(F_minus[..., 0]))):.2e}")

    return {
        'x_grid': x_grid,
        'x_u': x_u,
        'x_v': x_v,
        'omega_u': omega_u,
        'omega_v': omega_v,
        'F_plus': F_plus,
        'F_minus': F_minus,
        'closure_error': closure_err,
        'cross_error': cross_err,
        'conformality_error': conf_err,
        'tangency_error': tangent_err,
        'exactness_error': exact_err,
        'unit_error': unit_err,
        'orthogonality_error': orth_err,
        'analytic_derivatives': ad,
    }
