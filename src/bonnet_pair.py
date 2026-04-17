"""
Bonnet Pair Engine — Bonnet's Problem Phase 3

Implements the Bonnet pair formula (Theorem 5, Eq. 49):

  f±(u,v) = R(ω)² f(π-2ω+u,v) - ε² f(π-u,v)
            ± 2ε ( Φ⁻¹(v) B̂(u,w(v)) i Φ(v) + B̃(v) )

where:
  - f(u,v) is the isothermic torus (Phase 2)
  - R(ω) = 2ϑ₂(ω)² / (ϑ₁'(0)·ϑ₁(2ω))  (eq. 42)
  - B̂(u,w) is the planar component (eq. 121, Theorem 10)
  - B̃(v) is the axial component (eq. 122-123, Appendix A)
  - ε is the Bonnet parameter

The pair f⁺, f⁻ are isometric with equal mean curvature but non-congruent.

Components:
  1. B_hat_function     — B̂(u,w) via eq. (121)
  2. b_tilde_scalar     — b̃(w) complex scalar (eq. 123)
  3. compute_bonnet_pair — full pipeline f⁺, f⁻
  4. verify_isometry     — metric tensor comparison (Task 3.5)
  5. verify_mean_curvature — H⁺ vs H⁻ (Task 3.6)
  6. verify_non_congruence — Procrustes (Task 3.7)
  7. export_bonnet_pair_obj — dual OBJ export (Task 3.8)
  8. closure_gate        — rationality + axial vanishing (Task 3.9)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from scipy.interpolate import CubicSpline
from scipy.spatial import procrustes as scipy_procrustes

from . import theta_functions as TF
from . import quaternion_ops as Q
from .frame_integrator import integrate_frame, integrate_B_tilde, FrameResult, AxialBResult
from .isothermic_torus import (
    TorusParameters, TorusResult, compute_torus,
    build_torus_faces, compute_vertex_normals, verify_euler_characteristic,
    export_torus_obj, _write_obj_standalone,
)


# ---------------------------------------------------------------------------
# Discrete differential geometry
# ---------------------------------------------------------------------------

def compute_metric_tensor_periodic_central(f_grid, u_grid, v_grid):
    """First fundamental form via central periodic differences."""
    n_u, n_v = f_grid.shape[:2]
    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0
    dv = v_grid[1] - v_grid[0] if len(v_grid) > 1 else 1.0
    f_r3 = f_grid[:, :, 1:4]
    f_u = np.zeros_like(f_r3)
    f_v = np.zeros_like(f_r3)
    for iu in range(n_u):
        ip = (iu + 1) % n_u
        im = (iu - 1) % n_u
        f_u[iu] = (f_r3[ip] - f_r3[im]) / (2 * du)
    for iv in range(n_v):
        jp = (iv + 1) % n_v
        jm = (iv - 1) % n_v
        f_v[:, iv] = (f_r3[:, jp] - f_r3[:, jm]) / (2 * dv)
    EE = np.sum(f_u**2, axis=2)
    F = np.sum(f_u * f_v, axis=2)
    G = np.sum(f_v**2, axis=2)
    return {'E': EE, 'F': F, 'G': G, 'f_u': f_u, 'f_v': f_v}

# Convenience alias
compute_metric_tensor = compute_metric_tensor_periodic_central


def compute_mean_curvature_fd(f_grid, u_grid, v_grid):
    """Mean curvature H via finite differences."""
    n_u, n_v = f_grid.shape[:2]
    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0
    dv = v_grid[1] - v_grid[0] if len(v_grid) > 1 else 1.0
    f_r3 = f_grid[:, :, 1:4]
    f_u = np.zeros_like(f_r3)
    f_v = np.zeros_like(f_r3)
    for iu in range(n_u):
        f_u[iu] = (f_r3[(iu + 1) % n_u] - f_r3[(iu - 1) % n_u]) / (2 * du)
    for iv in range(n_v):
        f_v[:, iv] = (f_r3[:, (iv + 1) % n_v] - f_r3[:, (iv - 1) % n_v]) / (2 * dv)
    f_uu = np.zeros_like(f_r3)
    f_vv = np.zeros_like(f_r3)
    f_uv = np.zeros_like(f_r3)
    for iu in range(n_u):
        ip, im = (iu + 1) % n_u, (iu - 1) % n_u
        f_uu[iu] = (f_r3[ip] - 2 * f_r3[iu] + f_r3[im]) / du**2
    for iv in range(n_v):
        jp, jm = (iv + 1) % n_v, (iv - 1) % n_v
        f_vv[:, iv] = (f_r3[:, jp] - 2 * f_r3[:, iv] + f_r3[:, jm]) / dv**2
    for iu in range(n_u):
        ip, im = (iu + 1) % n_u, (iu - 1) % n_u
        for iv in range(n_v):
            jp, jm = (iv + 1) % n_v, (iv - 1) % n_v
            f_uv[iu, iv] = (
                f_r3[ip, jp] - f_r3[ip, jm] - f_r3[im, jp] + f_r3[im, jm]
            ) / (4 * du * dv)
    n_vec = np.cross(f_u, f_v)
    n_norm = np.linalg.norm(n_vec, axis=2, keepdims=True)
    n_vec = n_vec / np.maximum(n_norm, 1e-30)
    EE = np.sum(f_u**2, axis=2)
    F = np.sum(f_u * f_v, axis=2)
    G = np.sum(f_v**2, axis=2)
    e = np.sum(f_uu * n_vec, axis=2)
    f_coeff = np.sum(f_uv * n_vec, axis=2)
    g = np.sum(f_vv * n_vec, axis=2)
    denom = 2 * (EE * G - F**2)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return (e * G - 2 * f_coeff * F + g * EE) / denom


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BonnetPairResult:
    """Result of Bonnet pair generation."""
    f_plus: TorusResult           # f⁺ surface
    f_minus: TorusResult          # f⁻ surface
    f_base: TorusResult           # base isothermic torus f
    epsilon: float                # Bonnet parameter ε
    R_omega: complex              # R(ω) from eq. (42)
    B_hat_grid: np.ndarray        # (N_u, N_v) B̂ values
    B_tilde_result: AxialBResult  # B̃(v) integration
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# B̂(u,w) — Planar component (Theorem 10, Eq. 121)
# ---------------------------------------------------------------------------

def B_hat_function(u: float, w: float, omega: float, tau: complex) -> float:
    """
    B̂(u, w) — planar component of the Bonnet pair.

    Paper Eq. (121):
      B̂(u,w) = R(ω) · (2ϑ₁(2ω)/ϑ₁'(0)) ·
                ( ϑ₂''(ω)/ϑ₂(ω) · w/2
                  - Im_ℂ[ ϑ₂'((u+iw-ω)/2) / ϑ₂((u+iw-ω)/2) ] )

    Returns a real number (B̂ lives in ℝ as it measures the planar displacement).
    """
    R_om = TF.R_omega(omega, tau)
    th1_2om = TF.theta1(2 * omega, tau)
    th1p_0 = TF.theta1_prime_zero(tau)
    th2pp_om = TF.theta2(omega, tau, derivative=2)
    th2_om = TF.theta2(omega, tau)

    prefactor = complex(R_om) * (2 * th1_2om / th1p_0)

    # First term: ϑ₂''(ω)/ϑ₂(ω) · w/2
    term1 = (th2pp_om / th2_om) * w / 2

    # Second term: Im_ℂ[ ϑ₂'(z)/ϑ₂(z) ] where z = (u + iw - ω)/2
    z = 0.5 * (u + 1j * w - omega)
    th2p_z = TF.theta2(z, tau, derivative=1)
    th2_z = TF.theta2(z, tau)
    log_deriv = th2p_z / th2_z
    term2 = log_deriv.imag  # Im_ℂ

    result = prefactor * (term1 - term2)
    return result.real


def B_hat_vec(u_array: np.ndarray, w: float, omega: float,
              tau: complex) -> np.ndarray:
    """Vectorized B̂(u,w) over array of u values at fixed w."""
    R_om = TF.R_omega(omega, tau)
    th1_2om = TF.theta1(2 * omega, tau)
    th1p_0 = TF.theta1_prime_zero(tau)
    th2pp_om = TF.theta2(omega, tau, derivative=2)
    th2_om = TF.theta2(omega, tau)

    prefactor = complex(R_om) * (2 * th1_2om / th1p_0)
    term1 = (th2pp_om / th2_om) * w / 2  # constant over u

    z_array = 0.5 * (u_array + 1j * w - omega)
    th2p_z = TF.theta2_vec(z_array, tau, derivative=1)
    th2_z = TF.theta2_vec(z_array, tau)
    log_deriv = th2p_z / th2_z
    term2 = np.imag(log_deriv)

    result = prefactor * (term1 - term2)
    return np.real(result)


# ---------------------------------------------------------------------------
# b̃(w) — scalar for B̃ ODE (Eq. 123)
# ---------------------------------------------------------------------------

def b_tilde_scalar(w: float, omega: float, tau: complex) -> complex:
    """
    b̃(w) — complex scalar for the B̃ ODE (Eq. 123).

    Paper Eq. (123), after evaluating the Appendix A expression at u = ω:

      b̃(w) = (2ϑ₂(ω)/ϑ₁'(0)) · (ϑ₂(iw-ω)/ϑ₁(iw)) ·
              ( ϑ₂''(ω)/ϑ₂(ω) · w/2 - Im_ℂ[ϑ₂'(iw/2)/ϑ₂(iw/2)] )
              - i · ϑ₁(iw/2-ω)² / ϑ₂(iw/2)²
    """
    th2_om = TF.theta2(omega, tau)
    th1p_0 = TF.theta1_prime_zero(tau)
    th2_shifted = TF.theta2(1j * w - omega, tau)
    th1_iw = TF.theta1(1j * w, tau)

    prefactor = (2 * th2_om / th1p_0) * (th2_shifted / th1_iw)

    # Real scalar factor multiplying the prefactor.
    th2pp_om = TF.theta2(omega, tau, derivative=2)
    th2p_iw2 = TF.theta2(1j * w / 2, tau, derivative=1)
    th2_iw2 = TF.theta2(1j * w / 2, tau)

    scalar_factor = (th2pp_om / th2_om) * (w / 2) - (th2p_iw2 / th2_iw2).imag

    # Extra purely imaginary correction term present in the paper formula.
    th1_iw2_shift = TF.theta1(1j * w / 2 - omega, tau)
    correction = -1j * (th1_iw2_shift ** 2) / (th2_iw2 ** 2)

    return prefactor * scalar_factor + correction


# ---------------------------------------------------------------------------
# Christoffel dual via symmetry (Eq. 44-45)
# ---------------------------------------------------------------------------

def christoffel_dual_analytic(f_grid: np.ndarray, u_grid: np.ndarray,
                              omega: float, tau: complex,
                              w_func, phi_interp) -> np.ndarray:
    """
    Christoffel dual using the paper's analytic identity (Eq. 45):
      f*(u,v) = -f(π-u, v)

    This is much more accurate than finite-difference integration.
    Returns quaternion grid (n_u, n_v, 4).
    """
    n_u, n_v = f_grid.shape[:2]
    f_star = np.zeros_like(f_grid)

    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0

    for iu in range(n_u):
        u_star = np.pi - u_grid[iu]
        # Map to [0, 2π) and find nearest grid index
        u_star_mod = u_star % (2 * np.pi)
        iu_star = int(round(u_star_mod / du)) % n_u

        for iv in range(n_v):
            f_star[iu, iv] = -f_grid[iu_star, iv]

    return f_star


# ---------------------------------------------------------------------------
# Core: Bonnet pair computation (Eq. 49)
# ---------------------------------------------------------------------------

def compute_bonnet_pair(torus_result: TorusResult, epsilon: float) -> BonnetPairResult:
    """
    Compute the Bonnet pair f⁺, f⁻ from the isothermic torus.

    Paper Eq. (49), Theorem 5:
      f±(u,v) = R(ω)² f(π-2ω+u, v) - ε² f(π-u, v)
                ± 2ε ( Φ⁻¹(v) B̂(u,w(v)) i Φ(v) + B̃(v) )

    The 𝒜-part = R² f(π-2ω+u,v) - ε² f(π-u,v) is symmetric (same for f⁺,f⁻).
    The ℬ-part = 2ε(Φ⁻¹ B̂ i Φ + B̃) is antisymmetric (sign flip).

    Parameters
    ----------
    torus_result : TorusResult
        Base isothermic torus from Phase 2
    epsilon : float
        Bonnet parameter ε > 0
    """
    params = torus_result.params
    tau = params.tau
    omega = torus_result.omega
    f_grid = torus_result.f_grid
    u_grid = torus_result.u_grid
    v_grid = torus_result.v_grid
    n_u = params.u_res
    n_v = params.v_res

    # Pre-compute R(ω)
    R_om = TF.R_omega(omega, tau)
    R_om_sq = complex(R_om)**2

    # w(v) functions
    if params.w_func is None:
        w_func = lambda v: params.w0
        w_prime_func = lambda v: 0.0
    else:
        w_func = params.w_func
        w_prime_func = params.w_prime_func
    speed_func = params.speed_func

    # Frame interpolator
    phi_interp = CubicSpline(
        torus_result.frame_result.v_values,
        torus_result.frame_result.phi_values,
        axis=0
    )

    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0

    # -----------------------------------------------------------------------
    # Part 𝒜: R(ω)² f(π-2ω+u, v) - ε² f(π-u, v)
    # -----------------------------------------------------------------------
    A_grid = np.zeros((n_u, n_v, 4))

    for iu in range(n_u):
        # Map π-2ω+u → grid index
        u1 = (np.pi - 2 * omega + u_grid[iu]) % (2 * np.pi)
        iu1 = int(round(u1 / du)) % n_u

        # Map π-u → grid index (this is f*)
        u2 = (np.pi - u_grid[iu]) % (2 * np.pi)
        iu2 = int(round(u2 / du)) % n_u

        for iv in range(n_v):
            f_shifted = f_grid[iu1, iv]   # f(π-2ω+u, v)
            f_star_pt = f_grid[iu2, iv]   # f(π-u, v) = -f*(u,v)

            A_grid[iu, iv] = R_om_sq.real * f_shifted - epsilon**2 * f_star_pt

    # -----------------------------------------------------------------------
    # Part ℬ: 2ε ( Φ⁻¹(v) B̂(u,w(v)) i Φ(v) + B̃(v) )
    # -----------------------------------------------------------------------

    # Compute B̂ on the grid (vectorized over u for each v)
    B_hat_grid = np.zeros((n_u, n_v))
    for iv in range(n_v):
        w_val = w_func(v_grid[iv])
        B_hat_grid[:, iv] = B_hat_vec(u_grid, w_val, omega, tau)

    # Integrate B̃(v)
    def b_tilde_w(w_val):
        return b_tilde_scalar(w_val, omega, tau)

    B_tilde_result = integrate_B_tilde(
        phi_interp=lambda v: phi_interp(v) / np.linalg.norm(phi_interp(v)),
        w_func=w_func,
        w_prime_func=w_prime_func,
        b_tilde_func=b_tilde_w,
        omega=omega,
        tau=tau,
        v_span=(v_grid[0], v_grid[-1]),
        speed_func=speed_func,
        n_points=n_v,
    )

    # Interpolate B̃(v)
    B_tilde_interp = CubicSpline(
        B_tilde_result.v_values, B_tilde_result.B_tilde, axis=0
    )

    # Assemble ℬ grid
    i_quat = Q.quat_i()
    B_grid = np.zeros((n_u, n_v, 4))

    for iv in range(n_v):
        phi_v = phi_interp(v_grid[iv])
        phi_v = phi_v / np.linalg.norm(phi_v)
        phi_inv = Q.qconj(phi_v)
        B_tilde_v = B_tilde_interp(v_grid[iv])

        for iu in range(n_u):
            # Φ⁻¹ · B̂ · i · Φ
            B_hat_val = B_hat_grid[iu, iv]
            B_hat_quat = Q.quat_from_scalar(B_hat_val)
            B_hat_i = Q.qmul(B_hat_quat, i_quat)  # B̂ · i
            rotated = Q.qmul(Q.qmul(phi_inv, B_hat_i), phi_v)

            B_grid[iu, iv] = rotated + B_tilde_v

    # -----------------------------------------------------------------------
    # Assemble f± = 𝒜 ± 2ε·ℬ
    # -----------------------------------------------------------------------
    f_plus_grid = A_grid + 2 * epsilon * B_grid
    f_minus_grid = A_grid - 2 * epsilon * B_grid

    # Extract ℝ³ vertices
    def grid_to_torus_result(f_grid_pm, label):
        verts_4d = f_grid_pm.reshape(-1, 4)
        verts = verts_4d[:, 1:4].copy()
        faces = build_torus_faces(n_u, n_v)
        normals = compute_vertex_normals(verts, faces)
        max_scalar = float(np.max(np.abs(verts_4d[:, 0])))

        return TorusResult(
            vertices=verts,
            faces=faces,
            normals=normals,
            u_grid=u_grid,
            v_grid=v_grid,
            f_grid=f_grid_pm,
            frame_result=torus_result.frame_result,
            omega=omega,
            params=params,
            metrics={
                'n_vertices': len(verts),
                'n_faces': len(faces),
                'max_scalar_part': max_scalar,
                'label': label,
            },
        )

    f_plus = grid_to_torus_result(f_plus_grid, 'f_plus')
    f_minus = grid_to_torus_result(f_minus_grid, 'f_minus')

    # Metrics
    metrics = {
        'epsilon': epsilon,
        'R_omega': complex(R_om),
        'R_omega_sq': R_om_sq.real,
        'B_hat_range': [float(B_hat_grid.min()), float(B_hat_grid.max())],
        'B_tilde_max_norm': float(np.max(np.linalg.norm(B_tilde_result.B_tilde, axis=1))),
        'A_part_max_norm': float(np.max(np.linalg.norm(A_grid.reshape(-1, 4), axis=1))),
        'B_part_max_norm': float(np.max(np.linalg.norm(B_grid.reshape(-1, 4), axis=1))),
    }

    return BonnetPairResult(
        f_plus=f_plus,
        f_minus=f_minus,
        f_base=torus_result,
        epsilon=epsilon,
        R_omega=R_om,
        B_hat_grid=B_hat_grid,
        B_tilde_result=B_tilde_result,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 3.5 Isometry verification
# ---------------------------------------------------------------------------

# compute_metric_tensor — central periodic differences


def verify_isometry(pair: BonnetPairResult) -> dict:
    """
    Verify that f⁺ and f⁻ are isometric: same first fundamental form.

    Compares E, F, G coefficients pointwise.
    """
    u_grid = pair.f_base.u_grid
    v_grid = pair.f_base.v_grid

    g_plus = compute_metric_tensor_periodic_central(pair.f_plus.f_grid, u_grid, v_grid)
    g_minus = compute_metric_tensor_periodic_central(pair.f_minus.f_grid, u_grid, v_grid)

    err_E = np.abs(g_plus['E'] - g_minus['E'])
    err_F = np.abs(g_plus['F'] - g_minus['F'])
    err_G = np.abs(g_plus['G'] - g_minus['G'])

    # Relative errors (avoid div by zero)
    scale_E = np.maximum(np.abs(g_plus['E']), 1e-30)
    scale_F = np.maximum(np.sqrt(np.abs(g_plus['E'] * g_plus['G'])), 1e-30)
    scale_G = np.maximum(np.abs(g_plus['G']), 1e-30)

    rel_E = err_E / scale_E
    rel_F = err_F / scale_F
    rel_G = err_G / scale_G

    n_u, n_v = err_E.shape
    margin = max(2, n_u // 10)
    interior_slice = (
        slice(margin, -margin if n_u > 2 * margin else None),
        slice(margin, -margin if n_v > 2 * margin else None),
    )
    if n_u > 2 * margin and n_v > 2 * margin:
        int_err_E = err_E[interior_slice]
        int_err_F = err_F[interior_slice]
        int_err_G = err_G[interior_slice]
        int_rel_E = rel_E[interior_slice]
        int_rel_F = rel_F[interior_slice]
        int_rel_G = rel_G[interior_slice]
    else:
        int_err_E, int_err_F, int_err_G = err_E, err_F, err_G
        int_rel_E, int_rel_F, int_rel_G = rel_E, rel_F, rel_G

    return {
        'E_max_abs_err': float(np.max(err_E)),
        'F_max_abs_err': float(np.max(err_F)),
        'G_max_abs_err': float(np.max(err_G)),
        'E_max_rel_err': float(np.max(rel_E)),
        'F_max_scaled_err': float(np.max(rel_F)),
        'G_max_rel_err': float(np.max(rel_G)),
        'metric_max_err': float(max(np.max(err_E), np.max(err_F), np.max(err_G))),
        'interior_E_max_abs_err': float(np.max(int_err_E)),
        'interior_F_max_abs_err': float(np.max(int_err_F)),
        'interior_G_max_abs_err': float(np.max(int_err_G)),
        'interior_E_max_rel_err': float(np.max(int_rel_E)),
        'interior_F_max_scaled_err': float(np.max(int_rel_F)),
        'interior_G_max_rel_err': float(np.max(int_rel_G)),
        'interior_metric_max_err': float(max(np.max(int_err_E), np.max(int_err_F), np.max(int_err_G))),
        'interior_metric_mean_err': float(np.mean(int_err_E) + np.mean(int_err_F) + np.mean(int_err_G)),
        'trim_margin': int(margin),
    }


# ---------------------------------------------------------------------------
# 3.6 Mean curvature verification
# ---------------------------------------------------------------------------

# compute_mean_curvature_fd — finite-difference mean curvature


def verify_mean_curvature(pair: BonnetPairResult) -> dict:
    """
    Verify f⁺ and f⁻ have equal mean curvature H⁺ = H⁻.
    """
    u_grid = pair.f_base.u_grid
    v_grid = pair.f_base.v_grid

    H_plus = compute_mean_curvature_fd(pair.f_plus.f_grid, u_grid, v_grid)
    H_minus = compute_mean_curvature_fd(pair.f_minus.f_grid, u_grid, v_grid)

    err = np.abs(H_plus - H_minus)

    # Interior points only (skip boundary-affected finite diff points)
    n_u, n_v = H_plus.shape
    margin = max(2, n_u // 10)
    interior = err[margin:-margin, margin:-margin] if n_u > 2 * margin else err

    return {
        'H_plus_mean': float(np.mean(np.abs(H_plus))),
        'H_minus_mean': float(np.mean(np.abs(H_minus))),
        'max_abs_diff': float(np.max(err)),
        'mean_abs_diff': float(np.mean(err)),
        'interior_max_diff': float(np.max(interior)),
        'interior_mean_diff': float(np.mean(interior)),
        'trim_margin': int(margin),
    }


# ---------------------------------------------------------------------------
# 3.7 Non-congruence verification
# ---------------------------------------------------------------------------

def verify_non_congruence(pair: BonnetPairResult) -> dict:
    """
    Verify f⁺ and f⁻ are NOT congruent (not related by rigid motion).

    Uses Procrustes analysis: finds the best rigid transformation aligning
    f⁺ to f⁻, then measures the residual. For a true Bonnet pair, the
    residual must be significantly > 0.
    """
    v_plus = pair.f_plus.vertices
    v_minus = pair.f_minus.vertices

    # Center both
    c_plus = v_plus - v_plus.mean(axis=0)
    c_minus = v_minus - v_minus.mean(axis=0)

    # Normalize scale
    s_plus = np.sqrt(np.sum(c_plus**2))
    s_minus = np.sqrt(np.sum(c_minus**2))

    if s_plus < 1e-15 or s_minus < 1e-15:
        return {'procrustes_distance': float('inf'), 'is_non_congruent': True}

    n_plus = c_plus / s_plus
    n_minus = c_minus / s_minus

    # Procrustes: find R minimizing ||n_plus - n_minus @ R||
    _, _, disparity = scipy_procrustes(n_plus, n_minus)

    # Direct Hausdorff-like distance (without rigid alignment)
    diffs = v_plus - v_minus
    pointwise_dist = np.linalg.norm(diffs, axis=1)
    direct_distance = float(np.mean(pointwise_dist))

    # Scale ratio
    scale_ratio = s_plus / s_minus

    return {
        'procrustes_disparity': float(disparity),
        'direct_mean_distance': direct_distance,
        'direct_max_distance': float(np.max(pointwise_dist)),
        'scale_ratio': float(scale_ratio),
        'is_non_congruent': disparity > 1e-6,
    }


# ---------------------------------------------------------------------------
# 3.8 OBJ Export
# ---------------------------------------------------------------------------

def export_bonnet_pair_obj(pair: BonnetPairResult, output_dir: str | Path,
                           prefix: str = "bonnet") -> tuple[Path, Path]:
    """
    Export both f⁺ and f⁻ as OBJ files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_plus = export_torus_obj(
        pair.f_plus,
        output_dir / f"{prefix}_f_plus.obj",
        object_name=f"{prefix}_f_plus"
    )
    path_minus = export_torus_obj(
        pair.f_minus,
        output_dir / f"{prefix}_f_minus.obj",
        object_name=f"{prefix}_f_minus"
    )

    return path_plus, path_minus


# ---------------------------------------------------------------------------
# 3.9 Closure gate
# ---------------------------------------------------------------------------

def closure_gate(pair: BonnetPairResult) -> dict:
    """
    Verify closure conditions for the Bonnet pair tori.

    Two conditions (Theorem 6 / Theorem 7):
    1. Rationality: kθ ∈ 2πℕ (frame rotation angle rational multiple of 2π)
    2. Axial vanishing: the relevant scalar is the projection of the axial
       B-part on the rotation axis A, not the full vector norm.

    We report both the legacy geometric closure proxies and the paper-aware
    axial projection computed from B̃(V) and the monodromy axis.
    """
    # 1. Rationality check
    theta = pair.f_base.frame_result.rotation_angle
    k = pair.f_base.params.symmetry_fold
    ratio = k * theta / (2 * np.pi)
    rationality_err = abs(ratio - round(ratio))

    # Paper-aware axial scalar: projection of B̃(V) onto the monodromy axis.
    monodromy = pair.f_base.frame_result.monodromy
    axis = monodromy[1:4].astype(float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-14:
        axis = axis / axis_norm
    else:
        axis = np.zeros(3)

    B_end = pair.B_tilde_result.B_tilde[-1, 1:4].astype(float)
    axial_projection = float(np.dot(axis, B_end))
    axial_norm = float(np.linalg.norm(B_end))

    paper_axial_scalar = None
    paper_axial_abs = None
    try:
        from .theorem7_periodicity import theorem7_lemma3_axial_scalar

        base_params = pair.f_base.params
        if abs(base_params.delta) > 0 and (base_params.s1 != 0 or base_params.s2 != 0):
            paper_axial_scalar = float(theorem7_lemma3_axial_scalar(
                tau_imag=base_params.tau_imag,
                delta=base_params.delta,
                s1=base_params.s1,
                s2=base_params.s2,
                n_half_samples=max(151, pair.f_base.params.v_res + 1),
            ))
            paper_axial_abs = float(abs(paper_axial_scalar))
    except Exception:
        paper_axial_scalar = None
        paper_axial_abs = None

    # 2. Closure of f⁺: check that the torus "closes"
    # f⁺(u, V) should ≈ f⁺(u, 0) after one period
    f_plus_grid = pair.f_plus.f_grid
    f_minus_grid = pair.f_minus.f_grid
    n_v = f_plus_grid.shape[1]

    # Closure error: difference between last and first v-row
    close_plus = np.linalg.norm(f_plus_grid[:, -1] - f_plus_grid[:, 0], axis=1)
    close_minus = np.linalg.norm(f_minus_grid[:, -1] - f_minus_grid[:, 0], axis=1)

    # 3. Euler characteristic of both meshes
    ec_plus = verify_euler_characteristic(
        pair.f_plus.faces, len(pair.f_plus.vertices), expected=0
    )
    ec_minus = verify_euler_characteristic(
        pair.f_minus.faces, len(pair.f_minus.vertices), expected=0
    )

    return {
        'rotation_angle': float(theta),
        'k_fold': k,
        'k_theta_over_2pi': float(ratio),
        'rationality_error': float(rationality_err),
        'is_rational': rationality_err < 0.1,
        'axis_direction': axis.tolist(),
        'B_tilde_end': B_end.tolist(),
        'axial_projection': axial_projection,
        'axial_projection_abs': float(abs(axial_projection)),
        'axial_B_norm': axial_norm,
        'paper_axial_scalar': paper_axial_scalar,
        'paper_axial_scalar_abs': paper_axial_abs,
        'f_plus_closure_max': float(np.max(close_plus)),
        'f_plus_closure_mean': float(np.mean(close_plus)),
        'f_minus_closure_max': float(np.max(close_minus)),
        'f_minus_closure_mean': float(np.mean(close_minus)),
        'euler_plus': ec_plus,
        'euler_minus': ec_minus,
        'all_ok': (rationality_err < 0.1 and ec_plus['ok'] and ec_minus['ok']),
    }
