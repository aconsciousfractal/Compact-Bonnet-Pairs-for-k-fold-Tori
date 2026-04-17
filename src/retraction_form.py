"""
Retraction Form Engine — Phase 12 (arXiv:2506.13312v2)

Implements the Burstall–Hoffmann–Pedit–Sageman-Furnas construction:
given an isothermic torus f ∈ Im(ℍ) ≅ ℝ³, lift to S³ via inverse
stereographic projection, construct the retraction 1-form ω, and
integrate dF⁺ = ½ x·ω̄,  dF⁻ = -½ ω̄·x  to obtain the Bonnet pair.

Key equations (Section 3, quaternionic formalism):
  - 𝔰𝔬(4) ≅ Im(ℍ) ⊕ Im(ℍ), acting on ℍ by (z_L, z_R)c = z_L·c − c·z_R
  - a∧b ↦ ½(Im(b̄a), −Im(āb))   [Eq. 3.2]
  - For isothermic x: Σ → S³ ⊂ ℍ, the retraction form ω satisfies:
      dω = 0     [closure, 1.5a]
      ω̄∧dx = 0  [cross condition, 1.5b]
  - Bonnet pair: dF⁺ = ½ x·ω̄,   dF⁻ = −½ ω̄·x    [Theorem 2.1]

The retraction form is the Christoffel dual differential:
  ω = (x_u / |x_u|²) du  −  (x_v / |x_v|²) dv

This satisfies both conditions for isothermic surfaces in conformal
curvature-line coordinates, and produces F⁺, F⁻ ∈ Im(ℍ) as the
Bonnet pair — matching the Eq. 49 construction up to rigid motion.

Reference: arXiv:2506.13312v2, Sections 1–3.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from . import quaternion_ops as Q


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetractionFormResult:
    """Result of retraction form construction."""
    x_grid: np.ndarray            # (N_u, N_v, 4) — surface on S³
    omega_u: np.ndarray           # (N_u, N_v, 4) — retraction form u-component
    omega_v: np.ndarray           # (N_u, N_v, 4) — retraction form v-component
    F_plus: np.ndarray            # (N_u, N_v, 4) — Bonnet surface F⁺ ∈ Im(ℍ)
    F_minus: np.ndarray           # (N_u, N_v, 4) — Bonnet surface F⁻ ∈ Im(ℍ)
    closure_error: float          # max relative |∂_v ω_u − ∂_u ω_v|
    cross_error: float            # max relative |ω̄_u · x_v − ω̄_v · x_u|
    conformality_error: float     # max ||x_u|² − |x_v|²| / max(|x_u|², |x_v|²)
    exactness_error: float        # max relative path-dependence of F integration
    metrics: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Stereographic projection  (south-pole convention: S = (−1,0,0,0))
# ═══════════════════════════════════════════════════════════════════

def inverse_stereographic(f_grid: np.ndarray) -> np.ndarray:
    """
    Lift f ∈ Im(ℍ) ≅ ℝ³  →  x ∈ S³ ⊂ ℍ.

    σ⁻¹(f): x₀ = (|f|²−1)/(|f|²+1),  x_im = 2f/(|f|²+1)

    Parameters
    ----------
    f_grid : ndarray, shape (..., 4), scalar part ≈ 0

    Returns
    -------
    x_grid : ndarray, shape (..., 4), |x| = 1
    """
    f_im = f_grid[..., 1:4]
    r_sq = np.sum(f_im ** 2, axis=-1)  # |f|²
    denom = r_sq + 1.0

    x = np.empty_like(f_grid)
    x[..., 0] = (r_sq - 1.0) / denom
    x[..., 1] = 2.0 * f_im[..., 0] / denom
    x[..., 2] = 2.0 * f_im[..., 1] / denom
    x[..., 3] = 2.0 * f_im[..., 2] / denom
    return x


def stereographic(x_grid: np.ndarray) -> np.ndarray:
    """
    Project x ∈ S³ ⊂ ℍ  →  f ∈ Im(ℍ) ≅ ℝ³.

    σ_N(x) = Im(x) / (1 − Re(x))

    Convention: projection from north pole N = (1,0,0,0).
    Pairs with inverse_stereographic (σ_N⁻¹).
    """
    denom = 1.0 - x_grid[..., 0]
    denom_safe = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    f = np.zeros_like(x_grid)
    f[..., 1] = x_grid[..., 1] / denom_safe
    f[..., 2] = x_grid[..., 2] / denom_safe
    f[..., 3] = x_grid[..., 3] / denom_safe
    return f


# ═══════════════════════════════════════════════════════════════════
# Analytic Christoffel dual — f*(u,v) = −f(π−u, v)
# ═══════════════════════════════════════════════════════════════════

def _christoffel_dual_grid(f_grid: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """
    Analytic Christoffel dual: f*(u,v) = −f(π−u, v).

    Paper Eq. 45 identity for isothermic torus family.
    Uses nearest-neighbor grid lookup (exact when N_u is even).
    """
    n_u = f_grid.shape[0]
    du = u_grid[1] - u_grid[0]
    f_star = np.empty_like(f_grid)
    for iu in range(n_u):
        u_star = np.pi - u_grid[iu]
        u_star_mod = u_star % (2.0 * np.pi)
        iu_star = int(round(u_star_mod / du)) % n_u
        f_star[iu] = -f_grid[iu_star]
    return f_star


# ═══════════════════════════════════════════════════════════════════
# Discrete derivatives (central differences, u-periodic)
# ═══════════════════════════════════════════════════════════════════

def _central_diff_u(g: np.ndarray, du: float) -> np.ndarray:
    """Central difference ∂g/∂u with periodic wrapping in u."""
    return (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / (2.0 * du)


def _central_diff_v(g: np.ndarray, dv: float) -> np.ndarray:
    """Central difference ∂g/∂v; forward/backward at v-boundaries."""
    N_v = g.shape[1]
    out = np.empty_like(g)
    if N_v >= 3:
        out[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / (2.0 * dv)
    out[:, 0] = (g[:, 1] - g[:, 0]) / dv
    out[:, -1] = (g[:, -1] - g[:, -2]) / dv
    return out


def compute_derivatives(x_grid: np.ndarray, du: float, dv: float):
    """
    Return dx/du, dx/dv via central finite differences.

    Returns
    -------
    dx_du, dx_dv : ndarray, shape (N_u, N_v, 4)
    """
    return _central_diff_u(x_grid, du), _central_diff_v(x_grid, dv)


# ═══════════════════════════════════════════════════════════════════
# Retraction form  ω = (x_u / |x_u|²) du  −  (x_v / |x_v|²) dv
# ═══════════════════════════════════════════════════════════════════

def compute_retraction_omega(dx_du: np.ndarray, dx_dv: np.ndarray):
    """
    Retraction 1-form (Christoffel dual differential on S³).

    For isothermic x in conformal curvature-line coords:
        ω_u = x_u / |x_u|²,   ω_v = −x_v / |x_v|²

    Satisfies dω = 0  and  ω̄∧dx = 0 = dx∧ω̄.

    Returns
    -------
    omega_u, omega_v : (N_u, N_v, 4)
    e2u : (N_u, N_v) — conformal factor |x_u|²
    """
    e2u_u = np.sum(dx_du ** 2, axis=-1)
    e2u_v = np.sum(dx_dv ** 2, axis=-1)

    e2u_u_safe = np.maximum(e2u_u, 1e-30)
    e2u_v_safe = np.maximum(e2u_v, 1e-30)

    omega_u = dx_du / e2u_u_safe[..., np.newaxis]
    omega_v = -dx_dv / e2u_v_safe[..., np.newaxis]

    e2u = 0.5 * (e2u_u + e2u_v)
    return omega_u, omega_v, e2u


# ═══════════════════════════════════════════════════════════════════
# Verification helpers
# ═══════════════════════════════════════════════════════════════════

def verify_closure(omega_u: np.ndarray, omega_v: np.ndarray,
                   du: float, dv: float) -> float:
    """
    Check dω = 0:  ∂_v(ω_u) should equal ∂_u(ω_v).

    Returns max relative residual.
    """
    d_omega_u_dv = _central_diff_v(omega_u, dv)
    d_omega_v_du = _central_diff_u(omega_v, du)

    residual = d_omega_u_dv - d_omega_v_du
    res_norm = np.sqrt(np.sum(residual ** 2, axis=-1))

    scale = np.maximum(
        np.sqrt(np.sum(d_omega_u_dv ** 2, axis=-1)),
        np.sqrt(np.sum(d_omega_v_du ** 2, axis=-1)),
    )
    scale_max = max(float(np.max(scale)), 1e-30)

    return float(np.max(res_norm)) / scale_max


def verify_cross_condition(omega_u: np.ndarray, omega_v: np.ndarray,
                           dx_du: np.ndarray, dx_dv: np.ndarray) -> float:
    """
    Check ω̄∧dx = 0:  ω̄_u · x_v  −  ω̄_v · x_u  = 0.

    Returns max relative residual.
    """
    N = omega_u.shape[0] * omega_u.shape[1]

    ob_u = Q.qconj_batch(omega_u.reshape(N, 4))
    ob_v = Q.qconj_batch(omega_v.reshape(N, 4))

    term1 = Q.qmul_batch(ob_u, dx_dv.reshape(N, 4))  # ω̄_u · x_v
    term2 = Q.qmul_batch(ob_v, dx_du.reshape(N, 4))  # ω̄_v · x_u

    residual = term1 - term2
    res_norm = np.sqrt(np.sum(residual ** 2, axis=1))

    t1_norm = np.sqrt(np.sum(term1 ** 2, axis=1))
    t2_norm = np.sqrt(np.sum(term2 ** 2, axis=1))
    scale = max(float(np.max(t1_norm)), float(np.max(t2_norm)), 1e-30)

    return float(np.max(res_norm)) / scale


# ═══════════════════════════════════════════════════════════════════
# F± integration:  dF⁺ = ½ x·ω̄,   dF⁻ = −½ ω̄·x
# ═══════════════════════════════════════════════════════════════════

def _compute_dF(x_grid: np.ndarray,
                omega_u: np.ndarray, omega_v: np.ndarray):
    """
    Build the integrand 1-forms dF⁺ and dF⁻.

    Returns
    -------
    dFp_u, dFp_v, dFm_u, dFm_v : (N_u, N_v, 4)
    """
    N_u, N_v, _ = x_grid.shape
    N = N_u * N_v

    flat_x = x_grid.reshape(N, 4)
    ob_u = Q.qconj_batch(omega_u.reshape(N, 4))
    ob_v = Q.qconj_batch(omega_v.reshape(N, 4))

    dFp_u = 0.5 * Q.qmul_batch(flat_x, ob_u)
    dFp_v = 0.5 * Q.qmul_batch(flat_x, ob_v)
    dFm_u = -0.5 * Q.qmul_batch(ob_u, flat_x)
    dFm_v = -0.5 * Q.qmul_batch(ob_v, flat_x)

    shape = (N_u, N_v, 4)
    return (dFp_u.reshape(shape), dFp_v.reshape(shape),
            dFm_u.reshape(shape), dFm_v.reshape(shape))


def _integrate_on_grid(dF_u: np.ndarray, dF_v: np.ndarray,
                       du: float, dv: float) -> np.ndarray:
    """
    Cumulative trapezoidal integration of a closed 1-form on the (u,v) grid.

    Path: (0,0) → (u,0) along u, then (u,0) → (u,v) along v.
    """
    N_u, N_v, _ = dF_u.shape
    F = np.zeros_like(dF_u)

    # Along u at v = 0
    for iu in range(1, N_u):
        F[iu, 0] = F[iu - 1, 0] + 0.5 * (dF_u[iu - 1, 0] + dF_u[iu, 0]) * du

    # Along v for each u
    for iu in range(N_u):
        for iv in range(1, N_v):
            F[iu, iv] = (F[iu, iv - 1]
                         + 0.5 * (dF_v[iu, iv - 1] + dF_v[iu, iv]) * dv)
    return F


def _integrate_on_grid_alt(dF_u: np.ndarray, dF_v: np.ndarray,
                           du: float, dv: float) -> np.ndarray:
    """
    Alternative path: (0,0) → (0,v) along v, then (0,v) → (u,v) along u.
    Used for path-independence sanity check.
    """
    N_u, N_v, _ = dF_u.shape
    F = np.zeros_like(dF_u)

    # Along v at u = 0
    for iv in range(1, N_v):
        F[0, iv] = F[0, iv - 1] + 0.5 * (dF_v[0, iv - 1] + dF_v[0, iv]) * dv

    # Along u for each v
    for iv in range(N_v):
        for iu in range(1, N_u):
            F[iu, iv] = (F[iu - 1, iv]
                         + 0.5 * (dF_u[iu - 1, iv] + dF_u[iu, iv]) * du)
    return F


def integrate_bonnet_pair(x_grid: np.ndarray,
                          omega_u: np.ndarray, omega_v: np.ndarray,
                          du: float, dv: float):
    """
    Integrate the Bonnet pair from the retraction form.

    Returns
    -------
    F_plus, F_minus : (N_u, N_v, 4) — Im(ℍ)-valued surfaces
    exactness_error : float — max relative path-dependence
    """
    dFp_u, dFp_v, dFm_u, dFm_v = _compute_dF(x_grid, omega_u, omega_v)

    # Primary path: u first, then v
    F_plus = _integrate_on_grid(dFp_u, dFp_v, du, dv)
    F_minus = _integrate_on_grid(dFm_u, dFm_v, du, dv)

    # Alternative path: v first, then u
    Fp_alt = _integrate_on_grid_alt(dFp_u, dFp_v, du, dv)
    Fm_alt = _integrate_on_grid_alt(dFm_u, dFm_v, du, dv)

    # Exactness error (path-independence)
    diff_p = np.sqrt(np.sum((F_plus - Fp_alt) ** 2, axis=-1))
    diff_m = np.sqrt(np.sum((F_minus - Fm_alt) ** 2, axis=-1))
    scale_p = max(float(np.max(np.sqrt(np.sum(F_plus ** 2, axis=-1)))), 1e-30)
    scale_m = max(float(np.max(np.sqrt(np.sum(F_minus ** 2, axis=-1)))), 1e-30)

    exactness = max(float(np.max(diff_p)) / scale_p,
                    float(np.max(diff_m)) / scale_m)

    return F_plus, F_minus, exactness


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def compute_retraction_bonnet(torus_result, method: str = 'analytic',
                              verbose: bool = True) -> RetractionFormResult:
    """
    Full retraction form pipeline: f → S³ → ω → F±.

    Parameters
    ----------
    torus_result : TorusResult from isothermic_torus.py
    method : 'analytic' (recommended), 'dual', or 'direct'
        'analytic' — exact derivatives from theta functions (Phase 16)
        'dual'     — ω = dx* from analytic Christoffel dual
        'direct'   — ω = x_u/|x_u|² du − x_v/|x_v|² dv via FD (lowest accuracy)
    verbose : print diagnostic info

    Returns
    -------
    RetractionFormResult
    """
    # ── Analytic path (Phase 16): delegate entirely ──
    if method == 'analytic':
        from .analytic_derivatives import compute_analytic_retraction_form
        ar = compute_analytic_retraction_form(torus_result, verbose=verbose)
        metrics = dict(
            method='analytic',
            unit_error=ar['unit_error'],
            conformality_error=ar['conformality_error'],
            orthogonality_error=ar['orthogonality_error'],
            closure_error=ar['closure_error'],
            cross_error=ar['cross_error'],
            exactness_error=ar['exactness_error'],
            tangency_error=ar['tangency_error'],
            F_plus_max_scalar=float(np.max(np.abs(ar['F_plus'][..., 0]))),
            F_minus_max_scalar=float(np.max(np.abs(ar['F_minus'][..., 0]))),
        )
        return RetractionFormResult(
            x_grid=ar['x_grid'],
            omega_u=ar['omega_u'],
            omega_v=ar['omega_v'],
            F_plus=ar['F_plus'],
            F_minus=ar['F_minus'],
            closure_error=ar['closure_error'],
            cross_error=ar['cross_error'],
            conformality_error=ar['conformality_error'],
            exactness_error=ar['exactness_error'],
            metrics=metrics,
        )
    f_grid = torus_result.f_grid
    N_u, N_v, _ = f_grid.shape

    u_grid = np.linspace(0, 2 * np.pi, N_u, endpoint=False)
    v_values = torus_result.frame_result.v_values
    du = u_grid[1] - u_grid[0]
    dv = (v_values[-1] - v_values[0]) / (len(v_values) - 1) if len(v_values) > 1 else 1.0

    if verbose:
        print(f"[Phase 12] Retraction form ({method}): grid {N_u}×{N_v}, "
              f"du={du:.6f}, dv={dv:.6f}")

    # ── Step 1: lift f to S³ ──
    x_grid = inverse_stereographic(f_grid)
    unit_err = float(np.max(np.abs(
        np.sqrt(np.sum(x_grid ** 2, axis=-1)) - 1.0
    )))
    if verbose:
        print(f"  S³ lift: max ||x|−1| = {unit_err:.2e}")

    # ── Step 2: x derivatives (needed for conformality and cross checks) ──
    dx_du, dx_dv = compute_derivatives(x_grid, du, dv)

    # ── Step 3: conformality check ──
    e2u_u = np.sum(dx_du ** 2, axis=-1)
    e2u_v = np.sum(dx_dv ** 2, axis=-1)
    conf_err = float(np.max(np.abs(e2u_u - e2u_v))
                     / max(float(np.max(np.maximum(e2u_u, e2u_v))), 1e-30))
    if verbose:
        print(f"  Conformality: max ||x_u|²−|x_v|²|/max = {conf_err:.2e}")

    # ── Step 4: retraction form ω ──
    if method == 'dual':
        # Analytic Christoffel dual: f*(u,v) = −f(π−u,v), then ω = dx*
        f_star = _christoffel_dual_grid(f_grid, u_grid)
        x_star = inverse_stereographic(f_star)
        omega_u = _central_diff_u(x_star, du)
        omega_v = _central_diff_v(x_star, dv)
        e2u = 0.5 * (e2u_u + e2u_v)
    else:
        omega_u, omega_v, e2u = compute_retraction_omega(dx_du, dx_dv)

    # ── Step 5: verify x ⊥ ω (ω takes values in T_xS³) ──
    N = N_u * N_v
    flat_x = x_grid.reshape(N, 4)
    flat_ou = omega_u.reshape(N, 4)
    flat_ov = omega_v.reshape(N, 4)
    dot_u = np.abs(np.sum(flat_x * flat_ou, axis=1))  # ⟨x, ω_u⟩
    dot_v = np.abs(np.sum(flat_x * flat_ov, axis=1))  # ⟨x, ω_v⟩
    orth_err = max(float(np.max(dot_u)), float(np.max(dot_v)))
    if verbose:
        print(f"  x⊥ω (tangent check): max |⟨x,ω⟩| = {orth_err:.2e}")

    # ── Step 6: verify dω = 0 ──
    closure_err = verify_closure(omega_u, omega_v, du, dv)
    if verbose:
        print(f"  Closure dω=0: relative error = {closure_err:.2e}")

    # ── Step 7: verify ω̄∧dx = 0 ──
    cross_err = verify_cross_condition(omega_u, omega_v, dx_du, dx_dv)
    if verbose:
        print(f"  Cross ω̄∧dx=0: relative error = {cross_err:.2e}")

    # ── Step 8: integrate F± ──
    F_plus, F_minus, exact_err = integrate_bonnet_pair(
        x_grid, omega_u, omega_v, du, dv,
    )
    if verbose:
        print(f"  Exactness (path independence): relative = {exact_err:.2e}")

    # ── Step 9: purity check ──
    fp_scalar = float(np.max(np.abs(F_plus[..., 0])))
    fm_scalar = float(np.max(np.abs(F_minus[..., 0])))
    if verbose:
        print(f"  F⁺ max |scalar part| = {fp_scalar:.2e}")
        print(f"  F⁻ max |scalar part| = {fm_scalar:.2e}")

    metrics = dict(
        method=method,
        unit_error=unit_err,
        conformality_error=conf_err,
        orthogonality_error=orth_err,
        closure_error=closure_err,
        cross_error=cross_err,
        exactness_error=exact_err,
        F_plus_max_scalar=fp_scalar,
        F_minus_max_scalar=fm_scalar,
        e2u_mean=float(np.mean(e2u)),
        e2u_std=float(np.std(e2u)),
        grid_N_u=N_u,
        grid_N_v=N_v,
        du=du,
        dv=dv,
    )

    return RetractionFormResult(
        x_grid=x_grid,
        omega_u=omega_u,
        omega_v=omega_v,
        F_plus=F_plus,
        F_minus=F_minus,
        closure_error=closure_err,
        cross_error=cross_err,
        conformality_error=conf_err,
        exactness_error=exact_err,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════
# Comparison with Eq. 49  (Task 12.6)
# ═══════════════════════════════════════════════════════════════════

def _procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Procrustes disparity between two (N,3) point clouds.
    Centers, scales to unit Frobenius, then finds best rotation.
    Returns residual sum-of-squares (0 = identical shape).
    """
    from scipy.spatial import procrustes as scipy_procrustes
    _, _, d = scipy_procrustes(A, B)
    return float(d)


def compare_retraction_vs_eq49(retraction_result: RetractionFormResult,
                                bonnet_result,
                                verbose: bool = True) -> dict:
    """
    Compare F± from retraction with f± from Eq. 49 via Procrustes.

    Accounts for:
      - scaling ambiguity (Procrustes normalises)
      - rotational ambiguity (Procrustes finds best rotation)
      - possible ±/swap of labelling

    Returns dict with Procrustes disparities.
    """
    Fp = retraction_result.F_plus[..., 1:4].reshape(-1, 3)
    Fm = retraction_result.F_minus[..., 1:4].reshape(-1, 3)

    fp = bonnet_result.f_plus.f_grid[..., 1:4].reshape(-1, 3)
    fm = bonnet_result.f_minus.f_grid[..., 1:4].reshape(-1, 3)

    d_Fp_fp = _procrustes_disparity(fp, Fp)
    d_Fm_fm = _procrustes_disparity(fm, Fm)
    d_Fp_fm = _procrustes_disparity(fm, Fp)
    d_Fm_fp = _procrustes_disparity(fp, Fm)

    best_match = min(d_Fp_fp + d_Fm_fm, d_Fp_fm + d_Fm_fp)
    swapped = (d_Fp_fm + d_Fm_fp) < (d_Fp_fp + d_Fm_fm)

    if verbose:
        print(f"\n[Phase 12] Retraction F± vs Eq.49 f±")
        print(f"  Procrustes  F⁺↔f⁺ : {d_Fp_fp:.6e}")
        print(f"  Procrustes  F⁻↔f⁻ : {d_Fm_fm:.6e}")
        print(f"  Procrustes  F⁺↔f⁻ : {d_Fp_fm:.6e}")
        print(f"  Procrustes  F⁻↔f⁺ : {d_Fm_fp:.6e}")
        print(f"  Best match sum     : {best_match:.6e}"
              f"  {'(swapped)' if swapped else '(direct)'}")

    return dict(
        procrustes_Fp_fp=d_Fp_fp,
        procrustes_Fm_fm=d_Fm_fm,
        procrustes_Fp_fm=d_Fp_fm,
        procrustes_Fm_fp=d_Fm_fp,
        best_match=best_match,
        labels_swapped=swapped,
    )


# ═══════════════════════════════════════════════════════════════════
# Validation gate  (Task 12.7)
# ═══════════════════════════════════════════════════════════════════

def retraction_validation_gate(retraction_result: RetractionFormResult,
                                bonnet_result=None,
                                tol_closure: float = 0.05,
                                tol_cross: float = 0.05,
                                tol_exactness: float = 0.1,
                                tol_procrustes: float = 0.01,
                                verbose: bool = True) -> dict:
    """
    Validation gate for retraction form (resolution-independent check).

    Returns dict with pass/fail per criterion and overall status.
    """
    R = retraction_result
    checks = {}

    # Gate 1: closure dω = 0
    checks['closure'] = R.closure_error < tol_closure

    # Gate 2: cross condition ω̄∧dx = 0
    checks['cross'] = R.cross_error < tol_cross

    # Gate 3: path-independence
    checks['exactness'] = R.exactness_error < tol_exactness

    # Gate 4: F± purity (∈ Im(ℍ))
    checks['purity'] = (R.metrics.get('F_plus_max_scalar', 1) < 0.1 and
                        R.metrics.get('F_minus_max_scalar', 1) < 0.1)

    # Gate 5: comparison with Eq. 49 (if available)
    if bonnet_result is not None:
        comp = compare_retraction_vs_eq49(R, bonnet_result, verbose=False)
        checks['procrustes'] = comp['best_match'] < tol_procrustes
        checks['procrustes_value'] = comp['best_match']
    else:
        checks['procrustes'] = None

    checks['all_pass'] = all(
        v for v in checks.values() if isinstance(v, bool)
    )

    if verbose:
        status = "PASS ✓" if checks['all_pass'] else "FAIL ✗"
        print(f"\n[Phase 12] Validation gate: {status}")
        print(f"  dω=0 closure  : {'✓' if checks['closure'] else '✗'}  "
              f"({R.closure_error:.2e} < {tol_closure})")
        print(f"  ω̄∧dx=0 cross  : {'✓' if checks['cross'] else '✗'}  "
              f"({R.cross_error:.2e} < {tol_cross})")
        print(f"  Exactness      : {'✓' if checks['exactness'] else '✗'}  "
              f"({R.exactness_error:.2e} < {tol_exactness})")
        print(f"  F± purity      : {'✓' if checks['purity'] else '✗'}")
        if checks['procrustes'] is not None:
            print(f"  Procrustes     : {'✓' if checks['procrustes'] else '✗'}  "
                  f"({checks.get('procrustes_value', '?'):.2e} < {tol_procrustes})")

    return checks


# ═══════════════════════════════════════════════════════════════════
# Isometry check on F± directly
# ═══════════════════════════════════════════════════════════════════

def verify_retraction_isometry(F_plus: np.ndarray, F_minus: np.ndarray,
                                du: float, dv: float) -> dict:
    """
    Check that F⁺ and F⁻ are isometric:  |dF⁺| ≈ |dF⁻| at each point.

    Returns dict with max/mean metric discrepancy.
    """
    Fp_u = _central_diff_u(F_plus, du)
    Fp_v = _central_diff_v(F_plus, dv)
    Fm_u = _central_diff_u(F_minus, du)
    Fm_v = _central_diff_v(F_minus, dv)

    # g_uu = |F_u|², g_vv = |F_v|², g_uv = ⟨F_u, F_v⟩
    gp_uu = np.sum(Fp_u ** 2, axis=-1)
    gp_vv = np.sum(Fp_v ** 2, axis=-1)
    gp_uv = np.sum(Fp_u * Fp_v, axis=-1)

    gm_uu = np.sum(Fm_u ** 2, axis=-1)
    gm_vv = np.sum(Fm_v ** 2, axis=-1)
    gm_uv = np.sum(Fm_u * Fm_v, axis=-1)

    scale = np.maximum(gp_uu, gm_uu)
    scale = np.maximum(scale, 1e-30)

    d_uu = np.abs(gp_uu - gm_uu) / scale
    d_vv = np.abs(gp_vv - gm_vv) / np.maximum(np.maximum(gp_vv, gm_vv), 1e-30)
    d_uv = np.abs(gp_uv - gm_uv) / np.maximum(scale, 1e-30)

    return dict(
        max_d_guu=float(np.max(d_uu)),
        max_d_gvv=float(np.max(d_vv)),
        max_d_guv=float(np.max(d_uv)),
        mean_d_guu=float(np.mean(d_uu)),
        mean_d_gvv=float(np.mean(d_vv)),
        mean_d_guv=float(np.mean(d_uv)),
    )
