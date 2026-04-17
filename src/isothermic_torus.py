"""
Isothermic Torus Generator — Bonnet's Problem Phase 2

Implements Theorem 3 of the paper: constructs the isothermic torus
f(u,v) = Φ(v)⁻¹ · γ(u, w(v)) · j · Φ(v)

where:
  - γ(u,w) is the family of planar curves (theta_functions.gamma_curve)
  - Φ(v) is the quaternionic frame (frame_integrator.integrate_frame)
  - j is the unit quaternion j
  - w(v) is the reparametrization function

The torus lives in Im(ℍ) ≅ ℝ³.

Outputs:
  - vertices: (N_u × N_v, 3) array of ℝ³ points
  - faces: list of quad faces with toroidal topology
  - normals: per-vertex normals
  - metrics: curvature data, isothermic quality, mesh stats
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from scipy.interpolate import CubicSpline

from . import theta_functions as TF
from . import quaternion_ops as Q
from .frame_integrator import integrate_frame, FrameResult


def build_torus_faces(n_u: int, n_v: int) -> list[list[int]]:
    """Build quad faces for a doubly-periodic toroidal mesh."""
    faces = []
    for iu in range(n_u):
        iu_next = (iu + 1) % n_u
        for iv in range(n_v):
            iv_next = (iv + 1) % n_v
            v00 = iu * n_v + iv
            v10 = iu_next * n_v + iv
            v11 = iu_next * n_v + iv_next
            v01 = iu * n_v + iv_next
            faces.append([v00, v10, v11, v01])
    return faces


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TorusParameters:
    """Parameters for isothermic torus construction."""
    tau_imag: float               # Im(τ), range (0, ~0.3547)
    delta: float = 0.0            # perturbation parameter (0 = simplest case)
    s1: float = 0.0               # first modulus parameter
    s2: float = 0.0               # second modulus parameter
    u_res: int = 100              # resolution in u direction
    v_res: int = 100              # resolution in v direction
    v_periods: int = 1            # number of v-periods to cover
    symmetry_fold: int = 3        # k-fold symmetry (3 or 4 for paper examples)
    w_func: Callable | None = None     # custom w(v), if None uses constant
    w_prime_func: Callable | None = None  # custom w'(v)
    w0: float = 0.2               # constant w value (if w_func is None)
    speed_func: Callable | None = None   # signed sqrt(1 - w'^2), if needed
    v_period: float | None = None        # known period of w(v), if available

    @property
    def tau(self) -> complex:
        return 0.5 + 1j * self.tau_imag


@dataclass
class TorusResult:
    """Result of isothermic torus generation."""
    vertices: np.ndarray           # (N_u * N_v, 3) mesh vertices
    faces: list[list[int]]         # quad faces
    normals: np.ndarray            # (N_u * N_v, 3) per-vertex normals
    u_grid: np.ndarray             # (N_u,) u parameter values
    v_grid: np.ndarray             # (N_v,) v parameter values
    f_grid: np.ndarray             # (N_u, N_v, 4) quaternion values on grid
    frame_result: FrameResult      # Φ(v) integration result
    omega: float                   # critical omega found
    params: TorusParameters        # input parameters
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_torus(params: TorusParameters) -> TorusResult:
    """
    Generate isothermic torus from parameters.

    Paper Eq. (32): f(u,v) = Φ(v)⁻¹ · γ(u, w(v)) · j · Φ(v)

    Steps:
      1. Find critical ω from τ
      2. Integrate frame Φ(v) over one full period
      3. Evaluate γ(u, w(v)) on (u,v) grid
      4. Assemble f(u,v) = Φ⁻¹ · γ · j · Φ
      5. Extract ℝ³ coordinates from Im(ℍ)
      6. Build quad mesh with toroidal topology
    """
    tau = params.tau

    # Step 1: Find critical omega
    omega = TF.find_critical_omega(tau)

    # Step 2: Determine v-period from frame integration
    # For constant w, the period V comes from the monodromy
    # For general w, we need to find V such that kθ = 2πn
    if params.w_func is None:
        # Auto-build Theorem 7 w(v) profile when δ/s₁/s₂ are set
        if abs(params.delta) > 1e-14 and (abs(params.s1) > 1e-14 or abs(params.s2) > 1e-14):
            from .theorem7_periodicity import theorem7_w_functions
            w_func, w_prime_func, speed_func, _profile = theorem7_w_functions(
                tau_imag=params.tau_imag,
                delta=params.delta,
                s1=params.s1,
                s2=params.s2,
            )
            if params.v_period is None:
                params = TorusParameters(
                    tau_imag=params.tau_imag,
                    delta=params.delta,
                    s1=params.s1,
                    s2=params.s2,
                    u_res=params.u_res,
                    v_res=params.v_res,
                    v_periods=params.v_periods,
                    symmetry_fold=params.symmetry_fold,
                    w_func=w_func,
                    w_prime_func=w_prime_func,
                    speed_func=speed_func,
                    v_period=_profile.period,
                    w0=params.w0,
                )
        else:
            w_func = lambda v: params.w0
            w_prime_func = lambda v: 0.0
            speed_func = params.speed_func
    else:
        w_func = params.w_func
        w_prime_func = params.w_prime_func
        speed_func = params.speed_func

    if params.v_period is not None:
        V_period = float(params.v_period)
    else:
        # Probe integration to find the rotation rate
        probe_V = 2 * np.pi  # initial probe period
        frame_probe = integrate_frame(
            w_func=w_func,
            w_prime_func=w_prime_func,
            omega=omega,
            tau=tau,
            v_span=(0, probe_V),
            speed_func=speed_func,
            n_points=50,
        )

        # The rotation angle over probe_V
        theta_probe = frame_probe.rotation_angle
        if theta_probe < 1e-10:
            raise RuntimeError("Frame rotation angle is zero — degenerate case")

        # For k-fold symmetry, we need kθ = 2πn → V = (2π/θ_probe) * probe_V * n/k
        # We want the smallest V such that the torus closes
        # θ per unit v = theta_probe / probe_V
        rotation_rate = theta_probe / probe_V
        # V_period = 2π / (k * rotation_rate) for k-fold closure
        k = params.symmetry_fold
        V_period = 2 * np.pi / (k * rotation_rate) if rotation_rate > 1e-10 else probe_V
    V_total = V_period * params.v_periods

    # Step 3: Full frame integration
    frame_result = integrate_frame(
        w_func=w_func,
        w_prime_func=w_prime_func,
        omega=omega,
        tau=tau,
        v_span=(0, V_total),
        speed_func=speed_func,
        n_points=params.v_res,
    )

    # Build interpolator for Φ(v)
    phi_interp = CubicSpline(frame_result.v_values, frame_result.phi_values, axis=0)

    # Step 4: u and v grids
    u_grid = np.linspace(0, 2 * np.pi, params.u_res, endpoint=False)
    v_grid = frame_result.v_values

    # Step 5: Evaluate f(u,v) on the grid — vectorized over u
    j_quat = Q.quat_j()
    f_grid = np.zeros((params.u_res, params.v_res, 4))

    for iv in range(params.v_res):
        v = v_grid[iv]
        w_val = w_func(v)
        phi_v = phi_interp(v)
        phi_v = phi_v / np.linalg.norm(phi_v)  # ensure unit
        phi_inv = Q.qconj(phi_v)  # for unit quaternion, inv = conj

        # Vectorized gamma: compute all u values at once
        gamma_vals = TF.gamma_curve_vec(u_grid, w_val, omega, tau)

        for iu in range(params.u_res):
            gamma_quat = np.array([gamma_vals[iu].real, gamma_vals[iu].imag, 0.0, 0.0])

            # f(u,v) = Φ⁻¹ · γ · j · Φ
            gamma_j = Q.qmul(gamma_quat, j_quat)
            phi_inv_gamma_j = Q.qmul(phi_inv, gamma_j)
            f_uv = Q.qmul(phi_inv_gamma_j, phi_v)

            f_grid[iu, iv] = f_uv

    # Step 6: Extract ℝ³ vertices (imaginary part of quaternion)
    # f should be pure imaginary (scalar part ≈ 0)
    vertices_4d = f_grid.reshape(-1, 4)
    scalar_parts = np.abs(vertices_4d[:, 0])
    max_scalar = np.max(scalar_parts)

    vertices = vertices_4d[:, 1:4].copy()  # Im(ℍ) ≅ ℝ³

    # Step 7: Build quad faces with toroidal topology
    faces = build_torus_faces(params.u_res, params.v_res)

    # Step 8: Compute normals
    normals = compute_vertex_normals(vertices, faces)

    # Step 9: Metrics
    metrics = {
        'n_vertices': len(vertices),
        'n_faces': len(faces),
        'max_scalar_part': float(max_scalar),
        'omega': float(omega),
        'rotation_angle': float(frame_result.rotation_angle),
        'V_period': float(V_period),
        'V_total': float(V_total),
        'frame_unitarity_max_err': float(np.max(frame_result.unitarity_errors)),
        'bbox_min': vertices.min(axis=0).tolist(),
        'bbox_max': vertices.max(axis=0).tolist(),
    }

    return TorusResult(
        vertices=vertices,
        faces=faces,
        normals=normals,
        u_grid=u_grid,
        v_grid=v_grid,
        f_grid=f_grid,
        frame_result=frame_result,
        omega=omega,
        params=params,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Mesh topology
# ---------------------------------------------------------------------------

# build_torus_faces — defined above


def compute_vertex_normals(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """Compute per-vertex normals by averaging face normals."""
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        fn = np.cross(e1, e2)
        fn_norm = np.linalg.norm(fn)
        if fn_norm > 1e-30:
            fn = fn / fn_norm
        for idx in face:
            normals[idx] += fn
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    normals = normals / norms
    return normals


# ---------------------------------------------------------------------------
# Isothermic validation (Task 2.4)
# ---------------------------------------------------------------------------

def compute_cross_ratio(f: np.ndarray, iu: int, iv: int,
                        n_u: int, n_v: int) -> complex:
    """
    Compute discrete cross-ratio on a quad (iu,iv) → (iu+1,iv) → (iu+1,iv+1) → (iu,iv+1).

    For an isothermic net: (f₁-f)(f₁₂-f₁)⁻¹(f₁₂-f₂)(f₂-f)⁻¹ = -1

    where products are quaternionic.

    Parameters
    ----------
    f : ndarray (n_u, n_v, 4)
        Quaternion-valued surface
    """
    iu1 = (iu + 1) % n_u
    iv1 = (iv + 1) % n_v

    f00 = f[iu, iv]
    f10 = f[iu1, iv]
    f11 = f[iu1, iv1]
    f01 = f[iu, iv1]

    # Cross-ratio = (f₁-f)·(f₁₂-f₁)⁻¹·(f₁₂-f₂)·(f₂-f)⁻¹
    d1 = f10 - f00        # f₁ - f
    d12 = f11 - f10       # f₁₂ - f₁
    d2_back = f11 - f01   # f₁₂ - f₂
    d2 = f01 - f00        # f₂ - f

    cr = Q.qmul(Q.qmul(Q.qmul(d1, Q.qinv(d12)), d2_back), Q.qinv(d2))
    return complex(cr[0], np.linalg.norm(cr[1:4]))  # scalar + |vector|


def validate_isothermic(f_grid: np.ndarray, n_u: int, n_v: int) -> dict:
    """
    Validate isothermic property: cross-ratio = -1 on all quads.

    Returns dict with:
      - cross_ratios: array of cross-ratio scalars
      - mean_error: average |cr + 1|
      - max_error: maximum |cr + 1|
      - is_isothermic: bool (max_error < threshold)
    """
    cross_ratios = []
    errors = []

    for iu in range(n_u):
        for iv in range(n_v):
            iu1 = (iu + 1) % n_u
            iv1 = (iv + 1) % n_v

            f00 = f_grid[iu, iv]
            f10 = f_grid[iu1, iv]
            f11 = f_grid[iu1, iv1]
            f01 = f_grid[iu, iv1]

            d1 = f10 - f00
            d12 = f11 - f10
            d2_back = f11 - f01
            d2 = f01 - f00

            # Skip degenerate quads
            if np.linalg.norm(d12) < 1e-15 or np.linalg.norm(d2) < 1e-15:
                continue

            cr = Q.qmul(Q.qmul(Q.qmul(d1, Q.qinv(d12)), d2_back), Q.qinv(d2))

            # For isothermic: cr should == -1 = [-1, 0, 0, 0]
            target = np.array([-1.0, 0.0, 0.0, 0.0])
            err = np.linalg.norm(cr - target)
            cross_ratios.append(cr)
            errors.append(err)

    errors = np.array(errors)
    return {
        'n_quads_checked': len(errors),
        'mean_error': float(np.mean(errors)) if len(errors) > 0 else float('inf'),
        'max_error': float(np.max(errors)) if len(errors) > 0 else float('inf'),
        'median_error': float(np.median(errors)) if len(errors) > 0 else float('inf'),
        'is_isothermic': bool(np.max(errors) < 0.5) if len(errors) > 0 else False,
    }


# ---------------------------------------------------------------------------
# Symmetry verification (Task 2.5 — Theorem 4)
# ---------------------------------------------------------------------------

def verify_symmetry_inversion(f_grid: np.ndarray, u_grid: np.ndarray,
                              omega: float, tau: complex) -> dict:
    """
    Verify Theorem 4 inversion symmetry:
      f^{inv}(u, v) = f(2ω - u, v)

    The isothermic torus with planar curvature lines has an inversion
    symmetry exchanging the two curvature line families.

    We check: f(u, v) ≈ R · f(2ω - u, v) for some rigid transformation R.
    Since the inversion includes a scaling, we check shape similarity.
    """
    n_u, n_v = f_grid.shape[:2]
    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0

    # Map u → 2ω - u (mod 2π)
    u_reflected = 2 * omega - u_grid
    u_reflected = u_reflected % (2 * np.pi)

    # For each reflected u, find nearest grid index
    errors = []
    for iu in range(n_u):
        u_ref = u_reflected[iu]
        # Find nearest index
        iu_ref = int(round(u_ref / du)) % n_u

        for iv in range(n_v):
            f_original = f_grid[iu, iv, 1:4]  # ℝ³ part
            f_reflected = f_grid[iu_ref, iv, 1:4]

            # f and f_reflected should have same shape up to rigid motion
            # Simple check: compare norms (preserved by inversion up to scaling)
            r1 = np.linalg.norm(f_original)
            r2 = np.linalg.norm(f_reflected)
            if r1 > 1e-10 and r2 > 1e-10:
                errors.append(abs(r1 / r2 - 1.0) if r2 > r1 else abs(r2 / r1 - 1.0))

    errors = np.array(errors) if errors else np.array([float('inf')])
    return {
        'mean_ratio_deviation': float(np.mean(errors)),
        'max_ratio_deviation': float(np.max(errors)),
        'n_points_checked': len(errors),
    }


# ---------------------------------------------------------------------------
# Christoffel dual (Task 2.5 — prep for Phase 3)
# ---------------------------------------------------------------------------

def compute_christoffel_dual(f_grid: np.ndarray, u_grid: np.ndarray,
                             v_grid: np.ndarray) -> np.ndarray:
    """
    Compute the Christoffel dual f* of an isothermic surface.

    Paper Eq. (23): df* = -(f_u)⁻¹ du + (f_v)⁻¹ dv

    For the discrete case, we use finite differences to approximate f_u and f_v,
    then integrate the dual 1-form.

    Returns f_star: (n_u, n_v, 4) quaternion array.
    """
    n_u, n_v = f_grid.shape[:2]
    du = u_grid[1] - u_grid[0] if len(u_grid) > 1 else 1.0
    dv = v_grid[1] - v_grid[0] if len(v_grid) > 1 else 1.0

    f_star = np.zeros_like(f_grid)

    # Integrate along u first (at v=0), then along v
    # f*(0,0) = 0 (arbitrary choice)
    f_star[0, 0] = np.zeros(4)

    # Along u: df* = -(f_u)⁻¹ du
    for iu in range(1, n_u):
        f_u = (f_grid[iu, 0] - f_grid[iu - 1, 0]) / du
        if np.linalg.norm(f_u) > 1e-15:
            f_u_inv = Q.qinv(f_u)
            f_star[iu, 0] = f_star[iu - 1, 0] - f_u_inv * du
        else:
            f_star[iu, 0] = f_star[iu - 1, 0]

    # Along v: df* = (f_v)⁻¹ dv
    for iu in range(n_u):
        for iv in range(1, n_v):
            f_v = (f_grid[iu, iv] - f_grid[iu, iv - 1]) / dv
            if np.linalg.norm(f_v) > 1e-15:
                f_v_inv = Q.qinv(f_v)
                f_star[iu, iv] = f_star[iu, iv - 1] + f_v_inv * dv
            else:
                f_star[iu, iv] = f_star[iu, iv - 1]

    return f_star


# ---------------------------------------------------------------------------
# OBJ Export (Task 2.6)
# ---------------------------------------------------------------------------

def export_torus_obj(result: TorusResult, path: str | Path,
                     object_name: str = "isothermic_torus") -> Path:
    """Export torus mesh to OBJ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_obj_standalone(path, result, object_name)


def _write_obj_standalone(path: Path, result: TorusResult,
                          object_name: str) -> Path:
    """Standalone OBJ writer."""
    with open(path, 'w') as f:
        f.write(f"# Isothermic torus — Bonnet's Problem\n")
        f.write(f"# tau = {result.params.tau}, omega = {result.omega:.10f}\n")
        f.write(f"# Resolution: {result.params.u_res} x {result.params.v_res}\n")
        f.write(f"# Vertices: {len(result.vertices)}, Faces: {len(result.faces)}\n")
        f.write(f"o {object_name}\n")

        # Vertices
        for v in result.vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        # Normals
        for n in result.normals:
            f.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")

        # Faces (OBJ is 1-indexed)
        for face in result.faces:
            indices = ' '.join(f"{idx + 1}//{idx + 1}" for idx in face)
            f.write(f"f {indices}\n")

    return path


# ---------------------------------------------------------------------------
# Euler characteristic for torus
# ---------------------------------------------------------------------------

def verify_euler_characteristic(faces: list[list[int]], n_vertices: int,
                                expected: int = 0) -> dict:
    """
    Verify Euler characteristic χ = V - E + F for torus (expected = 0).
    """
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            e = tuple(sorted((face[i], face[(i + 1) % n])))
            edges.add(e)

    V = n_vertices
    E = len(edges)
    F = len(faces)
    chi = V - E + F

    return {
        'vertices': V,
        'edges': E,
        'faces': F,
        'euler_characteristic': chi,
        'expected': expected,
        'ok': chi == expected,
    }
