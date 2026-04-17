"""
Theorem 7 periodicity conditions for spherical v-curves.

This module implements the elliptic-integral formulation of the closure conditions
from Theorem 7 of Bobenko-Hoffmann-Sageman-Furnas (2025).

Unlike the Phase 8 heuristic scripts, the residuals here are written directly in the
paper coordinates (Im tau, delta, s1, s2), using the second elliptic curve

    Q(s) = -(s-s1)^2 (s-s2)^2 + delta^2 Q3(s)

and the corresponding rationality / axial integrals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from . import theta_functions as TF


_REAL_TOL = 1e-7


def _real_scalar(value: complex, tol: float = _REAL_TOL) -> float:
    """Extract a real scalar from a nearly-real complex value."""
    z = complex(value)
    scale = max(abs(z.real), abs(z.imag), 1.0)
    if abs(z.imag) > tol * scale:
        raise ValueError(f"expected nearly-real value, got {z!r}")
    return float(z.real)


def _paper_constants(omega: float, tau: complex) -> dict[str, float]:
    """Return the real scalar constants used in Theorem 7."""
    return {
        'U': _real_scalar(TF.U_omega(omega, tau)),
        'Up': _real_scalar(TF.U_prime_omega(omega, tau)),
        'U1p': _real_scalar(TF.U1_prime_omega(omega, tau)),
        'U2': _real_scalar(TF.U2_omega(omega, tau)),
    }


def theorem7_s0(omega: float, tau: complex) -> float:
    """Left endpoint s0 of the real oval of Q3."""
    return _real_scalar(TF.theta1(omega, tau) ** 2 / TF.theta2(0, tau) ** 2)


def theorem7_Q(omega: float, tau: complex, delta: float,
               s1: float, s2: float, s: float) -> float:
    """Real evaluation of Q(s), paper Eq. (84)."""
    return _real_scalar(TF.Q_polynomial(s, s1, s2, delta, omega, tau))


def theorem7_Q2(omega: float, tau: complex, delta: float,
                s1: float, s2: float, s: float) -> float:
    """Q2(s), paper Eq. (86)."""
    const = _paper_constants(omega, tau)
    return float(-(s - s1) * (s - s2) + (delta ** 2) * const['U1p'] * s)


def theorem7_Z0_squared(omega: float, tau: complex, delta: float,
                        s1: float, s2: float) -> float:
    """Z0^2, paper Eq. (88)."""
    const = _paper_constants(omega, tau)
    U = const['U']
    Up = const['Up']
    U1p = const['U1p']
    U2 = const['U2']

    return float(
        (U ** -2) * (2.0 * (s1 + s2) * U1p + (delta ** 2) * (U1p ** 2) - U2)
        + (U ** -4) * ((Up + s1 * s2 * U1p) ** 2)
    )


def theorem7_Z0(omega: float, tau: complex, delta: float,
                s1: float, s2: float) -> float:
    """Positive branch Z0 = |Z'(omega)|."""
    z0_sq = theorem7_Z0_squared(omega, tau, delta, s1, s2)
    if z0_sq <= 0:
        raise ValueError(f"Z0^2 must be positive, got {z0_sq}")
    return float(np.sqrt(z0_sq))


def theorem7_Qtilde2(omega: float, tau: complex, delta: float,
                     s1: float, s2: float, s: float) -> float:
    """
    Qtilde2(s), paper Eq. (87).

    The OCR text drops parentheses in Eq. (87). This implementation matches the
    printed formula reconstructed from Proposition 4 and Theorem 7:

        Qtilde2(s) = Z0^2 s^2 - (1 + s U^-2 (U' + s1 s2 U1'))^2

    In Eq. (82) the denominator is then Qtilde2(s) * sqrt(Q(s)), with an
    overall numerator factor Z0.
    """
    const = _paper_constants(omega, tau)
    U = const['U']
    affine = 1.0 + s * (U ** -2) * (const['Up'] + s1 * s2 * const['U1p'])
    z0_sq = theorem7_Z0_squared(omega, tau, delta, s1, s2)
    return float(z0_sq * (s ** 2) - affine ** 2)


def theorem7_a1(omega: float, tau: complex, delta: float,
                s1: float, s2: float, s: float) -> float:
    """Axis coefficient a1(s) from Proposition 4, up to the common sign choice."""
    const = _paper_constants(omega, tau)
    R = _real_scalar(TF.R_omega(omega, tau))
    z0 = theorem7_Z0(omega, tau, delta, s1, s2)
    return float((1.0 + s * (R ** 2) * (const['Up'] + s1 * s2 * const['U1p'])) / (s * z0))


def theorem7_a3(omega: float, tau: complex, delta: float,
                s1: float, s2: float, s: float) -> float:
    """Axis coefficient a3(s) from Proposition 4, up to the common sign choice."""
    R = _real_scalar(TF.R_omega(omega, tau))
    z0 = theorem7_Z0(omega, tau, delta, s1, s2)
    return float(-(R / (delta * z0 * s)) * theorem7_Q2(omega, tau, delta, s1, s2, s))


def theorem7_lemma3_axial_density(omega: float, tau: complex, delta: float,
                                  s1: float, s2: float, s: float) -> float:
    """
    Scalar density <A, e^{-h(omega,w(v))} n(omega,v)> from Lemma 3.

    Along the spherical curve u = omega, Eq. (80) gives

        <A, e^{-h} n> = s * a3(s),

    with a3(s) supplied by Proposition 4.
    """
    return float(s * theorem7_a3(omega, tau, delta, s1, s2, s))


def theorem7_lemma3_axial_scalar(tau_imag: float, delta: float,
                                 s1: float, s2: float,
                                 profile: "Theorem7WProfile" | None = None,
                                 n_half_samples: int = 401) -> float:
    """
    Lemma 3 axial scalar along the reconstructed theorem-7 profile.

    This is the paper-faithful scalar to compare inside the pipeline:

        <A, ∫_0^V e^{-h(omega,w(v))} n(omega,v) dv>

    evaluated through the moving-frame coefficient a3(s(v)), rather than through
    the less direct proxy dot(monodromy_axis, B_tilde(V)).
    """
    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    if profile is None:
        profile = theorem7_w_profile(
            tau_imag=tau_imag,
            delta=delta,
            s1=s1,
            s2=s2,
            n_half_samples=n_half_samples,
        )

    density = np.array([
        theorem7_lemma3_axial_density(omega, tau, delta, s1, s2, float(s))
        for s in profile.s_values
    ])
    values = cumulative_simpson(density, x=profile.v_values)
    return float(values[-1])


def theorem7_Q_coefficients(omega: float, tau: complex, delta: float,
                            s1: float, s2: float) -> np.ndarray:
    """Quartic coefficients of Q(s) in descending powers."""
    const = _paper_constants(omega, tau)
    quartic = np.poly1d([1.0, -(s1 + s2), s1 * s2])
    q_part = -np.polymul(quartic, quartic)
    q3_part = np.poly1d([
        2.0 * const['U1p'],
        -const['U2'],
        -2.0 * const['Up'],
        -(const['U'] ** 2),
    ])
    total = q_part + (delta ** 2) * q3_part
    return np.asarray(total.coeffs, dtype=float)


def theorem7_real_oval(omega: float, tau: complex, delta: float,
                       s1: float, s2: float,
                       real_tol: float = 1e-8) -> tuple[float, float]:
    """
    Return the real roots (s1-, s1+) that bound the real oval of Q.

    If several real pairs exist, prefer the positive interval that contains s1.
    """
    roots = np.roots(theorem7_Q_coefficients(omega, tau, delta, s1, s2))
    real_roots = sorted(float(r.real) for r in roots if abs(r.imag) < real_tol)
    if len(real_roots) < 2:
        raise ValueError(f"expected at least two real roots, got {roots!r}")

    containing_s1 = []
    other_pairs = []
    for a, b in zip(real_roots, real_roots[1:]):
        mid = 0.5 * (a + b)
        q_mid = theorem7_Q(omega, tau, delta, s1, s2, mid)
        if q_mid <= 0:
            continue
        item = (a, b, q_mid)
        if a <= s1 <= b:
            containing_s1.append(item)
        else:
            other_pairs.append(item)

    candidates = containing_s1 or other_pairs
    if not candidates:
        raise ValueError(f"no positive real oval found for delta={delta}, s1={s1}, s2={s2}")

    a, b, _ = max(candidates, key=lambda item: item[2])
    return float(a), float(b)


def _desingularized_integral(integrand_s, s_lower: float, s_upper: float,
                             epsabs: float = 1e-10,
                             epsrel: float = 1e-10) -> tuple[float, float]:
    """
    Integrate on [s_lower, s_upper] after s = mid + half*sin(phi).

    This removes the square-root endpoint singularity that appears in ds/sqrt(Q(s)).
    """
    mid = 0.5 * (s_lower + s_upper)
    half = 0.5 * (s_upper - s_lower)

    def integrand_phi(phi: float) -> float:
        cos_phi = math.cos(phi)
        if abs(cos_phi) < 1e-14:
            return 0.0
        s = mid + half * math.sin(phi)
        return float(integrand_s(s) * half * cos_phi)

    return quad(integrand_phi, -0.5 * math.pi, 0.5 * math.pi,
                epsabs=epsabs, epsrel=epsrel, limit=200)


def theorem7_theta_half(omega: float, tau: complex, delta: float,
                        s1: float, s2: float,
                        s_lower: float | None = None,
                        s_upper: float | None = None) -> tuple[float, float]:
    """Compute theta/2 from Theorem 7, Eq. (82)."""
    if s_lower is None or s_upper is None:
        s_lower, s_upper = theorem7_real_oval(omega, tau, delta, s1, s2)

    z0 = theorem7_Z0(omega, tau, delta, s1, s2)

    def integrand(s: float) -> float:
        q = theorem7_Q(omega, tau, delta, s1, s2, s)
        if q <= 0:
            return 0.0
        qtilde = theorem7_Qtilde2(omega, tau, delta, s1, s2, s)
        if abs(qtilde) < 1e-14:
            raise ZeroDivisionError(f"Qtilde2 nearly vanishes at s={s}")
        return z0 * theorem7_Q2(omega, tau, delta, s1, s2, s) / (qtilde * math.sqrt(q))

    return _desingularized_integral(integrand, s_lower, s_upper)


def theorem7_axial_integral(omega: float, tau: complex, delta: float,
                            s1: float, s2: float,
                            s_lower: float | None = None,
                            s_upper: float | None = None) -> tuple[float, float]:
    """Compute the axial integral from Theorem 7, Eq. (83)."""
    if s_lower is None or s_upper is None:
        s_lower, s_upper = theorem7_real_oval(omega, tau, delta, s1, s2)

    def integrand(s: float) -> float:
        q = theorem7_Q(omega, tau, delta, s1, s2, s)
        if q <= 0:
            return 0.0
        return theorem7_Q2(omega, tau, delta, s1, s2, s) / math.sqrt(q)

    return _desingularized_integral(integrand, s_lower, s_upper)


@dataclass
class Theorem7Residuals:
    tau_imag: float
    omega: float
    delta: float
    s1: float
    s2: float
    symmetry_fold: int
    s1_minus: float
    s1_plus: float
    theta: float
    theta_half: float
    ratio: float
    target_ratio: float | None
    rationality_residual: float
    nearest_multiple: int
    axial_integral: float
    axial_residual: float
    theta_error: float
    axial_error: float


@dataclass
class Theorem7WProfile:
    tau_imag: float
    omega: float
    delta: float
    s1: float
    s2: float
    s0: float
    s1_minus: float
    s1_plus: float
    period: float
    v_values: np.ndarray
    s_values: np.ndarray
    w_values: np.ndarray
    w_prime_values: np.ndarray
    signed_speed_values: np.ndarray


@dataclass
class Theorem7PipelineResiduals:
    theorem7: Theorem7Residuals
    profile: Theorem7WProfile
    frame_theta: float
    frame_ratio: float
    frame_rationality_residual: float
    lemma3_axial_scalar: float
    lemma3_axial_abs: float
    monodromy_axis: np.ndarray
    B_tilde_end: np.ndarray
    B_tilde_norm: float
    axial_projection: float


@dataclass
class Theorem7LocalSolveResult:
    success: bool
    status: int
    message: str
    method: str
    tau_imag: float
    s1: float
    symmetry_fold: int
    target_ratio: float
    initial_delta: float
    initial_s2: float
    delta: float
    s2: float
    residual_vector: np.ndarray
    residual_norm: float
    nfev: int
    theorem7: Theorem7Residuals
    pipeline: Theorem7PipelineResiduals | None


@dataclass
class Theorem7NondegeneracyDiagnostics:
    tau_imag: float
    delta: float
    s1: float
    s2: float
    symmetry_fold: int
    target_ratio: float
    lemma8_real_sqrt_expr: float
    lemma8_rationality_expr: float
    lemma8_vanishing_expr: float
    jacobian: np.ndarray
    jacobian_det: float
    jacobian_condition: float
    residual_vector: np.ndarray
    safe: bool


@dataclass
class Theorem7BranchSeed:
    label: str
    tau_imag: float
    s1: float
    delta: float
    s2: float
    symmetry_fold: int
    target_ratio: float = 1.0


@dataclass
class Theorem7ContinuationStep:
    index: int
    tau_imag: float
    predictor_delta: float
    predictor_s2: float
    used_secant_predictor: bool
    corrector: Theorem7LocalSolveResult
    accepted_step_size: float | None = None
    retry_count: int = 0
    nondegeneracy: Theorem7NondegeneracyDiagnostics | None = None


@dataclass
class Theorem7ContinuationResult:
    success: bool
    message: str
    method: str
    s1: float
    symmetry_fold: int
    target_ratio: float
    tau_values: np.ndarray
    steps: list[Theorem7ContinuationStep]
    n_success: int
    failed_index: int | None
    adaptive: bool = False
    start_tau_imag: float | None = None
    end_tau_imag: float | None = None
    final_step_size: float | None = None


@dataclass
class Theorem7BranchCatalogEntry:
    seed: Theorem7BranchSeed
    backward: Theorem7ContinuationResult
    forward: Theorem7ContinuationResult


@dataclass
class Theorem7BranchCatalog:
    entries: list[Theorem7BranchCatalogEntry]


@dataclass
class Theorem7BonnetVerification:
    tau_imag: float
    delta: float
    s1: float
    s2: float
    symmetry_fold: int
    epsilon: float
    torus: object
    pair: object
    isometry: dict
    mean_curvature: dict
    non_congruence: dict
    closure: dict


def theorem7_residuals(tau_imag: float, delta: float,
                       s1: float, s2: float,
                       symmetry_fold: int = 3,
                       target_ratio: float | None = None) -> Theorem7Residuals:
    """Evaluate the two Theorem 7 residuals at a given parameter point."""
    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    s1_minus, s1_plus = theorem7_real_oval(omega, tau, delta, s1, s2)

    theta_half, theta_error = theorem7_theta_half(
        omega, tau, delta, s1, s2, s1_minus, s1_plus,
    )
    axial_integral, axial_error = theorem7_axial_integral(
        omega, tau, delta, s1, s2, s1_minus, s1_plus,
    )

    theta = 2.0 * theta_half
    ratio = symmetry_fold * theta / (2.0 * math.pi)
    nearest_multiple = int(round(ratio))
    rationality_target = float(target_ratio) if target_ratio is not None else float(nearest_multiple)

    return Theorem7Residuals(
        tau_imag=float(tau_imag),
        omega=float(omega),
        delta=float(delta),
        s1=float(s1),
        s2=float(s2),
        symmetry_fold=int(symmetry_fold),
        s1_minus=float(s1_minus),
        s1_plus=float(s1_plus),
        theta=float(theta),
        theta_half=float(theta_half),
        ratio=float(ratio),
        target_ratio=target_ratio,
        rationality_residual=float(abs(ratio - rationality_target)),
        nearest_multiple=nearest_multiple,
        axial_integral=float(axial_integral),
        axial_residual=float(abs(axial_integral)),
        theta_error=float(theta_error),
        axial_error=float(axial_error),
    )


def theorem7_s_to_w_lookup(omega: float, tau: complex,
                           n_lookup: int = 4096,
                           eps: float = 1e-6) -> CubicSpline:
    """Monotone spline that inverts s(w) on the admissible real interval."""
    w_max = 2.0 * math.pi * float(tau.imag)
    w_grid = np.linspace(eps, w_max - eps, n_lookup)
    s_grid = np.array([
        _real_scalar(TF.s_of_w(float(w), omega, tau))
        for w in w_grid
    ])
    return CubicSpline(s_grid, w_grid, extrapolate=False)


def theorem7_w_profile(tau_imag: float, delta: float,
                       s1: float, s2: float,
                       n_half_samples: int = 401) -> Theorem7WProfile:
    """
    Build a periodic sample of w(v) for the spherical-v-curves family.

    The construction follows the monotone elliptic integral v(s) on the real oval
    [s1-, s1+] and inverts s = s(w) numerically using the exact Eq. (65).
    """
    if n_half_samples < 3:
        raise ValueError("n_half_samples must be at least 3")

    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    s0 = theorem7_s0(omega, tau)
    s1_minus, s1_plus = theorem7_real_oval(omega, tau, delta, s1, s2)
    if s1_minus <= s0:
        raise ValueError(
            f"real oval of Q must satisfy s1- > s0; got s1-={s1_minus}, s0={s0}"
        )

    phi_half = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_half_samples)
    mid = 0.5 * (s1_minus + s1_plus)
    half = 0.5 * (s1_plus - s1_minus)
    s_half = mid + half * np.sin(phi_half)
    q_half = np.array([theorem7_Q(omega, tau, delta, s1, s2, float(s)) for s in s_half])
    if np.any(q_half < -1e-10):
        raise ValueError(f"Q(s) must stay non-negative on the real oval, min={q_half.min()}")
    q_half = np.maximum(q_half, 0.0)

    dv_dphi = abs(delta) * half * np.cos(phi_half) / np.sqrt(np.maximum(q_half, 1e-30))
    v_half = np.zeros_like(phi_half)
    v_half[1:] = cumulative_simpson(dv_dphi, x=phi_half)
    v_half -= v_half[0]
    v_period = 2.0 * v_half[-1]

    v_second_half = v_period - v_half[-2::-1]
    v_full = np.concatenate([v_half, v_second_half])
    s_full = np.concatenate([s_half, s_half[-2::-1]])

    s_to_w = theorem7_s_to_w_lookup(omega, tau)
    w_full = np.asarray(s_to_w(s_full), dtype=float)

    q3_full = np.array([
        _real_scalar(TF.Q3_polynomial(float(s), omega, tau))
        for s in s_full
    ])
    q_full = np.array([
        theorem7_Q(omega, tau, delta, s1, s2, float(s))
        for s in s_full
    ])
    if np.any(q3_full <= 0):
        raise ValueError(f"Q3(s) must stay positive on the Q real oval, min={q3_full.min()}")

    speed_abs = np.sqrt(np.maximum(q_full, 0.0)) / (abs(delta) * np.sqrt(q3_full))
    sign = np.ones_like(speed_abs)
    sign[len(s_half):] = -1.0
    w_prime_full = sign * speed_abs
    w_prime_full[-1] = w_prime_full[0]

    signed_speed_full = ((s_full - s1) * (s_full - s2)) / (delta * np.sqrt(q3_full))
    signed_speed_full[-1] = signed_speed_full[0]

    return Theorem7WProfile(
        tau_imag=float(tau_imag),
        omega=float(omega),
        delta=float(delta),
        s1=float(s1),
        s2=float(s2),
        s0=float(s0),
        s1_minus=float(s1_minus),
        s1_plus=float(s1_plus),
        period=float(v_period),
        v_values=v_full,
        s_values=s_full,
        w_values=w_full,
        w_prime_values=w_prime_full,
        signed_speed_values=signed_speed_full,
    )


def theorem7_w_functions(tau_imag: float, delta: float,
                         s1: float, s2: float,
                         n_half_samples: int = 401):
    """Return periodic w(v), w'(v), signed-speed callables and profile."""
    profile = theorem7_w_profile(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        n_half_samples=n_half_samples,
    )
    w_spline = CubicSpline(profile.v_values, profile.w_values, bc_type='periodic')
    wp_spline = CubicSpline(profile.v_values, profile.w_prime_values, bc_type='periodic')
    speed_spline = CubicSpline(profile.v_values, profile.signed_speed_values, bc_type='periodic')

    def w_func(v: float) -> float:
        return float(w_spline(v % profile.period))

    def w_prime_func(v: float) -> float:
        return float(wp_spline(v % profile.period))

    def speed_func(v: float) -> float:
        return float(speed_spline(v % profile.period))

    return w_func, w_prime_func, speed_func, profile


def build_theorem7_torus_parameters(tau_imag: float, delta: float,
                                    s1: float, s2: float,
                                    symmetry_fold: int,
                                    u_res: int = 40,
                                    v_res: int = 240,
                                    v_periods: int = 1,
                                    n_half_samples: int = 401):
    """Construct TorusParameters wired for the spherical-v-curves family."""
    from .isothermic_torus import TorusParameters

    w_func, w_prime_func, speed_func, profile = theorem7_w_functions(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        n_half_samples=n_half_samples,
    )

    params = TorusParameters(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        u_res=u_res,
        v_res=v_res,
        v_periods=v_periods,
        symmetry_fold=symmetry_fold,
        w_func=w_func,
        w_prime_func=w_prime_func,
        speed_func=speed_func,
        v_period=profile.period,
    )
    return params, profile


def verify_theorem7_bonnet_pipeline(tau_imag: float, delta: float,
                                    s1: float, s2: float,
                                    symmetry_fold: int,
                                    epsilon: float = 0.3,
                                    u_res: int = 8,
                                    v_res: int = 80,
                                    n_half_samples: int = 121) -> Theorem7BonnetVerification:
    """End-to-end theorem-7 verification through torus and Bonnet pair pipeline."""
    from .isothermic_torus import compute_torus
    from .bonnet_pair import (
        compute_bonnet_pair,
        verify_isometry,
        verify_mean_curvature,
        verify_non_congruence,
        closure_gate,
    )

    params, _ = build_theorem7_torus_parameters(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        symmetry_fold=symmetry_fold,
        u_res=u_res,
        v_res=v_res,
        n_half_samples=n_half_samples,
    )
    torus = compute_torus(params)
    pair = compute_bonnet_pair(torus, epsilon=epsilon)
    isometry = verify_isometry(pair)
    mean_curvature = verify_mean_curvature(pair)
    non_congruence = verify_non_congruence(pair)
    closure = closure_gate(pair)

    return Theorem7BonnetVerification(
        tau_imag=float(tau_imag),
        delta=float(delta),
        s1=float(s1),
        s2=float(s2),
        symmetry_fold=int(symmetry_fold),
        epsilon=float(epsilon),
        torus=torus,
        pair=pair,
        isometry=isometry,
        mean_curvature=mean_curvature,
        non_congruence=non_congruence,
        closure=closure,
    )


def theorem7_pipeline_residuals(tau_imag: float, delta: float,
                                s1: float, s2: float,
                                symmetry_fold: int,
                                n_half_samples: int = 201,
                                n_points: int = 240) -> Theorem7PipelineResiduals:
    """Evaluate the Theorem 7 examples through the frame/B-tilde ODE pipeline."""
    from .frame_integrator import integrate_frame, integrate_B_tilde
    from .bonnet_pair import b_tilde_scalar

    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    theorem7_res = theorem7_residuals(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        symmetry_fold=symmetry_fold,
        target_ratio=1.0,
    )
    w_func, w_prime_func, speed_func, profile = theorem7_w_functions(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        n_half_samples=n_half_samples,
    )

    frame = integrate_frame(
        w_func=w_func,
        w_prime_func=w_prime_func,
        omega=omega,
        tau=tau,
        v_span=(0.0, profile.period),
        speed_func=speed_func,
        n_points=n_points,
        rtol=1e-11,
        atol=1e-13,
    )
    axis = frame.monodromy[1:4].copy()
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-14:
        axis /= axis_norm

    phi_interp = CubicSpline(frame.v_values, frame.phi_values, axis=0)
    B_result = integrate_B_tilde(
        phi_interp=lambda v: phi_interp(v) / np.linalg.norm(phi_interp(v)),
        w_func=w_func,
        w_prime_func=w_prime_func,
        b_tilde_func=lambda w: b_tilde_scalar(w, omega, tau),
        omega=omega,
        tau=tau,
        v_span=(0.0, profile.period),
        speed_func=speed_func,
        n_points=n_points,
        rtol=1e-10,
        atol=1e-12,
    )
    B_end = B_result.B_tilde[-1][1:4].copy()
    frame_ratio = symmetry_fold * frame.rotation_angle / (2.0 * math.pi)
    lemma3_scalar = theorem7_lemma3_axial_scalar(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        profile=profile,
    )

    return Theorem7PipelineResiduals(
        theorem7=theorem7_res,
        profile=profile,
        frame_theta=float(frame.rotation_angle),
        frame_ratio=float(frame_ratio),
        frame_rationality_residual=float(abs(frame_ratio - 1.0)),
        lemma3_axial_scalar=float(lemma3_scalar),
        lemma3_axial_abs=float(abs(lemma3_scalar)),
        monodromy_axis=axis,
        B_tilde_end=B_end,
        B_tilde_norm=float(np.linalg.norm(B_end)),
        axial_projection=float(np.dot(axis, B_end)),
    )


def theorem7_local_residual_vector(tau_imag: float, delta: float,
                                   s1: float, s2: float,
                                   symmetry_fold: int,
                                   target_ratio: float = 1.0) -> np.ndarray:
    """Residual vector [F_rat, F_ax] for the local theorem-7 solver."""
    res = theorem7_residuals(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        symmetry_fold=symmetry_fold,
        target_ratio=target_ratio,
    )
    return np.array([
        res.ratio - target_ratio,
        res.axial_integral,
    ], dtype=float)


def theorem7_lemma8_real_sqrt_expr(omega: float, tau: complex,
                                   s1: float, s2: float) -> float:
    """Lemma 8 (iii): positivity of the square root appearing in Eq. (95)."""
    const = _paper_constants(omega, tau)
    U = const['U']
    Up = const['Up']
    U1p = const['U1p']
    U2 = const['U2']
    return float((Up + s1 * s2 * U1p) ** 2 + (U ** 2) * (2.0 * (s1 + s2) * U1p - U2))


def theorem7_lemma8_rationality_expr(omega: float, tau: complex,
                                     s1: float, s2: float) -> float:
    """Lemma 8 (iv): non-degeneracy of the rationality condition."""
    const = _paper_constants(omega, tau)
    return float((2.0 * s1 + s2) * s1 * const['U1p'] - s1 * const['U2'] - const['Up'])


def theorem7_lemma8_vanishing_expr(omega: float, tau: complex,
                                   s1: float, s2: float) -> float:
    """Lemma 8 (v): non-degeneracy of the axial vanishing condition."""
    const = _paper_constants(omega, tau)
    return float((s1 + 2.0 * s2) * s1 * const['U1p'] - s1 * const['U2'] - const['Up'])


def theorem7_local_jacobian(tau_imag: float, delta: float,
                            s1: float, s2: float,
                            symmetry_fold: int,
                            target_ratio: float = 1.0,
                            delta_step: float | None = None,
                            s2_step: float | None = None) -> np.ndarray:
    """Finite-difference Jacobian of [F_rat, F_ax] with respect to [delta, s2]."""
    h_delta = float(delta_step if delta_step is not None else max(1e-6, 1e-6 * max(abs(delta), 1.0)))
    h_s2 = float(s2_step if s2_step is not None else max(1e-6, 1e-6 * max(abs(s2), 1.0)))

    f_pp = theorem7_local_residual_vector(tau_imag, delta + h_delta, s1, s2, symmetry_fold, target_ratio)
    f_pm = theorem7_local_residual_vector(tau_imag, delta - h_delta, s1, s2, symmetry_fold, target_ratio)
    f_sp = theorem7_local_residual_vector(tau_imag, delta, s1, s2 + h_s2, symmetry_fold, target_ratio)
    f_sm = theorem7_local_residual_vector(tau_imag, delta, s1, s2 - h_s2, symmetry_fold, target_ratio)

    d_delta = (f_pp - f_pm) / (2.0 * h_delta)
    d_s2 = (f_sp - f_sm) / (2.0 * h_s2)
    return np.column_stack([d_delta, d_s2])


def theorem7_local_nondegeneracy_diagnostics(tau_imag: float, delta: float,
                                             s1: float, s2: float,
                                             symmetry_fold: int,
                                             target_ratio: float = 1.0,
                                             formula_tol: float = 1e-6,
                                             jacobian_det_tol: float = 1e-8,
                                             jacobian_max_condition: float = 1e8) -> Theorem7NondegeneracyDiagnostics:
    """
    Numerical monitor inspired by Lemma 8 (iii)-(v), plus the local Jacobian.

    This is used to detect when a continuation step is approaching a numerically
    degenerate regime where the local corrector becomes unreliable.
    """
    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    real_sqrt_expr = theorem7_lemma8_real_sqrt_expr(omega, tau, s1, s2)
    rationality_expr = theorem7_lemma8_rationality_expr(omega, tau, s1, s2)
    vanishing_expr = theorem7_lemma8_vanishing_expr(omega, tau, s1, s2)
    residual_vector = theorem7_local_residual_vector(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        symmetry_fold=symmetry_fold,
        target_ratio=target_ratio,
    )

    try:
        jacobian = theorem7_local_jacobian(
            tau_imag=tau_imag,
            delta=delta,
            s1=s1,
            s2=s2,
            symmetry_fold=symmetry_fold,
            target_ratio=target_ratio,
        )
        jacobian_det = float(np.linalg.det(jacobian))
        jacobian_condition = float(np.linalg.cond(jacobian))
    except Exception:
        jacobian = np.full((2, 2), np.nan)
        jacobian_det = float('nan')
        jacobian_condition = float('inf')

    safe = bool(
        np.isfinite(real_sqrt_expr)
        and np.isfinite(rationality_expr)
        and np.isfinite(vanishing_expr)
        and real_sqrt_expr > formula_tol
        and abs(rationality_expr) > formula_tol
        and abs(vanishing_expr) > formula_tol
        and np.isfinite(jacobian_det)
        and abs(jacobian_det) > jacobian_det_tol
        and np.isfinite(jacobian_condition)
        and jacobian_condition < jacobian_max_condition
    )

    return Theorem7NondegeneracyDiagnostics(
        tau_imag=float(tau_imag),
        delta=float(delta),
        s1=float(s1),
        s2=float(s2),
        symmetry_fold=int(symmetry_fold),
        target_ratio=float(target_ratio),
        lemma8_real_sqrt_expr=float(real_sqrt_expr),
        lemma8_rationality_expr=float(rationality_expr),
        lemma8_vanishing_expr=float(vanishing_expr),
        jacobian=jacobian,
        jacobian_det=float(jacobian_det),
        jacobian_condition=float(jacobian_condition),
        residual_vector=residual_vector,
        safe=safe,
    )


def solve_theorem7_local_fixed_tau_s1(tau_imag: float, s1: float,
                                      initial_delta: float,
                                      initial_s2: float,
                                      symmetry_fold: int,
                                      target_ratio: float = 1.0,
                                      method: str = 'hybr',
                                      evaluate_pipeline: bool = False,
                                      pipeline_n_half_samples: int = 151,
                                      pipeline_n_points: int = 160) -> Theorem7LocalSolveResult:
    """
    Local solver at fixed Im(tau) and s1, solving (F_rat, F_ax)=0 in (delta, s2).

    This is the first Newton-style solver for Phase 8.4. It is intentionally local:
    the caller is expected to start from a nearby seed, such as a published paper example
    or a continuation step from a previous solution.
    """
    x0 = np.array([float(initial_delta), float(initial_s2)], dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        delta, s2 = float(x[0]), float(x[1])
        try:
            return theorem7_local_residual_vector(
                tau_imag=tau_imag,
                delta=delta,
                s1=s1,
                s2=s2,
                symmetry_fold=symmetry_fold,
                target_ratio=target_ratio,
            )
        except Exception:
            scale = 1.0 + np.linalg.norm(x)
            return np.array([1e3 * scale, 1e3 * scale], dtype=float)

    sol = root(residuals, x0, method=method)
    delta_sol = float(sol.x[0])
    s2_sol = float(sol.x[1])
    theorem7_sol = theorem7_residuals(
        tau_imag=tau_imag,
        delta=delta_sol,
        s1=s1,
        s2=s2_sol,
        symmetry_fold=symmetry_fold,
        target_ratio=target_ratio,
    )
    residual_vector = np.array([
        theorem7_sol.ratio - target_ratio,
        theorem7_sol.axial_integral,
    ], dtype=float)

    pipeline_sol = None
    if evaluate_pipeline:
        pipeline_sol = theorem7_pipeline_residuals(
            tau_imag=tau_imag,
            delta=delta_sol,
            s1=s1,
            s2=s2_sol,
            symmetry_fold=symmetry_fold,
            n_half_samples=pipeline_n_half_samples,
            n_points=pipeline_n_points,
        )

    return Theorem7LocalSolveResult(
        success=bool(sol.success),
        status=int(sol.status),
        message=str(sol.message),
        method=method,
        tau_imag=float(tau_imag),
        s1=float(s1),
        symmetry_fold=int(symmetry_fold),
        target_ratio=float(target_ratio),
        initial_delta=float(initial_delta),
        initial_s2=float(initial_s2),
        delta=delta_sol,
        s2=s2_sol,
        residual_vector=residual_vector,
        residual_norm=float(np.linalg.norm(residual_vector)),
        nfev=int(sol.nfev),
        theorem7=theorem7_sol,
        pipeline=pipeline_sol,
    )


def continue_theorem7_branch_fixed_s1(tau_imag_values,
                                      s1: float,
                                      initial_delta: float,
                                      initial_s2: float,
                                      symmetry_fold: int,
                                      target_ratio: float = 1.0,
                                      method: str = 'hybr',
                                      evaluate_pipeline: bool = False,
                                      pipeline_n_half_samples: int = 151,
                                      pipeline_n_points: int = 160,
                                      stop_on_failure: bool = True) -> Theorem7ContinuationResult:
    """
    Predictor-corrector continuation in Im(tau) with fixed s1.

    Predictor:
    - first step: user-supplied seed `(initial_delta, initial_s2)`
    - second step: previous corrected solution
    - later steps: secant extrapolation in `(delta, s2)` versus `tau_imag`

    Corrector:
    - `solve_theorem7_local_fixed_tau_s1(...)`
    """
    tau_values = np.asarray(list(tau_imag_values), dtype=float)
    if tau_values.ndim != 1 or len(tau_values) == 0:
        raise ValueError("tau_imag_values must be a non-empty 1D sequence")

    steps: list[Theorem7ContinuationStep] = []
    previous_result: Theorem7LocalSolveResult | None = None
    previous_previous_result: Theorem7LocalSolveResult | None = None
    failed_index: int | None = None
    message = 'continuation completed successfully'

    for index, tau_imag in enumerate(tau_values):
        if previous_result is None:
            predictor_delta = float(initial_delta)
            predictor_s2 = float(initial_s2)
            used_secant = False
        elif previous_previous_result is None:
            predictor_delta = previous_result.delta
            predictor_s2 = previous_result.s2
            used_secant = False
        else:
            tau_prev = previous_result.tau_imag
            tau_prevprev = previous_previous_result.tau_imag
            dt_prev = tau_prev - tau_prevprev
            if abs(dt_prev) < 1e-14:
                predictor_delta = previous_result.delta
                predictor_s2 = previous_result.s2
                used_secant = False
            else:
                alpha = (tau_imag - tau_prev) / dt_prev
                predictor_delta = previous_result.delta + alpha * (previous_result.delta - previous_previous_result.delta)
                predictor_s2 = previous_result.s2 + alpha * (previous_result.s2 - previous_previous_result.s2)
                used_secant = True

        corrector = solve_theorem7_local_fixed_tau_s1(
            tau_imag=float(tau_imag),
            s1=s1,
            initial_delta=predictor_delta,
            initial_s2=predictor_s2,
            symmetry_fold=symmetry_fold,
            target_ratio=target_ratio,
            method=method,
            evaluate_pipeline=evaluate_pipeline,
            pipeline_n_half_samples=pipeline_n_half_samples,
            pipeline_n_points=pipeline_n_points,
        )
        steps.append(Theorem7ContinuationStep(
            index=int(index),
            tau_imag=float(tau_imag),
            predictor_delta=float(predictor_delta),
            predictor_s2=float(predictor_s2),
            used_secant_predictor=bool(used_secant),
            corrector=corrector,
        ))

        if not corrector.success:
            failed_index = int(index)
            message = f'continuation failed at index {index} (tau_imag={tau_imag}): {corrector.message}'
            if stop_on_failure:
                break
            continue

        previous_previous_result = previous_result
        previous_result = corrector

    n_success = sum(1 for step in steps if step.corrector.success)
    success = failed_index is None

    return Theorem7ContinuationResult(
        success=bool(success),
        message=message,
        method=method,
        s1=float(s1),
        symmetry_fold=int(symmetry_fold),
        target_ratio=float(target_ratio),
        tau_values=tau_values,
        steps=steps,
        n_success=int(n_success),
        failed_index=failed_index,
    )


def continue_theorem7_branch_adaptive_fixed_s1(start_tau_imag: float,
                                               end_tau_imag: float,
                                               s1: float,
                                               initial_delta: float,
                                               initial_s2: float,
                                               symmetry_fold: int,
                                               target_ratio: float = 1.0,
                                               initial_step: float = 0.002,
                                               min_step: float = 0.0005,
                                               max_step: float = 0.004,
                                               step_growth: float = 1.25,
                                               step_shrink: float = 0.5,
                                               method: str = 'hybr',
                                               evaluate_pipeline: bool = False,
                                               pipeline_n_half_samples: int = 151,
                                               pipeline_n_points: int = 160,
                                               formula_tol: float = 1e-6,
                                               jacobian_det_tol: float = 1e-8,
                                               jacobian_max_condition: float = 1e8,
                                               max_steps: int = 100) -> Theorem7ContinuationResult:
    """
    Adaptive predictor-corrector continuation in Im(tau) with fixed s1.

    The step size is reduced when:
    - the corrector fails
    - the Lemma 8-inspired non-degeneracy monitor becomes unsafe

    and can grow again in well-conditioned regions.
    """
    start_tau = float(start_tau_imag)
    end_tau = float(end_tau_imag)
    if abs(end_tau - start_tau) < 1e-15:
        raise ValueError("start_tau_imag and end_tau_imag must differ")
    if initial_step <= 0 or min_step <= 0 or max_step <= 0:
        raise ValueError("step sizes must be positive")

    direction = 1.0 if end_tau > start_tau else -1.0
    step_size = min(max(float(initial_step), float(min_step)), float(max_step))
    current_tau = start_tau
    current_delta = float(initial_delta)
    current_s2 = float(initial_s2)
    steps: list[Theorem7ContinuationStep] = []
    previous_result: Theorem7LocalSolveResult | None = None
    previous_previous_result: Theorem7LocalSolveResult | None = None
    failed_index: int | None = None
    message = 'adaptive continuation completed successfully'

    for index in range(max_steps):
        if direction * (end_tau - current_tau) <= 1e-14:
            break

        remaining = abs(end_tau - current_tau)
        local_step = min(step_size, remaining)
        tau_candidate = current_tau + direction * local_step
        retry_count = 0

        while True:
            if previous_result is None:
                predictor_delta = current_delta
                predictor_s2 = current_s2
                used_secant = False
            elif previous_previous_result is None:
                predictor_delta = previous_result.delta
                predictor_s2 = previous_result.s2
                used_secant = False
            else:
                tau_prev = previous_result.tau_imag
                tau_prevprev = previous_previous_result.tau_imag
                dt_prev = tau_prev - tau_prevprev
                if abs(dt_prev) < 1e-14:
                    predictor_delta = previous_result.delta
                    predictor_s2 = previous_result.s2
                    used_secant = False
                else:
                    alpha = (tau_candidate - tau_prev) / dt_prev
                    predictor_delta = previous_result.delta + alpha * (previous_result.delta - previous_previous_result.delta)
                    predictor_s2 = previous_result.s2 + alpha * (previous_result.s2 - previous_previous_result.s2)
                    used_secant = True

            try:
                corrector = solve_theorem7_local_fixed_tau_s1(
                    tau_imag=tau_candidate,
                    s1=s1,
                    initial_delta=predictor_delta,
                    initial_s2=predictor_s2,
                    symmetry_fold=symmetry_fold,
                    target_ratio=target_ratio,
                    method=method,
                    evaluate_pipeline=evaluate_pipeline,
                    pipeline_n_half_samples=pipeline_n_half_samples,
                    pipeline_n_points=pipeline_n_points,
                )
            except (ValueError, RuntimeError) as exc:
                # Treat hard crash (e.g. ω bracket failure) as corrector failure
                corrector = None

            diagnostics = None
            if corrector is not None and corrector.success:
                diagnostics = theorem7_local_nondegeneracy_diagnostics(
                    tau_imag=corrector.tau_imag,
                    delta=corrector.delta,
                    s1=s1,
                    s2=corrector.s2,
                    symmetry_fold=symmetry_fold,
                    target_ratio=target_ratio,
                    formula_tol=formula_tol,
                    jacobian_det_tol=jacobian_det_tol,
                    jacobian_max_condition=jacobian_max_condition,
                )

            if corrector is not None and corrector.success and diagnostics is not None and diagnostics.safe:
                steps.append(Theorem7ContinuationStep(
                    index=len(steps),
                    tau_imag=float(tau_candidate),
                    predictor_delta=float(predictor_delta),
                    predictor_s2=float(predictor_s2),
                    used_secant_predictor=bool(used_secant),
                    corrector=corrector,
                    accepted_step_size=float(local_step),
                    retry_count=int(retry_count),
                    nondegeneracy=diagnostics,
                ))
                previous_previous_result = previous_result
                previous_result = corrector
                current_tau = tau_candidate
                current_delta = corrector.delta
                current_s2 = corrector.s2

                # Conservative growth only in numerically comfortable regions.
                if diagnostics.jacobian_condition < jacobian_max_condition / 50.0:
                    step_size = min(max_step, local_step * step_growth)
                else:
                    step_size = local_step
                break

            # Failed or numerically unsafe step: try a smaller step.
            local_step *= step_shrink
            retry_count += 1
            if local_step < min_step:
                failed_index = len(steps)
                if corrector is not None and corrector.success and diagnostics is not None and not diagnostics.safe:
                    message = (
                        f'adaptive continuation stopped near tau_imag={tau_candidate}: '
                        'non-degeneracy monitor became unsafe'
                    )
                else:
                    message = (
                        f'adaptive continuation failed near tau_imag={tau_candidate}: '
                        f'corrector did not converge before min_step={min_step}'
                    )
                step_size = local_step
                break

            tau_candidate = current_tau + direction * local_step

        if failed_index is not None:
            break
    else:
        message = f'adaptive continuation reached max_steps={max_steps} before end_tau_imag'
        failed_index = len(steps)

    success = failed_index is None and direction * (end_tau - current_tau) <= 1e-14
    if success:
        message = 'adaptive continuation completed successfully'

    tau_values = np.array([start_tau] + [step.corrector.tau_imag for step in steps], dtype=float)
    return Theorem7ContinuationResult(
        success=bool(success),
        message=message,
        method=method,
        s1=float(s1),
        symmetry_fold=int(symmetry_fold),
        target_ratio=float(target_ratio),
        tau_values=tau_values,
        steps=steps,
        n_success=len(steps),
        failed_index=failed_index,
        adaptive=True,
        start_tau_imag=float(start_tau),
        end_tau_imag=float(end_tau),
        final_step_size=float(step_size),
    )


def build_theorem7_branch_catalog(seed_specs: list[Theorem7BranchSeed],
                                  tau_radius: float = 0.004,
                                  initial_step: float = 0.002,
                                  min_step: float = 0.0005,
                                  max_step: float = 0.004,
                                  evaluate_pipeline: bool = False,
                                  pipeline_n_half_samples: int = 151,
                                  pipeline_n_points: int = 160) -> Theorem7BranchCatalog:
    """Build a small local branch catalog around a list of trusted seeds."""
    entries: list[Theorem7BranchCatalogEntry] = []
    for seed in seed_specs:
        backward = continue_theorem7_branch_adaptive_fixed_s1(
            start_tau_imag=seed.tau_imag,
            end_tau_imag=seed.tau_imag - tau_radius,
            s1=seed.s1,
            initial_delta=seed.delta,
            initial_s2=seed.s2,
            symmetry_fold=seed.symmetry_fold,
            target_ratio=seed.target_ratio,
            initial_step=initial_step,
            min_step=min_step,
            max_step=max_step,
            evaluate_pipeline=evaluate_pipeline,
            pipeline_n_half_samples=pipeline_n_half_samples,
            pipeline_n_points=pipeline_n_points,
        )
        forward = continue_theorem7_branch_adaptive_fixed_s1(
            start_tau_imag=seed.tau_imag,
            end_tau_imag=seed.tau_imag + tau_radius,
            s1=seed.s1,
            initial_delta=seed.delta,
            initial_s2=seed.s2,
            symmetry_fold=seed.symmetry_fold,
            target_ratio=seed.target_ratio,
            initial_step=initial_step,
            min_step=min_step,
            max_step=max_step,
            evaluate_pipeline=evaluate_pipeline,
            pipeline_n_half_samples=pipeline_n_half_samples,
            pipeline_n_points=pipeline_n_points,
        )
        entries.append(Theorem7BranchCatalogEntry(
            seed=seed,
            backward=backward,
            forward=forward,
        ))

    return Theorem7BranchCatalog(entries=entries)
