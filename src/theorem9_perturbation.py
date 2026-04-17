"""
Theorem 9 perturbative closure solver.

This module implements the first constructive step beyond the spherical-v-curves family:
starting from a closed Theorem 7 seed, we perturb the angle function

    psi(v) = psi0(v) + alpha1 * phi1(v) + alpha2 * phi2(v) + alpha3 * phi3(v) + eps * chi(v)

and solve the three periodicity conditions from Theorem 6 / Section 7.3 numerically:

    theta = theta0,
    b = <A, ∫ e^{-h} n dv> = 0,
    c = ∫ cos psi(v) dv = 0.

The implementation is intentionally local and low-dimensional. It is meant to provide a
practical, testable perturbative corrector before introducing broader functional freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, root

from . import quaternion_ops as Q
from . import theta_functions as TF
from .frame_integrator import integrate_frame, FrameResult
from .theorem7_periodicity import theorem7_residuals, theorem7_w_functions


@dataclass
class Theorem9SeedState:
    tau_imag: float
    tau: complex
    omega: float
    delta: float
    s1: float
    s2: float
    symmetry_fold: int
    target_theta: float
    target_ratio: float
    period: float
    v_values: np.ndarray
    w0_values: np.ndarray
    w0_prime_values: np.ndarray
    speed0_values: np.ndarray
    psi0_values: np.ndarray
    w0_start: float
    admissible_w_min: float
    admissible_w_max: float
    lookup: "Theorem9Lookup"


@dataclass
class Theorem9Lookup:
    w_grid: np.ndarray
    W1_re_spline: CubicSpline
    W1_im_spline: CubicSpline
    sigma_re_spline: CubicSpline
    sigma_im_spline: CubicSpline
    s_spline: CubicSpline

    def W1(self, w):
        return self.W1_re_spline(w) + 1j * self.W1_im_spline(w)

    def e_sigma(self, w):
        return self.sigma_re_spline(w) + 1j * self.sigma_im_spline(w)

    def s(self, w):
        return self.s_spline(w)


@dataclass
class Theorem9Basis:
    labels: list[str]
    values: np.ndarray         # shape (3, N)
    driver_label: str
    driver_values: np.ndarray  # shape (N,)


@dataclass
class Theorem9DriverSpec:
    label: str
    harmonic: int
    phase: float


@dataclass
class Theorem9ForcingSpec:
    label: str
    family: str
    harmonic: int
    phase: float
    amplitude: float = 1.0


@dataclass
class Theorem9PerturbationEvaluation:
    alpha: np.ndarray
    epsilon: float
    theta: float
    ratio: float
    b_scalar: float
    c_scalar: float
    residual_vector: np.ndarray
    w_values: np.ndarray
    w_prime_values: np.ndarray
    speed_values: np.ndarray
    frame: FrameResult
    axis: np.ndarray
    w_min: float
    w_max: float


@dataclass
class Theorem9Linearization:
    jacobian: np.ndarray
    determinant: float
    condition_number: float


@dataclass
class Theorem9PerturbationSolveResult:
    success: bool
    status: int
    message: str
    method: str
    epsilon: float
    alpha: np.ndarray
    residual_vector: np.ndarray
    residual_norm: float
    nfev: int
    evaluation: Theorem9PerturbationEvaluation
    linearization: Theorem9Linearization


@dataclass
class Theorem9ContinuationStep:
    index: int
    epsilon: float
    solve: Theorem9PerturbationSolveResult


@dataclass
class Theorem9ContinuationResult:
    success: bool
    message: str
    method: str
    eps_values: np.ndarray
    steps: list[Theorem9ContinuationStep]
    n_success: int


@dataclass
class Theorem9DriverScanEntry:
    spec: Theorem9DriverSpec
    epsilon: float
    basis: Theorem9Basis | None
    success: bool
    residual_norm: float
    alpha_norm: float
    alpha_predictor: np.ndarray | None
    residual_vector: np.ndarray | None
    error: str | None


@dataclass
class Theorem9DriverScanResult:
    epsilon: float
    entries: list[Theorem9DriverScanEntry]
    best_index: int | None


@dataclass
class Theorem9BestDriverContinuation:
    scan: Theorem9DriverScanResult
    basis: Theorem9Basis | None
    continuation: Theorem9ContinuationResult | None


@dataclass
class Theorem9RefinedScanEntry:
    screen_entry: Theorem9DriverScanEntry
    solve: Theorem9PerturbationSolveResult


@dataclass
class Theorem9RefinedScanResult:
    screen: Theorem9DriverScanResult
    refined_entries: list[Theorem9RefinedScanEntry]
    best_index: int | None


@dataclass
class Theorem9ForcingContinuationResult:
    forcing: Theorem9ForcingSpec
    basis: Theorem9Basis
    continuation: Theorem9ContinuationResult


@dataclass
class Theorem9BonnetVerification:
    seed: Theorem9SeedState
    solve: Theorem9PerturbationSolveResult
    torus: object
    pair: object
    isometry: dict
    mean_curvature: dict
    non_congruence: dict


def _complex_to_quat(z: complex) -> np.ndarray:
    return np.array([float(np.real(z)), float(np.imag(z)), 0.0, 0.0], dtype=float)


def _e_sigma_quaternion(w: float, omega: float, tau: complex, h: float = 1e-6) -> np.ndarray:
    """Unit complex number e^{i sigma(omega,w)} embedded in quaternions."""
    gamma_plus = TF.gamma_curve(omega + h, w, omega, tau)
    gamma_minus = TF.gamma_curve(omega - h, w, omega, tau)
    gamma_u = (gamma_plus - gamma_minus) / (2.0 * h)
    gamma_u /= abs(gamma_u)
    return _complex_to_quat(gamma_u)


def build_theorem9_lookup(tau: complex, omega: float,
                          w_min: float, w_max: float,
                          n_table: int = 1024) -> Theorem9Lookup:
    """Pre-tabulate expensive theta-based functions used in Theorem 9 evaluation."""
    w_grid = np.linspace(float(w_min), float(w_max), int(n_table))
    W1_vals = np.array([TF.W1_function(float(w), omega, tau) for w in w_grid], dtype=complex)
    sigma_vals = np.array([
        complex(*_e_sigma_quaternion(float(w), omega, tau)[1:3])
        for w in w_grid
    ], dtype=complex)
    s_vals = np.array([complex(TF.s_of_w(float(w), omega, tau)).real for w in w_grid], dtype=float)

    return Theorem9Lookup(
        w_grid=w_grid,
        W1_re_spline=CubicSpline(w_grid, W1_vals.real),
        W1_im_spline=CubicSpline(w_grid, W1_vals.imag),
        sigma_re_spline=CubicSpline(w_grid, sigma_vals.real),
        sigma_im_spline=CubicSpline(w_grid, sigma_vals.imag),
        s_spline=CubicSpline(w_grid, s_vals),
    )


def _integrate_frame_fast(seed: Theorem9SeedState,
                          w_values: np.ndarray,
                          w_prime_values: np.ndarray,
                          speed_values: np.ndarray,
                          n_points: int,
                          rtol: float,
                          atol: float) -> FrameResult:
    """Local fast frame integrator using theorem9 lookup tables instead of direct theta calls."""
    v_grid = seed.v_values
    w_spline = CubicSpline(v_grid, w_values)
    wp_spline = CubicSpline(v_grid, w_prime_values)
    speed_spline = CubicSpline(v_grid, speed_values)
    lookup = seed.lookup

    def rhs(v, phi):
        w_val = float(w_spline(v))
        speed = float(speed_spline(v))
        W1_val = lookup.W1(w_val)
        W1k = np.array([0.0, 0.0, -float(np.imag(W1_val)), float(np.real(W1_val))], dtype=float)
        return speed * Q.qmul(W1k, phi)

    v_eval = np.linspace(float(v_grid[0]), float(v_grid[-1]), int(n_points))
    sol = solve_ivp(
        fun=rhs,
        t_span=(float(v_grid[0]), float(v_grid[-1])),
        y0=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        method='DOP853',
        t_eval=v_eval,
        rtol=rtol,
        atol=atol,
        max_step=float(v_grid[-1] - v_grid[0]) / max(int(n_points), 2),
    )
    if not sol.success:
        raise RuntimeError(f"fast theorem9 frame integration failed: {sol.message}")

    phi_values = sol.y.T
    norms = np.linalg.norm(phi_values, axis=1, keepdims=True)
    phi_values = phi_values / norms
    unitarity_errors = np.abs(np.linalg.norm(phi_values, axis=1) - 1.0)
    monodromy = Q.qmul(phi_values[-1], Q.qinv(phi_values[0]))
    cos_half = np.clip(monodromy[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(abs(cos_half))

    return FrameResult(
        v_values=sol.t,
        phi_values=phi_values,
        unitarity_errors=unitarity_errors,
        monodromy=monodromy,
        rotation_angle=float(theta),
        n_renormalizations=0,
    )


def _weighted_axial_scalar(frame: FrameResult,
                           omega: float,
                           tau: complex,
                           w_values: np.ndarray,
                           w_prime_values: np.ndarray,
                           speed_values: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute b = <A, ∫ e^{-h} n dv> from Theorem 6 / Section 7.3."""
    axis = frame.monodromy[1:4].astype(float).copy()
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-14:
        raise ValueError("monodromy axis is degenerate")
    axis /= axis_norm

    i_quat = Q.quat_i()
    k_quat = Q.quat_k()
    density = np.zeros_like(frame.v_values, dtype=float)

    for idx, (w, wp, speed, phi) in enumerate(zip(w_values, w_prime_values, speed_values, frame.phi_values)):
        phi_u = phi / np.linalg.norm(phi)
        phi_inv = Q.qinv(phi_u)
        e_sigma = _e_sigma_quaternion(float(w), omega, tau)
        n_hat = wp * i_quat - speed * Q.qmul(e_sigma, k_quat)
        n_vec = Q.qmul(Q.qmul(phi_inv, n_hat), phi_u)[1:4]
        s_val = float(complex(TF.s_of_w(float(w), omega, tau)).real)
        density[idx] = s_val * float(np.dot(axis, n_vec))

    b_scalar = float(cumulative_simpson(density, x=frame.v_values)[-1])
    return b_scalar, axis


def build_theorem9_seed_from_theorem7(tau_imag: float, delta: float,
                                      s1: float, s2: float,
                                      symmetry_fold: int,
                                      n_half_samples: int = 101) -> Theorem9SeedState:
    """Build a perturbative seed state from a verified Theorem 7 example."""
    tau = 0.5 + 1j * tau_imag
    omega = TF.find_critical_omega(tau)
    theorem7 = theorem7_residuals(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        symmetry_fold=symmetry_fold,
        target_ratio=1.0,
    )
    _, _, _, profile = theorem7_w_functions(
        tau_imag=tau_imag,
        delta=delta,
        s1=s1,
        s2=s2,
        n_half_samples=n_half_samples,
    )
    psi0 = np.unwrap(np.arctan2(profile.signed_speed_values, profile.w_prime_values))
    w_max = 2.0 * math.pi * tau.imag
    lookup = build_theorem9_lookup(
        tau=tau,
        omega=omega,
        w_min=1e-4,
        w_max=float(w_max - 1e-4),
    )

    return Theorem9SeedState(
        tau_imag=float(tau_imag),
        tau=tau,
        omega=float(omega),
        delta=float(delta),
        s1=float(s1),
        s2=float(s2),
        symmetry_fold=int(symmetry_fold),
        target_theta=float(theorem7.theta),
        target_ratio=float(theorem7.ratio),
        period=float(profile.period),
        v_values=profile.v_values.copy(),
        w0_values=profile.w_values.copy(),
        w0_prime_values=profile.w_prime_values.copy(),
        speed0_values=profile.signed_speed_values.copy(),
        psi0_values=psi0,
        w0_start=float(profile.w_values[0]),
        admissible_w_min=1e-4,
        admissible_w_max=float(w_max - 1e-4),
        lookup=lookup,
    )


def build_theorem9_basis(seed: Theorem9SeedState,
                         basis_harmonics=(1, 2, 3),
                         basis_phases=(0.11, 0.23, 0.37),
                         driver_harmonic: int = 4,
                         driver_phase: float = 0.19,
                         driver_label: str | None = None) -> Theorem9Basis:
    """
    Build a low-dimensional perturbation basis and a single forcing driver.

    The common factor `sin(2πv/V)` keeps the two zeroes of `sin(psi0(v))` fixed in the
    current theorem-7 parametrization, while the phase-shifted cosine factors break the
    accidental symmetries of plain Fourier modes and yield a well-conditioned 3x3
    linearization in practice.
    """
    if len(tuple(basis_harmonics)) != 3 or len(tuple(basis_phases)) != 3:
        raise ValueError("basis_harmonics and basis_phases must have length 3")

    basis_harmonics = tuple(int(h) for h in basis_harmonics)
    basis_phases = tuple(float(p) for p in basis_phases)
    x = seed.v_values / seed.period
    zero_factor = np.sin(2.0 * math.pi * x)
    basis = np.vstack([
        zero_factor * np.cos(2.0 * math.pi * harmonic * (x - phase))
        for harmonic, phase in zip(basis_harmonics, basis_phases)
    ])
    driver = zero_factor * np.cos(2.0 * math.pi * driver_harmonic * (x - driver_phase))
    if driver_label is None:
        driver_label = f'chi_h{driver_harmonic}_p{driver_phase:.2f}'
    return Theorem9Basis(
        labels=['phi1', 'phi2', 'phi3'],
        values=basis,
        driver_label=driver_label,
        driver_values=driver,
    )


def build_default_theorem9_basis(seed: Theorem9SeedState) -> Theorem9Basis:
    """Repository default perturbation basis used by the first constructive tests."""
    return build_theorem9_basis(seed)


def build_default_theorem9_driver_specs() -> list[Theorem9DriverSpec]:
    """Small family of forcing drivers used to robustify the first perturbative scan."""
    return [
        Theorem9DriverSpec(label='chi_h4_p019', harmonic=4, phase=0.19),
        Theorem9DriverSpec(label='chi_h5_p013', harmonic=5, phase=0.13),
        Theorem9DriverSpec(label='chi_h6_p029', harmonic=6, phase=0.29),
        Theorem9DriverSpec(label='chi_h7_p041', harmonic=7, phase=0.41),
    ]


def build_default_theorem9_forcing_specs() -> list[Theorem9ForcingSpec]:
    """
    Small structured forcing library for 8.5 exploration.

    Families:
    - smooth_low_freq: gentle low-frequency perturbations
    - phase_shifted: same qualitative forcing with shifted phase
    - symmetry_aware: harmonics compatible with the seed periodic structure
    - anti_reflection: intended to break residual reflection symmetry more strongly
    """
    return [
        Theorem9ForcingSpec(label='smooth_h4_p019', family='smooth_low_freq', harmonic=4, phase=0.19),
        Theorem9ForcingSpec(label='smooth_h5_p013', family='smooth_low_freq', harmonic=5, phase=0.13),
        Theorem9ForcingSpec(label='phase_h6_p029', family='phase_shifted', harmonic=6, phase=0.29),
        Theorem9ForcingSpec(label='phase_h7_p041', family='phase_shifted', harmonic=7, phase=0.41),
        Theorem9ForcingSpec(label='sym_h8_p011', family='symmetry_aware', harmonic=8, phase=0.11),
        Theorem9ForcingSpec(label='sym_h10_p017', family='symmetry_aware', harmonic=10, phase=0.17),
        Theorem9ForcingSpec(label='anti_h9_p033', family='anti_reflection', harmonic=9, phase=0.33),
        Theorem9ForcingSpec(label='anti_h11_p027', family='anti_reflection', harmonic=11, phase=0.27),
    ]


def forcing_specs_to_driver_specs(specs: list[Theorem9ForcingSpec]) -> list[Theorem9DriverSpec]:
    return [
        Theorem9DriverSpec(label=spec.label, harmonic=spec.harmonic, phase=spec.phase)
        for spec in specs
    ]


def build_theorem9_basis_from_forcing(seed: Theorem9SeedState,
                                      forcing: Theorem9ForcingSpec,
                                      basis_harmonics=(1, 2, 3),
                                      basis_phases=(0.11, 0.23, 0.37)) -> Theorem9Basis:
    return build_theorem9_basis(
        seed=seed,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
        driver_harmonic=forcing.harmonic,
        driver_phase=forcing.phase,
        driver_label=forcing.label,
    )


def predict_theorem9_alpha_linear(seed: Theorem9SeedState,
                                  epsilon: float,
                                  basis: Theorem9Basis | None = None,
                                  linearization: Theorem9Linearization | None = None,
                                  n_points: int = 40) -> tuple[np.ndarray, Theorem9PerturbationEvaluation]:
    """
    Linear predictor for alpha using the Jacobian at epsilon = 0.

    This is the cheap screening stage: one low-resolution evaluation of F(0, epsilon)
    plus a 3x3 linear solve.
    """
    if basis is None:
        basis = build_default_theorem9_basis(seed)
    if linearization is None:
        linearization = linearize_theorem9_basis(seed, basis=basis, n_points=max(20, n_points))

    eval0 = evaluate_theorem9_perturbation(
        seed=seed,
        alpha=np.zeros(3, dtype=float),
        epsilon=epsilon,
        basis=basis,
        use_fast_frame=True,
        n_points=n_points,
    )
    alpha = -np.linalg.solve(linearization.jacobian, eval0.residual_vector)
    evaluation = evaluate_theorem9_perturbation(
        seed=seed,
        alpha=alpha,
        epsilon=epsilon,
        basis=basis,
        use_fast_frame=True,
        n_points=n_points,
    )
    return alpha, evaluation


def refine_theorem9_scan_result(seed: Theorem9SeedState,
                                scan: Theorem9DriverScanResult,
                                top_k: int = 1,
                                method: str = 'hybr',
                                n_points: int = 80,
                                linearization_n_points: int = 60) -> Theorem9RefinedScanResult:
    """Refine the best screening candidates with the accurate nonlinear corrector."""
    successful = [entry for entry in scan.entries if entry.success and entry.basis is not None and entry.alpha_predictor is not None]
    successful.sort(key=lambda entry: (entry.residual_norm, entry.alpha_norm))
    refined_entries: list[Theorem9RefinedScanEntry] = []

    for entry in successful[:max(1, int(top_k))]:
        solve = solve_theorem9_perturbation(
            seed=seed,
            epsilon=float(entry.epsilon),
            basis=entry.basis,
            initial_alpha=entry.alpha_predictor,
            method=method,
            n_points=n_points,
            linearization_n_points=linearization_n_points,
        )
        refined_entries.append(Theorem9RefinedScanEntry(screen_entry=entry, solve=solve))

    best_index = None
    if refined_entries:
        best_index = int(np.argmin([entry.solve.residual_norm for entry in refined_entries]))

    return Theorem9RefinedScanResult(
        screen=scan,
        refined_entries=refined_entries,
        best_index=best_index,
    )


def screen_then_refine_theorem9(seed: Theorem9SeedState,
                                epsilon: float,
                                driver_specs: list[Theorem9DriverSpec] | None = None,
                                basis_harmonics=(1, 2, 3),
                                basis_phases=(0.11, 0.23, 0.37),
                                top_k: int = 1,
                                method: str = 'hybr',
                                screen_n_points: int = 40,
                                screen_linearization_n_points: int = 30,
                                refine_n_points: int = 80,
                                refine_linearization_n_points: int = 60) -> Theorem9RefinedScanResult:
    """Two-stage perturbative workflow: cheap screening, then accurate refinement."""
    scan = scan_theorem9_driver_family(
        seed=seed,
        epsilon=epsilon,
        driver_specs=driver_specs,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
        method=method,
        n_points=screen_n_points,
        linearization_n_points=screen_linearization_n_points,
    )
    return refine_theorem9_scan_result(
        seed=seed,
        scan=scan,
        top_k=top_k,
        method=method,
        n_points=refine_n_points,
        linearization_n_points=refine_linearization_n_points,
    )


def scan_theorem9_driver_family(seed: Theorem9SeedState,
                                epsilon: float,
                                driver_specs: list[Theorem9DriverSpec] | None = None,
                                basis_harmonics=(1, 2, 3),
                                basis_phases=(0.11, 0.23, 0.37),
                                method: str = 'hybr',
                                n_points: int = 60,
                                linearization_n_points: int = 40) -> Theorem9DriverScanResult:
    """Try a small family of forcing drivers and rank them by residual quality."""
    if driver_specs is None:
        driver_specs = build_default_theorem9_driver_specs()

    reference_basis = build_theorem9_basis(
        seed,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
        driver_harmonic=driver_specs[0].harmonic,
        driver_phase=driver_specs[0].phase,
        driver_label=driver_specs[0].label,
    )
    shared_linearization = linearize_theorem9_basis(
        seed,
        basis=reference_basis,
        n_points=linearization_n_points,
    )

    entries: list[Theorem9DriverScanEntry] = []
    for spec in driver_specs:
        basis = build_theorem9_basis(
            seed,
            basis_harmonics=basis_harmonics,
            basis_phases=basis_phases,
            driver_harmonic=spec.harmonic,
            driver_phase=spec.phase,
            driver_label=spec.label,
        )
        try:
            alpha_predictor, evaluation = predict_theorem9_alpha_linear(
                seed=seed,
                epsilon=epsilon,
                basis=basis,
                linearization=shared_linearization,
                n_points=n_points,
            )
            entries.append(Theorem9DriverScanEntry(
                spec=spec,
                epsilon=float(epsilon),
                basis=basis,
                success=True,
                residual_norm=float(np.linalg.norm(evaluation.residual_vector)),
                alpha_norm=float(np.linalg.norm(alpha_predictor)),
                alpha_predictor=alpha_predictor,
                residual_vector=evaluation.residual_vector,
                error=None,
            ))
        except Exception as exc:
            entries.append(Theorem9DriverScanEntry(
                spec=spec,
                epsilon=float(epsilon),
                basis=basis,
                success=False,
                residual_norm=float('inf'),
                alpha_norm=float('inf'),
                alpha_predictor=None,
                residual_vector=None,
                error=str(exc),
            ))

    successful = [(idx, entry) for idx, entry in enumerate(entries) if entry.success and entry.alpha_predictor is not None]
    if not successful:
        return Theorem9DriverScanResult(epsilon=float(epsilon), entries=entries, best_index=None)

    best_index, _ = min(
        successful,
        key=lambda item: (item[1].residual_norm, item[1].alpha_norm, item[0]),
    )
    return Theorem9DriverScanResult(epsilon=float(epsilon), entries=entries, best_index=int(best_index))


def continue_theorem9_on_best_driver(seed: Theorem9SeedState,
                                     eps_values,
                                     driver_specs: list[Theorem9DriverSpec] | None = None,
                                     basis_harmonics=(1, 2, 3),
                                     basis_phases=(0.11, 0.23, 0.37),
                                     method: str = 'hybr',
                                     n_points: int = 60,
                                     linearization_n_points: int = 40) -> Theorem9BestDriverContinuation:
    """
    Pick the best forcing driver on the first epsilon and continue using that basis.
    """
    eps_values = np.asarray(list(eps_values), dtype=float)
    if eps_values.ndim != 1 or len(eps_values) == 0:
        raise ValueError("eps_values must be a non-empty 1D sequence")

    scan = scan_theorem9_driver_family(
        seed=seed,
        epsilon=float(eps_values[0]),
        driver_specs=driver_specs,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
        method=method,
        n_points=n_points,
        linearization_n_points=linearization_n_points,
    )
    if scan.best_index is None:
        return Theorem9BestDriverContinuation(scan=scan, basis=None, continuation=None)

    best_spec = scan.entries[scan.best_index].spec
    basis = build_theorem9_basis(
        seed,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
        driver_harmonic=best_spec.harmonic,
        driver_phase=best_spec.phase,
        driver_label=best_spec.label,
    )
    continuation = continue_theorem9_in_epsilon(
        seed=seed,
        eps_values=eps_values,
        basis=basis,
        method=method,
        n_points=n_points,
        linearization_n_points=linearization_n_points,
    )
    return Theorem9BestDriverContinuation(scan=scan, basis=basis, continuation=continuation)


def continue_theorem9_forcing(seed: Theorem9SeedState,
                              forcing: Theorem9ForcingSpec,
                              eps_values,
                              basis_harmonics=(1, 2, 3),
                              basis_phases=(0.11, 0.23, 0.37),
                              method: str = 'hybr',
                              n_points: int = 60,
                              linearization_n_points: int = 40) -> Theorem9ForcingContinuationResult:
    """Continuation in epsilon for one explicitly chosen forcing spec."""
    basis = build_theorem9_basis_from_forcing(
        seed=seed,
        forcing=forcing,
        basis_harmonics=basis_harmonics,
        basis_phases=basis_phases,
    )
    continuation = continue_theorem9_in_epsilon(
        seed=seed,
        eps_values=eps_values,
        basis=basis,
        method=method,
        n_points=n_points,
        linearization_n_points=linearization_n_points,
    )
    return Theorem9ForcingContinuationResult(
        forcing=forcing,
        basis=basis,
        continuation=continuation,
    )


def evaluate_theorem9_perturbation(seed: Theorem9SeedState,
                                   alpha: np.ndarray,
                                   epsilon: float,
                                   basis: Theorem9Basis | None = None,
                                   use_fast_frame: bool = False,
                                   n_points: int = 120,
                                   rtol: float = 1e-9,
                                   atol: float = 1e-11) -> Theorem9PerturbationEvaluation:
    """Evaluate the three periodicity conditions (theta, b, c) for a perturbation."""
    if basis is None:
        basis = build_default_theorem9_basis(seed)

    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (3,):
        raise ValueError("alpha must have shape (3,)")

    psi = seed.psi0_values.copy()
    for coeff, values in zip(alpha, basis.values):
        psi = psi + coeff * values
    psi = psi + float(epsilon) * basis.driver_values

    w_prime_values = np.cos(psi)
    speed_values = np.sin(psi)
    w_values = np.zeros_like(seed.v_values)
    w_values[0] = seed.w0_start
    w_values[1:] = seed.w0_start + cumulative_simpson(w_prime_values, x=seed.v_values)

    w_min = float(np.min(w_values))
    w_max = float(np.max(w_values))
    if w_min <= seed.admissible_w_min or w_max >= seed.admissible_w_max:
        raise ValueError(f"perturbed w(v) left admissible interval: [{w_min}, {w_max}]")

    w_spline = CubicSpline(seed.v_values, w_values)
    wp_spline = CubicSpline(seed.v_values, w_prime_values)
    speed_spline = CubicSpline(seed.v_values, speed_values)

    if use_fast_frame:
        frame = _integrate_frame_fast(
            seed=seed,
            w_values=w_values,
            w_prime_values=w_prime_values,
            speed_values=speed_values,
            n_points=n_points,
            rtol=rtol,
            atol=atol,
        )
    else:
        frame = integrate_frame(
            w_func=lambda v: float(w_spline(v)),
            w_prime_func=lambda v: float(wp_spline(v)),
            omega=seed.omega,
            tau=seed.tau,
            v_span=(float(seed.v_values[0]), float(seed.v_values[-1])),
            speed_func=lambda v: float(speed_spline(v)),
            n_points=n_points,
            rtol=rtol,
            atol=atol,
        )

    w_eval = np.asarray(w_spline(frame.v_values), dtype=float)
    wp_eval = np.asarray(wp_spline(frame.v_values), dtype=float)
    speed_eval = np.asarray(speed_spline(frame.v_values), dtype=float)
    b_scalar, axis = _weighted_axial_scalar(
        frame=frame,
        omega=seed.omega,
        tau=seed.tau,
        w_values=w_eval,
        w_prime_values=wp_eval,
        speed_values=speed_eval,
    )
    c_scalar = float(cumulative_simpson(w_prime_values, x=seed.v_values)[-1])
    ratio = seed.symmetry_fold * frame.rotation_angle / (2.0 * math.pi)
    residual_vector = np.array([
        frame.rotation_angle - seed.target_theta,
        b_scalar,
        c_scalar,
    ], dtype=float)

    return Theorem9PerturbationEvaluation(
        alpha=alpha,
        epsilon=float(epsilon),
        theta=float(frame.rotation_angle),
        ratio=float(ratio),
        b_scalar=float(b_scalar),
        c_scalar=float(c_scalar),
        residual_vector=residual_vector,
        w_values=w_values,
        w_prime_values=w_prime_values,
        speed_values=speed_values,
        frame=frame,
        axis=axis,
        w_min=w_min,
        w_max=w_max,
    )


def linearize_theorem9_basis(seed: Theorem9SeedState,
                             basis: Theorem9Basis | None = None,
                             alpha_step: float = 1e-4,
                             n_points: int = 100) -> Theorem9Linearization:
    """Finite-difference Jacobian of (theta-theta0, b, c) with respect to alpha."""
    if basis is None:
        basis = build_default_theorem9_basis(seed)

    jacobian = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        direction = np.zeros(3, dtype=float)
        direction[idx] = alpha_step
        plus = evaluate_theorem9_perturbation(seed, direction, epsilon=0.0, basis=basis, n_points=n_points)
        minus = evaluate_theorem9_perturbation(seed, -direction, epsilon=0.0, basis=basis, n_points=n_points)
        jacobian[:, idx] = (plus.residual_vector - minus.residual_vector) / (2.0 * alpha_step)

    return Theorem9Linearization(
        jacobian=jacobian,
        determinant=float(np.linalg.det(jacobian)),
        condition_number=float(np.linalg.cond(jacobian)),
    )


def solve_theorem9_perturbation(seed: Theorem9SeedState,
                                epsilon: float,
                                basis: Theorem9Basis | None = None,
                                initial_alpha: np.ndarray | None = None,
                                linearization: Theorem9Linearization | None = None,
                                method: str = 'hybr',
                                n_points: int = 120,
                                linearization_n_points: int = 100) -> Theorem9PerturbationSolveResult:
    """Solve the Theorem 9 perturbative corrector for a fixed small epsilon."""
    if basis is None:
        basis = build_default_theorem9_basis(seed)

    if linearization is None:
        linearization = linearize_theorem9_basis(seed, basis=basis, n_points=linearization_n_points)
    if not np.isfinite(linearization.determinant) or abs(linearization.determinant) < 1e-10:
        raise ValueError("default perturbation basis is numerically degenerate at this seed")

    x0 = np.zeros(3, dtype=float) if initial_alpha is None else np.asarray(initial_alpha, dtype=float)
    if x0.shape != (3,):
        raise ValueError("initial_alpha must have shape (3,)")

    # Linear predictor from the Jacobian at epsilon = 0.
    try:
        eval0 = evaluate_theorem9_perturbation(seed, x0, epsilon, basis=basis, n_points=max(40, n_points // 2))
        correction = np.linalg.solve(linearization.jacobian, eval0.residual_vector)
        x0 = x0 - correction
    except Exception:
        pass

    def residuals(alpha: np.ndarray) -> np.ndarray:
        try:
            evaluation = evaluate_theorem9_perturbation(seed, alpha, epsilon, basis=basis, n_points=n_points)
            return evaluation.residual_vector
        except Exception:
            scale = 1.0 + np.linalg.norm(alpha)
            return np.array([1e3 * scale, 1e3 * scale, 1e3 * scale], dtype=float)

    sol = root(residuals, x0, method=method)
    x_sol = np.asarray(sol.x, dtype=float)
    success = bool(sol.success)
    status = int(sol.status)
    message = str(sol.message)
    nfev = int(sol.nfev)

    if not success:
        ls = least_squares(residuals, x0, method='lm', max_nfev=100)
        if ls.success or np.linalg.norm(ls.fun) < np.linalg.norm(residuals(x_sol)):
            x_sol = np.asarray(ls.x, dtype=float)
            success = bool(ls.success) or np.linalg.norm(ls.fun) < 1e-6
            status = int(ls.status)
            message = str(ls.message)
            nfev = int(ls.nfev)

    evaluation = evaluate_theorem9_perturbation(seed, x_sol, epsilon, basis=basis, n_points=n_points)
    return Theorem9PerturbationSolveResult(
        success=bool(success),
        status=int(status),
        message=message,
        method=method,
        epsilon=float(epsilon),
        alpha=x_sol,
        residual_vector=evaluation.residual_vector,
        residual_norm=float(np.linalg.norm(evaluation.residual_vector)),
        nfev=nfev,
        evaluation=evaluation,
        linearization=linearization,
    )


def continue_theorem9_in_epsilon(seed: Theorem9SeedState,
                                 eps_values,
                                 basis: Theorem9Basis | None = None,
                                 method: str = 'hybr',
                                 n_points: int = 120,
                                 linearization_n_points: int = 100) -> Theorem9ContinuationResult:
    """Simple continuation in epsilon using the previous alpha as predictor."""
    eps_values = np.asarray(list(eps_values), dtype=float)
    if eps_values.ndim != 1 or len(eps_values) == 0:
        raise ValueError("eps_values must be a non-empty 1D sequence")

    if basis is None:
        basis = build_default_theorem9_basis(seed)
    linearization = linearize_theorem9_basis(
        seed,
        basis=basis,
        n_points=linearization_n_points,
    )

    steps: list[Theorem9ContinuationStep] = []
    alpha = np.zeros(3, dtype=float)
    message = 'epsilon continuation completed successfully'

    for index, epsilon in enumerate(eps_values):
        solve = solve_theorem9_perturbation(
            seed=seed,
            epsilon=float(epsilon),
            basis=basis,
            initial_alpha=alpha,
            linearization=linearization,
            method=method,
            n_points=n_points,
            linearization_n_points=linearization_n_points,
        )
        steps.append(Theorem9ContinuationStep(
            index=int(index),
            epsilon=float(epsilon),
            solve=solve,
        ))
        if not solve.success:
            message = f'epsilon continuation failed at epsilon={epsilon}: {solve.message}'
            return Theorem9ContinuationResult(
                success=False,
                message=message,
                method=method,
                eps_values=eps_values,
                steps=steps,
                n_success=sum(1 for step in steps if step.solve.success),
            )
        alpha = solve.alpha

    return Theorem9ContinuationResult(
        success=True,
        message=message,
        method=method,
        eps_values=eps_values,
        steps=steps,
        n_success=len(steps),
    )


def build_theorem9_torus_parameters(seed: Theorem9SeedState,
                                    evaluation: Theorem9PerturbationEvaluation,
                                    u_res: int = 20,
                                    v_res: int = 120):
    """Convert a perturbative evaluation into TorusParameters for the main pipeline."""
    from .isothermic_torus import TorusParameters

    w_values = evaluation.w_values.copy()
    w_prime_values = evaluation.w_prime_values.copy()
    speed_values = evaluation.speed_values.copy()

    # The corrector enforces periodicity only up to solver tolerance; make the endpoint
    # exactly periodic so CubicSpline can use periodic boundary conditions robustly.
    w_values[-1] = w_values[0]
    w_prime_values[-1] = w_prime_values[0]
    speed_values[-1] = speed_values[0]

    w_spline = CubicSpline(seed.v_values, w_values, bc_type='periodic')
    wp_spline = CubicSpline(seed.v_values, w_prime_values, bc_type='periodic')
    speed_spline = CubicSpline(seed.v_values, speed_values, bc_type='periodic')

    def w_func(v: float) -> float:
        return float(w_spline(v % seed.period))

    def w_prime_func(v: float) -> float:
        return float(wp_spline(v % seed.period))

    def speed_func(v: float) -> float:
        return float(speed_spline(v % seed.period))

    return TorusParameters(
        tau_imag=seed.tau_imag,
        u_res=u_res,
        v_res=v_res,
        v_periods=1,
        symmetry_fold=seed.symmetry_fold,
        w_func=w_func,
        w_prime_func=w_prime_func,
        speed_func=speed_func,
        v_period=seed.period,
    )


def verify_theorem9_bonnet_pipeline(seed: Theorem9SeedState,
                                    solve: Theorem9PerturbationSolveResult,
                                    epsilon_geom: float = 0.3,
                                    u_res: int = 8,
                                    v_res: int = 80) -> Theorem9BonnetVerification:
    """Run a perturbatively corrected example through torus and Bonnet pair pipeline."""
    from .isothermic_torus import compute_torus
    from .bonnet_pair import (
        compute_bonnet_pair,
        verify_isometry,
        verify_mean_curvature,
        verify_non_congruence,
    )

    params = build_theorem9_torus_parameters(
        seed=seed,
        evaluation=solve.evaluation,
        u_res=u_res,
        v_res=v_res,
    )
    torus = compute_torus(params)
    pair = compute_bonnet_pair(torus, epsilon=epsilon_geom)
    isometry = verify_isometry(pair)
    mean_curvature = verify_mean_curvature(pair)
    non_congruence = verify_non_congruence(pair)

    return Theorem9BonnetVerification(
        seed=seed,
        solve=solve,
        torus=torus,
        pair=pair,
        isometry=isometry,
        mean_curvature=mean_curvature,
        non_congruence=non_congruence,
    )
