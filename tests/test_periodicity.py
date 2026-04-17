"""Tests for the Theorem 7 periodicity layer."""
import os
import sys
import types
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# The Bonnet scripts reuse core utilities via package imports that also expose
# optional TDA dependencies. Stub them here so this focused test can run even if
# gudhi / persim are not installed in the current environment.
if 'gudhi' not in sys.modules:
    gudhi_stub = types.ModuleType('gudhi')

    class _GudhiUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('gudhi is not available in this test environment')

    gudhi_stub.AlphaComplex = _GudhiUnavailable
    gudhi_stub.RipsComplex = _GudhiUnavailable
    sys.modules['gudhi'] = gudhi_stub

if 'persim' not in sys.modules:
    persim_stub = types.ModuleType('persim')

    def _persim_unavailable(*args, **kwargs):
        raise RuntimeError('persim is not available in this test environment')

    persim_stub.wasserstein = _persim_unavailable
    sys.modules['persim'] = persim_stub

from src import theta_functions as TF
from src.theorem7_periodicity import (
    continue_theorem7_branch_adaptive_fixed_s1,
    build_theorem7_torus_parameters,
    continue_theorem7_branch_fixed_s1,
    solve_theorem7_local_fixed_tau_s1,
    theorem7_local_nondegeneracy_diagnostics,
    theorem7_lemma3_axial_scalar,
    verify_theorem7_bonnet_pipeline,
    theorem7_pipeline_residuals,
    theorem7_Z0_squared,
    theorem7_Q,
    theorem7_Q2,
    theorem7_Qtilde2,
    theorem7_real_oval,
    theorem7_residuals,
    theorem7_w_functions,
    theorem7_w_profile,
)
from src.frame_integrator import integrate_frame
from src.isothermic_torus import compute_torus


TAU_PAPER = 0.5 + 0.3205128205j
THREEFOLD = {
    'tau_imag': 0.3205128205,
    'delta': 1.897366596,
    's1': -3.601381552,
    's2': 0.5965202011,
    'symmetry_fold': 3,
}
FOURFOLD = {
    'tau_imag': 0.3205128205,
    'delta': 1.61245155,
    's1': -3.13060628,
    's2': 0.5655771591,
    'symmetry_fold': 4,
}


class TestTheorem7ParameterLayer(unittest.TestCase):
    def test_parameter_functions_finite_at_paper_values(self):
        omega = TF.find_critical_omega(TAU_PAPER)
        z0_sq = theorem7_Z0_squared(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
        )
        q2 = theorem7_Q2(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
            s=THREEFOLD['s1'],
        )
        qtilde2 = theorem7_Qtilde2(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
            s=THREEFOLD['s1'],
        )

        self.assertTrue(np.isfinite(z0_sq))
        self.assertGreater(z0_sq, 0.0)
        self.assertTrue(np.isfinite(q2))
        self.assertTrue(np.isfinite(qtilde2))

    def test_real_oval_is_positive_interval_for_paper_3fold(self):
        omega = TF.find_critical_omega(TAU_PAPER)
        s_lo, s_hi = theorem7_real_oval(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
        )
        s_mid = 0.5 * (s_lo + s_hi)
        self.assertLess(s_lo, s_hi)
        self.assertGreater(theorem7_Q(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
            s_mid,
        ), 0.0)

    def test_s_of_w_starts_at_s0_and_is_increasing(self):
        omega = TF.find_critical_omega(TAU_PAPER)
        tau_imag = TAU_PAPER.imag
        s0 = complex(TF.theta1(omega, TAU_PAPER) ** 2 / TF.theta2(0, TAU_PAPER) ** 2).real

        w_values = np.linspace(1e-3, 2 * np.pi * tau_imag - 1e-3, 12)
        s_values = np.array([
            complex(TF.s_of_w(float(w), omega, TAU_PAPER)).real
            for w in w_values
        ])
        s_lo, _ = theorem7_real_oval(
            omega, TAU_PAPER,
            THREEFOLD['delta'], THREEFOLD['s1'], THREEFOLD['s2'],
        )

        self.assertAlmostEqual(s_values[0], s0, delta=1e-3)
        self.assertTrue(np.all(np.diff(s_values) > 0), f"s(w) not increasing: {s_values}")
        self.assertLess(s0, s_lo)


class TestTheorem7ResidualLayer(unittest.TestCase):
    def test_residuals_run_for_paper_3fold(self):
        res = theorem7_residuals(**THREEFOLD, target_ratio=1.0)
        self.assertTrue(np.isfinite(res.theta))
        self.assertTrue(np.isfinite(res.ratio))
        self.assertTrue(np.isfinite(res.rationality_residual))
        self.assertTrue(np.isfinite(res.axial_integral))
        self.assertLess(res.axial_residual, 1e-6)
        self.assertAlmostEqual(res.theta, 2 * np.pi / 3, delta=1e-6)
        self.assertAlmostEqual(res.ratio, 1.0, delta=1e-6)
        self.assertLess(res.rationality_residual, 1e-6)
        self.assertLess(res.s1_minus, res.s1_plus)

    def test_residuals_run_for_paper_4fold(self):
        res = theorem7_residuals(**FOURFOLD, target_ratio=1.0)
        self.assertTrue(np.isfinite(res.theta))
        self.assertTrue(np.isfinite(res.ratio))
        self.assertTrue(np.isfinite(res.rationality_residual))
        self.assertTrue(np.isfinite(res.axial_integral))
        self.assertLess(res.axial_residual, 1e-6)
        self.assertAlmostEqual(res.theta, np.pi / 2, delta=1e-6)
        self.assertAlmostEqual(res.ratio, 1.0, delta=1e-6)
        self.assertLess(res.rationality_residual, 1e-6)
        self.assertLess(res.s1_minus, res.s1_plus)


class TestTheorem7WProfile(unittest.TestCase):
    def test_w_profile_builds_for_paper_3fold(self):
        profile = theorem7_w_profile(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            n_half_samples=121,
        )
        s_back = np.array([
            complex(TF.s_of_w(float(w), profile.omega, TAU_PAPER)).real
            for w in profile.w_values
        ])

        self.assertGreater(profile.period, 0.0)
        self.assertAlmostEqual(profile.w_values[0], profile.w_values[-1], delta=1e-8)
        self.assertAlmostEqual(profile.s_values[0], profile.s_values[-1], delta=1e-8)
        self.assertGreater(np.min(profile.w_values), 0.0)
        self.assertLess(np.max(profile.w_values), 2 * np.pi * THREEFOLD['tau_imag'])
        self.assertLessEqual(np.max(np.abs(profile.w_prime_values)), 1.0 + 1e-6)
        self.assertLess(
            np.max(np.abs(profile.w_prime_values**2 + profile.signed_speed_values**2 - 1.0)),
            1e-6,
        )
        self.assertLess(np.max(np.abs(s_back - profile.s_values)), 1e-6)

    def test_signed_speed_recovers_frame_monodromy(self):
        omega = TF.find_critical_omega(TAU_PAPER)
        for params in (THREEFOLD, FOURFOLD):
            w_func, wp_func, speed_func, profile = theorem7_w_functions(
                tau_imag=params['tau_imag'],
                delta=params['delta'],
                s1=params['s1'],
                s2=params['s2'],
                n_half_samples=201,
            )
            frame = integrate_frame(
                w_func=w_func,
                w_prime_func=wp_func,
                omega=omega,
                tau=TAU_PAPER,
                v_span=(0.0, profile.period),
                speed_func=speed_func,
                n_points=220,
                rtol=1e-11,
                atol=1e-13,
            )
            ratio = params['symmetry_fold'] * frame.rotation_angle / (2 * np.pi)
            self.assertAlmostEqual(ratio, 1.0, delta=1e-2)

    def test_compute_torus_accepts_theorem7_period_and_speed(self):
        params, profile = build_theorem7_torus_parameters(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            u_res=8,
            v_res=120,
            n_half_samples=201,
        )
        torus = compute_torus(params)
        ratio = THREEFOLD['symmetry_fold'] * torus.frame_result.rotation_angle / (2 * np.pi)
        self.assertAlmostEqual(ratio, 1.0, delta=1e-2)

    def test_pipeline_bridge_stays_close_on_3fold_example(self):
        bridge = theorem7_pipeline_residuals(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            n_half_samples=151,
            n_points=160,
        )
        self.assertAlmostEqual(bridge.theorem7.ratio, 1.0, delta=1e-6)
        self.assertAlmostEqual(bridge.frame_ratio, 1.0, delta=1e-2)
        self.assertLess(bridge.lemma3_axial_abs, 1e-3)
        self.assertLess(bridge.lemma3_axial_abs, abs(bridge.axial_projection))

    def test_lemma3_scalar_is_small_on_paper_examples(self):
        for params in (THREEFOLD, FOURFOLD):
            scalar = theorem7_lemma3_axial_scalar(
                tau_imag=params['tau_imag'],
                delta=params['delta'],
                s1=params['s1'],
                s2=params['s2'],
                n_half_samples=151,
            )
            self.assertLess(abs(scalar), 2e-3)


class TestTheorem7LocalSolver(unittest.TestCase):
    def test_local_solver_recovers_3fold_from_nearby_seed(self):
        result = solve_theorem7_local_fixed_tau_s1(
            tau_imag=THREEFOLD['tau_imag'],
            s1=THREEFOLD['s1'],
            initial_delta=1.80,
            initial_s2=0.61,
            symmetry_fold=THREEFOLD['symmetry_fold'],
            target_ratio=1.0,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertAlmostEqual(result.delta, THREEFOLD['delta'], delta=1e-6)
        self.assertAlmostEqual(result.s2, THREEFOLD['s2'], delta=1e-6)
        self.assertLess(result.residual_norm, 1e-8)

    def test_local_solver_recovers_4fold_and_can_evaluate_pipeline(self):
        result = solve_theorem7_local_fixed_tau_s1(
            tau_imag=FOURFOLD['tau_imag'],
            s1=FOURFOLD['s1'],
            initial_delta=1.55,
            initial_s2=0.62,
            symmetry_fold=FOURFOLD['symmetry_fold'],
            target_ratio=1.0,
            evaluate_pipeline=True,
            pipeline_n_half_samples=151,
            pipeline_n_points=160,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertAlmostEqual(result.delta, FOURFOLD['delta'], delta=1e-6)
        self.assertAlmostEqual(result.s2, FOURFOLD['s2'], delta=1e-6)
        self.assertLess(result.residual_norm, 1e-8)
        self.assertIsNotNone(result.pipeline)
        self.assertAlmostEqual(result.pipeline.frame_ratio, 1.0, delta=1e-2)
        self.assertLess(result.pipeline.lemma3_axial_abs, 2e-3)


class TestTheorem7Continuation(unittest.TestCase):
    def test_continuation_tracks_3fold_branch_forward_in_tau(self):
        tau_values = [
            THREEFOLD['tau_imag'],
            THREEFOLD['tau_imag'] + 0.002,
            THREEFOLD['tau_imag'] + 0.004,
        ]
        result = continue_theorem7_branch_fixed_s1(
            tau_imag_values=tau_values,
            s1=THREEFOLD['s1'],
            initial_delta=THREEFOLD['delta'],
            initial_s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            target_ratio=1.0,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(len(result.steps), len(tau_values))
        self.assertTrue(all(step.corrector.success for step in result.steps))
        self.assertTrue(result.steps[2].used_secant_predictor)
        self.assertLess(result.steps[-1].corrector.delta, result.steps[0].corrector.delta)
        self.assertLess(result.steps[-1].corrector.s2, result.steps[0].corrector.s2)
        self.assertLess(result.steps[-1].corrector.residual_norm, 1e-8)

    def test_continuation_tracks_4fold_branch_backward_in_tau(self):
        tau_values = [
            FOURFOLD['tau_imag'],
            FOURFOLD['tau_imag'] - 0.002,
            FOURFOLD['tau_imag'] - 0.004,
        ]
        result = continue_theorem7_branch_fixed_s1(
            tau_imag_values=tau_values,
            s1=FOURFOLD['s1'],
            initial_delta=FOURFOLD['delta'],
            initial_s2=FOURFOLD['s2'],
            symmetry_fold=FOURFOLD['symmetry_fold'],
            target_ratio=1.0,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(result.n_success, len(tau_values))
        self.assertTrue(result.steps[2].used_secant_predictor)
        self.assertGreater(result.steps[-1].corrector.delta, result.steps[0].corrector.delta)
        self.assertGreater(result.steps[-1].corrector.s2, result.steps[0].corrector.s2)
        self.assertLess(result.steps[-1].corrector.residual_norm, 1e-8)


class TestTheorem7RobustContinuation(unittest.TestCase):
    def test_nondegeneracy_monitor_is_safe_on_paper_seed(self):
        diag = theorem7_local_nondegeneracy_diagnostics(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            target_ratio=1.0,
        )
        self.assertTrue(diag.safe)
        self.assertGreater(diag.lemma8_real_sqrt_expr, 0.0)
        self.assertGreater(abs(diag.lemma8_rationality_expr), 1e-3)
        self.assertGreater(abs(diag.lemma8_vanishing_expr), 1e-3)
        self.assertGreater(abs(diag.jacobian_det), 1e-6)
        self.assertLess(diag.jacobian_condition, 1e3)

    def test_adaptive_continuation_tracks_3fold_branch(self):
        result = continue_theorem7_branch_adaptive_fixed_s1(
            start_tau_imag=THREEFOLD['tau_imag'],
            end_tau_imag=THREEFOLD['tau_imag'] + 0.006,
            s1=THREEFOLD['s1'],
            initial_delta=THREEFOLD['delta'],
            initial_s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            target_ratio=1.0,
            initial_step=0.003,
            min_step=0.001,
            max_step=0.003,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertTrue(result.adaptive)
        self.assertGreaterEqual(result.n_success, 2)
        self.assertIsNotNone(result.final_step_size)
        self.assertTrue(all(step.nondegeneracy is not None and step.nondegeneracy.safe for step in result.steps))
        self.assertLess(result.steps[-1].corrector.delta, result.steps[0].corrector.delta)
        self.assertLess(result.steps[-1].corrector.s2, result.steps[0].corrector.s2)


class TestTheorem7BonnetEndToEnd(unittest.TestCase):
    def test_bonnet_pipeline_verification_on_3fold(self):
        verification = verify_theorem7_bonnet_pipeline(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            epsilon=0.3,
            u_res=8,
            v_res=80,
            n_half_samples=121,
        )
        self.assertAlmostEqual(
            verification.closure['k_theta_over_2pi'],
            1.0,
            delta=2e-2,
        )
        self.assertLess(verification.isometry['E_max_rel_err'], 1e-10)
        self.assertGreater(verification.non_congruence['procrustes_disparity'], 0.1)
        self.assertIsNotNone(verification.closure['paper_axial_scalar_abs'])
        self.assertLess(verification.closure['paper_axial_scalar_abs'], 2e-3)
        self.assertLess(
            verification.closure['paper_axial_scalar_abs'],
            verification.closure['axial_projection_abs'],
        )

    def test_bonnet_pipeline_verification_on_4fold(self):
        verification = verify_theorem7_bonnet_pipeline(
            tau_imag=FOURFOLD['tau_imag'],
            delta=FOURFOLD['delta'],
            s1=FOURFOLD['s1'],
            s2=FOURFOLD['s2'],
            symmetry_fold=FOURFOLD['symmetry_fold'],
            epsilon=0.3,
            u_res=8,
            v_res=80,
            n_half_samples=121,
        )
        self.assertAlmostEqual(
            verification.closure['k_theta_over_2pi'],
            1.0,
            delta=2e-2,
        )
        self.assertLess(verification.isometry['E_max_rel_err'], 1e-10)
        self.assertGreater(verification.non_congruence['procrustes_disparity'], 0.1)
        self.assertIsNotNone(verification.closure['paper_axial_scalar_abs'])
        self.assertLess(verification.closure['paper_axial_scalar_abs'], 3e-3)
        self.assertLess(
            verification.closure['paper_axial_scalar_abs'],
            verification.closure['axial_projection_abs'],
        )


if __name__ == '__main__':
    unittest.main()
