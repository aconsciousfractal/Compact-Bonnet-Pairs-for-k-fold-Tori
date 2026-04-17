"""Tests for the first constructive Theorem 9 perturbation step."""
import os
import sys
import types
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

from src.theorem9_perturbation import (
    Theorem9ForcingSpec,
    build_default_theorem9_forcing_specs,
    build_theorem9_basis_from_forcing,
    build_default_theorem9_basis,
    build_theorem9_seed_from_theorem7,
    continue_theorem9_forcing,
    verify_theorem9_bonnet_pipeline,
    continue_theorem9_in_epsilon,
    linearize_theorem9_basis,
    solve_theorem9_perturbation,
)


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

class TestTheorem9Perturbation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = build_theorem9_seed_from_theorem7(
            tau_imag=THREEFOLD['tau_imag'],
            delta=THREEFOLD['delta'],
            s1=THREEFOLD['s1'],
            s2=THREEFOLD['s2'],
            symmetry_fold=THREEFOLD['symmetry_fold'],
            n_half_samples=61,
        )
        cls.basis = build_default_theorem9_basis(cls.seed)

    def test_default_basis_linearization_is_invertible(self):
        linearization = linearize_theorem9_basis(
            self.seed,
            basis=self.basis,
            alpha_step=1e-4,
            n_points=80,
        )
        self.assertGreater(abs(linearization.determinant), 1e-3)
        self.assertLess(linearization.condition_number, 100.0)

    def test_default_forcing_library_has_expected_families(self):
        specs = build_default_theorem9_forcing_specs()
        self.assertEqual(len(specs), 8)
        families = {spec.family for spec in specs}
        self.assertEqual(
            families,
            {'smooth_low_freq', 'phase_shifted', 'symmetry_aware', 'anti_reflection'},
        )

    def test_basis_can_be_built_from_explicit_forcing_spec(self):
        forcing = Theorem9ForcingSpec(
            label='smooth_h4_p019',
            family='smooth_low_freq',
            harmonic=4,
            phase=0.19,
        )
        basis = build_theorem9_basis_from_forcing(self.seed, forcing)
        self.assertEqual(basis.driver_label, forcing.label)
        self.assertEqual(basis.values.shape[0], 3)
        self.assertEqual(len(basis.driver_values), len(self.seed.v_values))

    def test_local_perturbative_corrector_solves_small_forcing(self):
        result = solve_theorem9_perturbation(
            seed=self.seed,
            epsilon=5e-4,
            basis=self.basis,
            n_points=80,
            linearization_n_points=60,
        )
        self.assertTrue(result.success, msg=result.message)
        self.assertLess(result.residual_norm, 1e-6)
        self.assertLess(abs(result.evaluation.b_scalar), 1e-6)
        self.assertLess(abs(result.evaluation.c_scalar), 1e-6)
        self.assertGreater(np.linalg.norm(result.alpha), 1e-3)

    def test_epsilon_continuation_tracks_small_branch(self):
        continuation = continue_theorem9_in_epsilon(
            seed=self.seed,
            eps_values=[2.5e-4, 5e-4],
            basis=self.basis,
            n_points=50,
            linearization_n_points=40,
        )
        self.assertTrue(continuation.success, msg=continuation.message)
        self.assertEqual(continuation.n_success, 2)
        self.assertLess(continuation.steps[-1].solve.residual_norm, 1e-5)
        self.assertGreater(
            np.linalg.norm(continuation.steps[-1].solve.alpha - continuation.steps[0].solve.alpha),
            1e-5,
        )

    def test_perturbed_example_runs_through_bonnet_pipeline(self):
        solve = solve_theorem9_perturbation(
            seed=self.seed,
            epsilon=5e-4,
            basis=self.basis,
            n_points=80,
            linearization_n_points=60,
        )
        verification = verify_theorem9_bonnet_pipeline(
            seed=self.seed,
            solve=solve,
            epsilon_geom=0.3,
            u_res=8,
            v_res=80,
        )
        ratio = self.seed.symmetry_fold * verification.torus.frame_result.rotation_angle / (2 * np.pi)
        self.assertAlmostEqual(ratio, self.seed.target_ratio, delta=2e-2)
        self.assertLess(verification.isometry['E_max_rel_err'], 1e-10)
        self.assertGreater(verification.non_congruence['procrustes_disparity'], 0.1)
        self.assertLess(abs(solve.evaluation.b_scalar), 1e-6)
        self.assertLess(abs(solve.evaluation.c_scalar), 1e-6)

    def test_forcing_specific_continuation_on_3fold(self):
        forcing = Theorem9ForcingSpec(
            label='smooth_h4_p019',
            family='smooth_low_freq',
            harmonic=4,
            phase=0.19,
        )
        result = continue_theorem9_forcing(
            seed=self.seed,
            forcing=forcing,
            eps_values=[1e-4, 2.5e-4, 5e-4],
            n_points=50,
            linearization_n_points=40,
        )
        self.assertTrue(result.continuation.success, msg=result.continuation.message)
        self.assertEqual(result.continuation.n_success, 3)
        self.assertLess(result.continuation.steps[-1].solve.residual_norm, 1e-3)

    def test_forcing_specific_continuation_on_4fold(self):
        forcing = Theorem9ForcingSpec(
            label='smooth_h4_p019',
            family='smooth_low_freq',
            harmonic=4,
            phase=0.19,
        )
        seed4 = build_theorem9_seed_from_theorem7(
            tau_imag=FOURFOLD['tau_imag'],
            delta=FOURFOLD['delta'],
            s1=FOURFOLD['s1'],
            s2=FOURFOLD['s2'],
            symmetry_fold=FOURFOLD['symmetry_fold'],
            n_half_samples=61,
        )
        result = continue_theorem9_forcing(
            seed=seed4,
            forcing=forcing,
            eps_values=[2.5e-4, 5e-4],
            n_points=50,
            linearization_n_points=40,
        )
        self.assertTrue(result.continuation.success, msg=result.continuation.message)
        self.assertEqual(result.continuation.n_success, 2)
        self.assertLess(result.continuation.steps[-1].solve.residual_norm, 1e-5)


if __name__ == '__main__':
    unittest.main()
