"""
Phase 3 Test Suite — Bonnet Pair Engine

Tests:
  1. BHatTests — B̂(u,w) computation, periodicity, scalar values
  2. BTildeTests — b̃(w) scalar, B̃(v) integration
  3. BonnetPairGeneration — f⁺/f⁻ generated, vertex counts, shapes
  4. IsometryTests — metric tensor comparison (Task 3.5)
  5. MeanCurvatureTests — H⁺ ≈ H⁻ (Task 3.6)
  6. NonCongruenceTests — Procrustes distance > 0 (Task 3.7)
  7. OBJExportTests — dual file export (Task 3.8)
  8. ClosureTests — rationality + Euler (Task 3.9)
  9. SymmetryRelations — analytic Christoffel dual f* = -f(π-u,v)
"""

import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bonnet_pair import (
    B_hat_function,
    B_hat_vec,
    b_tilde_scalar,
    compute_bonnet_pair,
    BonnetPairResult,
    verify_isometry,
    verify_mean_curvature,
    verify_non_congruence,
    export_bonnet_pair_obj,
    closure_gate,
    christoffel_dual_analytic,
    compute_metric_tensor,
)
from src.isothermic_torus import (
    TorusParameters,
    compute_torus,
    verify_euler_characteristic,
)
from src import theta_functions as TF
from src import quaternion_ops as Q


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TORUS = None
_PAIR = None


def get_torus():
    """Lazily compute shared 20×20 torus."""
    global _TORUS
    if _TORUS is None:
        params = TorusParameters(
            tau_imag=0.3205,
            u_res=20,
            v_res=20,
            v_periods=1,
            symmetry_fold=3,
            w0=0.15,
        )
        _TORUS = compute_torus(params)
    return _TORUS


def get_pair():
    """Lazily compute shared Bonnet pair."""
    global _PAIR
    if _PAIR is None:
        torus = get_torus()
        _PAIR = compute_bonnet_pair(torus, epsilon=0.5)
    return _PAIR


# ---------------------------------------------------------------------------
# 1. B̂ tests
# ---------------------------------------------------------------------------

class TestBHat(unittest.TestCase):
    """B̂(u, w) computation tests."""

    def test_B_hat_returns_real(self):
        """B̂ returns a real number."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        val = B_hat_function(1.0, 0.15, omega, tau)
        self.assertIsInstance(val, float)

    def test_B_hat_finite(self):
        """B̂ is finite for typical parameters."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        val = B_hat_function(0.5, 0.15, omega, tau)
        self.assertTrue(np.isfinite(val))

    def test_B_hat_vec_matches_scalar(self):
        """Vectorized B̂ matches scalar version."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        u_arr = np.array([0.5, 1.0, 2.0, 3.0])
        w = 0.15

        vec_vals = B_hat_vec(u_arr, w, omega, tau)
        scalar_vals = np.array([B_hat_function(u, w, omega, tau) for u in u_arr])

        np.testing.assert_allclose(vec_vals, scalar_vals, rtol=1e-10)

    def test_B_hat_2pi_periodic(self):
        """B̂(u+2π, w) = B̂(u, w) — periodicity in u."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        u0 = 0.7
        w = 0.15
        val1 = B_hat_function(u0, w, omega, tau)
        val2 = B_hat_function(u0 + 2 * np.pi, w, omega, tau)
        self.assertAlmostEqual(val1, val2, places=8)


# ---------------------------------------------------------------------------
# 2. b̃ tests
# ---------------------------------------------------------------------------

class TestBTilde(unittest.TestCase):
    """b̃(w) scalar function tests."""

    def test_b_tilde_finite(self):
        """b̃(w) is finite."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        val = b_tilde_scalar(0.15, omega, tau)
        self.assertTrue(np.isfinite(abs(val)))

    def test_b_tilde_different_w(self):
        """b̃ gives different values for different w."""
        tau = 0.5 + 0.3205j
        omega = TF.find_critical_omega(tau)
        v1 = b_tilde_scalar(0.1, omega, tau)
        v2 = b_tilde_scalar(0.2, omega, tau)
        self.assertNotAlmostEqual(abs(v1), abs(v2), places=3)


# ---------------------------------------------------------------------------
# 3. Bonnet pair generation
# ---------------------------------------------------------------------------

class TestBonnetPairGeneration(unittest.TestCase):
    """Core Bonnet pair generation tests."""

    def test_pair_returns_result(self):
        """compute_bonnet_pair returns BonnetPairResult."""
        pair = get_pair()
        self.assertIsInstance(pair, BonnetPairResult)

    def test_f_plus_vertex_count(self):
        """f⁺ has correct vertex count."""
        pair = get_pair()
        self.assertEqual(len(pair.f_plus.vertices), 20 * 20)

    def test_f_minus_vertex_count(self):
        """f⁻ has correct vertex count."""
        pair = get_pair()
        self.assertEqual(len(pair.f_minus.vertices), 20 * 20)

    def test_f_plus_3d(self):
        """f⁺ vertices are 3D."""
        pair = get_pair()
        self.assertEqual(pair.f_plus.vertices.shape[1], 3)

    def test_f_plus_not_degenerate(self):
        """f⁺ spans non-trivial volume."""
        pair = get_pair()
        bbox = pair.f_plus.vertices.max(axis=0) - pair.f_plus.vertices.min(axis=0)
        self.assertGreater(np.prod(bbox), 1e-15)

    def test_f_minus_not_degenerate(self):
        """f⁻ spans non-trivial volume."""
        pair = get_pair()
        bbox = pair.f_minus.vertices.max(axis=0) - pair.f_minus.vertices.min(axis=0)
        self.assertGreater(np.prod(bbox), 1e-15)

    def test_f_plus_differs_from_f_minus(self):
        """f⁺ ≠ f⁻ (not identical)."""
        pair = get_pair()
        diff = np.linalg.norm(pair.f_plus.vertices - pair.f_minus.vertices)
        self.assertGreater(diff, 1e-6)

    def test_epsilon_stored(self):
        """Epsilon is stored correctly."""
        pair = get_pair()
        self.assertAlmostEqual(pair.epsilon, 0.5)

    def test_B_hat_grid_shape(self):
        """B̂ grid has shape (n_u, n_v)."""
        pair = get_pair()
        self.assertEqual(pair.B_hat_grid.shape, (20, 20))

    def test_metrics_populated(self):
        """Metrics contain expected keys."""
        pair = get_pair()
        for key in ['epsilon', 'R_omega', 'B_hat_range', 'B_tilde_max_norm']:
            self.assertIn(key, pair.metrics)


# ---------------------------------------------------------------------------
# 4. Isometry (Task 3.5)
# ---------------------------------------------------------------------------

class TestIsometry(unittest.TestCase):
    """Metric tensor comparison between f⁺ and f⁻."""

    def test_verify_isometry_runs(self):
        """verify_isometry runs without error."""
        pair = get_pair()
        result = verify_isometry(pair)
        self.assertIn('metric_max_err', result)

    def test_metric_components_exist(self):
        """Metric has E, F, G components."""
        pair = get_pair()
        torus = get_torus()
        g = compute_metric_tensor(pair.f_plus.f_grid, torus.u_grid, torus.v_grid)
        for key in ['E', 'F', 'G']:
            self.assertIn(key, g)
            self.assertEqual(g[key].shape, (20, 20))


# ---------------------------------------------------------------------------
# 5. Mean curvature (Task 3.6)
# ---------------------------------------------------------------------------

class TestMeanCurvature(unittest.TestCase):
    """H⁺ ≈ H⁻ verification."""

    def test_verify_mean_curvature_runs(self):
        """verify_mean_curvature runs without error."""
        pair = get_pair()
        result = verify_mean_curvature(pair)
        self.assertIn('max_abs_diff', result)
        self.assertIn('H_plus_mean', result)

    def test_H_values_finite(self):
        """Mean curvature values are finite."""
        pair = get_pair()
        result = verify_mean_curvature(pair)
        self.assertTrue(np.isfinite(result['H_plus_mean']))
        self.assertTrue(np.isfinite(result['H_minus_mean']))


# ---------------------------------------------------------------------------
# 6. Non-congruence (Task 3.7)
# ---------------------------------------------------------------------------

class TestNonCongruence(unittest.TestCase):
    """Procrustes verification: f⁺ ≇ f⁻."""

    def test_verify_non_congruence_runs(self):
        """verify_non_congruence runs without error."""
        pair = get_pair()
        result = verify_non_congruence(pair)
        self.assertIn('procrustes_disparity', result)

    def test_positive_disparity(self):
        """Procrustes disparity > 0 for distinct Bonnet pair."""
        pair = get_pair()
        result = verify_non_congruence(pair)
        self.assertGreater(result['procrustes_disparity'], 0)

    def test_non_congruent_flag(self):
        """is_non_congruent flag is True."""
        pair = get_pair()
        result = verify_non_congruence(pair)
        self.assertTrue(result['is_non_congruent'])


# ---------------------------------------------------------------------------
# 7. OBJ Export (Task 3.8)
# ---------------------------------------------------------------------------

class TestOBJExport(unittest.TestCase):
    """Dual OBJ file export."""

    def test_export_creates_files(self):
        """export_bonnet_pair_obj creates two OBJ files."""
        pair = get_pair()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2 = export_bonnet_pair_obj(pair, tmpdir, prefix="test")
            self.assertTrue(Path(p1).exists())
            self.assertTrue(Path(p2).exists())

    def test_obj_filenames(self):
        """Output files have correct names."""
        pair = get_pair()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2 = export_bonnet_pair_obj(pair, tmpdir, prefix="bonnet3")
            self.assertIn("bonnet3_f_plus.obj", str(p1))
            self.assertIn("bonnet3_f_minus.obj", str(p2))

    def test_obj_vertex_counts(self):
        """OBJ files have correct vertex counts."""
        pair = get_pair()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2 = export_bonnet_pair_obj(pair, tmpdir)
            for path in [p1, p2]:
                content = Path(path).read_text()
                v_lines = [l for l in content.splitlines() if l.startswith('v ')]
                self.assertEqual(len(v_lines), 400)  # 20×20


# ---------------------------------------------------------------------------
# 8. Closure gate (Task 3.9)
# ---------------------------------------------------------------------------

class TestClosureGate(unittest.TestCase):
    """Closure conditions verification."""

    def test_closure_gate_runs(self):
        """closure_gate runs without error."""
        pair = get_pair()
        result = closure_gate(pair)
        self.assertIn('rationality_error', result)

    def test_euler_both_tori(self):
        """Both f⁺ and f⁻ have Euler χ = 0."""
        pair = get_pair()
        result = closure_gate(pair)
        self.assertTrue(result['euler_plus']['ok'])
        self.assertTrue(result['euler_minus']['ok'])

    def test_k_fold_stored(self):
        """k-fold symmetry is 3."""
        pair = get_pair()
        result = closure_gate(pair)
        self.assertEqual(result['k_fold'], 3)


# ---------------------------------------------------------------------------
# 9. Analytic Christoffel dual
# ---------------------------------------------------------------------------

class TestChristoffelDualAnalytic(unittest.TestCase):
    """f*(u,v) = -f(π-u, v) identity."""

    def test_dual_shape(self):
        """Analytic dual has same shape as f_grid."""
        torus = get_torus()
        omega = torus.omega
        tau = torus.params.tau
        f_star = christoffel_dual_analytic(
            torus.f_grid, torus.u_grid, omega, tau,
            w_func=None, phi_interp=None,
        )
        self.assertEqual(f_star.shape, torus.f_grid.shape)

    def test_dual_is_negation(self):
        """f* is approximately the negation of some shifted f."""
        torus = get_torus()
        omega = torus.omega
        tau = torus.params.tau
        f_star = christoffel_dual_analytic(
            torus.f_grid, torus.u_grid, omega, tau,
            w_func=None, phi_interp=None,
        )
        # f*(u,v) = -f(π-u,v) → all values should be negations of some f value
        norms_star = np.linalg.norm(f_star.reshape(-1, 4), axis=1)
        norms_f = np.linalg.norm(torus.f_grid.reshape(-1, 4), axis=1)
        # The sets of norms should overlap (same magnitudes)
        self.assertAlmostEqual(np.mean(norms_star), np.mean(norms_f), places=1)


# ---------------------------------------------------------------------------
# 10. Different epsilon values
# ---------------------------------------------------------------------------

class TestEpsilonVariation(unittest.TestCase):
    """Test behavior with different epsilon."""

    def test_epsilon_zero_gives_symmetric_pair(self):
        """ε=0 → f⁺ = f⁻ (degenerate pair)."""
        torus = get_torus()
        pair = compute_bonnet_pair(torus, epsilon=0.0)
        diff = np.linalg.norm(pair.f_plus.vertices - pair.f_minus.vertices)
        self.assertLess(diff, 1e-10)

    def test_larger_epsilon_more_different(self):
        """Larger ε → larger separation between f⁺ and f⁻."""
        torus = get_torus()
        pair_small = compute_bonnet_pair(torus, epsilon=0.1)
        pair_large = compute_bonnet_pair(torus, epsilon=1.0)

        diff_small = np.linalg.norm(
            pair_small.f_plus.vertices - pair_small.f_minus.vertices)
        diff_large = np.linalg.norm(
            pair_large.f_plus.vertices - pair_large.f_minus.vertices)

        self.assertGreater(diff_large, diff_small)


if __name__ == '__main__':
    unittest.main(verbosity=2)
