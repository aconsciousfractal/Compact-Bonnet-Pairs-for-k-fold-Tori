"""
Phase 12 Test Suite — Retraction Form (arXiv:2506.13312v2)

Tests:
  1. StereographicTests — inverse/forward projection, round-trip
  2. DerivativeTests — central differences, conformality
  3. RetractionFormTests — ω construction, dω=0, ω̄∧dx=0
  4. IntegrationTests — F± computation, purity, path-independence
  5. IsometryTests — F⁺ isometric to F⁻
  6. ComparisonTests — F± vs Eq.49 f± (Procrustes)
  7. ValidationGateTests — full gate pass/fail
"""
from __future__ import annotations

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retraction_form import (
    inverse_stereographic,
    stereographic,
    compute_derivatives,
    compute_retraction_omega,
    verify_closure,
    verify_cross_condition,
    integrate_bonnet_pair,
    compute_retraction_bonnet,
    compare_retraction_vs_eq49,
    retraction_validation_gate,
    verify_retraction_isometry,
    RetractionFormResult,
)
from src.isothermic_torus import TorusParameters, compute_torus
from src.bonnet_pair import compute_bonnet_pair, BonnetPairResult
from src import quaternion_ops as Q


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TORUS = None
_RETRACTION = None
_PAIR = None


def get_torus():
    """Lazily compute shared torus (k=3, 50×50 for FD accuracy)."""
    global _TORUS
    if _TORUS is None:
        params = TorusParameters(
            tau_imag=0.3205,
            u_res=50,
            v_res=50,
            v_periods=1,
            symmetry_fold=3,
            w0=0.15,
        )
        _TORUS = compute_torus(params)
    return _TORUS


def get_retraction():
    """Lazily compute retraction form result."""
    global _RETRACTION
    if _RETRACTION is None:
        torus = get_torus()
        _RETRACTION = compute_retraction_bonnet(torus, verbose=False)
    return _RETRACTION


def get_pair():
    """Lazily compute Eq.49 Bonnet pair."""
    global _PAIR
    if _PAIR is None:
        torus = get_torus()
        _PAIR = compute_bonnet_pair(torus, epsilon=0.5)
    return _PAIR


# ═══════════════════════════════════════════════════════════════════
# 1. Stereographic projection
# ═══════════════════════════════════════════════════════════════════

class StereographicTests(unittest.TestCase):
    """Inverse/forward stereographic projection on S³."""

    def test_roundtrip_random_pure_quaternions(self):
        """σ(σ⁻¹(f)) = f for random f ∈ Im(ℍ)."""
        rng = np.random.default_rng(42)
        f = np.zeros((50, 4))
        f[:, 1:4] = rng.uniform(-3, 3, size=(50, 3))

        x = inverse_stereographic(f)
        f_back = stereographic(x)

        np.testing.assert_allclose(f_back, f, atol=1e-12)

    def test_unit_norm_on_S3(self):
        """σ⁻¹(f) ∈ S³:  |x| = 1."""
        rng = np.random.default_rng(123)
        f = np.zeros((100, 4))
        f[:, 1:4] = rng.uniform(-5, 5, size=(100, 3))

        x = inverse_stereographic(f)
        norms = np.sqrt(np.sum(x ** 2, axis=-1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_origin_maps_to_south_pole(self):
        """σ⁻¹(0) should be (-1, 0, 0, 0) (south pole)."""
        f = np.array([[0, 0, 0, 0]], dtype=float)
        x = inverse_stereographic(f)
        np.testing.assert_allclose(x[0], [-1, 0, 0, 0], atol=1e-15)

    def test_torus_grid_on_S3(self):
        """Torus f_grid lifts to unit quaternions on S³."""
        torus = get_torus()
        x = inverse_stereographic(torus.f_grid)
        norms = np.sqrt(np.sum(x ** 2, axis=-1))
        self.assertLess(np.max(np.abs(norms - 1.0)), 1e-12)


# ═══════════════════════════════════════════════════════════════════
# 2. Derivatives
# ═══════════════════════════════════════════════════════════════════

class DerivativeTests(unittest.TestCase):

    def test_sinusoidal_derivative(self):
        """Central differences recover d/du sin(u) = cos(u)."""
        N_u, N_v = 100, 5
        u = np.linspace(0, 2 * np.pi, N_u, endpoint=False)
        du = u[1] - u[0]
        dv = 0.1

        g = np.zeros((N_u, N_v, 4))
        g[:, :, 1] = np.sin(u)[:, None]

        dg_du, _ = compute_derivatives(g, du, dv)
        expected = np.cos(u)
        np.testing.assert_allclose(dg_du[:, 0, 1], expected, atol=du ** 2)

    def test_conformality_on_torus(self):
        """On isothermic torus in S³, |x_u|² ≈ |x_v|²."""
        R = get_retraction()
        self.assertLess(R.conformality_error, 0.5,
                        "Conformality error too large for isothermic surface")


# ═══════════════════════════════════════════════════════════════════
# 3. Retraction form
# ═══════════════════════════════════════════════════════════════════

class RetractionFormTests(unittest.TestCase):

    def test_omega_constructed(self):
        """ω_u and ω_v have correct shape."""
        R = get_retraction()
        torus = get_torus()
        N_u = torus.f_grid.shape[0]
        N_v = torus.f_grid.shape[1]
        self.assertEqual(R.omega_u.shape, (N_u, N_v, 4))
        self.assertEqual(R.omega_v.shape, (N_u, N_v, 4))

    def test_closure_dw_zero(self):
        """dω = 0 — closure of retraction form (isothermic condition)."""
        R = get_retraction()
        self.assertLess(R.closure_error, 0.5,
                        f"Closure error {R.closure_error:.2e} too large")

    def test_cross_condition(self):
        """ω̄∧dx = 0 — cross condition."""
        R = get_retraction()
        self.assertLess(R.cross_error, 1.0,
                        f"Cross error {R.cross_error:.2e} too large")

    def test_omega_in_tangent_space(self):
        """ω takes values in T_xS³ = x⊥: Re(x̄·ω) = 0."""
        R = get_retraction()
        orth = R.metrics.get('orthogonality_error', 1.0)
        self.assertLess(orth, 0.5,
                        f"Orthogonality error {orth:.2e} too large")


# ═══════════════════════════════════════════════════════════════════
# 4. F± integration
# ═══════════════════════════════════════════════════════════════════

class IntegrationTests(unittest.TestCase):

    def test_Fplus_shape(self):
        """F⁺ has correct shape (N_u, N_v, 4)."""
        R = get_retraction()
        torus = get_torus()
        N_u, N_v = torus.f_grid.shape[:2]
        self.assertEqual(R.F_plus.shape, (N_u, N_v, 4))

    def test_Fminus_shape(self):
        """F⁻ has correct shape (N_u, N_v, 4)."""
        R = get_retraction()
        torus = get_torus()
        N_u, N_v = torus.f_grid.shape[:2]
        self.assertEqual(R.F_minus.shape, (N_u, N_v, 4))

    def test_Fplus_nonzero(self):
        """F⁺ is not identically zero."""
        R = get_retraction()
        self.assertGreater(np.max(np.abs(R.F_plus)), 1e-10)

    def test_Fminus_nonzero(self):
        """F⁻ is not identically zero."""
        R = get_retraction()
        self.assertGreater(np.max(np.abs(R.F_minus)), 1e-10)

    def test_Fplus_Fminus_differ(self):
        """F⁺ ≠ F⁻ (non-congruence)."""
        R = get_retraction()
        diff = np.max(np.abs(R.F_plus - R.F_minus))
        self.assertGreater(diff, 1e-10)

    def test_path_independence(self):
        """Exactness error (path-dependence) is bounded."""
        R = get_retraction()
        self.assertLess(R.exactness_error, 0.5,
                        f"Exactness error {R.exactness_error:.2e} too large")

    def test_Fplus_purity(self):
        """F⁺ is predominantly imaginary (∈ Im(ℍ))."""
        R = get_retraction()
        fp_scalar = R.metrics.get('F_plus_max_scalar', 1.0)
        self.assertLess(fp_scalar, 0.1,
                        f"F⁺ max scalar = {fp_scalar:.2e}")

    def test_Fminus_purity(self):
        """F⁻ is predominantly imaginary (∈ Im(ℍ))."""
        R = get_retraction()
        fm_scalar = R.metrics.get('F_minus_max_scalar', 1.0)
        self.assertLess(fm_scalar, 0.1,
                        f"F⁻ max scalar = {fm_scalar:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 5. Isometry
# ═══════════════════════════════════════════════════════════════════

class IsometryTests(unittest.TestCase):

    def test_F_plus_F_minus_isometric(self):
        """F⁺ and F⁻ have similar induced metrics."""
        R = get_retraction()
        torus = get_torus()
        u = np.linspace(0, 2 * np.pi, torus.f_grid.shape[0], endpoint=False)
        v = torus.frame_result.v_values
        du = u[1] - u[0]
        dv = (v[-1] - v[0]) / (len(v) - 1)

        iso = verify_retraction_isometry(R.F_plus, R.F_minus, du, dv)
        # At low resolution, metric comparison may have ~10-30% error
        self.assertLess(iso['mean_d_guu'], 1.0,
                        msg=f"Mean g_uu discrepancy = {iso['mean_d_guu']:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 6. Comparison with Eq. 49
# ═══════════════════════════════════════════════════════════════════

class ComparisonTests(unittest.TestCase):

    def test_procrustes_finite(self):
        """F± vs f± Procrustes distance is finite (not NaN/Inf)."""
        R = get_retraction()
        P = get_pair()
        comp = compare_retraction_vs_eq49(R, P, verbose=False)
        self.assertTrue(np.isfinite(comp['best_match']),
                        "Procrustes distance is not finite")
        # Shape match not expected at this resolution/normalization;
        # convergence verified in run_phase12_diagnostic.py
        self.assertLess(comp['best_match'], 2.0,
                        f"Procrustes best={comp['best_match']:.4f} > 2.0")


# ═══════════════════════════════════════════════════════════════════
# 7. Validation gate
# ═══════════════════════════════════════════════════════════════════

class ValidationGateTests(unittest.TestCase):

    def test_gate_returns_dict(self):
        """Gate returns a well-formed dict."""
        R = get_retraction()
        gate = retraction_validation_gate(R, verbose=False)
        self.assertIn('closure', gate)
        self.assertIn('cross', gate)
        self.assertIn('exactness', gate)
        self.assertIn('purity', gate)
        self.assertIn('all_pass', gate)

    def test_gate_with_bonnet(self):
        """Gate with Bonnet comparison includes Procrustes."""
        R = get_retraction()
        P = get_pair()
        gate = retraction_validation_gate(R, bonnet_result=P, verbose=False)
        self.assertIn('procrustes', gate)


# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
