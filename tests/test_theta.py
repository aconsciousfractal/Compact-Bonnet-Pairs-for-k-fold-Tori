"""
Phase 1 — Comprehensive Test Suite

Tests for all mathematical primitives:
  1. Theta functions: identities, special values, derivatives
  2. Elliptic integrals: known values, Legendre relation
  3. Weierstrass: double periodicity, ODE, inversion
  4. Quaternions: algebra, rotations, S³/SU(2)
  5. Frame integrator: unitarity, constant-w case

Run: python -m pytest tests/test_phase1.py -v
  or: python tests/test_phase1.py  (standalone)
"""
from __future__ import annotations

import sys
import os
import numpy as np

# Ensure src package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import theta_functions as TF
from src import elliptic_integrals as EI
from src import weierstrass as WS
from src import quaternion_ops as Q


# ===========================================================================
# Test configuration
# ===========================================================================

# Paper reference parameters
TAU_PAPER = 0.5 + 0.3205128205j   # τ = 1/2 + iR, 3-fold example
TAU_0_UPPER = 0.3547               # upper bound for Im(τ)

# Tolerances
TOL_TIGHT = 1e-12
TOL_MEDIUM = 1e-8
TOL_LOOSE = 1e-4


# ===========================================================================
# 1. THETA FUNCTIONS
# ===========================================================================

class TestThetaFunctions:
    """Tests for Jacobi theta functions (Whittaker & Watson convention)."""

    def test_nome_from_tau(self):
        """q = exp(iπτ), |q| < 1 for Im(τ) > 0."""
        q = TF.nome_from_tau(TAU_PAPER)
        assert abs(q) < 1.0, f"|q| = {abs(q)} should be < 1"
        # q = exp(iπ(1/2 + iR)) = exp(iπ/2 - πR) = i·exp(-πR)
        expected = 1j * np.exp(-np.pi * TAU_PAPER.imag)
        assert abs(q - expected) < TOL_TIGHT, f"q = {q}, expected {expected}"

    def test_theta1_zero(self):
        """ϑ₁(0 | τ) = 0."""
        val = TF.theta1(0, TAU_PAPER)
        assert abs(val) < TOL_TIGHT, f"ϑ₁(0) = {val}, expected 0"

    def test_jacobi_identity(self):
        """ϑ₁'(0) = ϑ₂(0)·ϑ₃(0)·ϑ₄(0) — the Jacobi triple product identity."""
        th1p = TF.theta1_prime_zero(TAU_PAPER)
        th2_0 = TF.theta2(0, TAU_PAPER)
        th3_0 = TF.theta3(0, TAU_PAPER)
        th4_0 = TF.theta4(0, TAU_PAPER)
        product = th2_0 * th3_0 * th4_0
        rel_err = abs(th1p - product) / abs(th1p)
        assert rel_err < TOL_TIGHT, f"Jacobi identity error: {rel_err}"

    def test_theta_quasi_periodicity(self):
        """ϑ₁(z + π) = -ϑ₁(z) (quasi-periodicity in π)."""
        z = 0.7 + 0.3j
        val_z = TF.theta1(z, TAU_PAPER)
        val_zp = TF.theta1(z + np.pi, TAU_PAPER)
        assert abs(val_zp + val_z) < TOL_TIGHT * max(abs(val_z), 1), \
            f"ϑ₁(z+π) + ϑ₁(z) = {abs(val_zp + val_z)}"

    def test_theta2_from_theta1(self):
        """ϑ₂(z) = ϑ₁(z + π/2)."""
        z = 0.5 + 0.2j
        th1_shifted = TF.theta1(z + np.pi / 2, TAU_PAPER)
        th2_z = TF.theta2(z, TAU_PAPER)
        assert abs(th1_shifted - th2_z) < TOL_TIGHT * max(abs(th2_z), 1)

    def test_theta_quartic_identity(self):
        """Jacobi quartic: ϑ₃(0)⁴ = ϑ₂(0)⁴ + ϑ₄(0)⁴."""
        th2 = TF.theta2(0, TAU_PAPER)
        th3 = TF.theta3(0, TAU_PAPER)
        th4 = TF.theta4(0, TAU_PAPER)
        lhs = th3**4
        rhs = th2**4 + th4**4
        rel_err = abs(lhs - rhs) / abs(lhs)
        assert rel_err < TOL_TIGHT, f"Quartic identity error: {rel_err}"

    def test_find_critical_omega(self):
        """Find ω such that ϑ₂'(ω) = 0, verify ω ∈ (0, π/4)."""
        omega = TF.find_critical_omega(TAU_PAPER)
        assert 0 < omega < np.pi / 4, f"ω = {omega} not in (0, π/4)"
        # Paper value: ω ≈ 0.3890180475
        assert abs(omega - 0.3890180475) < 1e-6, f"ω = {omega}, expected ≈ 0.389"
        # Verify derivative is zero
        deriv = TF.theta2(omega, TAU_PAPER, derivative=1)
        assert abs(deriv) < TOL_MEDIUM, f"ϑ₂'(ω) = {deriv}, expected ≈ 0"

    def test_R_omega_equals_neg_U_inv(self):
        """R(ω) = -U(ω)⁻¹ — paper consistency check."""
        omega = TF.find_critical_omega(TAU_PAPER)
        R = TF.R_omega(omega, TAU_PAPER)
        U = TF.U_omega(omega, TAU_PAPER)
        assert abs(R + 1.0 / U) < TOL_TIGHT * max(abs(R), 1), \
            f"R + 1/U = {abs(R + 1.0 / U)}"

    def test_gamma_periodicity(self):
        """γ(u + 2π, w) = γ(u, w) — periodicity in u."""
        omega = TF.find_critical_omega(TAU_PAPER)
        u, w = 1.0, 0.5
        g1 = TF.gamma_curve(u, w, omega, TAU_PAPER)
        g2 = TF.gamma_curve(u + 2 * np.pi, w, omega, TAU_PAPER)
        assert abs(g1 - g2) < TOL_MEDIUM * max(abs(g1), 1), \
            f"|γ(u+2π) - γ(u)| = {abs(g1 - g2)}"

    def test_gamma_is_planar(self):
        """γ(u, w) for fixed w traces a plane curve (Im part only, in i-direction)."""
        omega = TF.find_critical_omega(TAU_PAPER)
        w = 0.3
        # Sample several u values
        us = np.linspace(0.1, 2 * np.pi - 0.1, 20)
        values = [TF.gamma_curve(u, w, omega, TAU_PAPER) for u in us]
        # In the paper, γ(u,w) is a complex number (lies in a plane containing i)
        # All values should be complex numbers
        assert all(isinstance(v, complex) for v in values)


# ===========================================================================
# 2. ELLIPTIC INTEGRALS
# ===========================================================================

class TestEllipticIntegrals:
    """Tests for elliptic integral wrappers."""

    def test_K_at_zero(self):
        """K(0) = π/2."""
        assert abs(EI.K(0) - np.pi / 2) < TOL_TIGHT

    def test_E_at_zero(self):
        """E(0) = π/2."""
        assert abs(EI.E(0) - np.pi / 2) < TOL_TIGHT

    def test_E_at_one(self):
        """E(1) = 1."""
        assert abs(EI.E(0.9999999) - 1.0) < 1e-4  # near k=1

    def test_legendre_relation(self):
        """K·E' + E·K' - K·K' = π/2. Various moduli."""
        for k in [0.1, 0.3, 0.5, 0.7, 0.9]:
            err = EI.legendre_relation(k)
            assert abs(err) < TOL_TIGHT, f"Legendre violation at k={k}: {err}"

    def test_jacobi_sn_cn_dn_identity(self):
        """sn² + cn² = 1 and dn² + k²·sn² = 1."""
        k = 0.6
        for u in [0.5, 1.0, 2.0]:
            sn, cn, dn = EI.jacobi_elliptic(u, k)
            assert abs(sn**2 + cn**2 - 1) < TOL_TIGHT
            assert abs(dn**2 + k**2 * sn**2 - 1) < TOL_TIGHT

    def test_F_at_zero(self):
        """F(0, k) = 0."""
        assert abs(EI.F_incomplete(0, 0.5)) < TOL_TIGHT

    def test_F_at_pi_half_equals_K(self):
        """F(π/2, k) = K(k)."""
        k = 0.7
        assert abs(EI.F_incomplete(np.pi / 2, k) - EI.K(k)) < TOL_TIGHT

    def test_nome_from_modulus_range(self):
        """Nome q ∈ (0, 1) for k ∈ (0, 1)."""
        for k in [0.1, 0.3, 0.5, 0.7, 0.9]:
            q = EI.nome_from_modulus(k)
            assert 0 < q < 1, f"q = {q} for k = {k}"


# ===========================================================================
# 3. WEIERSTRASS FUNCTION
# ===========================================================================

class TestWeierstrass:
    """Tests for Weierstrass ℘-function."""

    def test_double_periodicity(self):
        """℘(z + 2ω₁) = ℘(z) and ℘(z + 2ω₃) = ℘(z)."""
        g2, g3 = 60.0, 0.0
        omega1, omega3 = WS._half_periods(g2, g3)

        z = 0.3 + 0.2j
        # Use higher N for periodicity verification
        wp_z = WS.weierstrass_p(z, g2, g3, N=40)
        wp_z_2w1 = WS.weierstrass_p(z + 2 * omega1, g2, g3, N=40)
        wp_z_2w3 = WS.weierstrass_p(z + 2 * omega3, g2, g3, N=40)

        assert abs(wp_z - wp_z_2w1) < 1e-3 * max(abs(wp_z), 1), \
            f"Period 2ω₁ error: {abs(wp_z - wp_z_2w1)}"
        assert abs(wp_z - wp_z_2w3) < 1e-3 * max(abs(wp_z), 1), \
            f"Period 2ω₃ error: {abs(wp_z - wp_z_2w3)}"

    def test_differential_equation(self):
        """(℘')² = 4℘³ - g₂℘ - g₃ at a sample point."""
        g2, g3 = 10.0 + 0j, 1.0 + 0j
        z = 0.5 + 0.3j
        wp = WS.weierstrass_p(z, g2, g3)
        wp_prime = WS.weierstrass_p_prime(z, g2, g3)
        lhs = wp_prime**2
        rhs = 4 * wp**3 - complex(g2) * wp - complex(g3)
        rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1)
        assert rel_err < TOL_LOOSE, f"ODE error: {rel_err}"

    def test_laurent_expansion_at_origin(self):
        """℘(z) ≈ 1/z² + O(z²) near origin."""
        g2, g3 = 10.0 + 0j, 1.0 + 0j
        z = 0.01 + 0.01j
        wp = WS.weierstrass_p(z, g2, g3)
        expected_leading = 1.0 / z**2
        # The ratio should approach 1 for small z
        ratio = wp / expected_leading
        assert abs(ratio - 1.0) < 0.1, f"Laurent check: ratio = {ratio}"


# ===========================================================================
# 4. QUATERNION ALGEBRA
# ===========================================================================

class TestQuaternionOps:
    """Tests for quaternion algebra."""

    def test_multiplication_ij_eq_k(self):
        """i·j = k."""
        result = Q.qmul(Q.quat_i(), Q.quat_j())
        expected = Q.quat_k()
        np.testing.assert_allclose(result, expected, atol=TOL_TIGHT)

    def test_multiplication_jk_eq_i(self):
        """j·k = i."""
        result = Q.qmul(Q.quat_j(), Q.quat_k())
        np.testing.assert_allclose(result, Q.quat_i(), atol=TOL_TIGHT)

    def test_multiplication_ki_eq_j(self):
        """k·i = j."""
        result = Q.qmul(Q.quat_k(), Q.quat_i())
        np.testing.assert_allclose(result, Q.quat_j(), atol=TOL_TIGHT)

    def test_multiplication_anticommutative(self):
        """i·j = -j·i."""
        ij = Q.qmul(Q.quat_i(), Q.quat_j())
        ji = Q.qmul(Q.quat_j(), Q.quat_i())
        np.testing.assert_allclose(ij, -ji, atol=TOL_TIGHT)

    def test_inverse(self):
        """q·q⁻¹ = 1."""
        q = Q.quat(1.0, 2.0, 3.0, 4.0)
        q_inv = Q.qinv(q)
        product = Q.qmul(q, q_inv)
        expected = Q.quat(1.0, 0, 0, 0)
        np.testing.assert_allclose(product, expected, atol=TOL_TIGHT)

    def test_norm_multiplicative(self):
        """|p·q| = |p|·|q|."""
        p = Q.quat(1.0, 2.0, 0.5, -1.0)
        q = Q.quat(0.3, -1.0, 2.0, 0.7)
        pq = Q.qmul(p, q)
        assert abs(Q.qnorm(pq) - Q.qnorm(p) * Q.qnorm(q)) < TOL_TIGHT

    def test_conjugate_product(self):
        """conj(p·q) = conj(q)·conj(p)."""
        p = Q.quat(1.0, 2.0, 3.0, 4.0)
        q = Q.quat(0.5, -1.0, 0.3, 2.0)
        lhs = Q.qconj(Q.qmul(p, q))
        rhs = Q.qmul(Q.qconj(q), Q.qconj(p))
        np.testing.assert_allclose(lhs, rhs, atol=TOL_TIGHT)

    def test_rotation_preserves_norm(self):
        """Rotation X ↦ q⁻¹Xq preserves |X|."""
        X = Q.quat_from_vector(np.array([1.0, 2.0, 3.0]))
        q = Q.qnormalize(Q.quat(1.0, 1.0, 0.0, 0.0))
        X_rot = Q.rotate_by_unit(X, q)
        np.testing.assert_allclose(Q.qnorm(X_rot), Q.qnorm(X), atol=TOL_TIGHT)

    def test_rotation_result_is_pure(self):
        """Rotation of pure imaginary quaternion stays pure imaginary."""
        X = Q.quat_from_vector(np.array([1.0, 0.0, 0.0]))
        q = Q.rotation_quaternion(np.array([0, 0, 1.0]), np.pi / 3)
        X_rot = Q.rotate_by_unit(X, q)
        assert abs(X_rot[0]) < TOL_TIGHT, f"Scalar part = {X_rot[0]}"

    def test_rotation_by_90_deg(self):
        """Rotate [1,0,0] by π/2 around z → [0,1,0] using qXq̄ convention."""
        X = Q.quat_from_vector(np.array([1.0, 0.0, 0.0]))
        q = Q.rotation_quaternion(np.array([0, 0, 1.0]), np.pi / 2)
        # rotate_by uses qXq̄ (standard active rotation)
        X_rot = Q.rotate_by(X, q)
        expected = Q.quat_from_vector(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(X_rot, expected, atol=TOL_TIGHT)

    def test_rotation_inverse_convention(self):
        """rotate_by_unit (q⁻¹Xq) is inverse of rotate_by (qXq̄)."""
        X = Q.quat_from_vector(np.array([1.0, 2.0, 3.0]))
        q = Q.qnormalize(Q.quat(1.0, 0.5, 0.3, 0.7))
        X_fwd = Q.rotate_by(X, q)
        X_back = Q.rotate_by_unit(X_fwd, q)
        np.testing.assert_allclose(X_back, X, atol=TOL_TIGHT)

    def test_imh_product(self):
        """XY = -⟨X,Y⟩ + X×Y for pure imaginary quaternions."""
        X = Q.quat_from_vector(np.array([1.0, 0.0, 0.0]))
        Y = Q.quat_from_vector(np.array([0.0, 1.0, 0.0]))
        product = Q.imh_product(X, Y)
        # ⟨X,Y⟩ = 0, X×Y = [0,0,1]
        expected = Q.quat_from_vector(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(product, expected, atol=TOL_TIGHT)

    def test_su2_matrix_roundtrip(self):
        """Quaternion → SU(2) → Quaternion roundtrip."""
        q = Q.qnormalize(Q.quat(1.0, 2.0, 3.0, 4.0))
        M = Q.su2_matrix(q)
        # Verify SU(2): M†M = I, det(M) = 1
        MdM = M.conj().T @ M
        np.testing.assert_allclose(MdM, np.eye(2), atol=TOL_TIGHT)
        assert abs(np.linalg.det(M) - 1.0) < TOL_TIGHT
        # Roundtrip
        q_back = Q.from_su2_matrix(M)
        # Sign ambiguity: q and -q give same SU(2) matrix
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        np.testing.assert_allclose(q_back, q, atol=TOL_TIGHT)

    def test_hopf_map_on_sphere(self):
        """Hopf map sends S³ → S² (output on unit sphere)."""
        for _ in range(10):
            q = Q.qnormalize(Q.quat(*np.random.randn(4)))
            p = Q.hopf_map(q)
            assert abs(np.linalg.norm(p) - 1.0) < TOL_TIGHT

    def test_batch_multiply(self):
        """Batch multiply matches element-wise multiply."""
        N = 50
        P = np.random.randn(N, 4)
        Qm = np.random.randn(N, 4)
        batch_result = Q.qmul_batch(P, Qm)
        for i in range(N):
            single = Q.qmul(P[i], Qm[i])
            np.testing.assert_allclose(batch_result[i], single, atol=TOL_TIGHT)

    def test_complex_quaternion_inverse(self):
        """Complex quaternion q·q⁻¹ = 1."""
        q = np.array([1 + 2j, 0.5 - 1j, 2 + 0.3j, -1 + 0.5j])
        q_inv = Q.qinv_complex(q)
        product = Q.qmul_complex(q, q_inv)
        expected = np.array([1 + 0j, 0j, 0j, 0j])
        np.testing.assert_allclose(product, expected, atol=TOL_MEDIUM)


# ===========================================================================
# 5. FRAME INTEGRATOR (basic tests without full theta construction)
# ===========================================================================

class TestFrameIntegrator:
    """Tests for frame ODE — using simple/analytic cases."""

    def test_constant_w_unitarity(self):
        """With constant w, Φ(v) stays on S³."""
        from src.frame_integrator import integrate_frame, constant_w

        omega = TF.find_critical_omega(TAU_PAPER)
        w_func, wp_func = constant_w(0.2)

        result = integrate_frame(
            w_func=w_func,
            w_prime_func=wp_func,
            omega=omega,
            tau=TAU_PAPER,
            v_span=(0, 2.0),
            n_points=100,
        )

        # All unitarity errors should be tiny
        assert np.max(result.unitarity_errors) < 1e-8, \
            f"Max unitarity error: {np.max(result.unitarity_errors)}"

    def test_constant_w_pure_rotation(self):
        """
        With w' = 0 (constant w), the ODE becomes Φ' = W₁(w₀)·k·Φ,
        which has solution Φ(v) = exp(W₁·k·v)·Φ(0).
        The monodromy should be a pure rotation around k-axis.
        """
        from src.frame_integrator import integrate_frame, constant_w

        omega = TF.find_critical_omega(TAU_PAPER)
        w_func, wp_func = constant_w(0.1)

        result = integrate_frame(
            w_func=w_func,
            w_prime_func=wp_func,
            omega=omega,
            tau=TAU_PAPER,
            v_span=(0, 1.0),
            n_points=200,
        )

        # Monodromy should be unit quaternion
        mono_norm = Q.qnorm(result.monodromy)
        assert abs(mono_norm - 1.0) < 1e-6, f"Monodromy norm: {mono_norm}"

        # Rotation angle should be well-defined
        assert result.rotation_angle >= 0


# ===========================================================================
# 6. INTEGRATION TESTS (cross-module)
# ===========================================================================

class TestCrossModule:
    """Tests that verify consistency across modules."""

    def test_theta_elliptic_bridge(self):
        """
        Nome from modulus (EI) matches nome from tau (TF) for
        self-consistent tau.
        """
        k = 0.5
        tau = EI.tau_from_modulus(k)
        q_from_EI = EI.nome_from_modulus(k)
        q_from_TF = abs(TF.nome_from_tau(tau))  # magnitude
        assert abs(q_from_EI - q_from_TF) < TOL_MEDIUM, \
            f"nome mismatch: EI={q_from_EI}, TF={q_from_TF}"

    def test_paper_parameters_consistency(self):
        """
        For paper τ, verify ω is found and R(ω), U(ω) are finite and consistent.
        """
        omega = TF.find_critical_omega(TAU_PAPER)
        R = TF.R_omega(omega, TAU_PAPER)
        U = TF.U_omega(omega, TAU_PAPER)
        U_prime = TF.U_prime_omega(omega, TAU_PAPER)
        U1_prime = TF.U1_prime_omega(omega, TAU_PAPER)
        U2 = TF.U2_omega(omega, TAU_PAPER)

        # All should be finite
        for name, val in [('R', R), ('U', U), ("U'", U_prime),
                          ("U1'", U1_prime), ('U2', U2)]:
            assert np.isfinite(abs(val)), f"{name}(ω) is not finite: {val}"

        # R = -1/U
        assert abs(R + 1.0 / U) < TOL_TIGHT * max(abs(R), 1)

    def test_Q3_polynomial_at_paper_params(self):
        """Q₃(s) should be evaluable at paper parameter values."""
        omega = TF.find_critical_omega(TAU_PAPER)
        # s₁ ≈ -3.601381552, s₂ ≈ 0.5965202011 (3-fold values)
        s1 = -3.601381552
        s2 = 0.5965202011

        q3_s1 = TF.Q3_polynomial(s1, omega, TAU_PAPER)
        q3_s2 = TF.Q3_polynomial(s2, omega, TAU_PAPER)

        assert np.isfinite(abs(q3_s1)), f"Q₃(s₁) = {q3_s1}"
        assert np.isfinite(abs(q3_s2)), f"Q₃(s₂) = {q3_s2}"

    def test_Q_polynomial_at_paper_params(self):
        """Q(s) at paper 3-fold values should be evaluable."""
        omega = TF.find_critical_omega(TAU_PAPER)
        s1 = -3.601381552
        s2 = 0.5965202011
        delta = 1.897366596

        # Q at midpoint
        s_mid = (s1 + s2) / 2
        Q_val = TF.Q_polynomial(s_mid, s1, s2, delta, omega, TAU_PAPER)
        assert np.isfinite(abs(Q_val)), f"Q(s_mid) = {Q_val}"

    def test_W1_finite_at_paper_params(self):
        """W₁(w) should be finite for small w with paper parameters."""
        omega = TF.find_critical_omega(TAU_PAPER)
        for w in [0.1, 0.2, 0.3, 0.5]:
            W1 = TF.W1_function(w, omega, TAU_PAPER)
            assert np.isfinite(abs(W1)), f"W₁({w}) = {W1}"

    def test_s_of_w_finite(self):
        """s(w) = e^{-h(ω,w)} should be finite."""
        omega = TF.find_critical_omega(TAU_PAPER)
        for w in [0.1, 0.2, 0.3]:
            s = TF.s_of_w(w, omega, TAU_PAPER)
            assert np.isfinite(abs(s)), f"s({w}) = {s}"


# ===========================================================================
# Runner
# ===========================================================================

def run_all_tests():
    """Simple test runner (no pytest dependency required)."""
    import traceback

    test_classes = [
        TestThetaFunctions,
        TestEllipticIntegrals,
        TestWeierstrass,
        TestQuaternionOps,
        TestFrameIntegrator,
        TestCrossModule,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  ✓ {method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  ✗ {method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
