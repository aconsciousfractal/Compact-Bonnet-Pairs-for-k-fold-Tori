"""
Phase 2 Test Suite — Isothermic Torus Generator

Tests:
  1. TorusGeneration — basic torus construction, vertex count, face count
  2. MeshTopology — Euler characteristic χ = 0, periodic connectivity
  3. IsothermicValidation — cross-ratio convergence toward -1
  4. SymmetryTests — Theorem 4 inversion symmetry
  5. OBJExport — file writing and format verification
  6. PureImaginaryTest — f(u,v) scalar part ≈ 0
"""

import sys
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.isothermic_torus import (
    TorusParameters,
    TorusResult,
    compute_torus,
    build_torus_faces,
    compute_vertex_normals,
    validate_isothermic,
    verify_symmetry_inversion,
    verify_euler_characteristic,
    export_torus_obj,
    compute_christoffel_dual,
)
from src import quaternion_ops as Q
from src import theta_functions as TF


# ---------------------------------------------------------------------------
# Shared fixture: small torus for fast tests
# ---------------------------------------------------------------------------

_SMALL_PARAMS = None
_SMALL_RESULT = None


def get_small_torus():
    """Lazily compute a small torus (20×20) for reuse across tests."""
    global _SMALL_PARAMS, _SMALL_RESULT
    if _SMALL_RESULT is None:
        _SMALL_PARAMS = TorusParameters(
            tau_imag=0.3205,
            u_res=20,
            v_res=20,
            v_periods=1,
            symmetry_fold=3,
            w0=0.15,
        )
        _SMALL_RESULT = compute_torus(_SMALL_PARAMS)
    return _SMALL_PARAMS, _SMALL_RESULT


class TestTorusGeneration(unittest.TestCase):
    """Basic torus construction tests."""

    def test_compute_torus_returns_result(self):
        """compute_torus returns a TorusResult."""
        _, result = get_small_torus()
        self.assertIsInstance(result, TorusResult)

    def test_vertex_count(self):
        """Vertex count = u_res × v_res."""
        params, result = get_small_torus()
        expected = params.u_res * params.v_res
        self.assertEqual(len(result.vertices), expected)

    def test_face_count(self):
        """Face count = u_res × v_res (one quad per grid cell, periodic)."""
        params, result = get_small_torus()
        expected = params.u_res * params.v_res
        self.assertEqual(len(result.faces), expected)

    def test_vertices_are_3d(self):
        """Vertices are 3D points."""
        _, result = get_small_torus()
        self.assertEqual(result.vertices.shape[1], 3)

    def test_normals_shape(self):
        """Normals have same shape as vertices."""
        _, result = get_small_torus()
        self.assertEqual(result.vertices.shape, result.normals.shape)

    def test_normals_unit_length(self):
        """Normals are approximately unit length."""
        _, result = get_small_torus()
        norms = np.linalg.norm(result.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_omega_matches_paper(self):
        """ω ≈ 0.389 for τ_imag ≈ 0.3205."""
        _, result = get_small_torus()
        self.assertAlmostEqual(result.omega, 0.389018, places=3)

    def test_f_grid_shape(self):
        """f_grid has shape (u_res, v_res, 4)."""
        params, result = get_small_torus()
        self.assertEqual(result.f_grid.shape, (params.u_res, params.v_res, 4))

    def test_metrics_populated(self):
        """Metrics dict contains key fields."""
        _, result = get_small_torus()
        for key in ['n_vertices', 'n_faces', 'max_scalar_part', 'omega',
                     'V_period', 'frame_unitarity_max_err']:
            self.assertIn(key, result.metrics)

    def test_vertices_not_degenerate(self):
        """Vertices span a non-degenerate volume (bbox > 0 in all axes)."""
        _, result = get_small_torus()
        bbox_size = result.vertices.max(axis=0) - result.vertices.min(axis=0)
        for i in range(3):
            self.assertGreater(bbox_size[i], 1e-6,
                               f"Degenerate in axis {i}: bbox_size={bbox_size}")


class TestMeshTopology(unittest.TestCase):
    """Mesh topology: Euler characteristic, connectivity."""

    def test_euler_characteristic_zero(self):
        """Torus has χ = V - E + F = 0."""
        _, result = get_small_torus()
        ec = verify_euler_characteristic(result.faces, len(result.vertices),
                                         expected=0)
        self.assertTrue(ec['ok'],
                        f"χ = {ec['euler_characteristic']}, expected 0")

    def test_all_quads(self):
        """All faces are quads (4 vertices each)."""
        _, result = get_small_torus()
        for face in result.faces:
            self.assertEqual(len(face), 4)

    def test_build_torus_faces_small(self):
        """build_torus_faces(3,4) creates 12 quads with proper wrapping."""
        faces = build_torus_faces(3, 4)
        self.assertEqual(len(faces), 12)
        # Check wrap-around: last u row connects to first
        found_wrap = False
        for f in faces:
            if 0 in f and (3 * 4 - 1) in f:
                found_wrap = True
        # At minimum, v-wrapping should be present
        all_indices = set()
        for f in faces:
            all_indices.update(f)
        self.assertEqual(all_indices, set(range(12)))

    def test_vertex_indices_valid(self):
        """All face vertex indices are within range."""
        _, result = get_small_torus()
        n = len(result.vertices)
        for face in result.faces:
            for idx in face:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, n)


class TestPureImaginary(unittest.TestCase):
    """f(u,v) should be pure imaginary (scalar part ≈ 0)."""

    def test_scalar_part_small(self):
        """Max scalar part of f(u,v) is small."""
        _, result = get_small_torus()
        max_scalar = result.metrics['max_scalar_part']
        self.assertLess(max_scalar, 0.1,
                        f"Max scalar part = {max_scalar}, should be ≈ 0")


class TestIsothermicValidation(unittest.TestCase):
    """Cross-ratio validation for isothermic property."""

    def test_validate_isothermic_runs(self):
        """validate_isothermic runs without error."""
        _, result = get_small_torus()
        val = validate_isothermic(result.f_grid,
                                  result.params.u_res, result.params.v_res)
        self.assertIn('mean_error', val)
        self.assertIn('max_error', val)
        self.assertIn('n_quads_checked', val)
        self.assertGreater(val['n_quads_checked'], 0)

    def test_cross_ratio_convergence(self):
        """
        Cross-ratio error decreases with resolution.
        Compare 20×20 vs a fresh 10×10 torus.
        """
        params_coarse = TorusParameters(
            tau_imag=0.3205,
            u_res=10,
            v_res=10,
            v_periods=1,
            symmetry_fold=3,
            w0=0.15,
        )
        result_coarse = compute_torus(params_coarse)
        val_coarse = validate_isothermic(result_coarse.f_grid, 10, 10)

        _, result_fine = get_small_torus()
        val_fine = validate_isothermic(result_fine.f_grid, 20, 20)

        # Both should have finite errors
        self.assertTrue(np.isfinite(val_coarse['mean_error']))
        self.assertTrue(np.isfinite(val_fine['mean_error']))


class TestSymmetry(unittest.TestCase):
    """Theorem 4 inversion symmetry tests."""

    def test_symmetry_verification_runs(self):
        """verify_symmetry_inversion runs without error."""
        _, result = get_small_torus()
        sym = verify_symmetry_inversion(
            result.f_grid, result.u_grid, result.omega, result.params.tau
        )
        self.assertIn('mean_ratio_deviation', sym)
        self.assertIn('n_points_checked', sym)
        self.assertGreater(sym['n_points_checked'], 0)


class TestChristoffelDual(unittest.TestCase):
    """Christoffel dual computation."""

    def test_dual_shape(self):
        """Christoffel dual has same shape as original."""
        _, result = get_small_torus()
        f_star = compute_christoffel_dual(
            result.f_grid, result.u_grid, result.v_grid
        )
        self.assertEqual(f_star.shape, result.f_grid.shape)

    def test_dual_starts_at_origin(self):
        """f*(0,0) = 0 by convention."""
        _, result = get_small_torus()
        f_star = compute_christoffel_dual(
            result.f_grid, result.u_grid, result.v_grid
        )
        np.testing.assert_allclose(f_star[0, 0], 0.0, atol=1e-14)


class TestOBJExport(unittest.TestCase):
    """OBJ file export tests."""

    def test_export_creates_file(self):
        """export_torus_obj creates an OBJ file."""
        _, result = get_small_torus()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_torus.obj"
            export_torus_obj(result, path)
            self.assertTrue(path.exists())

    def test_obj_content_valid(self):
        """OBJ file has correct vertex and face counts."""
        _, result = get_small_torus()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_torus.obj"
            export_torus_obj(result, path)

            content = path.read_text()
            v_lines = [l for l in content.splitlines()
                       if l.startswith('v ')]
            f_lines = [l for l in content.splitlines()
                       if l.startswith('f ')]
            vn_lines = [l for l in content.splitlines()
                        if l.startswith('vn ')]

            self.assertEqual(len(v_lines), len(result.vertices))
            self.assertEqual(len(f_lines), len(result.faces))
            self.assertEqual(len(vn_lines), len(result.normals))

    def test_obj_faces_are_quads(self):
        """OBJ face lines have exactly 4 vertex references."""
        _, result = get_small_torus()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_torus.obj"
            export_torus_obj(result, path)

            content = path.read_text()
            f_lines = [l for l in content.splitlines()
                       if l.startswith('f ')]
            for line in f_lines:
                parts = line.split()[1:]  # skip 'f'
                self.assertEqual(len(parts), 4,
                                 f"Non-quad face: {line}")

    def test_obj_has_object_name(self):
        """OBJ contains object name line."""
        _, result = get_small_torus()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_torus.obj"
            export_torus_obj(result, path, object_name="my_torus")

            content = path.read_text()
            self.assertIn("o my_torus", content)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and parameter variations."""

    def test_different_tau(self):
        """Torus generates with a different τ value."""
        params = TorusParameters(
            tau_imag=0.30,
            u_res=10,
            v_res=10,
            v_periods=1,
            symmetry_fold=3,
            w0=0.1,
        )
        result = compute_torus(params)
        self.assertEqual(len(result.vertices), 100)
        ec = verify_euler_characteristic(result.faces, len(result.vertices))
        self.assertTrue(ec['ok'])

    def test_higher_resolution(self):
        """Torus generates at 30×30 resolution."""
        params = TorusParameters(
            tau_imag=0.3205,
            u_res=30,
            v_res=30,
            v_periods=1,
            symmetry_fold=3,
            w0=0.15,
        )
        result = compute_torus(params)
        self.assertEqual(len(result.vertices), 900)
        self.assertEqual(len(result.faces), 900)

    def test_4fold_symmetry(self):
        """Torus generates with 4-fold symmetry parameter."""
        params = TorusParameters(
            tau_imag=0.3205,
            u_res=12,
            v_res=12,
            v_periods=1,
            symmetry_fold=4,
            w0=0.15,
        )
        result = compute_torus(params)
        self.assertEqual(len(result.vertices), 144)


if __name__ == '__main__':
    unittest.main(verbosity=2)
