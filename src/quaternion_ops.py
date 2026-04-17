"""
Quaternion Algebra — Pure ℍ ≅ ℝ⁴ and Im(ℍ) ≅ ℝ³.

Representation: q = [q₀, q₁, q₂, q₃] as numpy array (float64 or complex128).
  q = q₀ + q₁i + q₂j + q₃k

Conventions:
  - ij = k, jk = i, ki = j  (right-handed Hamilton)
  - Im(ℍ) product: XY = -⟨X,Y⟩ + X×Y
  - Rotation: X ↦ q⁻¹Xq  rotates around Im(q)/|Im(q)| by 2·arccos(q₀/|q|)
  - S³ ≅ ℍ₁ = {q ∈ ℍ : |q| = 1} ≅ SU(2)
"""
from __future__ import annotations

import numpy as np
from typing import Union

Quat = np.ndarray  # shape (4,) or (N, 4)


def quat(w: float = 0, x: float = 0, y: float = 0, z: float = 0) -> Quat:
    return np.array([w, x, y, z], dtype=np.float64)

def quat_from_scalar(s: float) -> Quat:
    return np.array([s, 0, 0, 0], dtype=np.float64)

def quat_from_vector(v: np.ndarray) -> Quat:
    return np.array([0, v[0], v[1], v[2]], dtype=np.float64)

def quat_from_complex(c: complex) -> Quat:
    return np.array([c.real, c.imag, 0, 0], dtype=np.float64)

def quat_i() -> Quat:
    return np.array([0, 1, 0, 0], dtype=np.float64)

def quat_j() -> Quat:
    return np.array([0, 0, 1, 0], dtype=np.float64)

def quat_k() -> Quat:
    return np.array([0, 0, 0, 1], dtype=np.float64)

def qmul(p: Quat, q: Quat) -> Quat:
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dtype=p.dtype)

def qconj(q: Quat) -> Quat:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

def qnorm_sq(q: Quat) -> float:
    return float(np.dot(q, q))

def qnorm(q: Quat) -> float:
    return float(np.sqrt(np.dot(q, q)))

def qinv(q: Quat) -> Quat:
    nsq = qnorm_sq(q)
    if nsq < 1e-30:
        raise ValueError("Cannot invert zero quaternion")
    return qconj(q) / nsq

def qnormalize(q: Quat) -> Quat:
    n = qnorm(q)
    if n < 1e-30:
        raise ValueError("Cannot normalize zero quaternion")
    return q / n

def scalar_part(q: Quat) -> float:
    return float(q[0])

def vector_part(q: Quat) -> np.ndarray:
    return q[1:4].copy()

def is_pure(q: Quat, tol: float = 1e-12) -> bool:
    return abs(q[0]) < tol

def is_unit(q: Quat, tol: float = 1e-10) -> bool:
    return abs(qnorm(q) - 1.0) < tol

def imh_product(X: Quat, Y: Quat) -> Quat:
    return qmul(X, Y)

def dot_r3(X: Quat, Y: Quat) -> float:
    return -scalar_part(qmul(X, Y))

def cross_r3(X: Quat, Y: Quat) -> Quat:
    product = qmul(X, Y)
    return np.array([0, product[1], product[2], product[3]], dtype=product.dtype)

def rotate_by_unit(X: Quat, q: Quat) -> Quat:
    return qmul(qmul(qconj(q), X), q)

def rotate_by(X: Quat, q: Quat) -> Quat:
    return qmul(qmul(q, X), qconj(q))

def rotation_quaternion(axis: np.ndarray, angle: float) -> Quat:
    axis_n = axis / np.linalg.norm(axis)
    half = angle / 2
    return np.array([
        np.cos(half),
        np.sin(half) * axis_n[0],
        np.sin(half) * axis_n[1],
        np.sin(half) * axis_n[2],
    ], dtype=np.float64)

def su2_matrix(q: Quat) -> np.ndarray:
    a, b, c, d = q
    return np.array([
        [a + 1j*b, c + 1j*d],
        [-c + 1j*d, a - 1j*b],
    ], dtype=np.complex128)

def from_su2_matrix(M: np.ndarray) -> Quat:
    a = M[0, 0].real
    b = M[0, 0].imag
    c = M[0, 1].real
    d = M[0, 1].imag
    q = np.array([a, b, c, d], dtype=np.float64)
    return qnormalize(q)

def hopf_map(q: Quat) -> np.ndarray:
    k = quat_k()
    rotated = rotate_by_unit(k, q)
    return vector_part(rotated)

def qmul_complex(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dtype=np.complex128)

def qconj_complex(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

def qinv_complex(q: np.ndarray) -> np.ndarray:
    nsq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    return qconj_complex(q) / nsq

def qmul_batch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    q0, q1, q2, q3 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.column_stack([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ])

def qconj_batch(Q: np.ndarray) -> np.ndarray:
    result = Q.copy()
    result[:, 1:] *= -1
    return result
