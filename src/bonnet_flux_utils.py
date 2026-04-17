"""
Bonnet Flux Utilities — shared helpers for flux OBJ generation.

Provides curvature-based coloring, flux wireframe export, and
quad triangulation for Bonnet-pair mesh visualization.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np


def triangulate_quads(faces) -> list[list[int]]:
    """Split quad faces into triangles.

    Each quad [a, b, c, d] -> two triangles [a, b, c], [a, c, d].
    Triangles are passed through unchanged.
    """
    triangles = []
    for f in faces:
        if len(f) == 3:
            triangles.append(list(f))
        elif len(f) >= 4:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
    return triangles


def compute_curvature_proxy(vertices: np.ndarray, n_u: int, n_v: int) -> np.ndarray:
    """
    Approximate mean curvature magnitude per vertex via discrete Laplacian.
    Returns (N,) array normalized to [0, 1].
    """
    N = len(vertices)
    curv = np.zeros(N)

    neighbor_sum = np.zeros_like(vertices)
    neighbor_count = np.zeros(N)

    for iu in range(n_u):
        for iv in range(n_v):
            idx = iu * n_v + iv
            neighbors = [
                ((iu + 1) % n_u) * n_v + iv,
                ((iu - 1) % n_u) * n_v + iv,
                iu * n_v + (iv + 1) % n_v,
                iu * n_v + (iv - 1) % n_v,
            ]
            for ni in neighbors:
                neighbor_sum[idx] += vertices[ni]
                neighbor_count[idx] += 1

    for i in range(N):
        if neighbor_count[i] > 0:
            lap = neighbor_sum[i] / neighbor_count[i] - vertices[i]
            curv[i] = np.linalg.norm(lap)

    c_min, c_max = curv.min(), curv.max()
    if c_max - c_min > 1e-10:
        curv = (curv - c_min) / (c_max - c_min)
    else:
        curv[:] = 0.5
    return curv


def color_map_surface(
    n_u: int, n_v: int,
    curvature: np.ndarray,
    hue_range: tuple,
    sat_base: float,
    val_base: float,
    hue_shift: float = 0.0,
) -> np.ndarray:
    """
    Map (u, v, curvature) to RGB colors via HSV.

    u -> hue sweep across hue_range
    v -> value modulation (brightness wave)
    curvature -> saturation boost + value accent

    Returns (N, 3) RGB in [0, 1].
    """
    N = n_u * n_v
    colors = np.zeros((N, 3))
    h_start, h_end = hue_range

    for iu in range(n_u):
        u_norm = iu / n_u
        for iv in range(n_v):
            v_norm = iv / n_v
            idx = iu * n_v + iv

            h = h_start + u_norm * (h_end - h_start) + curvature[idx] * 0.08
            h = (h + hue_shift) % 1.0
            s = min(sat_base + curvature[idx] * 0.15, 1.0)
            val = max(0.3, min(
                val_base + 0.15 * np.sin(v_norm * 2 * np.pi) + curvature[idx] * 0.15,
                1.0,
            ))
            r, g, b = colorsys.hsv_to_rgb(h, s, val)
            colors[idx] = [r, g, b]

    return colors


def compute_flux_edges(n_u: int, n_v: int,
                       diag_stride: int = 3) -> list[tuple[int, int]]:
    """
    Generate flux-standard edges:
      - Horizontal (u-direction, wrap around)
      - Vertical (v-direction, wrap around for torus)
      - Diagonal reinforcement every diag_stride steps
    """
    edges = []
    for iu in range(n_u):
        for iv in range(n_v):
            idx = iu * n_v + iv
            edges.append((idx, iu * n_v + (iv + 1) % n_v))
            edges.append((idx, ((iu + 1) % n_u) * n_v + iv))
            if iu % diag_stride == 0 and iv % diag_stride == 0:
                edges.append((idx, ((iu + 1) % n_u) * n_v + (iv + 1) % n_v))
    return edges


def write_flux_wireframe_obj(
    path: Path,
    vertices: np.ndarray,
    edges: list[tuple[int, int]],
    colors: np.ndarray,
    object_name: str = "flux_wireframe",
    header: list[str] = None,
):
    """Write OBJ with line elements (l) for flux wireframe visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        if header:
            for h in header:
                f.write(f"# {h}\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(edges)}\n")
        f.write(f"\no {object_name}\n\n")

        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")

        f.write("\n")
        for e in edges:
            f.write(f"l {e[0]+1} {e[1]+1}\n")
