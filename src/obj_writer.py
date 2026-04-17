"""
Wavefront OBJ writer for Bonnet-pair mesh export.

Supports vertex colors, vertex normals, face-only export,
optional MTL material reference, and header comments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_obj(
    path: str | Path,
    vertices: np.ndarray,
    faces: list[list[int]],
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    object_name: str = "bonnet_object",
    header: Sequence[str] = (),
    mtl_file: str | None = None,
    material_name: str | None = None,
    scale_mm: float = 1.0,
) -> Path:
    """Write a triangle mesh as Wavefront OBJ."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(vertices, dtype=float)

    if scale_mm != 1.0:
        verts = verts * float(scale_mm)

    has_normals = normals is not None and len(normals) == len(verts)
    has_colors = colors is not None and len(colors) == len(verts)

    with target.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write(f"# Vertices: {len(verts)}\n")
        f.write(f"# Faces: {len(faces)}\n")

        if mtl_file:
            f.write(f"\nmtllib {mtl_file}\n")
            if material_name:
                f.write(f"usemtl {material_name}\n")

        f.write(f"\no {object_name}\n\n")

        if has_colors:
            c = np.asarray(colors, dtype=float)
            for v, col in zip(verts, c):
                f.write(
                    f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n"
                )
        else:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if has_normals:
            f.write("\n")
            nrm = np.asarray(normals, dtype=float)
            for n in nrm:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("\n")
        for face in faces:
            indices_1 = [str(idx + 1) for idx in face]
            if has_normals:
                f.write("f " + " ".join(f"{i}//{i}" for i in indices_1) + "\n")
            else:
                f.write("f " + " ".join(indices_1) + "\n")

    return target


def write_mtl(
    path: str | Path,
    material_name: str,
    *,
    ka: tuple[float, float, float] = (0.05, 0.05, 0.05),
    kd: tuple[float, float, float] = (0.42, 0.38, 0.34),
    ks: tuple[float, float, float] = (0.22, 0.20, 0.18),
    ns: float = 50.0,
    d: float = 1.0,
    comment: str = "",
) -> Path:
    """Write a Wavefront MTL material file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {ka[0]:.3f} {ka[1]:.3f} {ka[2]:.3f}\n")
        f.write(f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n")
        f.write(f"Ks {ks[0]:.3f} {ks[1]:.3f} {ks[2]:.3f}\n")
        f.write(f"Ns {ns:.1f}\n")
        f.write(f"d {d:.2f}\n")
        f.write("illum 2\n")

    return target
