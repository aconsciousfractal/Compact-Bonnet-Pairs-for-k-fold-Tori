#!/usr/bin/env python3
"""
Bonnet Atlas → OBJ/MTL/Flux/JSON Batch Exporter — Google Colab Runner

Reads checkpoint colab.json (240 atlas entries) and generates for each entry:
  - {name}_f_plus.obj   (Bonnet f+ surface, triangulated, with vertex colors + normals)
  - {name}_f_minus.obj  (Bonnet f- surface)
  - {name}_flux.obj     (Flux wireframe on f+ surface)
  - {name}_material.mtl (Phong materials for f+ and f-)
  - {name}_metadata.json (full parameters + verification metrics)

Checkpoints every 5 OBJs.  Progress bar via tqdm.

Usage on Colab:
  1. Upload "Bonnet's Problem" folder to Google Drive
  2. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
  3. cd to the folder:
     import os; os.chdir("/content/drive/MyDrive/Bonnet's Problem")
  4. Run:  !python run_colab_export_atlas.py
  
  Optional arguments:
     --top N          Export only top N entries (default: all 240)
     --res N          Export resolution NxN (default: 80)
     --start N        Skip first N entries (resume after crash)
     --checkpoint P   Path to checkpoint colab.json
"""
from __future__ import annotations

# ============================================================================
# SECTION 0: Dependencies + PAPP Core Stubs
# ============================================================================
import importlib
import os
import sys
import types

def _install_deps():
    import subprocess
    for pkg in ['mpmath', 'tqdm']:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_install_deps()

import argparse
import colorsys
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path(os.getcwd()).resolve()

# Scan Colab mount points
if not (PROJECT_DIR / 'src').is_dir():
    for candidate in [
        Path('/content/drive/MyDrive/Bonnet'),
        Path("/content/drive/My Drive/Bonnet"),
        Path("/content/drive/MyDrive/Bonnet's Problem"),
        Path("/content/drive/My Drive/Bonnet's Problem"),
    ]:
        if (candidate / 'src').is_dir():
            PROJECT_DIR = candidate.resolve()
            os.chdir(str(PROJECT_DIR))
            break

if not (PROJECT_DIR / 'src').is_dir():
    raise FileNotFoundError(
        f"Cannot find 'src/' directory in {PROJECT_DIR}.\n"
        f"Make sure you cd to the repo root before running."
    )

print(f"  PROJECT_DIR = {PROJECT_DIR}")
sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# Core stubs (identical to run_colab_atlas_campaign.py)
# ---------------------------------------------------------------------------
def _ensure_core_stubs():
    """Build minimal core.* modules in-memory so Bonnet scripts import OK."""
    if 'core' not in sys.modules:
        core = types.ModuleType('core')
        core.__path__ = [str(PROJECT_DIR / '_core_stub')]
        sys.modules['core'] = core

    if 'core.utils' not in sys.modules:
        cu = types.ModuleType('core.utils')
        cu.__path__ = [str(PROJECT_DIR / '_core_stub' / 'utils')]
        sys.modules['core.utils'] = cu
        sys.modules['core'].__dict__['utils'] = cu

    if 'core.io' not in sys.modules:
        cio = types.ModuleType('core.io')
        cio.__path__ = [str(PROJECT_DIR / '_core_stub' / 'io')]
        sys.modules['core.io'] = cio
        sys.modules['core'].__dict__['io'] = cio

    # ---- core.utils.theta_functions ----
    import mpmath
    tf = types.ModuleType('core.utils.theta_functions')
    def _nome(tau):
        if tau.imag <= 0: raise ValueError(f"Im(τ) must be > 0, got {tau.imag}")
        return complex(mpmath.exp(1j * mpmath.pi * tau))
    def _t1(z, tau, derivative=0): return complex(mpmath.jtheta(1, z, _nome(tau), derivative=derivative))
    def _t2(z, tau, derivative=0): return complex(mpmath.jtheta(2, z, _nome(tau), derivative=derivative))
    def _t3(z, tau, derivative=0): return complex(mpmath.jtheta(3, z, _nome(tau), derivative=derivative))
    def _t4(z, tau, derivative=0): return complex(mpmath.jtheta(4, z, _nome(tau), derivative=derivative))
    def _t1v(za, tau, derivative=0):
        q = _nome(tau)
        return np.array([complex(mpmath.jtheta(1, z, q, derivative=derivative)) for z in za.flat]).reshape(za.shape)
    def _t2v(za, tau, derivative=0):
        q = _nome(tau)
        return np.array([complex(mpmath.jtheta(2, z, q, derivative=derivative)) for z in za.flat]).reshape(za.shape)
    def _t1p0(tau): return _t1(0, tau, derivative=1)
    tf.nome_from_tau = _nome
    tf.theta1 = _t1; tf.theta2 = _t2; tf.theta3 = _t3; tf.theta4 = _t4
    tf.theta1_vec = _t1v; tf.theta2_vec = _t2v; tf.theta1_prime_zero = _t1p0
    sys.modules['core.utils.theta_functions'] = tf

    # ---- core.utils.quaternion_ops ----
    qo = types.ModuleType('core.utils.quaternion_ops')
    Quat = np.ndarray
    qo.Quat = Quat
    def _quat(w=0,x=0,y=0,z=0): return np.array([w,x,y,z], dtype=np.float64)
    def _qfs(s): return np.array([s,0,0,0], dtype=np.float64)
    def _qfv(v): return np.array([0,v[0],v[1],v[2]], dtype=np.float64)
    def _qfc(c): return np.array([c.real,c.imag,0,0], dtype=np.float64)
    def _qi(): return np.array([0,1,0,0], dtype=np.float64)
    def _qj(): return np.array([0,0,1,0], dtype=np.float64)
    def _qk(): return np.array([0,0,0,1], dtype=np.float64)
    def _qmul(p,q):
        p0,p1,p2,p3=p; q0,q1,q2,q3=q
        return np.array([p0*q0-p1*q1-p2*q2-p3*q3, p0*q1+p1*q0+p2*q3-p3*q2,
                         p0*q2-p1*q3+p2*q0+p3*q1, p0*q3+p1*q2-p2*q1+p3*q0], dtype=p.dtype)
    def _qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]], dtype=q.dtype)
    def _qnsq(q): return float(np.dot(q,q))
    def _qn(q): return float(np.sqrt(np.dot(q,q)))
    def _qinv(q):
        nsq=_qnsq(q)
        if nsq<1e-30: raise ValueError("Cannot invert zero quaternion")
        return _qconj(q)/nsq
    def _qnorm(q):
        n=_qn(q)
        if n<1e-30: raise ValueError("Cannot normalize zero quaternion")
        return q/n
    def _sp(q): return float(q[0])
    def _vp(q): return q[1:4].copy()
    def _ip(q, tol=1e-12): return abs(q[0])<tol
    def _iu(q, tol=1e-10): return abs(_qn(q)-1.0)<tol
    def _imhp(X,Y): return _qmul(X,Y)
    def _dr3(X,Y): return -_sp(_qmul(X,Y))
    def _cr3(X,Y):
        p=_qmul(X,Y); return np.array([0,p[1],p[2],p[3]], dtype=p.dtype)
    def _rbu(X,q): return _qmul(_qmul(_qconj(q),X),q)
    def _rb(X,q): return _qmul(_qmul(q,X),_qconj(q))
    def _rq(axis,angle):
        an=axis/np.linalg.norm(axis); h=angle/2
        return np.array([np.cos(h),np.sin(h)*an[0],np.sin(h)*an[1],np.sin(h)*an[2]], dtype=np.float64)
    def _su2(q):
        a,b,c,d=q
        return np.array([[a+1j*b,c+1j*d],[-c+1j*d,a-1j*b]], dtype=np.complex128)
    def _fsu2(M):
        a,b,c,d=M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag
        return _qnorm(np.array([a,b,c,d], dtype=np.float64))
    def _hm(q): return _vp(_rbu(_qk(),q))
    def _qmc(p,q):
        p0,p1,p2,p3=p; q0,q1,q2,q3=q
        return np.array([p0*q0-p1*q1-p2*q2-p3*q3, p0*q1+p1*q0+p2*q3-p3*q2,
                         p0*q2-p1*q3+p2*q0+p3*q1, p0*q3+p1*q2-p2*q1+p3*q0], dtype=np.complex128)
    def _qcc(q): return np.array([q[0],-q[1],-q[2],-q[3]], dtype=q.dtype)
    def _qic(q): return _qcc(q)/(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
    def _qmb(P,Q):
        p0,p1,p2,p3=P[:,0],P[:,1],P[:,2],P[:,3]; q0,q1,q2,q3=Q[:,0],Q[:,1],Q[:,2],Q[:,3]
        return np.column_stack([p0*q0-p1*q1-p2*q2-p3*q3, p0*q1+p1*q0+p2*q3-p3*q2,
                                p0*q2-p1*q3+p2*q0+p3*q1, p0*q3+p1*q2-p2*q1+p3*q0])
    def _qcb(Q): r=Q.copy(); r[:,1:]*=-1; return r

    for n,f in [('quat',_quat),('quat_from_scalar',_qfs),('quat_from_vector',_qfv),
                ('quat_from_complex',_qfc),('quat_i',_qi),('quat_j',_qj),('quat_k',_qk),
                ('qmul',_qmul),('qconj',_qconj),('qnorm_sq',_qnsq),('qnorm',_qn),
                ('qinv',_qinv),('qnormalize',_qnorm),('scalar_part',_sp),('vector_part',_vp),
                ('is_pure',_ip),('is_unit',_iu),('imh_product',_imhp),('dot_r3',_dr3),
                ('cross_r3',_cr3),('rotate_by_unit',_rbu),('rotate_by',_rb),
                ('rotation_quaternion',_rq),('su2_matrix',_su2),('from_su2_matrix',_fsu2),
                ('hopf_map',_hm),('qmul_complex',_qmc),('qconj_complex',_qcc),
                ('qinv_complex',_qic),('qmul_batch',_qmb),('qconj_batch',_qcb)]:
        qo.__dict__[n] = f
    sys.modules['core.utils.quaternion_ops'] = qo

    # ---- core.utils.discrete_diffgeo ----
    ddg = types.ModuleType('core.utils.discrete_diffgeo')
    def _cmt_pc(f_grid, u_grid, v_grid):
        n_u,n_v=f_grid.shape[:2]; du=u_grid[1]-u_grid[0] if len(u_grid)>1 else 1.0; dv=v_grid[1]-v_grid[0] if len(v_grid)>1 else 1.0
        f_r3=f_grid[:,:,1:4]; fu=np.zeros_like(f_r3); fv=np.zeros_like(f_r3)
        for iu in range(n_u): fu[iu]=(f_r3[(iu+1)%n_u]-f_r3[(iu-1)%n_u])/(2*du)
        for iv in range(n_v): fv[:,iv]=(f_r3[:,(iv+1)%n_v]-f_r3[:,(iv-1)%n_v])/(2*dv)
        E=np.sum(fu**2,axis=2); F=np.sum(fu*fv,axis=2); G=np.sum(fv**2,axis=2)
        return {'E':E,'F':F,'G':G,'f_u':fu,'f_v':fv}
    def _cmcf(f_grid, u_grid, v_grid):
        n_u,n_v=f_grid.shape[:2]; du=u_grid[1]-u_grid[0] if len(u_grid)>1 else 1.0; dv=v_grid[1]-v_grid[0] if len(v_grid)>1 else 1.0
        f_r3=f_grid[:,:,1:4]
        fu=np.zeros_like(f_r3); fv=np.zeros_like(f_r3)
        for iu in range(n_u): fu[iu]=(f_r3[(iu+1)%n_u]-f_r3[(iu-1)%n_u])/(2*du)
        for iv in range(n_v): fv[:,iv]=(f_r3[:,(iv+1)%n_v]-f_r3[:,(iv-1)%n_v])/(2*dv)
        fuu=np.zeros_like(f_r3); fvv=np.zeros_like(f_r3); fuv=np.zeros_like(f_r3)
        for iu in range(n_u):
            ip,im=(iu+1)%n_u,(iu-1)%n_u; fuu[iu]=(f_r3[ip]-2*f_r3[iu]+f_r3[im])/du**2
        for iv in range(n_v):
            jp,jm=(iv+1)%n_v,(iv-1)%n_v; fvv[:,iv]=(f_r3[:,jp]-2*f_r3[:,iv]+f_r3[:,jm])/dv**2
        for iu in range(n_u):
            ip,im=(iu+1)%n_u,(iu-1)%n_u
            for iv in range(n_v):
                jp,jm=(iv+1)%n_v,(iv-1)%n_v
                fuv[iu,iv]=(f_r3[ip,jp]-f_r3[ip,jm]-f_r3[im,jp]+f_r3[im,jm])/(4*du*dv)
        nv=np.cross(fu,fv); nn=np.linalg.norm(nv,axis=2,keepdims=True); nv=nv/np.maximum(nn,1e-30)
        E=np.sum(fu**2,axis=2); F=np.sum(fu*fv,axis=2); G=np.sum(fv**2,axis=2)
        e=np.sum(fuu*nv,axis=2); fc=np.sum(fuv*nv,axis=2); g=np.sum(fvv*nv,axis=2)
        dn=2*(E*G-F**2); dn=np.where(np.abs(dn)<1e-30,1e-30,dn)
        return (e*G-2*fc*F+g*E)/dn
    def _crq(f00,f10,f11,f01):
        def v2q(v): return np.array([0,v[0],v[1],v[2]], dtype=np.float64)
        d1=v2q(f10-f00); d12=v2q(f11-f10); d2b=v2q(f11-f01); d2=v2q(f01-f00)
        return _qmul(_qmul(_qmul(d1,_qinv(d12)),d2b),_qinv(d2))
    ddg.compute_metric_tensor_periodic_central = _cmt_pc
    ddg.compute_mean_curvature_fd = _cmcf
    ddg.cross_ratio_quad = _crq
    sys.modules['core.utils.discrete_diffgeo'] = ddg

    # ---- core.utils.mesh_ops ----
    mo = types.ModuleType('core.utils.mesh_ops')
    def _btf(n_u, n_v):
        faces=[]
        for iu in range(n_u):
            iu_next=(iu+1)%n_u
            for iv in range(n_v):
                iv_next=(iv+1)%n_v
                faces.append([iu*n_v+iv, iu_next*n_v+iv, iu_next*n_v+iv_next, iu*n_v+iv_next])
        return faces
    def _tq(faces):
        tris=[]
        for f in faces:
            if len(f)==3: tris.append(list(f))
            elif len(f)>=4: tris.append([f[0],f[1],f[2]]); tris.append([f[0],f[2],f[3]])
        return tris
    mo.build_torus_faces = _btf
    mo.triangulate_quads = _tq
    sys.modules['core.utils.mesh_ops'] = mo

    # ---- core.utils.elliptic (stub) ----
    sys.modules['core.utils.elliptic'] = types.ModuleType('core.utils.elliptic')
    # ---- core.utils.weierstrass (stub) ----
    sys.modules['core.utils.weierstrass'] = types.ModuleType('core.utils.weierstrass')

    # ---- core.io.obj_writer ----
    ow = types.ModuleType('core.io.obj_writer')
    def _write_obj(path, vertices, faces, *, normals=None, colors=None,
                   object_name="papp_object", header=(), mtl_file=None, material_name=None,
                   scale_mm=None):
        target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        verts=np.asarray(vertices, dtype=float)
        if scale_mm is not None:
            verts = verts * float(scale_mm)
        hn=normals is not None and len(normals)==len(verts)
        hc=colors is not None and len(colors)==len(verts)
        with target.open("w", encoding="utf-8") as f:
            for l in header: f.write(f"# {l}\n")
            f.write(f"# Vertices: {len(verts)}\n# Faces: {len(faces)}\n")
            if mtl_file:
                f.write(f"\nmtllib {mtl_file}\n")
                if material_name: f.write(f"usemtl {material_name}\n")
            f.write(f"\no {object_name}\n\n")
            if hc:
                c=np.asarray(colors, dtype=float)
                for v,co in zip(verts,c): f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {co[0]:.4f} {co[1]:.4f} {co[2]:.4f}\n")
            else:
                for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            if hn:
                f.write("\n"); nrm=np.asarray(normals, dtype=float)
                for n in nrm: f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("\n")
            for face in faces:
                idx1=[str(i+1) for i in face]
                if hn: f.write("f "+" ".join(f"{i}//{i}" for i in idx1)+"\n")
                else: f.write("f "+" ".join(idx1)+"\n")
        return target
    def _write_mtl(path, material_name, *, ka=(0.05,0.05,0.05), kd=(0.42,0.38,0.34),
                   ks=(0.22,0.20,0.18), ns=50.0, d=1.0, comment=""):
        target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            if comment: f.write(f"# {comment}\n")
            f.write(f"newmtl {material_name}\nKa {ka[0]:.3f} {ka[1]:.3f} {ka[2]:.3f}\n")
            f.write(f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\nKs {ks[0]:.3f} {ks[1]:.3f} {ks[2]:.3f}\n")
            f.write(f"Ns {ns:.1f}\nd {d:.2f}\nillum 2\n")
        return target
    ow.write_obj = _write_obj
    ow.write_mtl = _write_mtl
    sys.modules['core.io.obj_writer'] = ow

_ensure_core_stubs()

# ============================================================================
# SECTION 1: Imports
# ============================================================================
from tqdm import tqdm

from src.bonnet_pair import closure_gate
from src.isothermic_torus import compute_vertex_normals
from src.theorem9_perturbation import (
    Theorem9ForcingSpec,
    build_theorem9_basis_from_forcing,
    build_theorem9_seed_from_theorem7,
    solve_theorem9_perturbation,
    verify_theorem9_bonnet_pipeline,
)
from src.obj_writer import write_obj, write_mtl
from src.bonnet_flux_utils import triangulate_quads

# ============================================================================
# SECTION 2: Forcing Library (must match run_colab_atlas_campaign.py)
# ============================================================================
FORCING_LIBRARY = [
    Theorem9ForcingSpec(label='smooth_h3_p011', family='smooth_low_freq', harmonic=3, phase=0.11),
    Theorem9ForcingSpec(label='smooth_h4_p019', family='smooth_low_freq', harmonic=4, phase=0.19),
    Theorem9ForcingSpec(label='smooth_h5_p013', family='smooth_low_freq', harmonic=5, phase=0.13),
    Theorem9ForcingSpec(label='smooth_h5_p031', family='smooth_low_freq', harmonic=5, phase=0.31),
    Theorem9ForcingSpec(label='phase_h6_p029', family='phase_shifted', harmonic=6, phase=0.29),
    Theorem9ForcingSpec(label='phase_h7_p041', family='phase_shifted', harmonic=7, phase=0.41),
    Theorem9ForcingSpec(label='phase_h7_p017', family='phase_shifted', harmonic=7, phase=0.17),
    Theorem9ForcingSpec(label='phase_h8_p023', family='phase_shifted', harmonic=8, phase=0.23),
    Theorem9ForcingSpec(label='sym_h8_p011', family='symmetry_aware', harmonic=8, phase=0.11),
    Theorem9ForcingSpec(label='sym_h9_p007', family='symmetry_aware', harmonic=9, phase=0.07),
    Theorem9ForcingSpec(label='sym_h10_p017', family='symmetry_aware', harmonic=10, phase=0.17),
    Theorem9ForcingSpec(label='sym_h12_p013', family='symmetry_aware', harmonic=12, phase=0.13),
    Theorem9ForcingSpec(label='anti_h9_p033', family='anti_reflection', harmonic=9, phase=0.33),
    Theorem9ForcingSpec(label='anti_h11_p027', family='anti_reflection', harmonic=11, phase=0.27),
    Theorem9ForcingSpec(label='anti_h11_p043', family='anti_reflection', harmonic=11, phase=0.43),
    Theorem9ForcingSpec(label='anti_h13_p019', family='anti_reflection', harmonic=13, phase=0.19),
    Theorem9ForcingSpec(label='hh_h14_p011', family='high_harmonic', harmonic=14, phase=0.11),
    Theorem9ForcingSpec(label='hh_h16_p023', family='high_harmonic', harmonic=16, phase=0.23),
    Theorem9ForcingSpec(label='hh_h18_p007', family='high_harmonic', harmonic=18, phase=0.07),
    Theorem9ForcingSpec(label='hh_h20_p031', family='high_harmonic', harmonic=20, phase=0.31),
]
FORCING_MAP = {f.label: f for f in FORCING_LIBRARY}

# ============================================================================
# SECTION 3: Utility functions
# ============================================================================

def _jsonable(obj):
    if isinstance(obj, dict): return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, Path): return str(obj)
    return obj


def compute_curvature_proxy(vertices: np.ndarray, n_u: int, n_v: int) -> np.ndarray:
    """Approximate mean curvature per vertex via discrete Laplacian, normalized to [0,1]."""
    N = len(vertices)
    neighbor_sum = np.zeros_like(vertices)
    neighbor_count = np.zeros(N)
    for iu in range(n_u):
        for iv in range(n_v):
            idx = iu * n_v + iv
            for ni in [
                ((iu + 1) % n_u) * n_v + iv,
                ((iu - 1) % n_u) * n_v + iv,
                iu * n_v + (iv + 1) % n_v,
                iu * n_v + (iv - 1) % n_v,
            ]:
                neighbor_sum[idx] += vertices[ni]
                neighbor_count[idx] += 1
    curv = np.zeros(N)
    for i in range(N):
        if neighbor_count[i] > 0:
            curv[i] = np.linalg.norm(neighbor_sum[i] / neighbor_count[i] - vertices[i])
    c_min, c_max = curv.min(), curv.max()
    if c_max - c_min > 1e-10:
        curv = (curv - c_min) / (c_max - c_min)
    else:
        curv[:] = 0.5
    return curv


def color_map_surface(n_u, n_v, curvature, hue_range, sat_base, val_base, hue_shift=0.0):
    """Map (u, v, curvature) → RGB via HSV."""
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
            val = max(0.3, min(val_base + 0.15 * np.sin(v_norm * 2 * np.pi) + curvature[idx] * 0.15, 1.0))
            colors[idx] = colorsys.hsv_to_rgb(h, s, val)
    return colors


def compute_flux_edges(n_u, n_v, diag_stride=3):
    """PAPP Flux wireframe edges (horizontal + vertical + diagonal)."""
    edges = []
    for iu in range(n_u):
        for iv in range(n_v):
            idx = iu * n_v + iv
            edges.append((idx, iu * n_v + (iv + 1) % n_v))
            edges.append((idx, ((iu + 1) % n_u) * n_v + iv))
            if iu % diag_stride == 0 and iv % diag_stride == 0:
                edges.append((idx, ((iu + 1) % n_u) * n_v + (iv + 1) % n_v))
    return edges


def write_flux_wireframe_obj(path, vertices, edges, colors, object_name="flux", header=None):
    """OBJ with line elements for flux wireframe visualization."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            for h in header: f.write(f"# {h}\n")
        f.write(f"# Vertices: {len(vertices)}\n# Edges: {len(edges)}\n\no {object_name}\n\n")
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        f.write("\n")
        for e in edges:
            f.write(f"l {e[0]+1} {e[1]+1}\n")


# ============================================================================
# SECTION 4: Export Checkpoint
# ============================================================================

class ExportCheckpoint:
    """Tracks which atlas entries have been exported. Saves every N items."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            self.data = {'exported': {}, 'errors': {}, 'started_at': time.time()}

    def is_done(self, key: str) -> bool:
        return key in self.data.get('exported', {})

    def mark_done(self, key: str, info: dict):
        self.data.setdefault('exported', {})[key] = _jsonable(info)

    def mark_error(self, key: str, error: str):
        self.data.setdefault('errors', {})[key] = error

    def save(self):
        self.data['saved_at'] = time.time()
        self.data['n_exported'] = len(self.data.get('exported', {}))
        self.data['n_errors'] = len(self.data.get('errors', {}))
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    @property
    def n_done(self):
        return len(self.data.get('exported', {}))


# ============================================================================
# SECTION 5: Single Entry Export
# ============================================================================

def export_single_entry(
    item: dict,
    seed_spec: dict,
    forcing: Theorem9ForcingSpec,
    export_dir: Path,
    export_res: int,
    n_half_samples: int = 51,
    solve_n_points: int = 50,
    solve_lin_points: int = 35,
    epsilon_geom: float = 0.3,
) -> dict:
    """
    Full pipeline for one atlas entry: seed → solve → torus → Bonnet pair → OBJ/MTL/flux/JSON.
    Returns metadata dict.
    """
    t0 = time.time()
    rank = item['rank']
    fold = item['symmetry_fold']
    name = f"rank{rank:03d}_fold{fold}_{item['seed_label']}__{item['forcing_label']}_eps{item['epsilon']:.0e}"

    entry_dir = export_dir / name
    entry_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Build seed ---
    seed = build_theorem9_seed_from_theorem7(
        tau_imag=seed_spec['tau_imag'],
        delta=seed_spec['delta'],
        s1=seed_spec['s1'],
        s2=seed_spec['s2'],
        symmetry_fold=seed_spec['symmetry_fold'],
        n_half_samples=n_half_samples,
    )

    # --- Step 2: Build basis + solve (with stored alpha as initial guess) ---
    basis = build_theorem9_basis_from_forcing(seed, forcing)
    solve = solve_theorem9_perturbation(
        seed=seed,
        epsilon=float(item['epsilon']),
        basis=basis,
        initial_alpha=np.asarray(item['alpha'], dtype=float),
        n_points=solve_n_points,
        linearization_n_points=solve_lin_points,
    )

    # --- Step 3: Full Bonnet pipeline at export resolution ---
    verification = verify_theorem9_bonnet_pipeline(
        seed=seed,
        solve=solve,
        epsilon_geom=epsilon_geom,
        u_res=export_res,
        v_res=export_res,
    )
    pair = verification.pair
    n_u, n_v = export_res, export_res
    ratio = fold * verification.torus.frame_result.rotation_angle / (2 * np.pi)

    # --- Step 4: Curvature + colors ---
    curv_plus = compute_curvature_proxy(pair.f_plus.vertices, n_u, n_v)
    curv_minus = compute_curvature_proxy(pair.f_minus.vertices, n_u, n_v)

    # Fold-dependent color scheme: 3-fold → blue-teal, 4-fold → amber-gold
    if fold == 3:
        colors_plus = color_map_surface(n_u, n_v, curv_plus, hue_range=(0.50, 0.62), sat_base=0.80, val_base=0.72)
        colors_minus = color_map_surface(n_u, n_v, curv_minus, hue_range=(0.50, 0.62), sat_base=0.70, val_base=0.66, hue_shift=0.06)
        kd_plus = (0.22, 0.54, 0.72)
        kd_minus = (0.18, 0.42, 0.60)
    else:
        colors_plus = color_map_surface(n_u, n_v, curv_plus, hue_range=(0.10, 0.24), sat_base=0.84, val_base=0.74)
        colors_minus = color_map_surface(n_u, n_v, curv_minus, hue_range=(0.10, 0.24), sat_base=0.74, val_base=0.68, hue_shift=0.06)
        kd_plus = (0.82, 0.54, 0.22)
        kd_minus = (0.72, 0.30, 0.18)

    # --- Step 5: Triangulate + normals ---
    tris_plus = triangulate_quads(pair.f_plus.faces)
    tris_minus = triangulate_quads(pair.f_minus.faces)
    nrm_plus = compute_vertex_normals(pair.f_plus.vertices, tris_plus)
    nrm_minus = compute_vertex_normals(pair.f_minus.vertices, tris_minus)

    # --- Step 6: Write OBJs ---
    header = [
        "PAPP Bonnet Atlas — Theorem 9 Perturbative Export",
        f"Rank: {rank}",
        f"seed: {item['seed_label']}",
        f"forcing: {item['forcing_label']} (family={item.get('forcing_family','')})",
        f"tau_imag = {seed_spec['tau_imag']}",
        f"delta = {seed_spec['delta']}",
        f"s1 = {seed_spec['s1']}",
        f"s2 = {seed_spec['s2']}",
        f"symmetry_fold = {fold}",
        f"epsilon = {item['epsilon']}",
        f"alpha = {item['alpha']}",
        f"validation_score = {item['validation_score']:.6e}",
    ]

    mtl_name = f"{name}_material.mtl"
    write_obj(
        entry_dir / f"{name}_f_plus.obj",
        pair.f_plus.vertices, tris_plus,
        normals=nrm_plus, colors=colors_plus,
        object_name=f"{name}_f_plus",
        header=header + ["Surface: f_plus (Bonnet + branch)"],
        mtl_file=mtl_name, material_name=f"{name}_plus",
    )
    write_obj(
        entry_dir / f"{name}_f_minus.obj",
        pair.f_minus.vertices, tris_minus,
        normals=nrm_minus, colors=colors_minus,
        object_name=f"{name}_f_minus",
        header=header + ["Surface: f_minus (Bonnet - branch)"],
        mtl_file=mtl_name, material_name=f"{name}_minus",
    )

    # --- Step 7: Write MTL ---
    mtl_path = entry_dir / mtl_name
    write_mtl(mtl_path, f"{name}_plus",
              ka=(0.03, 0.03, 0.03), kd=kd_plus, ks=(0.35, 0.30, 0.25), ns=80.0,
              comment=f"Bonnet atlas rank {rank}")
    with mtl_path.open("a", encoding="utf-8") as f:
        f.write(f"\nnewmtl {name}_minus\n")
        f.write("Ka 0.030 0.030 0.030\n")
        f.write(f"Kd {kd_minus[0]:.3f} {kd_minus[1]:.3f} {kd_minus[2]:.3f}\n")
        f.write("Ks 0.300 0.250 0.200\nNs 60.0\nd 1.00\nillum 2\n")

    # --- Step 8: Flux wireframe ---
    flux_edges = compute_flux_edges(n_u, n_v)
    write_flux_wireframe_obj(
        entry_dir / f"{name}_flux.obj",
        pair.f_plus.vertices, flux_edges, colors_plus,
        object_name=f"{name}_flux",
        header=header + ["Flux wireframe (on f_plus surface)"],
    )

    # --- Step 9: Metadata JSON ---
    closure = closure_gate(pair)
    metadata = {
        "name": name,
        "rank": rank,
        "kind": "bonnet_atlas_theorem9",
        "seed_label": item['seed_label'],
        "forcing_label": item['forcing_label'],
        "forcing_family": item.get('forcing_family', ''),
        "symmetry_fold": fold,
        "tau_imag": seed_spec['tau_imag'],
        "delta": seed_spec['delta'],
        "s1": seed_spec['s1'],
        "s2": seed_spec['s2'],
        "epsilon": item['epsilon'],
        "epsilon_bonnet": epsilon_geom,
        "alpha": item['alpha'],
        "residual_norm": float(solve.residual_norm),
        "ratio": float(solve.evaluation.ratio),
        "verified_ratio": float(ratio),
        "validation_score": item['validation_score'],
        "b_scalar": float(solve.evaluation.b_scalar),
        "c_scalar": float(solve.evaluation.c_scalar),
        "closure": _jsonable(closure),
        "isometry": _jsonable(verification.isometry),
        "mean_curvature": _jsonable(verification.mean_curvature),
        "non_congruence": _jsonable(verification.non_congruence),
        "n_vertices": int(len(pair.f_plus.vertices)),
        "n_triangles": int(len(tris_plus)),
        "n_flux_edges": int(len(flux_edges)),
        "resolution": export_res,
        "output_files": [
            f"{name}_f_plus.obj",
            f"{name}_f_minus.obj",
            f"{name}_flux.obj",
            mtl_name,
            f"{name}_metadata.json",
        ],
        "time_seconds": round(time.time() - t0, 2),
    }
    with (entry_dir / f"{name}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2)

    return metadata


# ============================================================================
# SECTION 6: Main
# ============================================================================

def main(top=0, res=80, start=0, checkpoint=None):
    """
    Entry point.  Parameters can be passed directly (Colab / notebook) or via CLI.

    Args:
        top:        Export only top N entries (0 = all 240).
        res:        Export resolution NxN (default 80).
        start:      Skip first N entries (resume after crash).
        checkpoint:  Path to checkpoint colab.json (auto-detected if None).
    """
    # --- CLI support (only when running as a script, not inside Jupyter) ---
    _in_notebook = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    if not _in_notebook:
        parser = argparse.ArgumentParser(description="Bonnet Atlas → OBJ batch exporter")
        parser.add_argument('--top', type=int, default=top)
        parser.add_argument('--res', type=int, default=res)
        parser.add_argument('--start', type=int, default=start)
        parser.add_argument('--checkpoint', type=str, default=checkpoint)
        args = parser.parse_args()
        top, res, start, checkpoint = args.top, args.res, args.start, args.checkpoint

    # --- Locate checkpoint ---
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        # Try common locations
        candidates = [
            PROJECT_DIR / "results" / "colab_atlas_campaign" / "checkpoint.json",
            PROJECT_DIR / "results" / "checkpoint.json",
            PROJECT_DIR / "results" / "checkpoint colab.json",
        ]
        ckpt_path = candidates[-1]  # fallback for error message
        for candidate in candidates:
            if candidate.exists():
                ckpt_path = candidate
                break
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = json.loads(ckpt_path.read_text(encoding='utf-8'))
    seeds = ckpt['seeds']
    atlas = ckpt['atlas']
    seed_map = {s['label']: s for s in seeds}

    n_total = len(atlas)
    if top > 0:
        atlas = atlas[:top]
    if start > 0:
        atlas = atlas[start:]

    export_dir = PROJECT_DIR / "results" / "obj" / "atlas_obj"
    export_ckpt = ExportCheckpoint(export_dir / "export_checkpoint.json")

    print("=" * 70)
    print("  BONNET ATLAS → OBJ BATCH EXPORT")
    print("=" * 70)
    print(f"  Checkpoint:    {ckpt_path.name} ({n_total} atlas entries)")
    print(f"  Exporting:     {len(atlas)} entries (start={start}, top={top or 'all'})")
    print(f"  Resolution:    {res}×{res}")
    print(f"  Output:        {export_dir}")
    print(f"  Already done:  {export_ckpt.n_done}")
    print("=" * 70)

    t_start = time.time()
    exported = 0
    skipped = 0
    errors = 0

    pbar = tqdm(total=len(atlas), desc="Exporting OBJs", unit="entry",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for i, item in enumerate(atlas):
        entry_key = f"rank{item['rank']:03d}"
        pbar.set_postfix_str(f"rank={item['rank']} fold={item['symmetry_fold']} {item['seed_label'][:25]}")

        # Skip if already exported
        if export_ckpt.is_done(entry_key):
            skipped += 1
            pbar.update(1)
            continue

        # Lookup seed + forcing
        seed_spec = seed_map.get(item['seed_label'])
        if seed_spec is None:
            export_ckpt.mark_error(entry_key, f"seed not found: {item['seed_label']}")
            errors += 1
            pbar.update(1)
            continue

        forcing = FORCING_MAP.get(item['forcing_label'])
        if forcing is None:
            export_ckpt.mark_error(entry_key, f"forcing not found: {item['forcing_label']}")
            errors += 1
            pbar.update(1)
            continue

        try:
            meta = export_single_entry(
                item=item,
                seed_spec=seed_spec,
                forcing=forcing,
                export_dir=export_dir,
                export_res=res,
            )
            export_ckpt.mark_done(entry_key, {
                'name': meta['name'],
                'rank': meta['rank'],
                'fold': meta['symmetry_fold'],
                'time_s': meta['time_seconds'],
                'n_verts': meta['n_vertices'],
                'n_tris': meta['n_triangles'],
            })
            exported += 1
        except Exception as exc:
            tb = traceback.format_exc()
            export_ckpt.mark_error(entry_key, str(exc)[:500])
            errors += 1
            tqdm.write(f"  ✗ rank {item['rank']}: {str(exc)[:120]}")

        pbar.update(1)

        # Checkpoint every 5 entries
        if (exported + errors) % 5 == 0 or i == len(atlas) - 1:
            export_ckpt.save()

    pbar.close()
    export_ckpt.save()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  EXPORT COMPLETE")
    print(f"  Exported: {exported}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/max(exported,1):.1f}s per entry)")
    print(f"  Files per entry: 5 (f_plus.obj, f_minus.obj, flux.obj, material.mtl, metadata.json)")
    print(f"  Output: {export_dir}")
    print("=" * 70)

    if errors > 0:
        print(f"\n  ⚠ {errors} errors — see {export_dir / 'export_checkpoint.json'} for details")


if __name__ == '__main__':
    main()
