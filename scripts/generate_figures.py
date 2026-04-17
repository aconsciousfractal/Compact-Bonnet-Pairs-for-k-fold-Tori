"""
Generate publication figures for the Bonnet pairs paper.

Produces:
  figures/delta_vs_k.pdf       — log-log plot of δ(k)
  figures/running_exponent.pdf — α(k) convergence to -0.5
  figures/richardson.pdf       — A_k = δ√k convergence
  figures/procrustes.pdf       — Procrustes decay with floor

Requires: matplotlib, numpy, json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (5.5, 4.0),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load data ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
with open(os.path.join(DATA_DIR, "full_series_k3_1000.json")) as f:
    raw = json.load(f)

entries = raw if isinstance(raw, list) else raw.get("seeds", raw.get("data", []))
ks = np.array([e["k"] for e in entries], dtype=float)
deltas = np.array([e["delta"] for e in entries], dtype=float)
s2s = np.array([e["s2"] for e in entries], dtype=float)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Figure 1: δ vs k (log–log) ────────────────────────────────────
fig, ax = plt.subplots()
ax.loglog(ks, deltas, ".", ms=1.5, color="C0", label=r"$\delta_k$ (data)")
# Fit line
A_est = 7.0814165
ax.loglog(ks, A_est / np.sqrt(ks), "--", color="C3", lw=1,
          label=rf"$A/\sqrt{{k}}$, $A={A_est:.4f}$")
ax.set_xlabel(r"Fold number $k$")
ax.set_ylabel(r"Closure parameter $\delta_k$")
ax.legend()
ax.set_title(r"$\delta_k$ vs $k$ (log–log)")
fig.savefig(os.path.join(FIG_DIR, "delta_vs_k.pdf"))
plt.close(fig)
print("  → delta_vs_k.pdf")


# ── Figure 2: Running exponent ─────────────────────────────────────
alpha = np.log(deltas[1:] / deltas[:-1]) / np.log(ks[:-1] / ks[1:])
k_mid = 0.5 * (ks[:-1] + ks[1:])

fig, ax = plt.subplots()
ax.plot(k_mid, alpha, "-", lw=0.5, color="C0", alpha=0.6)
# Smoothed (running mean over 20)
win = 20
if len(alpha) > win:
    kernel = np.ones(win) / win
    alpha_smooth = np.convolve(alpha, kernel, mode="valid")
    k_smooth = np.convolve(k_mid, kernel, mode="valid")
    ax.plot(k_smooth, alpha_smooth, "-", lw=1.5, color="C1",
            label=f"Running mean (window {win})")
ax.axhline(-0.5, ls="--", color="C3", lw=1, label=r"$\alpha = -1/2$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"Running exponent $\alpha(k)$")
ax.set_ylim(-0.55, -0.40)
ax.legend()
ax.set_title(r"Running exponent $\alpha(k) \to -1/2$")
fig.savefig(os.path.join(FIG_DIR, "running_exponent.pdf"))
plt.close(fig)
print("  → running_exponent.pdf")


# ── Figure 3: Richardson (A_k = δ√k) ──────────────────────────────
A_k = deltas * np.sqrt(ks)

fig, ax = plt.subplots()
ax.plot(ks, A_k, "-", lw=0.5, color="C0", label=r"$A_k = \delta_k\sqrt{k}$")
ax.axhline(A_est, ls="--", color="C3", lw=1,
           label=rf"$A = {A_est}$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\delta_k \sqrt{k}$")
ax.set_ylim(A_est - 0.3, A_est + 1.5)
ax.legend()
ax.set_title(r"Convergence of $\delta_k\sqrt{k} \to A$")
fig.savefig(os.path.join(FIG_DIR, "richardson.pdf"))
plt.close(fig)
print("  → richardson.pdf")


# ── Figure 4: Procrustes decay (synthetic from fit) ────────────────
k_proc = np.arange(3, 201)
d_proc = 3.0/64.0 + 0.216 / k_proc**1.236

fig, ax = plt.subplots()
ax.plot(k_proc, d_proc, "-", lw=1.5, color="C0",
        label=r"$d_P = 3/64 + 0.216\,k^{-1.236}$")
ax.axhline(3.0/64.0, ls="--", color="C3", lw=1,
           label=r"Floor $= 3/64$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"Procrustes disparity $d_P$")
ax.legend()
ax.set_title("Procrustes decay with geometric floor")
fig.savefig(os.path.join(FIG_DIR, "procrustes.pdf"))
plt.close(fig)
print("  → procrustes.pdf")


print(f"\nAll figures saved to {FIG_DIR}")
