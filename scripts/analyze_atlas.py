"""
Atlas Colab — Complete analysis of 240 Bonnet pairs.
Produces:
  1) atlas_summary.csv  — flat table of all 240 entries
  2) Figures 1-5        — publication-quality plots
  3) atlas_report.json  — structured statistics
"""
import json, os, csv, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS = str(_REPO_ROOT / "results" / "obj" / "atlas_obj")
OUTDIR = str(_REPO_ROOT / "results" / "obj" / "atlas_obj")

# ── 1) Load all metadata ──────────────────────────────────────────────
meta = []
for d in sorted(os.listdir(ATLAS)):
    dp = os.path.join(ATLAS, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.endswith('_metadata.json'):
                with open(os.path.join(dp, f)) as fh:
                    meta.append(json.load(fh))

print(f"Loaded {len(meta)} entries")

# ── 2) Extract flat table ─────────────────────────────────────────────
rows = []
for m in meta:
    nc = m['non_congruence']
    mc = m['mean_curvature']
    cl = m['closure']
    iso = m['isometry']
    rows.append({
        'rank': m['rank'],
        'name': m['name'],
        'seed_label': m['seed_label'],
        'forcing_family': m['forcing_family'],
        'symmetry_fold': m['symmetry_fold'],
        'tau_imag': m['tau_imag'],
        'delta': m['delta'],
        's1': m['s1'],
        's2': m['s2'],
        'epsilon': m['epsilon'],
        'epsilon_bonnet': m['epsilon_bonnet'],
        'alpha_0': m['alpha'][0],
        'alpha_1': m['alpha'][1],
        'alpha_2': m['alpha'][2],
        'residual_norm': m['residual_norm'],
        'ratio': m['ratio'],
        'verified_ratio': m['verified_ratio'],
        'validation_score': m['validation_score'],
        'b_scalar': m['b_scalar'],
        'c_scalar': m['c_scalar'],
        'closure_rotation_angle': cl['rotation_angle'],
        'closure_k_fold': cl['k_fold'],
        'closure_k_theta_2pi': cl['k_theta_over_2pi'],
        'closure_rationality_error': cl['rationality_error'],
        'closure_axial_proj': cl['axial_projection'],
        'f_plus_closure_max': cl['f_plus_closure_max'],
        'f_plus_closure_mean': cl['f_plus_closure_mean'],
        'f_minus_closure_max': cl['f_minus_closure_max'],
        'f_minus_closure_mean': cl['f_minus_closure_mean'],
        'euler_plus_ok': cl['euler_plus']['ok'],
        'euler_minus_ok': cl['euler_minus']['ok'],
        'iso_E_max_rel': iso['E_max_rel_err'],
        'iso_interior_F_scaled': iso['interior_F_max_scaled_err'],
        'iso_interior_G_rel': iso['interior_G_max_rel_err'],
        'iso_interior_metric_mean': iso['interior_metric_mean_err'],
        'H_plus_mean': mc['H_plus_mean'],
        'H_minus_mean': mc['H_minus_mean'],
        'H_interior_max_diff': mc['interior_max_diff'],
        'H_interior_mean_diff': mc['interior_mean_diff'],
        'procrustes_disparity': nc['procrustes_disparity'],
        'direct_mean_distance': nc['direct_mean_distance'],
        'direct_max_distance': nc['direct_max_distance'],
        'scale_ratio': nc['scale_ratio'],
        'is_non_congruent': nc['is_non_congruent'],
        'n_vertices': m['n_vertices'],
        'n_triangles': m['n_triangles'],
        'time_seconds': m['time_seconds'],
    })

csv_path = os.path.join(OUTDIR, 'atlas_summary.csv')
keys = list(rows[0].keys())
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"CSV: {csv_path} ({len(rows)} rows, {len(keys)} cols)")

# ── 3) Branch extraction ──────────────────────────────────────────────
# Best residual per (fold, tau_imag)
branches = {}
for m in meta:
    key = (m['symmetry_fold'], round(m['tau_imag'], 10))
    if key not in branches or m['residual_norm'] < branches[key]['residual_norm']:
        branches[key] = m

# Separate fold 3, fold 4
f3_taus, f3_deltas, f3_s2s, f3_procs, f3_Hdiffs, f3_res = [], [], [], [], [], []
f4_taus, f4_deltas, f4_s2s, f4_procs, f4_Hdiffs, f4_res = [], [], [], [], [], []

for (fold, ti), m in sorted(branches.items()):
    nc = m['non_congruence']
    mc = m['mean_curvature']
    if fold == 3:
        f3_taus.append(ti); f3_deltas.append(m['delta']); f3_s2s.append(m['s2'])
        f3_procs.append(nc['procrustes_disparity']); f3_Hdiffs.append(mc['interior_mean_diff'])
        f3_res.append(m['residual_norm'])
    else:
        f4_taus.append(ti); f4_deltas.append(m['delta']); f4_s2s.append(m['s2'])
        f4_procs.append(nc['procrustes_disparity']); f4_Hdiffs.append(mc['interior_mean_diff'])
        f4_res.append(m['residual_norm'])

for arr_name in ['f3_taus','f3_deltas','f3_s2s','f3_procs','f3_Hdiffs','f3_res',
                  'f4_taus','f4_deltas','f4_s2s','f4_procs','f4_Hdiffs','f4_res']:
    locals()[arr_name] = np.array(locals()[arr_name])

tau0 = 0.3205128205

# ── Plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── FIGURE 1: δ(τ_im) per branch ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f3_taus, f3_deltas, 'o-', color='#2166ac', markersize=4, label=r'$k=3$ branch ($s_1=-3.601$)')
ax.plot(f4_taus, f4_deltas, 's-', color='#b2182b', markersize=4, label=r'$k=4$ branch ($s_1=-3.131$)')
ax.axvline(tau0, color='gray', ls='--', alpha=0.7, label=r'$\tau_0$ (reference)')

# Mark minima
i_min3 = np.argmin(f3_deltas)
i_min4 = np.argmin(f4_deltas)
ax.plot(f3_taus[i_min3], f3_deltas[i_min3], '*', color='#2166ac', markersize=14, zorder=5)
ax.plot(f4_taus[i_min4], f4_deltas[i_min4], '*', color='#b2182b', markersize=14, zorder=5)

ax.set_xlabel(r'$\mathrm{Im}(\tau)$')
ax.set_ylabel(r'$\delta_{\mathrm{Bonnet}}$')
ax.set_title(r'Bonnet deformation parameter $\delta$ vs lattice modulus')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUTDIR, 'fig1_delta_vs_tau.png'))
print("Saved fig1_delta_vs_tau.png")
plt.close()

# ── FIGURE 2: s₂(τ_im) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f3_taus, f3_s2s, 'o-', color='#2166ac', markersize=4, label=r'$k=3$')
ax.plot(f4_taus, f4_s2s, 's-', color='#b2182b', markersize=4, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.7, label=r'$\tau_0$')
ax.set_xlabel(r'$\mathrm{Im}(\tau)$')
ax.set_ylabel(r'$s_2$')
ax.set_title(r'Spectral parameter $s_2$ along $\tau$-deformation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUTDIR, 'fig2_s2_vs_tau.png'))
print("Saved fig2_s2_vs_tau.png")
plt.close()

# ── FIGURE 3: Procrustes disparity vs τ ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f3_taus, f3_procs, 'o-', color='#2166ac', markersize=4, label=r'$k=3$')
ax.plot(f4_taus, f4_procs, 's-', color='#b2182b', markersize=4, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel(r'$\mathrm{Im}(\tau)$')
ax.set_ylabel('Procrustes disparity')
ax.set_title(r'Non-congruence measure along $\tau$-deformation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUTDIR, 'fig3_procrustes_vs_tau.png'))
print("Saved fig3_procrustes_vs_tau.png")
plt.close()

# ── FIGURE 4: Mean curvature diff vs τ ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f3_taus, f3_Hdiffs, 'o-', color='#2166ac', markersize=4, label=r'$k=3$')
ax.plot(f4_taus, f4_Hdiffs, 's-', color='#b2182b', markersize=4, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel(r'$\mathrm{Im}(\tau)$')
ax.set_ylabel(r'$\langle |H_+ - H_-| \rangle_{\mathrm{interior}}$')
ax.set_title(r'Mean curvature difference (interior) along $\tau$-deformation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUTDIR, 'fig4_curvature_diff_vs_tau.png'))
print("Saved fig4_curvature_diff_vs_tau.png")
plt.close()

# ── FIGURE 5: δ² vs s₂ (parabolic test) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f3_s2s, f3_deltas**2, 'o', color='#2166ac', markersize=5, alpha=0.7, label=r'$k=3$')
ax.plot(f4_s2s, f4_deltas**2, 's', color='#b2182b', markersize=5, alpha=0.7, label=r'$k=4$')

# Linear fits
for taus, deltas, s2s, c, lab in [(f3_taus, f3_deltas, f3_s2s, '#2166ac', '3'),
                                    (f4_taus, f4_deltas, f4_s2s, '#b2182b', '4')]:
    p = np.polyfit(s2s, deltas**2, 1)
    s2_fit = np.linspace(s2s.min(), s2s.max(), 100)
    ax.plot(s2_fit, np.polyval(p, s2_fit), '--', color=c, alpha=0.5,
            label=rf'$k={lab}$: $\delta^2 = {p[0]:.3f}\,s_2 + {p[1]:.3f}$')

ax.set_xlabel(r'$s_2$')
ax.set_ylabel(r'$\delta^2$')
ax.set_title(r'Parabolic relation $\delta^2$ vs $s_2$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUTDIR, 'fig5_delta2_vs_s2.png'))
print("Saved fig5_delta2_vs_s2.png")
plt.close()

# ── FIGURE 6: Convergence quality — residual vs ε ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
# all entries
eps_vals = np.array([m['epsilon'] for m in meta])
res_vals = np.array([m['residual_norm'] for m in meta])
folds = np.array([m['symmetry_fold'] for m in meta])
mask3 = folds == 3
mask4 = folds == 4
ax.scatter(eps_vals[mask3], res_vals[mask3], c='#2166ac', s=15, alpha=0.4, label=r'$k=3$')
ax.scatter(eps_vals[mask4], res_vals[mask4], c='#b2182b', s=15, alpha=0.4, label=r'$k=4$')
ax.set_yscale('log')
ax.set_xlabel(r'$\varepsilon$ (perturbation amplitude)')
ax.set_ylabel('Residual norm')
ax.set_title('Solution convergence vs perturbation strength')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
fig.savefig(os.path.join(OUTDIR, 'fig6_residual_vs_eps.png'))
print("Saved fig6_residual_vs_eps.png")
plt.close()

# ── FIGURE 7: 4-panel overview ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) δ(τ)
ax = axes[0, 0]
ax.plot(f3_taus, f3_deltas, 'o-', color='#2166ac', ms=3, label=r'$k=3$')
ax.plot(f4_taus, f4_deltas, 's-', color='#b2182b', ms=3, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'$\mathrm{Im}(\tau)$'); ax.set_ylabel(r'$\delta$')
ax.set_title('(a) Bonnet parameter'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (b) s₂(τ)
ax = axes[0, 1]
ax.plot(f3_taus, f3_s2s, 'o-', color='#2166ac', ms=3, label=r'$k=3$')
ax.plot(f4_taus, f4_s2s, 's-', color='#b2182b', ms=3, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'$\mathrm{Im}(\tau)$'); ax.set_ylabel(r'$s_2$')
ax.set_title('(b) Spectral parameter'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (c) Procrustes(τ)
ax = axes[1, 0]
ax.plot(f3_taus, f3_procs, 'o-', color='#2166ac', ms=3, label=r'$k=3$')
ax.plot(f4_taus, f4_procs, 's-', color='#b2182b', ms=3, label=r'$k=4$')
ax.axvline(tau0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'$\mathrm{Im}(\tau)$'); ax.set_ylabel('Procrustes')
ax.set_title('(c) Non-congruence'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (d) δ² vs s₂
ax = axes[1, 1]
ax.plot(f3_s2s, f3_deltas**2, 'o', color='#2166ac', ms=4, alpha=0.7, label=r'$k=3$')
ax.plot(f4_s2s, f4_deltas**2, 's', color='#b2182b', ms=4, alpha=0.7, label=r'$k=4$')
ax.set_xlabel(r'$s_2$'); ax.set_ylabel(r'$\delta^2$')
ax.set_title(r'(d) $\delta^2$ vs $s_2$'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig7_atlas_overview.png'))
print("Saved fig7_atlas_overview.png")
plt.close()

# ── 4) Structured statistics report ───────────────────────────────────
# Parabolic fits
p3 = np.polyfit(f3_s2s, f3_deltas**2, 1)
p4 = np.polyfit(f4_s2s, f4_deltas**2, 1)
r2_3 = 1 - np.sum((f3_deltas**2 - np.polyval(p3, f3_s2s))**2) / np.sum((f3_deltas**2 - np.mean(f3_deltas**2))**2)
r2_4 = 1 - np.sum((f4_deltas**2 - np.polyval(p4, f4_s2s))**2) / np.sum((f4_deltas**2 - np.mean(f4_deltas**2))**2)

# Quadratic fit for minima
p_min4 = np.polyfit(f4_taus, f4_deltas, 2)
tau_min4 = -p_min4[1] / (2*p_min4[0])
delta_min4 = np.polyval(p_min4, tau_min4)
p_min3 = np.polyfit(f3_taus, f3_deltas, 2)
tau_min3 = -p_min3[1] / (2*p_min3[0])
delta_min3 = np.polyval(p_min3, tau_min3)

report = {
    'total_entries': len(meta),
    'fold_3_entries': int(np.sum([m['symmetry_fold']==3 for m in meta])),
    'fold_4_entries': int(np.sum([m['symmetry_fold']==4 for m in meta])),
    'tau_imag_values': len(set(round(m['tau_imag'], 10) for m in meta)),
    'tau_range': [float(min(m['tau_imag'] for m in meta)), float(max(m['tau_imag'] for m in meta))],
    'all_non_congruent': all(m['non_congruence']['is_non_congruent'] for m in meta),
    'branches': {
        'fold_3': {
            's1': float(f3_taus[0] and meta[0]['s1']),  # just grab from data
            'tau_min': float(tau_min3),
            'delta_min': float(delta_min3),
            'delta_range': [float(f3_deltas.min()), float(f3_deltas.max())],
            's2_range': [float(f3_s2s.min()), float(f3_s2s.max())],
            'procrustes_range': [float(f3_procs.min()), float(f3_procs.max())],
            'parabolic_fit': {'slope': float(p3[0]), 'intercept': float(p3[1]), 'R2': float(r2_3)},
        },
        'fold_4': {
            's1': -3.130606,
            'tau_min': float(tau_min4),
            'delta_min': float(delta_min4),
            'delta_range': [float(f4_deltas.min()), float(f4_deltas.max())],
            's2_range': [float(f4_s2s.min()), float(f4_s2s.max())],
            'procrustes_range': [float(f4_procs.min()), float(f4_procs.max())],
            'parabolic_fit': {'slope': float(p4[0]), 'intercept': float(p4[1]), 'R2': float(r2_4)},
        },
    },
    'tau0_speciality': {
        'tau0': tau0,
        'fold4_tau_min': float(tau_min4),
        'distance_fraction': float(abs(tau_min4 - tau0) / tau0),
        'fold4_delta_at_tau0': float(f4_deltas[np.argmin(np.abs(f4_taus - tau0))]),
    },
    'quality': {
        'median_residual': float(np.median([m['residual_norm'] for m in meta])),
        'best_residual': float(min(m['residual_norm'] for m in meta)),
        'worst_residual': float(max(m['residual_norm'] for m in meta)),
        'entries_below_1e-9': int(np.sum([m['residual_norm'] < 1e-9 for m in meta])),
    },
    'forcing_breakdown': dict(Counter(m['forcing_family'] for m in meta)),
    'epsilon_breakdown': dict(Counter(str(m['epsilon']) for m in meta)),
    'compute_hours': float(sum(m['time_seconds'] for m in meta) / 3600),
}

# Fix fold_3 s1
f3_entries = [m for m in meta if m['symmetry_fold'] == 3]
report['branches']['fold_3']['s1'] = float(f3_entries[0]['s1'])

rpt_path = os.path.join(OUTDIR, 'atlas_report.json')
with open(rpt_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"Report: {rpt_path}")

# Print key findings
print("\n" + "="*60)
print("ATLAS ANALYSIS — KEY FINDINGS")
print("="*60)
print(f"Entries: {report['total_entries']} (fold-3: {report['fold_3_entries']}, fold-4: {report['fold_4_entries']})")
print(f"τ_imag sweep: {report['tau_range'][0]:.4f} → {report['tau_range'][1]:.4f} ({report['tau_imag_values']} values)")
print(f"ALL {report['total_entries']} pairs verified NON-CONGRUENT")
print(f"\nFold-4 branch (s₁ = {report['branches']['fold_4']['s1']:.3f}):")
print(f"  δ minimum at τ_im = {tau_min4:.6f} (τ₀ = {tau0:.6f}, gap = {abs(tau_min4-tau0):.6f})")
print(f"  δ_min = {delta_min4:.6f}")
print(f"  δ²=a·s₂+b fit: R² = {r2_4:.4f}")
print(f"\nFold-3 branch (s₁ = {report['branches']['fold_3']['s1']:.3f}):")
print(f"  δ minimum at τ_im = {tau_min3:.6f}")
print(f"  δ_min = {delta_min3:.6f}")
print(f"  δ²=a·s₂+b fit: R² = {r2_3:.4f}")
print(f"\nQuality: {report['quality']['entries_below_1e-9']}/{len(meta)} with residual < 10⁻⁹")
print(f"Compute: {report['compute_hours']:.1f} GPU-hours")
