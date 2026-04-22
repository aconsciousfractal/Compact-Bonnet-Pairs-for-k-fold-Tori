# Pre-computed OBJ Meshes

Wavefront OBJ meshes of discrete Bonnet pairs, organised by generation pipeline.
Every directory follows the naming convention:

| File suffix | Content |
|---|---|
| `*_f_plus.obj` | f⁺ surface (Bonnet pair, positive rotation) |
| `*_f_minus.obj` | f⁻ surface (Bonnet pair, negative rotation) |
| `*_flux.obj` | Flux wireframe with curvature-based vertex colouring |
| `*_base.obj` | Pre-Bonnet isothermic torus (where applicable) |
| `*_material.mtl` | Wavefront MTL material library |
| `*_metadata.json` | Per-surface generation metadata |

Total: **~1 345 files, ~688 MB**.

---

## Directory layout

### `atlas_obj/` — Colab atlas (240 ranked surfaces)

| Metric | Value |
|---|---|
| Surfaces | 240 (125 three-fold, 115 four-fold) |
| Files per surface | 5 (`f_plus`, `f_minus`, `flux`, `mtl`, `metadata`) |
| Total files | 1 203 |
| Size | ~552 MB |

Systematic exploration of the (τ, ε, palette) parameter space for k = 3
and k = 4.  Each `rank###_fold{3,4}_…` directory contains one Bonnet
pair at the specified spectral parameters.

Top-level files:
- `atlas_report.json` — ranking metrics and summary statistics.
- `atlas_summary.csv` — tabular overview of all 240 entries.
- `export_checkpoint.json` — Colab export checkpoint.

**Generator:** `scripts/legacy/run_colab_export_atlas.py` (designed to run on
Google Colab; see docstring for instructions).

---

### `higher_fold/` — Intermediate fold orders (k = 5, 6, 7)

```
higher_fold/
├── visual_obj/
│   ├── k5/   (15 OBJ)
│   ├── k6/   (15 OBJ)
│   └── k7/   (15 OBJ)
├── all_seeds_validation.json
├── comparative_analysis.json
├── lemma9_sweep.json
├── step1_diagnostic.json
├── step2_frame_sweep.json
└── theorem9_higher_fold.json
```

Each k-directory contains **3 variants × 5 files**:

| Variant | Description |
|---|---|
| `seed` | Reference seed parameters |
| `shallow` | Shallow spectral deformation |
| `deep` | Deep spectral deformation |

Files per variant: `base`, `f_minus`, `f_minus_flux`, `f_plus`, `f_plus_flux`.

The top-level JSON files are validation diagnostics (Lemma 9 sweep,
frame analysis, comparative metrics).

**Generator:** `scripts/legacy/run_phase11_visual_obj.py` +
`scripts/legacy/run_phase11_flux_only.py` (flux overlay).

---

### `higher_k/` — High fold orders (k = 50 – 1 000)

```
higher_k/
└── visual_obj/
    ├── k50/    (9 files)
    ├── k100/   (9 files)
    ├── k250/   (9 files)
    ├── k500/   (9 files)
    ├── k750/   (9 files)
    ├── k1000/  (9 files)
    ├── base_torus_metadata.json
    └── high_k_metadata.json
```

Each k-directory contains:  `base.obj`, `base.mtl`, `base_flux.obj`,
`f_plus.obj`, `f_minus.obj`, `flux.obj`, `material.mtl`, plus two
metadata JSON files.

These surfaces demonstrate the asymptotic scaling law
δ(k) = A/√k · (1 − c₁/k + c₃/k²) established in the paper (§ 6).

**Generator:** `scripts/legacy/generate_obj.py` (Bonnet pair + flux) and
`scripts/legacy/generate_base_obj.py` (pre-Bonnet base torus).

---

### `paper_figures/` — Canonical candidates for paper figures

```
paper_figures/
├── paper_3fold/          (4 files)
├── paper_3fold_forward_2/ (4 files)
├── paper_4fold/          (4 files)
├── phase85_canonical_candidates.json
└── README.md
```

Best-quality 3-fold and 4-fold Bonnet pairs selected through the Phase 8.5
forcing and continuation pipeline.  Each sub-directory has `f_plus.obj`,
`f_minus.obj`, `flux.obj`, and `material.mtl`.

**Generator:** `scripts/legacy/run_phase85_export_canonical_candidates.py`.

---

### `theorem9/` — Theorem 9 perturbative visualisations

```
theorem9/
├── bonnet_theorem9_perturbed_3fold/       (5 files)
├── bonnet_theorem9_perturbed_4fold/       (5 files)
├── bonnet_theorem9_perturbed_4fold_highres/ (6 files)
└── bonnet_theorem9_perturbed_4fold_visual/  (5 files)
```

Perturbative Bonnet pairs constructed via Theorem 9, at standard and
high resolution, with a visual-exaggeration variant for presentation.
Each sub-directory has `f_plus.obj`, `f_minus.obj`, `flux.obj`,
`material.mtl`, and `metadata.json` (highres adds a `report.json`).

**Generators:**
- `scripts/legacy/generate_flux_theorem9_perturbed.py` (3-fold)
- `scripts/legacy/generate_flux_theorem9_perturbed_4fold.py` (4-fold standard)
- `scripts/legacy/generate_flux_theorem9_perturbed_4fold_visual.py` (4-fold visual)
- `scripts/legacy/run_phase85_highres_final_4fold.py` (4-fold high-resolution)

---

## Viewing the meshes

Any Wavefront OBJ viewer works.  Recommended:

- **[MeshLab](https://www.meshlab.net/)** — open-source, handles MTL materials.
- **[3D Viewer](https://apps.microsoft.com/detail/9nblggh42ths)** — built into Windows 10/11.
- **[Blender](https://www.blender.org/)** — import via File → Import → Wavefront (.obj).

Flux wireframes (`*_flux.obj`) use per-vertex RGB colour to encode
Gaussian curvature on a blue → red scale.

## Status

This `results/obj/` tree is an auxiliary / legacy visualization layer.
It is not the canonical paper-facing reproducibility archive, which now lives
under the top-level `results/` phase/audit directories cited in the paper and
documented in `REPRODUCE.md`.

## Regenerating

All meshes can be regenerated from the source code.  From the repo root:

```bash
# High-k Bonnet pairs (k = 50–1000)
python scripts/legacy/generate_obj.py

# Base isothermic tori (pre-Bonnet)
python scripts/legacy/generate_base_obj.py

# Flux visualisation (3 showcase pairs)
python scripts/legacy/generate_flux_obj.py
```

The atlas requires Google Colab (see `scripts/legacy/run_colab_export_atlas.py`).
The remaining generators are kept as repository-local legacy/auxiliary tools.
Their script paths should be read relative to this repository, not to the old
source workspace.
