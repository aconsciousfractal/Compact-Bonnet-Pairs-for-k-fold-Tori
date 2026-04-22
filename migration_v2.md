# Migration v2

## Scope

This document maps the migration work required to bring the final repository

- `Compact Bonnet Pairs for k-fold Tori`

into full alignment with the current paper and its paper-cited scripts,
artifacts, and documentation.

Source repository for the newer paper-support layer:

- the local development workspace used during the migration pass

Primary rule for this migration:

- every script path cited in the paper must resolve inside the final repo;
- every artifact path cited in the paper must resolve inside the final repo;
- every documentation file in the final repo must describe the migrated layout,
  not the old `data/`-only or `HAN`-absolute-path state.

## Progress Snapshot

Already completed in the target repo during this migration pass:

- compatibility wrappers added in `scripts/` for the paper-cited core modules
  now implemented canonically in `src/`;
- new runner scripts added:
  - `scripts/export_phase15_critical_recheck.py`
  - `scripts/export_B3_certificate_inputs.py`
  - `scripts/export_B4_high_precision_seed_data.py`
  - `scripts/export_C2_raw_grid_audit.py`
- paper-facing artifact trees created/copied:
  - `results/phase15_asymptotic/`
  - `results/B3/finite_certificate/`
  - `results/B4/high_precision/`
  - `results/C2/raw_grid/`
- support inputs copied:
  - `results/phase11_higher_fold/lemma9_sweep.json`
  - `results/phase17_moduli/phase17_345_results.json`
  - `results/P10/P10_2_spectral_periods.json`
  - `results/P5/P5v3_checkpoint.json`
- `scripts/generate_figures.py`, `scripts/run_full_series.py`,
  `paper/Compact Bonnet Pairs for k-fold Tori.tex`, `README.md`,
  `CITATION.cff`, and the new `REPRODUCE.md` have been aligned to the
  canonical release layout.
- the most obviously historical / auxiliary runners have been moved from the
  root `scripts/` directory into `scripts/legacy/`, and the OBJ README files
  now point to that legacy location.

Still remaining after this snapshot:

- optional final pruning of any additional non-paper-facing analysis runners,
- optional release-level metadata polish.

## Recommended Canonical Decisions

| Area | Recommended decision | Why |
|---|---|---|
| Core implementation modules | Keep `src/` as the implementation layer | The target repo is already organized around `src/` |
| Paper-cited module paths | Add thin compatibility wrappers under `scripts/` for the paper-cited module files | The paper cites `scripts/*.py`, not `src/*.py` |
| Figure generator path in the paper | Rewrite from `paper/paper/generate_figures.py` to `scripts/generate_figures.py` | The target repo already uses `scripts/generate_figures.py`; adding a fake nested `paper/paper/` tree is unnecessary churn |
| Figure output path in the paper | Rewrite from `paper/paper/figures/*.pdf` to `paper/figures/*.pdf` | The target repo already stores figures under `paper/figures/` |
| Canonical paper dataset path | Adopt `results/phase15_asymptotic/full_series_k3_1000.json` as the paper-facing archival path | The current paper and the new export scripts expect the phase-style `results/` tree |
| Existing `data/full_series_k3_1000.json` | Keep only as a temporary compatibility mirror, or deprecate after script/doc rewrites | Avoid dual source-of-truth ambiguity |
| Reproducibility guide | Keep a repo-root `REPRODUCE.md` specific to the final repo | The paper already cites it; it is now part of the canonical release surface |

## Mandatory Migration Items

### A. Paper-cited module-style scripts

These are cited in `paper/Compact Bonnet Pairs for k-fold Tori.tex:3433-3440`.
They now exist in `src/` canonically and also resolve at the cited `scripts/`
paths via compatibility wrappers.

| Paper-cited path | Target current state | Source / current implementation | Required action |
|---|---|---|---|
| `scripts/theta_functions.py` | present via wrapper | `src/theta_functions.py` | done |
| `scripts/theorem7_periodicity.py` | present via wrapper | `src/theorem7_periodicity.py` | done |
| `scripts/retraction_form.py` | present via wrapper | `src/retraction_form.py` | done |
| `scripts/analytic_derivatives.py` | present via wrapper | `src/analytic_derivatives.py` | done |
| `scripts/frame_integrator.py` | present via wrapper | `src/frame_integrator.py` | done |

Recommended implementation style:

- wrapper files in `scripts/` that re-export the corresponding `src.*` modules;
- do not fork the logic into duplicate long files unless a runner entrypoint is
  genuinely needed.

### B. Paper-cited runner scripts migrated into the target repo

| Script | Status in target repo | Source path | Required action |
|---|---|---|---|
| `scripts/export_phase15_critical_recheck.py` | present | `scripts/export_phase15_critical_recheck.py` in source repo | migrated |
| `scripts/export_B3_certificate_inputs.py` | present | `scripts/export_B3_certificate_inputs.py` in source repo | migrated |
| `scripts/export_B4_high_precision_seed_data.py` | present | `scripts/export_B4_high_precision_seed_data.py` in source repo | migrated |
| `scripts/export_C2_raw_grid_audit.py` | present | source version adapted to target repo | migrated and made self-contained |

### C. Existing target scripts that must be replaced or merged

| Script in target repo | Problem | Required action |
|---|---|---|
| `scripts/generate_figures.py` | now aligned to the current paper-facing layout and figure output path | done |
| `scripts/run_full_series.py` | now writes the canonical paper-facing results tree and keeps a `data/` mirror | done |

### C2. Existing target scripts with legacy workflow or stale path assumptions

These scripts already exist in the final repo, but they still carry old phase
labels, old path assumptions, Colab-era workflow assumptions, or fragile import
patterns. They must be reviewed during migration v2, even if they are not all
part of the paper's primary reproducibility path.

| Script | Legacy / stale issue | Required action |
|---|---|---|
| `scripts/legacy/run_tau_universality.py` | legacy analysis runner; canonical dataset handling normalized before move | moved to `scripts/legacy/` |
| `scripts/legacy/run_s1_sweep_extended.py` | legacy sweep runner; canonical path/import hygiene normalized before move | moved to `scripts/legacy/` |
| `scripts/legacy/run_moduli_analysis.py` | legacy moduli analysis runner | moved to `scripts/legacy/` |
| `scripts/legacy/run_spectral_analysis.py` | legacy spectral analysis runner | moved to `scripts/legacy/` |
| `scripts/legacy/compute_high_precision.py` | legacy Colab-era P5 workflow | moved to `scripts/legacy/` |
| `scripts/legacy/run_pslq_A.py` | legacy Colab-era P5v3 extraction workflow | moved to `scripts/legacy/` |
| `scripts/legacy/run_colab_export_atlas.py` | legacy Colab-only atlas exporter | moved to `scripts/legacy/` |
| `scripts/legacy/analyze_atlas.py` | legacy atlas analysis runner | moved to `scripts/legacy/` |
| `scripts/legacy/generate_flux_obj.py` | legacy visualization runner | moved to `scripts/legacy/` |
| `scripts/legacy/generate_obj.py` | legacy high-k OBJ export | moved to `scripts/legacy/` |

Recommended classification after migration:

- `scripts/` root: only canonical paper/repro runners and actively supported utilities;
- `scripts/legacy/`: Colab-era, atlas-only, or phase-history runners kept only for provenance;
- if a script stays in `scripts/`, its docstring, paths, and imports must match the
  final repo layout exactly.

Suggested `scripts/legacy/` cluster unless a specific public-facing use is
still intended:

- `run_colab_export_atlas.py`
- `analyze_atlas.py`
- `run_phase11_visual_obj.py`
- `run_phase11_flux_only.py`
- `run_phase85_export_canonical_candidates.py`
- `run_phase85_highres_final_4fold.py`
- `generate_flux_theorem9_perturbed.py`
- `generate_flux_theorem9_perturbed_4fold.py`
- `generate_flux_theorem9_perturbed_4fold_visual.py`
- `generate_flux_obj.py`
- `generate_obj.py`
- `generate_base_obj.py`

These scripts are not necessarily wrong, but they are phase-history or
visualization-specific and should not remain mixed with the canonical paper
repro runners unless the docs explicitly keep them there.

### D. Paper-cited artifact trees now present in the target repo

These are all explicitly cited by the current paper and must exist in the final
repo if the paper is left paper-first rather than rewritten around the old
target layout.

| Artifact path cited in paper | Current target state | Required action |
|---|---|---|
| `results/phase15_asymptotic/full_series_k3_1000.json` | present | canonical paper-facing branch archive |
| `results/phase15_asymptotic/critical_recheck/` | present | migrated |
| `results/B3/finite_certificate/` | present | migrated |
| `results/B4/high_precision/` | present | migrated |
| `results/C2/raw_grid/` | present | migrated |

### E. Concrete artifact files migrated into the target repo

#### Phase 15 branch audit

| Path | Source |
|---|---|
| `results/phase15_asymptotic/full_series_k3_1000.json` | source repo `results/phase15_asymptotic/full_series_k3_1000.json` |
| `results/phase15_asymptotic/critical_recheck/phase15_critical_recheck.csv` | source repo |
| `results/phase15_asymptotic/critical_recheck/phase15_critical_recheck_summary.json` | source repo |

#### B3 finite certificate layer

| Path | Source |
|---|---|
| `results/B3/finite_certificate/b3_full_series_998.csv` | source repo |
| `results/B3/finite_certificate/b3_sweep_raw_121.csv` | source repo |
| `results/B3/finite_certificate/b3_sweep_unique_91_from_lemma9.csv` | source repo |
| `results/B3/finite_certificate/b3_sweep_unique_91_from_phase17.csv` | source repo |
| `results/B3/finite_certificate/b3_export_summary.json` | source repo |
| `results/B3/finite_certificate/b3_full_series_998_certificate.json` | source repo |
| `results/B3/finite_certificate/b3_sweep_raw_121_certificate.json` | source repo |
| `results/B3/finite_certificate/b3_sweep_unique_91_certificate.json` | source repo |

#### B4 high-precision layer

| Path | Source |
|---|---|
| `results/B4/high_precision/B4_full_precision_seed_data.csv` | source repo |
| `results/B4/high_precision/B4_full_precision_seed_data_summary.json` | source repo |
| `results/B4/high_precision/B4_full_precision_seed_data_kle200.csv` | source repo |
| `results/B4/high_precision/B4_full_precision_seed_data_kle200_summary.json` | source repo |
| `results/B4/high_precision/B4_highk_probe.csv` | source repo |
| `results/B4/high_precision/B4_highk_probe_summary.json` | source repo |
| `results/B4/high_precision/B4_midk_probe.csv` | source repo |
| `results/B4/high_precision/B4_midk_probe_summary.json` | source repo |

#### C2 raw-grid layer

| Path | Source |
|---|---|
| `results/C2/raw_grid/c2_raw_grid_level1_k3_5_7_10_12.csv` | source repo |
| `results/C2/raw_grid/c2_raw_grid_level1_k3_5_7_10_12_summary.json` | source repo |
| `results/C2/raw_grid/c2_raw_grid_k5_level2.csv` | source repo |
| `results/C2/raw_grid/c2_raw_grid_k5_level2_summary.json` | source repo |
| `results/C2/raw_grid/c2_raw_grid_k5_level4.csv` | source repo |
| `results/C2/raw_grid/c2_raw_grid_k5_level4_summary.json` | source repo |

### F. Transitive support inputs used by the migrated export scripts

These are not optional if the migrated export scripts are supposed to run in the
final repo rather than merely be checked in as dead files.

| Needed by | Missing / mismatched dependency in target repo | Action |
|---|---|---|
| `export_B3_certificate_inputs.py` | `results/phase11_higher_fold/lemma9_sweep.json` | migrated |
| `export_B3_certificate_inputs.py` | `results/phase17_moduli/phase17_345_results.json` | migrated |
| `export_B4_high_precision_seed_data.py` | `results/P10/P10_2_spectral_periods.json` | migrated |
| `export_B4_high_precision_seed_data.py` | `results/P5/P5v3_checkpoint.json` | migrated |
| `export_C2_raw_grid_audit.py` | old source dependency on `scripts/P4_grid_refinement.py` | replaced by target-native self-contained implementation |
| `export_C2_raw_grid_audit.py` | old source dependency on `scripts/higher_fold_seeds.py` | replaced by `src/seed_catalog.py` |
| `export_C2_raw_grid_audit.py` | old source dependency on `paper/math work with agent.md/v1/verify_c2_retraction_normalization.py` | internalized into the target-native script |

## Existing Target Files That Need Surgical Cleanup or Replacement

These are code or artifact files already in the final repo that still encode the
old source-repo structure, old absolute paths, or old canonical data locations.

| Path | Current problem | Required action |
|---|---|---|
| `scripts/legacy/run_tau_universality.py` | retained only as a legacy analysis runner | done |
| `scripts/legacy/run_s1_sweep_extended.py` | retained only as a legacy sweep runner | done |
| `scripts/legacy/run_spectral_analysis.py` | retained only as a legacy analysis runner | done |
| `scripts/legacy/run_moduli_analysis.py` | retained only as a legacy analysis runner | done |
| `scripts/legacy/compute_high_precision.py` | retained only as a legacy Colab-era P5 workflow | done |
| `scripts/legacy/run_pslq_A.py` | retained only as a legacy high-precision extraction workflow | done |
| `scripts/legacy/run_colab_export_atlas.py` | retained only as a legacy Colab-only atlas exporter | done |
| `scripts/legacy/analyze_atlas.py` | retained only as a legacy atlas analysis runner | done |
| `scripts/legacy/generate_flux_obj.py` | retained only as a legacy visualization runner | done |
| `data/moduli_analysis.json` | provenance normalized to a repo-local path | done |
| `results/obj/paper_figures/phase85_canonical_candidates.json` | legacy phase artifact that should be marked explicitly as historical/auxiliary if kept | review and relabel in docs |

## Documentation Files To Update

This is the complete documentation rewrite list in the final repo.

### 1. `paper/Compact Bonnet Pairs for k-fold Tori.tex`

This file has been realigned to the current target-repo layout and should now be
kept in sync with the migrated `results/` and `scripts/` trees.

Mandatory edits:

| Current paper reference | Problem in target repo | Surgical rewrite |
|---|---|---|
| `paper/paper/generate_figures.py` (`:3441,3468`) | old mismatch | fixed to `scripts/generate_figures.py` |
| `paper/paper/figures/*.pdf` (`:3452`) | old mismatch | fixed to `paper/figures/*.pdf` |
| `results/phase15_asymptotic/full_series_k3_1000.json` | old mismatch | migrated and cited canonically |
| `REPRODUCE.md` (`:3462,3478`) | old missing file | added |
| commit `65083dc7` (`:3484`) | stale hardcoded revision | replaced by release-manifest wording |

Also verify after migration that every `\path{scripts/...}` cited in the paper
actually resolves inside the final repo.

### 2. `README.md`

Status:

- title/subtitle updated to the current paper packaging;
- manuscript path corrected;
- layout block rewritten toward the `results/` paper-facing tree;
- paper-facing runner list expanded;
- paper compile command clarified.

### 3. `REPRODUCE.md`

Status: added.

It now includes:

- environment setup;
- exact commands for `run_full_series.py`;
- exact commands for `export_phase15_critical_recheck.py`;
- exact commands for `export_B3_certificate_inputs.py`;
- exact commands for `export_B4_high_precision_seed_data.py`;
- exact commands for `export_C2_raw_grid_audit.py`;
- exact command for `scripts/generate_figures.py`;
- expected output locations in the final repo.

### 4. `CITATION.cff`

Status: title updated to the current paper packaging.

### 5. `results/obj/README.md`

Status: rewritten as a legacy / auxiliary visualization README and updated to
point at `scripts/legacy/`.

### 6. `results/obj/paper_figures/README.md`

Status: rewritten to repo-local relative paths and marked as legacy / auxiliary.

### 7. `results/obj/theorem9/bonnet_theorem9_perturbed_4fold_highres/README.md`

Current status:

- not broken, but still old-style and detached from the current paper-facing
  reproducibility layer.

Required action:

- keep only if Theorem 9 perturbative OBJ material remains part of the final
  public package;
- if kept, add a short note that it is auxiliary/legacy and not part of the
  main paper-reproduction path.

### 8. `.pytest_cache/README.md`

No action. Ignore: autogenerated tooling file.

## Migration Execution Order

1. Keep `results/phase15_asymptotic/...` as the paper-facing archive.
2. Keep the migrated export scripts and wrappers as the canonical paper-repro layer.
3. Maintain `scripts/legacy/` as the home for historical / auxiliary runners.
4. Keep public docs synchronized with the migrated layout.
5. Sweep old absolute provenance strings from checked-in public-facing docs/JSON.

## Verification Checklist After Migration

| Check | Pass condition |
|---|---|
| Paper-cited scripts | every `\path{scripts/...}` in the paper resolves in the final repo |
| Paper-cited results | every `\path{results/...}` in the paper resolves in the final repo |
| Figure paths | no `paper/paper/` paths remain; paper uses `scripts/generate_figures.py` and `paper/figures/*.pdf` |
| Repro guide | `REPRODUCE.md` exists and matches the final repo layout |
| Old absolute provenance | no `P:\GitHub_puba\HAN\...` paths remain in checked-in docs that are meant to be public-facing |
| Canonical dataset path | exactly one documented paper-facing location is declared canonical |
| Paper compile | `pdflatex` from the final repo paper directory succeeds |

## Bottom Line

The migration was not just “copy four new scripts”.

The final repo now has the paper-facing scripts, artifacts, and documentation
needed for a coherent release package.

What remains after this migration is much smaller:

- optional further pruning of additional historical analysis runners,
- optional release-level metadata polishing,
- and ordinary maintenance to keep docs aligned with future repo changes.

The main paper/code/docs package is now aligned around the canonical
`results/` paper-facing archive tree plus the `data/` compatibility mirror.
