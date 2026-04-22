# Reproduce The Paper Artifacts

This guide records the paper-facing commands and canonical artifact paths for

- `paper/Compact Bonnet Pairs for k-fold Tori.tex`

inside this repository.

## Environment

- Python 3.10+
- `numpy`, `scipy`, `mpmath`, `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```

## Canonical Paper-Facing Artifact Paths

| Artifact | Canonical path |
|---|---|
| Main 998-seed branch file | `results/phase15_asymptotic/full_series_k3_1000.json` |
| Compatibility mirror | `data/full_series_k3_1000.json` |
| Phase-15 critical-row recheck | `results/phase15_asymptotic/critical_recheck/` |
| B3 finite-certificate layer | `results/B3/finite_certificate/` |
| B4 high-precision audit | `results/B4/high_precision/` |
| C2 raw-grid audit | `results/C2/raw_grid/` |
| Paper figures | `paper/figures/*.pdf` |

## Core Commands

Run from the repository root.

### 1. Rebuild the main 998-seed branch file

```bash
python scripts/run_full_series.py
```

Writes:

- `results/phase15_asymptotic/full_series_k3_1000.json`
- `data/full_series_k3_1000.json`

### 2. Rebuild the residual recheck for the critical Phase-15 rows

```bash
python scripts/export_phase15_critical_recheck.py
```

Writes:

- `results/phase15_asymptotic/critical_recheck/phase15_critical_recheck.csv`
- `results/phase15_asymptotic/critical_recheck/phase15_critical_recheck_summary.json`

### 3. Rebuild the B3 finite-certificate input layer

```bash
python scripts/export_B3_certificate_inputs.py
```

Writes:

- `results/B3/finite_certificate/b3_full_series_998.csv`
- `results/B3/finite_certificate/b3_sweep_raw_121.csv`
- `results/B3/finite_certificate/b3_sweep_unique_91_from_lemma9.csv`
- `results/B3/finite_certificate/b3_sweep_unique_91_from_phase17.csv`
- `results/B3/finite_certificate/b3_export_summary.json`

### 4. Rebuild the B4 high-precision spectral audit layer

Default full export:

```bash
python scripts/export_B4_high_precision_seed_data.py
```

Paper-cited verified-range export:

```bash
python scripts/export_B4_high_precision_seed_data.py --ks 5,7,10,15,20,25,30,40,50,75,100,150,200 --out-csv results/B4/high_precision/B4_full_precision_seed_data_kle200.csv --out-json results/B4/high_precision/B4_full_precision_seed_data_kle200_summary.json
```

This reproduces the checked-in verified-range subset, which currently consists
of the `13` archived spectral-period seeds available from the P10 selection
inside `results/P10/P10_2_spectral_periods.json`.

### 5. Rebuild the C2 raw-grid audit layer

Baseline level:

```bash
python scripts/export_C2_raw_grid_audit.py --ks 3,5,7,10,12 --levels 1 --out-prefix c2_raw_grid_level1_k3_5_7_10_12
```

Refined `k=5` levels:

```bash
python scripts/export_C2_raw_grid_audit.py --ks 5 --levels 2 --out-prefix c2_raw_grid_k5_level2
python scripts/export_C2_raw_grid_audit.py --ks 5 --levels 4 --out-prefix c2_raw_grid_k5_level4
```

### 6. Regenerate the paper figures

```bash
python scripts/generate_figures.py
```

Writes:

- `paper/figures/delta_vs_k.pdf`
- `paper/figures/running_exponent.pdf`
- `paper/figures/richardson.pdf`
- `paper/figures/procrustes.pdf`

### 7. Compile the paper

From the `paper/` directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error "Compact Bonnet Pairs for k-fold Tori.tex"
pdflatex -interaction=nonstopmode -halt-on-error "Compact Bonnet Pairs for k-fold Tori.tex"
```

## Notes

- The final repo keeps `src/` as the canonical implementation layer.
- Thin compatibility wrappers also exist under `scripts/` for the paper-cited
  module paths.
- `results/obj/` is auxiliary / legacy visualization material, not the primary
  paper-facing artifact tree.
