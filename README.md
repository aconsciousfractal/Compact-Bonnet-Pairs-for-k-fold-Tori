# Compact Bonnet Pairs for *k*-fold Tori

**998 Computed Seeds and a High-Fold *k*<sup>−1/2</sup> Asymptotic Law**

> Companion code and data for the paper by O. Babanskyy (2026).
> Extends the Bobenko–Hoffmann–Sageman-Furnas construction to all fold
> numbers *k* = 3, …, 1000, yielding 998 compact Bonnet-pair seeds supported
> by the archived branch file and its critical-row residual recheck.

---

## Repository layout

```
.
├── src/                 # Library modules (self-contained, no external framework)
│   ├── theta_functions.py       # Jacobi theta functions and spectral-curve helpers
│   ├── elliptic_integrals.py    # Complete/incomplete elliptic integrals (scipy wrappers)
│   ├── weierstrass.py           # Weierstrass ℘-function (q-expansion)
│   ├── quaternion_ops.py        # ℍ-algebra, SU(2), Hopf map, batch operations
│   ├── frame_integrator.py      # Frame ODE Φ' = (W₁k + W₂k·w')Φ on S³
│   ├── isothermic_torus.py      # Isothermic torus generator (Eq. 49)
│   ├── bonnet_pair.py           # Bonnet-pair engine: f⁺, f⁻, isometry/curvature gates
│   ├── theorem7_periodicity.py  # Theorem 7 closing-condition solver & continuation
│   ├── theorem9_perturbation.py # Theorem 9 perturbative deformation layer
│   ├── retraction_form.py       # Retraction-form validation (§12)
│   └── analytic_derivatives.py  # Analytic derivative cross-checks
├── scripts/             # Reproducibility runners and compatibility shims
│   ├── run_full_series.py              # Solve k = 3…1000  →  results/phase15_asymptotic/
│   ├── export_phase15_critical_recheck.py
│   ├── export_B3_certificate_inputs.py
│   ├── export_B4_high_precision_seed_data.py
│   ├── export_C2_raw_grid_audit.py
│   ├── generate_figures.py             # Regenerate paper figures  →  paper/figures/
│   └── compute_constants.py            # Print spectral coefficients (Appendix A)
├── tests/               # Pytest suite (153 tests)
│   ├── test_theta.py            # Theta / elliptic / Weierstrass / quaternion primitives
│   ├── test_isothermic.py       # Torus generation, topology, isothermic cross-ratio
│   ├── test_bonnet_pair.py      # Bonnet pair, isometry, mean curvature, Procrustes
│   ├── test_periodicity.py      # Theorem 7 solver, continuation, end-to-end pipeline
│   ├── test_perturbation.py     # Theorem 9 perturbation layer
│   └── test_retraction.py       # Retraction-form validation gate
├── paper/
│   ├── compact_bonnet_pairs_k_fold_tori.tex
│   └── figures/                 # Publication figures (PDF)
├── results/
│   ├── phase15_asymptotic/      # Canonical paper-facing branch archive
│   ├── B3/                      # Finite-certificate layer
│   ├── B4/                      # High-precision spectral audit layer
│   ├── C2/                      # Raw-grid normalization audit layer
│   └── obj/                     # Legacy / auxiliary visualization outputs
├── data/
│   └── full_series_k3_1000.json # Compatibility mirror of the main branch file
├── requirements.txt
├── CITATION.cff
└── LICENSE                      # MIT
```

## Quick start

```bash
# Clone
git clone https://github.com/aconsciousfractal/Compact-Bonnet-Pairs-for-k-fold-Tori.git
cd Compact-Bonnet-Pairs-for-k-fold-Tori

# Create environment (Python ≥ 3.10)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest tests/ -v

# Reproduce the full k = 3…1000 dataset
python scripts/run_full_series.py

# Regenerate paper figures
python scripts/generate_figures.py
```

## Dependencies

| Package    | Version  | Purpose                              |
|------------|----------|--------------------------------------|
| numpy      | ≥ 1.24   | Array computation                    |
| scipy      | ≥ 1.10   | Elliptic integrals, ODE, optimization|
| mpmath     | ≥ 1.3    | Arbitrary-precision theta functions  |
| matplotlib | ≥ 3.7    | Figure generation                    |
| pytest     | ≥ 7.0    | Test runner (optional)               |

## Reproducing the paper results

1. **Dataset** — `python scripts/run_full_series.py` solves the Theorem 7
   closing conditions for every fold *k* = 3, …, 1000 and writes
   `results/phase15_asymptotic/full_series_k3_1000.json`
   (with a compatibility mirror in `data/full_series_k3_1000.json`).

2. **Phase-15 recheck** — `python scripts/export_phase15_critical_recheck.py`
   rebuilds the critical-row residual audit for the 998-seed branch.

3. **Figures** — `python scripts/generate_figures.py` reads the dataset and
   produces the four publication figures in `paper/figures/`.

4. **B3/B4/C2 paper audits** — run the three export scripts listed in
   `REPRODUCE.md` to rebuild the paper-cited B3 finite-certificate layer,
   B4 high-precision spectral audit, and C2 raw-grid audit.

5. **Constants** — `python scripts/compute_constants.py` prints the spectral
   coefficients tabulated in Appendix A.

6. **Paper** — From `paper/`, run:
   `pdflatex -interaction=nonstopmode -halt-on-error compact_bonnet_pairs_k_fold_tori.tex`
   twice (or use an equivalent multi-pass LaTeX toolchain).

See `REPRODUCE.md` for the paper-facing artifact map and exact commands.

## Tests

```bash
python -m pytest tests/ -v          # full suite
python -m pytest tests/ -x -q       # stop on first failure
python tests/test_theta.py          # standalone (no pytest needed)
```

The test suite covers:
- Jacobi theta identities and quasi-periodicity
- Elliptic integral known values and the Legendre relation
- Weierstrass ℘ double-periodicity and the ODE
- Quaternion algebra (associativity, inverses, Hopf map)
- Frame integrator unitarity on S³
- Isothermic torus topology (Euler χ = 0) and cross-ratio convergence
- Bonnet-pair isometry, equal mean curvature, non-congruence (Procrustes)
- Theorem 7 solver accuracy and branch continuation
- Theorem 9 perturbation and ε-continuation
- Retraction-form closure (dω = 0) and cross-condition (ω̄ ∧ dx = 0)

## Citation

```bibtex
@software{babanskyy2026bonnet,
  author    = {Babanskyy, Oleksiy},
  title     = {Compact Bonnet Pairs for k-fold Tori:
               998 Computed Seeds and a High-Fold k^{-1/2} Asymptotic Law},
  year      = {2026},
  url       = {https://github.com/aconsciousfractal/Compact-Bonnet-Pairs-for-k-fold-Tori},
  license   = {MIT}
}
```

## License

[MIT](LICENSE)
