# Changelog

All notable changes to the *Compact Bonnet Pairs for k-fold Tori* release
package are recorded in this file. The project follows a release-tag model;
the manuscript, code, data, and figures are versioned together.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2026-04-22] — Post-audit close-out

Post-audit release closing the red-team audit of 2026-04-22. Full
verification: pdflatex 2-pass clean (46 pages, 0 errors, 0 undefined
references), pytest `153 passed` on the tests/ suite, phase-15
critical-row recheck reproduced successfully, archive row count
verified at $998$ with all residuals $<10^{-9}$ and all
$\texttt{nfev} > 0$.

### Fixed

- **Data integrity (CRITICAL).** Regenerated
  `results/phase15_asymptotic/full_series_k3_1000.json` (and its
  `data/` compatibility mirror) from scratch via
  `scripts/run_full_series.py`. The previous archive contained
  $21$ placeholder rows at $k=10,\ldots,30$ with `residual=0` and
  `nfev=0` originating from an earlier solver-bootstrap failure mode;
  these rows have been replaced by genuine Newton-solver output.
  The `critical_recheck/` layer, which already re-solved these
  $21$ rows independently, is unaffected.
- **Manuscript prose aligned with regenerated archive.** §4.3
  "Archive convention" and the companion recheck remark now describe
  the uniform $998$-row dump ($950$ rows with residual $<10^{-10}$,
  $48$ rows with residuals in $[10^{-10},\,5.35\times 10^{-10}]$,
  $\texttt{nfev}\ge 8$ throughout); the "$108$ critical rows" narrative
  in §4.3, Appendix B, and `REPRODUCE.md` has been replaced by the
  actual $48$-row recheck, and the recheck-bound has been updated to
  $1.3\times 10^{-10}$.
- **Tables 1 and §5.2 residual columns** refreshed against the
  regenerated archive.
- **Duplicate label `prop:rigorous-asymp`** on
  `obs:rigorous-asymp` removed; the two stale `\cref{prop:rigorous-asymp}`
  call sites have been rewired to `obs:rigorous-asymp`.
- **Label prefixes aligned with environments.** `prop:conformality`
  \textrightarrow\ `cert:conformality`, `prop:simple-arc`
  \textrightarrow\ `cert:simple-arc`, `prop:tau-univ`
  \textrightarrow\ `cond:tau-univ`, `obs:gauge-invariant`
  \textrightarrow\ `rem:gauge-invariant` (9 `\cref` sites updated).
- **Wall-clock narrative** removed from §4.5 and from the README quick
  start; runtime is machine-dependent and was inconsistently quoted
  between the two documents.
- **README test count** updated from the stale "$\approx 120$" to the
  actual $153$.
- Cosmetic: stray `%% Placeholder for figure:` LaTeX comment removed.

### Added

- `CHANGELOG.md` (this file) documenting the post-audit close-out.
- Archive-provenance paragraph in `REPRODUCE.md` noting the regeneration
  date and the role of the `critical_recheck/` layer.
- `§2.3 Reality of the spectral coefficients on the rhombic lattice`
  in `paper/*.tex` (Lemma `lem:rhombic-parity` on the
  $p=q^{1/4}$ phase factorisation, Proposition `prop:spectral-reality`
  establishing $R,U,U',U_1',U_2\in\RR$, and the associated remark).
  Supplies the missing derivation behind the `float(np.real(...))`
  casts in `scripts/compute_constants.py`.
- Expanded docstring in `scripts/compute_constants.py` pointing at the
  new `§2.3` justification.
- ORCID for the author in `CITATION.cff`
  (`https://orcid.org/0009-0001-6176-6208`).

### Changed

- `REPRODUCE.md`: cross-references to the regenerated archive.
- **Paper filename.** `paper/Compact Bonnet Pairs for k-fold Tori.tex`
  renamed to `paper/compact_bonnet_pairs_k_fold_tori.tex` (no spaces,
  snake_case). `README.md`, `REPRODUCE.md`, and the release checklist
  below were updated. The rename affects only the build path; the
  arXiv/PDF artifact identity is unchanged.
- **`prop:tau-univ` hypotheses** restructured from prose into a
  numbered enumerate `(H1)/(H2)/(H3)` with labels
  `hyp:root-ordering`, `hyp:lambda-corrected`, `hyp:cusp-branch`; the
  proof now references each hypothesis explicitly via `\ref`.
- **Label `thm:asymptotic` renamed to `valid:asymptotic`** to match
  the `validatedclaim` environment it labels (1 definition +
  4 `\cref` call sites).
- **§1.3 Contributions (spectral-structure bullet).** Clarified that
  the finite-layer root-topology certification is obtained "by
  exact-rational Sturm certificates".
- **`requirements.txt`.** Added upper bounds
  (`numpy<3`, `scipy<2`, `mpmath<2`, `matplotlib<4`, `pytest<9`), a
  tested-with line (Python 3.10/3.11/3.12, Windows 10/11 and
  Ubuntu 22.04), and a one-line inline comment on each dependency.

---

## [2026-04-17] — Initial public release

Initial release, distilled from the HAN-internal working tree per
`migration_v2.md`. Summary of structural and mathematical changes
relative to the pre-release working drafts:

### Added

- **Public repository layout.** `src/` (canonical implementation),
  `scripts/` (paper-cited wrappers + reproducibility runners),
  `tests/`, `results/{phase15_asymptotic,B3,B4,C2}/`, `paper/`,
  `data/` (temporary compatibility mirror).
- **New theorem environment taxonomy** (`paper/*.tex`, preamble
  lines 29–43):
  `validatedclaim`, `computationalcertificate`, `conditionaltheorem`,
  and the existing `theorem`/`proposition`/`lemma`/`conjecture`/
  `observation`/`remark` classes. This distinguishes strict theorems
  from numerically validated and computer-certified claims.
- `REPRODUCE.md` with six canonical commands covering the full
  reproducibility path.
- `CITATION.cff`, MIT `LICENSE`, `requirements.txt`, pytest suite.
- Phase-15 critical-row recheck layer
  (`results/phase15_asymptotic/critical_recheck/`), B3 finite
  certificate layer (`results/B3/finite_certificate/`), B4
  high-precision layer (`results/B4/high_precision/`), and C2 raw-grid
  audit layer (`results/C2/raw_grid/`).
- Figures: `paper/figures/{delta_vs_k,running_exponent,richardson,procrustes}.pdf`.

### Changed (mathematical / manuscript corrections)

These correspond to the RT1 / RT2 / RT3 red-team cycles carried out
pre-release:

- **Running-exponent sign corrected.** Text now consistently reports
  $\alpha(k)\to 0.500$ matching the displayed formula; prior drafts
  carried a residual sign confusion. See §5 of the manuscript.
- **Möbius/cross-ratio relation corrected.** The branch relation is
  now `CR_k = 1/(1 − λ(τ_spec(k)))` everywhere it appears (abstract,
  §1.3, §6, §8). Earlier drafts had `λ(τ_spec) = CR_k`, which is
  incompatible with the boundary circle `|1 − λ(τ)| = 1`.
- **CM correction.** The statement that
  `τ₀ = 1/2 + 25i/78` is "not a CM point" has been removed. The paper
  now states correctly that it *is* a CM point, so the PSLQ negative
  searches around this base do not furnish a structural non-CM
  transcendence.
- **`prop:conformality` → `computationalcertificate`.** The statement
  is now scoped to a finite `24 × 120` grid on $k=3,\ldots,7$
  (no longer extrapolated to all 998 seeds).
- **`prop:simple-arc` softened at $k=7$.** "Cusp at $s_1 \approx -3.5$"
  replaced by "two numerically distinct continuation starts collapse".
- **`obs:CR-universal` sample size stated.** Honest report:
  $97$ stored points from the τ-deformation sweep ($41$ for $k=5$
  and $28$ each for $k=10, 20$).
- **Appendix A theta table labelled "real-normalized".** Explicit
  remark that in the raw Jacobi convention
  $\vt{4}(0) = \overline{\vt{3}(0)}$ for $\Re\tau = 1/2$.
- **`thm:asymptotic` env changed to `validatedclaim`.** The headline
  high-fold asymptotic law is a `validatedclaim`; the label is kept
  as `thm:asymptotic` only for backward-compatible cref resolution.

### Removed

- Historical/Colab-era runners moved from `scripts/` into
  `scripts/legacy/` (PSLQ Colab workflow, atlas exporter, legacy flux
  visualizers, etc.). See `migration_v2.md` (HAN-internal) for the
  complete list.
- Absolute HAN paths stripped from every public-facing document.

### Known issues at 2026-04-17 release

- `results/phase15_asymptotic/full_series_k3_1000.json` and its
  `data/` mirror contained $21$ placeholder rows at $k=10,\ldots,30$.
  Fixed in [Unreleased] above.
- No public-facing `CHANGELOG.md`. Fixed by this file.
- `data/` is a declared *temporary compatibility mirror*, slated
  for deprecation after script/doc rewrites complete. It is retained
  for the 2026-04-17 tag but may be removed in a future release.

---

## Release checklist

Post-fix verification for the next tag:

- [ ] `scripts/run_full_series.py` produces $998$ rows with
  `residual < 10^{-6}` and $\texttt{nfev} > 0$ throughout.
- [ ] `pytest` suite green.
- [ ] `pdflatex compact_bonnet_pairs_k_fold_tori.tex`
  compiles with $0$ errors and $0$ undefined references.
- [ ] `REPRODUCE.md` commands reproduce every paper-cited artifact.
- [ ] Git tag set on the release commit; tag referenced from
  `REPRODUCE.md` and `CITATION.cff`.
