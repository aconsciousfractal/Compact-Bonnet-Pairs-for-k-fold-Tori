# Bonnet Theorem 9 Perturbative 4-fold - High-Res Final

Auxiliary / legacy visualization artifact.  This directory is not part of the
canonical paper-facing reproduction path documented in `REPRODUCE.md`; it is
kept as a high-resolution Theorem 9 visualization/reference case.

## Summary

This directory contains the current high-resolution reference case for the first perturbative 4-fold Bonnet example beyond the spherical Theorem 7 family.

The workflow was:

1. build the 4-fold Theorem 7 seed
2. solve the local Theorem 9 perturbative corrector in `alpha`
3. sweep `epsilon_bonnet`
4. run a resolution ladder
5. export the best validation case as OBJ + Flux wireframe

## Seed

- `tau_imag = 0.3205128205`
- `delta = 1.61245155`
- `s1 = -3.13060628`
- `s2 = 0.5655771591`
- `symmetry_fold = 4`
- `epsilon_perturbation = 5e-4`

Solved perturbative coefficients:

- `alpha = [0.009192282477850252, -0.05631084078314392, 0.13657403415912184]`
- `residual_norm = 3.1152302293529824e-11`

## Epsilon Sweep

Reference resolution: `80 x 80`

| epsilon_bonnet | ratio | interior_metric_mean_err | interior_mean_diff(H) | Procrustes disparity | score |
|---|---:|---:|---:|---:|---:|
| 0.10 | 1.000004316 | 0.4000063 | 0.0480363 | 0.0552422 | 0.1453189 |
| 0.15 | 1.000004316 | 0.5992836 | 0.0687784 | 0.1192901 | 0.2126780 |
| 0.20 | 1.000004316 | 0.7977179 | 0.0869398 | 0.2005320 | 0.2763859 |
| 0.25 | 1.000004316 | 0.9950767 | 0.1022937 | 0.2923366 | 0.3364892 |
| 0.30 | 1.000004316 | 1.1911644 | 0.1146428 | 0.3881758 | 0.3930683 |

Chosen validation epsilon:

- `epsilon_bonnet = 0.10`

Reason:

- best combined validation score
- much cleaner interior metric / curvature behavior
- still non-congruent

## Resolution Ladder

Fixed `epsilon_bonnet = 0.10`

| resolution | ratio | interior_metric_mean_err | interior_mean_diff(H) | Procrustes disparity |
|---|---:|---:|---:|---:|
| 80  | 1.000004316 | 0.4000063 | 0.0480363 | 0.0552422 |
| 120 | 1.000004316 | 0.4009633 | 0.0480742 | 0.0551654 |
| 160 | 1.000004316 | 0.4014641 | 0.0480888 | 0.0551263 |

Observation:

- the important interior metrics are stable across the ladder
- the global maxima remain much noisier than the interior metrics
- the case is numerically stable enough to serve as the current reference validation case

## Final Exported Case

- `resolution = 160 x 160`
- `epsilon_bonnet = 0.10`
- `ratio = 1.000004315558235`
- `E_max_rel_err = 1.3042035988973282e-13`
- `interior_F_max_scaled_err = 0.4357000555611049`
- `interior_G_max_rel_err = 0.275142831081877`
- `interior_metric_mean_err = 0.4014640598381336`
- `interior_mean_diff(H) = 0.048088820464049026`
- `procrustes_disparity = 0.05512633052181219`

## Files

- `bonnet_theorem9_perturbed_4fold_highres_res160_eps10_f_plus.obj`
- `bonnet_theorem9_perturbed_4fold_highres_res160_eps10_f_minus.obj`
- `bonnet_theorem9_perturbed_4fold_highres_res160_eps10_flux.obj`
- `bonnet_theorem9_perturbed_4fold_highres_res160_eps10_material.mtl`
- `bonnet_theorem9_perturbed_4fold_highres_report.json`

## Interpretation

This is the current best **validation-oriented** high-resolution perturbative 4-fold case.

It is not the most visually separated case. A larger `epsilon_bonnet` produces stronger visual separation but worse validation metrics. For presentation / visualization, generate a separate visual companion with larger Bonnet epsilon.
