"""
Higher-fold Bonnet pair seed catalog.

All seeds validated via:
  - Theorem 7: rationality + vanishing axial residuals (≤ 5e-12)
  - Lemma 8: conditions (iii)-(v) all satisfied
  - Isometry: E_rel_err ~ 1e-14
  - Non-congruence: Procrustes disparity confirmed

Discovered 2026-04-02/03 during Phase 11 exploration.
Extended to k=8..12 during Phase 15 (2026-04-04).
Paper: Bobenko–Hoffmann–Sageman-Furnas (s10240-025-00159-z).

Key finding: s₂ ∝ k^2.01  (R² = 0.88)  — quadratic scaling law (k=3..7).
Key finding Phase 15: δ peaks at k=8 (2.11) then DECREASES — non-monotone δ(k).
"""

# All seeds share the same lattice parameter
TAU_IMAG = 0.3205128205  # τ = 0.5 + i·TAU_IMAG

SEEDS = {
    # ── Paper seeds (Bobenko et al.) ──
    3: {
        "label": "paper_3fold",
        "delta": 1.897366596066,
        "s1": -3.601381552,
        "s2": 0.596520201315,
        "theta_deg": 120.0,
        "residual_norm": 3.78e-12,
        "source": "paper",
    },
    4: {
        "label": "paper_4fold",
        "delta": 1.612451549616,
        "s1": -3.13060628,
        "s2": 0.565577159433,
        "theta_deg": 90.0,
        "residual_norm": 4.18e-13,
        "source": "paper",
    },
    # ── New discoveries (Phase 11) ──
    5: {
        "label": "new_5fold",
        "delta": 1.678757500643,
        "s1": -3.80,
        "s2": 1.188432661226,
        "theta_deg": 72.0,
        "residual_norm": 6.17e-13,
        "source": "phase11",
    },
    6: {
        "label": "new_6fold",
        "delta": 1.749166739658,
        "s1": -4.50,
        "s2": 1.813323472665,
        "theta_deg": 60.0,
        "residual_norm": 4.45e-12,
        "source": "phase11",
        "note": "±δ produces same surfaces (Remark 11); -δ variant: -1.749166739657",
    },
    7: {
        "label": "new_7fold_a",
        "delta": 1.959258407446,
        "s1": -6.00,
        "s2": 2.979872632766,
        "theta_deg": 51.4286,
        "residual_norm": 2.71e-13,
        "source": "phase11",
    },
    # ── Phase 15 discoveries ──
    8: {
        "label": "new_8fold",
        "delta": 2.110430633960,
        "s1": -7.5,
        "s2": 4.203859622936,
        "theta_deg": 45.0,
        "residual_norm": 3.33e-12,
        "source": "phase15",
    },
    9: {
        "label": "new_9fold",
        "delta": 2.085021670166,
        "s1": -8.0,
        "s2": 4.758109471537,
        "theta_deg": 40.0,
        "residual_norm": 1.74e-12,
        "source": "phase15",
    },
    10: {
        "label": "new_10fold",
        "delta": 2.063114436779,
        "s1": -8.5,
        "s2": 5.304037890320,
        "theta_deg": 36.0,
        "residual_norm": 1.60e-12,
        "source": "phase15",
    },
    11: {
        "label": "new_11fold",
        "delta": 1.981205459539,
        "s1": -8.5,
        "s2": 5.461485085276,
        "theta_deg": 32.7273,
        "residual_norm": 5.27e-13,
        "source": "phase15",
    },
    12: {
        "label": "new_12fold",
        "delta": 1.714196682214,
        "s1": -7.0,
        "s2": 4.421958233663,
        "theta_deg": 30.0,
        "residual_norm": 7.76e-13,
        "source": "phase15",
        "note": "δ decreasing from k=8 peak — non-monotone behavior confirmed",
    },
}

# Additional seeds at the same k (different s₁ — distinct parameter families)
EXTRA_SEEDS = {
    "7b": {
        "label": "new_7fold_b",
        "k": 7,
        "delta": 1.862353106277,
        "s1": -5.50,
        "s2": 2.647915844137,
        "theta_deg": 51.4286,
        "residual_norm": 2.05e-12,
        "source": "phase11",
        "note": "Same family as 7a — Phase 11.5 sweep confirmed both branches converge "
               "to identical (δ, s₂) at every s₁. Kept for cross-check only.",
    },
}


def get_seed(k: int) -> dict:
    """Get seed parameters for k-fold symmetry.

    Returns dict with: delta, s1, s2, tau_imag, symmetry_fold.
    """
    if k not in SEEDS:
        raise ValueError(f"No seed for k={k}. Available: {sorted(SEEDS.keys())}")
    s = SEEDS[k]
    return {
        "tau_imag": TAU_IMAG,
        "delta": s["delta"],
        "s1": s["s1"],
        "s2": s["s2"],
        "symmetry_fold": k,
    }


# Scaling law coefficients: param ≈ C · k^α
SCALING_LAWS = {
    "abs_delta": {"alpha": 0.04, "C": 1.667, "R2": 0.027, "note": "~constant"},
    "abs_s1":    {"alpha": 0.61, "C": 1.586, "R2": 0.661, "note": "moderate growth"},
    "s2":        {"alpha": 2.01, "C": 0.050, "R2": 0.883, "note": "quadratic growth"},
}
