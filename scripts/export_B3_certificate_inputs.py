#!/usr/bin/env python3
"""Export authoritative B3 finite-certificate inputs from archived repo data.

This script writes CSV inputs for the exact-rational B3 certificate from the
archived JSON files already present in the repository.

Outputs:
  - results/B3/finite_certificate/b3_full_series_998.csv
  - results/B3/finite_certificate/b3_sweep_raw_121.csv
  - results/B3/finite_certificate/b3_sweep_unique_91_from_lemma9.csv
  - results/B3/finite_certificate/b3_sweep_unique_91_from_phase17.csv
  - results/B3/finite_certificate/b3_export_summary.json

The summary file records the dataset counts and the max discrepancy between the
deduplicated k=5,6,7 sweep from ``lemma9_sweep.json`` and the unique curves
archived in ``phase17_345_results.json``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUT_DIR = RESULTS / "B3" / "finite_certificate"

PHASE15_PATH = RESULTS / "phase15_asymptotic" / "full_series_k3_1000.json"
LEMMA9_PATH = RESULTS / "phase11_higher_fold" / "lemma9_sweep.json"
PHASE17_PATH = RESULTS / "phase17_moduli" / "phase17_345_results.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh, parse_float=Decimal)


def as_text(value) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_text(row.get(key, "")) for key in fieldnames})


def build_full_series_rows(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        rows.append(
            {
                "k": entry["k"],
                "s1": entry["s1"],
                "delta": entry["delta"],
                "s2": entry["s2"],
                "source": "phase15_asymptotic/full_series_k3_1000.json",
            }
        )
    rows.sort(key=lambda row: int(str(row["k"])))
    return rows


def build_lemma9_raw_rows(lemma9: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k_text, entries in sorted(lemma9.items(), key=lambda item: int(item[0])):
        for entry in sorted(entries, key=lambda row: (Decimal(str(row["s1"])), int(row.get("branch", 0)))):
            rows.append(
                {
                    "k": int(k_text),
                    "s1": entry["s1"],
                    "delta": entry["delta"],
                    "s2": entry["s2"],
                    "branch": entry.get("branch", 0),
                    "source": "phase11_higher_fold/lemma9_sweep.json",
                }
            )
    return rows


def build_lemma9_unique_rows(
    lemma9: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    per_k_counts: dict[str, int] = {}
    duplicate_summary: dict[str, dict[str, str]] = {}

    for k_text, entries in sorted(lemma9.items(), key=lambda item: int(item[0])):
        grouped: dict[Decimal, list[dict[str, object]]] = defaultdict(list)
        for entry in entries:
            grouped[Decimal(str(entry["s1"]))].append(entry)

        unique_rows: list[dict[str, object]] = []
        max_delta_diff = Decimal("0")
        max_s2_diff = Decimal("0")

        for s1_value in sorted(grouped):
            candidates = sorted(grouped[s1_value], key=lambda row: int(row.get("branch", 0)))
            rep = candidates[0]
            if len(candidates) > 1:
                deltas = [Decimal(str(row["delta"])) for row in candidates]
                s2s = [Decimal(str(row["s2"])) for row in candidates]
                max_delta_diff = max(max_delta_diff, max(deltas) - min(deltas))
                max_s2_diff = max(max_s2_diff, max(s2s) - min(s2s))

            unique_rows.append(
                {
                    "k": int(k_text),
                    "s1": rep["s1"],
                    "delta": rep["delta"],
                    "s2": rep["s2"],
                    "branch": rep.get("branch", 0),
                    "duplicate_count": len(candidates),
                    "source": "phase11_higher_fold/lemma9_sweep.json (deduplicated by k,s1)",
                }
            )

        per_k_counts[k_text] = len(unique_rows)
        duplicate_summary[k_text] = {
            "raw_rows": str(len(entries)),
            "unique_rows": str(len(unique_rows)),
            "max_delta_diff_across_duplicates": as_text(max_delta_diff),
            "max_s2_diff_across_duplicates": as_text(max_s2_diff),
        }
        rows.extend(unique_rows)

    return rows, {
        "per_k_unique_counts": per_k_counts,
        "duplicate_summary": duplicate_summary,
    }


def build_phase17_unique_rows(phase17: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task = phase17["task_17_3_curvature"]
    for k_text in ("5", "6", "7"):
        entry = task[k_text]
        s1_values = entry["s1"]
        delta_values = entry["delta"]
        s2_values = entry["s2"]
        for s1_value, delta_value, s2_value in zip(s1_values, delta_values, s2_values):
            rows.append(
                {
                    "k": int(k_text),
                    "s1": s1_value,
                    "delta": delta_value,
                    "s2": s2_value,
                    "source": "phase17_moduli/phase17_345_results.json",
                }
            )
    rows.sort(key=lambda row: (int(str(row["k"])), Decimal(str(row["s1"]))))
    return rows


def compare_unique_rows(
    lemma9_rows: list[dict[str, object]], phase17_rows: list[dict[str, object]]
) -> dict[str, object]:
    lemma9_map = {(int(str(row["k"])), Decimal(str(row["s1"]))): row for row in lemma9_rows}
    phase17_map = {(int(str(row["k"])), Decimal(str(row["s1"]))): row for row in phase17_rows}

    common_keys = sorted(set(lemma9_map) & set(phase17_map))
    max_delta_diff = Decimal("0")
    max_s2_diff = Decimal("0")
    worst_delta = None
    worst_s2 = None
    for key in common_keys:
        l_row = lemma9_map[key]
        p_row = phase17_map[key]
        delta_diff = abs(Decimal(str(l_row["delta"])) - Decimal(str(p_row["delta"])))
        s2_diff = abs(Decimal(str(l_row["s2"])) - Decimal(str(p_row["s2"])))
        if delta_diff > max_delta_diff:
            max_delta_diff = delta_diff
            worst_delta = {"k": key[0], "s1": as_text(key[1]), "difference": as_text(delta_diff)}
        if s2_diff > max_s2_diff:
            max_s2_diff = s2_diff
            worst_s2 = {"k": key[0], "s1": as_text(key[1]), "difference": as_text(s2_diff)}

    return {
        "lemma9_unique_rows": len(lemma9_rows),
        "phase17_unique_rows": len(phase17_rows),
        "common_rows": len(common_keys),
        "max_abs_delta_diff": as_text(max_delta_diff),
        "max_abs_s2_diff": as_text(max_s2_diff),
        "worst_delta_row": worst_delta,
        "worst_s2_row": worst_s2,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase15 = load_json(PHASE15_PATH)
    lemma9 = load_json(LEMMA9_PATH)
    phase17 = load_json(PHASE17_PATH)

    full_series_rows = build_full_series_rows(phase15)
    lemma9_raw_rows = build_lemma9_raw_rows(lemma9)
    lemma9_unique_rows, lemma9_meta = build_lemma9_unique_rows(lemma9)
    phase17_unique_rows = build_phase17_unique_rows(phase17)
    comparison = compare_unique_rows(lemma9_unique_rows, phase17_unique_rows)

    write_csv(
        OUT_DIR / "b3_full_series_998.csv",
        ["k", "s1", "delta", "s2", "source"],
        full_series_rows,
    )
    write_csv(
        OUT_DIR / "b3_sweep_raw_121.csv",
        ["k", "s1", "delta", "s2", "branch", "source"],
        lemma9_raw_rows,
    )
    write_csv(
        OUT_DIR / "b3_sweep_unique_91_from_lemma9.csv",
        ["k", "s1", "delta", "s2", "branch", "duplicate_count", "source"],
        lemma9_unique_rows,
    )
    write_csv(
        OUT_DIR / "b3_sweep_unique_91_from_phase17.csv",
        ["k", "s1", "delta", "s2", "source"],
        phase17_unique_rows,
    )

    summary = {
        "datasets": {
            "phase15_full_series": {
                "path": str(PHASE15_PATH.relative_to(PROJECT)),
                "rows": len(full_series_rows),
            },
            "phase11_lemma9_raw": {
                "path": str(LEMMA9_PATH.relative_to(PROJECT)),
                "rows": len(lemma9_raw_rows),
            },
            "phase11_lemma9_unique": lemma9_meta,
            "phase17_unique": {
                "path": str(PHASE17_PATH.relative_to(PROJECT)),
                "rows": len(phase17_unique_rows),
            },
        },
        "lemma9_vs_phase17_unique_comparison": comparison,
        "written_files": [
            "results/B3/finite_certificate/b3_full_series_998.csv",
            "results/B3/finite_certificate/b3_sweep_raw_121.csv",
            "results/B3/finite_certificate/b3_sweep_unique_91_from_lemma9.csv",
            "results/B3/finite_certificate/b3_sweep_unique_91_from_phase17.csv",
            "results/B3/finite_certificate/b3_export_summary.json",
        ],
    }

    (OUT_DIR / "b3_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote B3 certificate input exports:")
    for rel in summary["written_files"]:
        print(f"  - {rel}")
    print()
    print("Dataset counts:")
    print(f"  full_series_998 = {len(full_series_rows)}")
    print(f"  sweep_raw_121   = {len(lemma9_raw_rows)}")
    print(f"  sweep_unique_91 = {len(lemma9_unique_rows)}")
    print(f"  phase17_unique  = {len(phase17_unique_rows)}")
    print()
    print("lemma9 unique vs phase17 unique:")
    print(f"  common rows      = {comparison['common_rows']}")
    print(f"  max |delta diff| = {comparison['max_abs_delta_diff']}")
    print(f"  max |s2 diff|    = {comparison['max_abs_s2_diff']}")


if __name__ == "__main__":
    main()
