#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tool_circuits.statistics import paired_permutation_pvalue


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare patching treatment to controls")
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--controls", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    controls = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.controls
    ]
    target_ids = [row["target_id"] for row in treatment["pairs"]]
    for control in controls:
        control_targets = [row["target_id"] for row in control["pairs"]]
        if control_targets != target_ids:
            raise SystemExit(
                "Patching files do not contain identical ordered targets. "
                "Use the same --seed and --pairs for each run."
            )

    rows = []
    for layer in range(len(treatment["summary"])):
        treatment_effect = np.asarray(
            [pair["layers"][layer]["effect"] for pair in treatment["pairs"]]
        )
        row = {
            "layer": layer,
            "treatment_source": treatment["source_label"],
            "treatment_mean_effect": float(treatment_effect.mean()),
            "treatment_flip_rate": treatment["summary"][layer]["flip_rate"],
        }
        for control in controls:
            label = control["source_label"]
            control_effect = np.asarray(
                [pair["layers"][layer]["effect"] for pair in control["pairs"]]
            )
            row[f"{label}_mean_effect"] = float(control_effect.mean())
            row[f"{label}_flip_rate"] = control["summary"][layer]["flip_rate"]
            row[f"delta_vs_{label}"] = float(
                (treatment_effect - control_effect).mean()
            )
            row[f"p_vs_{label}"] = paired_permutation_pvalue(
                treatment_effect, control_effect, seed=args.seed
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "patching_control_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ordered = sorted(rows, key=lambda row: row["treatment_mean_effect"], reverse=True)
    markdown = [
        "| Layer | Treatment effect | Flip rate | "
        + " | ".join(
            f"Delta vs {control['source_label']} | p"
            for control in controls
        )
        + " |",
        "|---:|---:|---:|" + "---:|---:|" * len(controls),
    ]
    for row in ordered[:10]:
        values = [
            str(row["layer"]),
            f"{row['treatment_mean_effect']:.3f}",
            f"{row['treatment_flip_rate']:.3f}",
        ]
        for control in controls:
            label = control["source_label"]
            values.extend(
                [f"{row[f'delta_vs_{label}']:.3f}", f"{row[f'p_vs_{label}']:.4g}"]
            )
        markdown.append("| " + " | ".join(values) + " |")
    markdown_path = args.output_dir / "patching_control_comparison.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
