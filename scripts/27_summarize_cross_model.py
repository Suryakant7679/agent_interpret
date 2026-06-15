#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def behavior_summary(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "invalid_outputs": metrics["invalid_outputs"],
        "calculator_recall": metrics["per_label"]["calculator"]["recall"],
        "python_recall": metrics["per_label"]["python"]["recall"],
        "calculator_to_python": metrics["confusion_matrix"]["calculator"]["python"],
    }


def probe_summary(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    best = max(rows, key=lambda row: row["macro_f1"])
    return {
        "best_layer": best["layer"],
        "best_macro_f1": best["macro_f1"],
        "layer_0_macro_f1": rows[0]["macro_f1"],
        "last_layer_macro_f1": rows[-1]["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize reduced cross-model replication outputs"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-behavior", type=Path, required=True)
    parser.add_argument("--ood-behavior", type=Path, required=True)
    parser.add_argument("--lexical-behavior", type=Path, required=True)
    parser.add_argument("--residual-probe", type=Path, required=True)
    parser.add_argument("--mlp-probe", type=Path, required=True)
    parser.add_argument("--patching-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.patching_comparison.open(newline="", encoding="utf-8") as handle:
        patching_rows = list(csv.DictReader(handle))
    strongest = max(
        patching_rows,
        key=lambda row: float(row["treatment_mean_effect"]),
    )

    summary = {
        "model": args.model,
        "behavior": {
            "test": behavior_summary(args.test_behavior),
            "ood": behavior_summary(args.ood_behavior),
            "lexical_control": behavior_summary(args.lexical_behavior),
        },
        "probes": {
            "residual": probe_summary(args.residual_probe),
            "mlp": probe_summary(args.mlp_probe),
        },
        "controlled_patching": {
            "best_layer": int(strongest["layer"]),
            "treatment_mean_effect": float(strongest["treatment_mean_effect"]),
            "treatment_flip_rate": float(strongest["treatment_flip_rate"]),
            "delta_vs_python": float(strongest["delta_vs_python"]),
            "p_vs_python": float(strongest["p_vs_python"]),
            "delta_vs_none": float(strongest["delta_vs_none"]),
            "p_vs_none": float(strongest["p_vs_none"]),
        },
        "checkpoint_note": (
            "Inspect behavior competence before interpreting probes or patching. "
            "A causal replication requires positive treatment deltas against "
            "both controls on paired targets."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
