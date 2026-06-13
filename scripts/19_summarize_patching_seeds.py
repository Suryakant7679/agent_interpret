#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check controlled patching results across random seeds"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        csv_path = (
            args.input_root
            / f"seed_{seed}"
            / "controlled_comparison"
            / "patching_control_comparison.csv"
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        best = max(rows, key=lambda row: float(row["treatment_mean_effect"]))
        delta_python = float(best["delta_vs_python"])
        delta_none = float(best["delta_vs_none"])
        passed = delta_python > 0 and delta_none > 0
        results.append(
            {
                "seed": seed,
                "best_layer": int(best["layer"]),
                "treatment_mean_effect": float(best["treatment_mean_effect"]),
                "delta_vs_python": delta_python,
                "delta_vs_none": delta_none,
                "passed": passed,
            }
        )

    summary = {
        "criterion": (
            "At each seed's strongest treatment layer, calculator-source "
            "patching has a larger mean effect than Python and none controls."
        ),
        "seeds": results,
        "all_seeds_passed": all(row["passed"] for row in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"seed={row['seed']} layer={row['best_layer']} "
            f"effect={row['treatment_mean_effect']:.4f} "
            f"delta_python={row['delta_vs_python']:.4f} "
            f"delta_none={row['delta_vs_none']:.4f} {status}"
        )
    print(f"ALL SEEDS PASSED: {summary['all_seeds_passed']}")


if __name__ == "__main__":
    main()
