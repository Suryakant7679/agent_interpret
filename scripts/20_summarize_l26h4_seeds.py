#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check L26H4 necessity and specificity across sampling seeds"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        path = args.input_root / f"seed_{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        calculator = result["results"]["calculator"]["target"]
        unrelated_flips = {
            label: result["results"][label]["target"]["flip_rate"]
            for label in ("web_search", "python", "none")
        }
        strong_calculator_drop = (
            calculator["mean_effect"] <= -2.0 and calculator["ci95"][1] < 0
        )
        unrelated_stable = all(rate == 0 for rate in unrelated_flips.values())
        rows.append(
            {
                "seed": seed,
                "calculator_mean_effect": calculator["mean_effect"],
                "calculator_ci95": calculator["ci95"],
                "calculator_flip_rate": calculator["flip_rate"],
                "unrelated_flip_rates": unrelated_flips,
                "specificity_score": result["specificity_score"],
                "strong_calculator_drop": strong_calculator_drop,
                "unrelated_stable": unrelated_stable,
                "passed": strong_calculator_drop and unrelated_stable,
            }
        )

    summary = {
        "criterion": (
            "For every seed, the calculator mean effect is at most -2 logits "
            "with a 95% CI below zero, and unrelated labels have zero flips."
        ),
        "seeds": rows,
        "all_seeds_passed": all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"seed={row['seed']} "
            f"calculator_effect={row['calculator_mean_effect']:.4f} "
            f"calculator_flip_rate={row['calculator_flip_rate']:.4f} "
            f"specificity={row['specificity_score']:.4f} "
            f"unrelated_flips={row['unrelated_flip_rates']} {status}"
        )
    print(f"ALL SEEDS PASSED: {summary['all_seeds_passed']}")


if __name__ == "__main__":
    main()
