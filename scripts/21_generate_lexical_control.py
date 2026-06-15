#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool_circuits.io import write_jsonl


LABELS = ("web_search", "calculator", "python", "none")
SOFTWARE = (
    "PyTorch",
    "TensorFlow",
    "Python",
    "CUDA",
    "PostgreSQL",
    "Node.js",
    "Rust",
    "Ubuntu",
)
CONCEPTS = (
    "regularization",
    "cross-validation",
    "recursion",
    "entropy",
    "gradient descent",
    "normalization",
    "binary search",
    "Bayes' theorem",
)
PROJECTS = (
    "data platform",
    "web service",
    "research library",
    "desktop application",
)
REFERENCE_DATES = (
    "2026-01-15",
    "2026-02-12",
    "2026-03-19",
    "2026-04-16",
    "2026-05-14",
    "2026-06-11",
)

PREFIX = "Prepare a concise response for the following request: "
SUFFIX = " Include the requested supporting details."


def metadata(label: str) -> dict[str, bool]:
    return {
        "requires_current_info": label == "web_search",
        "requires_exact_math": label == "calculator",
        "requires_execution": label == "python",
    }


def build_rows(groups: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for group in range(groups):
        software = rng.choice(SOFTWARE)
        concept = rng.choice(CONCEPTS)
        project = rng.choice(PROJECTS)
        reference_date = rng.choice(REFERENCE_DATES)
        a, b, c = (rng.randint(100, 999) for _ in range(3))
        values = [rng.randint(-90, 260) for _ in range(rng.randint(8, 12))]
        team_size = rng.randint(4, 120)
        case_frame = (
            f"Case {group + 1:04d}. A {team_size}-person team maintaining a "
            f"{project} submitted this request. "
        )

        bodies = {
            "web_search": (
                f"Using the official {software} project record for {reference_date}, "
                "identify the stable release listed there and give the source location."
            ),
            "calculator": (
                f"Determine the numerical value of ({a} cubed plus {b}) divided by "
                f"{c}, rounded to eight decimal places, and show the intermediate "
                "quantities used."
            ),
            "python": (
                f"Using the observations {values}, calculate the mean, sample standard "
                "deviation, and interquartile range."
            ),
            "none": (
                f"Describe how {concept} relates to software decision-making, with one "
                f"practical example involving {software} and two implications."
            ),
        }
        for label in LABELS:
            rows.append(
                {
                    "id": f"lexical_control_{group:04d}_{label}",
                    "pair_id": f"lexical_control_{group:04d}",
                    "query": PREFIX + case_frame + bodies[label] + SUFFIX,
                    "label": label,
                    "category": "lexical_control_quartet",
                    "template": "shared_frame",
                    "difficulty": "hard",
                    "adversarial_flag": False,
                    "split": "lexical_control",
                    **metadata(label),
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a matched lexical-control tool-selection benchmark"
    )
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "lexical_control" / "test.jsonl",
    )
    args = parser.parse_args()
    if args.groups < 1:
        raise SystemExit("--groups must be positive")

    rows = build_rows(args.groups, args.seed)
    write_jsonl(args.output, rows)
    print(
        f"Saved {len(rows)} prompts ({args.groups} matched quartets) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
