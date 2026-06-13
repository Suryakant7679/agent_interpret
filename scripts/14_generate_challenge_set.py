#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool_circuits.io import write_jsonl


SOFTWARE = [
    "PyTorch",
    "TensorFlow",
    "Python",
    "CUDA",
    "PostgreSQL",
    "Node.js",
    "Rust",
    "Ubuntu",
]
CONCEPTS = [
    "regularization",
    "cross-validation",
    "recursion",
    "entropy",
    "gradient descent",
    "normalization",
    "binary search",
    "Bayes' theorem",
]


def metadata(label: str) -> dict[str, bool]:
    return {
        "requires_current_info": label == "web_search",
        "requires_exact_math": label == "calculator",
        "requires_execution": label == "python",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched counterfactual tool-selection quartets"
    )
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "full" / "challenge_test.jsonl"
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []

    for group in range(args.groups):
        software = rng.choice(SOFTWARE)
        concept = rng.choice(CONCEPTS)
        a, b, c = (rng.randint(100, 999) for _ in range(3))
        values = [rng.randint(-50, 250) for _ in range(rng.randint(18, 30))]
        ubuntu_release = rng.choice(["22.04", "24.04", "26.04"])
        python_release = rng.choice(["3.10", "3.11", "3.12", "3.13"])
        contributors = group + 3
        fleet_size = group + 5
        project = rng.choice(
            ["data platform", "web service", "research library", "desktop application"]
        )
        queries = {
            "web_search": (
                f"For {software}, report the stable release available today and "
                f"verify it against an official source for a workstation using "
                f"Ubuntu {ubuntu_release} and Python {python_release} across a "
                f"fleet of {fleet_size} machines."
            ),
            "none": (
                f"For {software}, explain conceptually why software projects use "
                f"version numbers to a {contributors}-person team maintaining a "
                f"{project}, relating the explanation to {concept}."
            ),
            "calculator": (
                f"For the expression ({a} cubed plus {b}) divided by {c}, return "
                "one exact numerical result rounded to eight decimal places."
            ),
            "python": (
                f"For each value in {values}, compute (value cubed plus {b}) divided "
                "by the sample standard deviation, then summarize the resulting distribution."
            ),
        }
        for label, query in queries.items():
            rows.append(
                {
                    "id": f"challenge_{group:04d}_{label}",
                    "pair_id": f"challenge_{group:04d}",
                    "query": query,
                    "label": label,
                    "category": "counterfactual_quartet",
                    "template": "matched_challenge",
                    "difficulty": "hard",
                    "adversarial_flag": False,
                    "split": "challenge_test",
                    **metadata(label),
                }
            )

    rng.shuffle(rows)
    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} prompts ({args.groups} matched quartets) to {args.output}")


if __name__ == "__main__":
    main()
