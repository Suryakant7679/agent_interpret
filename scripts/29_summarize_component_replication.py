#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "samples": result["samples"],
        "top_k": result["top_k"],
        "mean_effect": result["summary"]["top_ranked"]["mean_effect"],
        "ci95": result["summary"]["top_ranked"]["ci95"],
        "random_control_mean_effect": result["summary"]["random_control"][
            "mean_effect"
        ],
        "paired_permutation_p": result["summary"]["paired_permutation_p"],
        "top_flip_rate": result["summary"]["top_flip_rate"],
        "control_flip_rate": result["summary"]["control_flip_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize cross-model neuron and head ablations"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--neurons", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "neurons": summarize(args.neurons),
        "heads": summarize(args.heads),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
