#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def probe_summary(rows: list[dict]) -> dict:
    best = max(rows, key=lambda row: row["macro_f1"])
    early = rows[: min(5, len(rows))]
    late = rows[max(0, len(rows) - 5) :]
    first_above_90 = next(
        (row["layer"] for row in rows if row["macro_f1"] >= 0.90),
        None,
    )
    return {
        "layers": len(rows),
        "best_layer": best["layer"],
        "best_macro_f1": best["macro_f1"],
        "layer_0_macro_f1": rows[0]["macro_f1"],
        "mean_first_5_macro_f1": sum(row["macro_f1"] for row in early) / len(early),
        "mean_last_5_macro_f1": sum(row["macro_f1"] for row in late) / len(late),
        "first_layer_macro_f1_at_least_0_90": first_above_90,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize lexical-control behavior and held-out probes"
    )
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    behavior = json.loads(args.behavior.read_text(encoding="utf-8"))
    residual = json.loads(args.residual.read_text(encoding="utf-8"))
    mlp = json.loads(args.mlp.read_text(encoding="utf-8"))
    summary = {
        "behavior": {
            "samples": behavior["samples"],
            "accuracy": behavior["accuracy"],
            "macro_f1": behavior["macro_f1"],
            "invalid_outputs": behavior["invalid_outputs"],
            "per_label": behavior["per_label"],
        },
        "residual_probe": probe_summary(residual),
        "mlp_probe": probe_summary(mlp),
        "interpretation_required": (
            "Compare layer-0 and first-five-layer performance with late-layer "
            "performance. This script reports the evidence but does not impose "
            "a pass threshold on the scientific result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
