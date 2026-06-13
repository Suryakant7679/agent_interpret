#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidate MLP-output residual features by selectivity"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    archive = np.load(args.input, allow_pickle=False)
    activations = archive["mlp"]
    labels = archive["labels"]
    label_names = archive["label_names"].tolist()
    if activations.ndim != 3:
        raise ValueError("Expected MLP activations with shape [samples, layers, hidden]")
    counts = np.bincount(labels, minlength=len(label_names))
    if np.any(counts < 2):
        raise SystemExit(
            "Feature ranking requires at least 2 samples per class. "
            f"Found {dict(zip(label_names, counts.tolist(), strict=True))}. "
            "The existing ranking is invalid; re-run balanced activation extraction."
        )

    rankings: dict[str, list[dict]] = {}
    for label_index, label_name in enumerate(label_names):
        positive = activations[labels == label_index]
        negative = activations[labels != label_index]
        mean_difference = positive.mean(axis=0) - negative.mean(axis=0)
        pooled_variance = (
            positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)
        ) / 2
        effect_size = mean_difference / np.sqrt(np.maximum(pooled_variance, 1e-12))
        flat_order = np.argsort(np.abs(effect_size), axis=None)[::-1][: args.top_k]
        layer_indices, feature_indices = np.unravel_index(
            flat_order, effect_size.shape
        )
        rankings[label_name] = [
            {
                "rank": rank,
                "layer": int(layer),
                "feature": int(feature),
                "effect_size": float(effect_size[layer, feature]),
                "mean_difference": float(mean_difference[layer, feature]),
            }
            for rank, (layer, feature) in enumerate(
                zip(layer_indices, feature_indices, strict=True), start=1
            )
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    print(f"Saved top {args.top_k} MLP-output features per tool to {args.output}")
    print(
        "These are residual-space MLP output dimensions, not intermediate MLP "
        "neurons. Causal ablation is still required."
    )


if __name__ == "__main__":
    main()
