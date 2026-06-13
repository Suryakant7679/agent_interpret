#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def standardized_difference(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    variance = (positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)) / 2
    return difference / np.sqrt(np.maximum(variance, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank true MLP neurons and heads")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    archive = np.load(args.input, allow_pickle=False)
    neurons = archive["neurons"].astype(np.float32)
    heads = archive["heads"].astype(np.float32)
    labels = archive["labels"]
    names = archive["label_names"].tolist()
    result = {"neurons": {}, "heads": {}}

    for label_index, name in enumerate(names):
        neuron_effect = standardized_difference(
            neurons[labels == label_index], neurons[labels != label_index]
        )
        order = np.argsort(np.abs(neuron_effect), axis=None)[::-1][: args.top_k]
        layers, indices = np.unravel_index(order, neuron_effect.shape)
        result["neurons"][name] = [
            {
                "rank": rank,
                "layer": int(layer),
                "neuron": int(neuron),
                "effect_size": float(neuron_effect[layer, neuron]),
            }
            for rank, (layer, neuron) in enumerate(
                zip(layers, indices, strict=True), start=1
            )
        ]

        head_vectors = np.linalg.norm(heads, axis=-1)
        head_effect = standardized_difference(
            head_vectors[labels == label_index], head_vectors[labels != label_index]
        )
        order = np.argsort(np.abs(head_effect), axis=None)[::-1][: args.top_k]
        layers, indices = np.unravel_index(order, head_effect.shape)
        result["heads"][name] = [
            {
                "rank": rank,
                "layer": int(layer),
                "head": int(head),
                "effect_size": float(head_effect[layer, head]),
            }
            for rank, (layer, head) in enumerate(
                zip(layers, indices, strict=True), start=1
            )
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved component rankings to {args.output}")


if __name__ == "__main__":
    main()
