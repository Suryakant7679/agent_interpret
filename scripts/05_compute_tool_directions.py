#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute one-vs-rest tool directions")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation", choices=["residual", "mlp"], default="residual")
    args = parser.parse_args()

    archive = np.load(args.input, allow_pickle=False)
    activations = archive[args.activation]
    labels = archive["labels"]
    label_names = archive["label_names"].tolist()
    counts = np.bincount(labels, minlength=len(label_names))
    if np.any(counts < 2):
        raise SystemExit(
            "Reliable tool directions require at least 2 samples per class. "
            f"Found {dict(zip(label_names, counts.tolist(), strict=True))}. "
            "Re-run balanced activation extraction."
        )

    directions = []
    for label_index in range(len(label_names)):
        positive = activations[labels == label_index].mean(axis=0)
        negative = activations[labels != label_index].mean(axis=0)
        direction = positive - negative
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        directions.append(direction / np.maximum(norm, 1e-12))
    directions_array = np.stack(directions)

    cosine_by_layer = np.einsum("clh,dlh->lcd", directions_array, directions_array)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"{args.activation}_tool_directions.npy", directions_array)
    np.save(args.output_dir / f"{args.activation}_direction_cosines.npy", cosine_by_layer)
    metadata = {
        "activation": args.activation,
        "labels": label_names,
        "directions_shape": list(directions_array.shape),
        "cosines_shape": list(cosine_by_layer.shape),
    }
    (args.output_dir / f"{args.activation}_direction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
