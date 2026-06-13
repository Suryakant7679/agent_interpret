#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train layer-wise linear probes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--test-input",
        type=Path,
        help="Optional held-out activation archive. When set, no random split is used.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation", choices=["residual", "mlp"], default="residual")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Install research dependencies: python3 -m pip install -e '.[research]'"
        ) from exc

    archive = np.load(args.input, allow_pickle=False)
    activations = archive[args.activation]
    labels = archive["labels"]
    label_names = archive["label_names"].tolist()
    counts = np.bincount(labels, minlength=len(label_names))
    insufficient = {
        label_names[index]: int(count)
        for index, count in enumerate(counts)
        if count < 2
    }
    if insufficient:
        raise SystemExit(
            "Each class needs at least 2 samples for a stratified probe split. "
            f"Found {dict(zip(label_names, counts.tolist(), strict=True))}. "
            "Re-run activation extraction with a balanced --limit of at least 16."
        )

    if args.test_input:
        test_archive = np.load(args.test_input, allow_pickle=False)
        test_activations = test_archive[args.activation]
        test_labels = test_archive["labels"]
        test_label_names = test_archive["label_names"].tolist()
        if test_label_names != label_names:
            raise SystemExit("Training and test archives have different label orders.")
        if test_activations.shape[1:] != activations.shape[1:]:
            raise SystemExit(
                "Training and test activation shapes are incompatible: "
                f"{activations.shape[1:]} vs {test_activations.shape[1:]}"
            )
        test_counts = np.bincount(test_labels, minlength=len(label_names))
        if np.any(test_counts < 1):
            raise SystemExit(
                "Held-out archive must contain every class. "
                f"Found {dict(zip(label_names, test_counts.tolist(), strict=True))}."
            )
        train_activations = activations
        train_labels = labels
        evaluation_activations = test_activations
        evaluation_labels = test_labels
        evaluation_mode = "held_out_archive"
    else:
        indices = np.arange(len(labels))
        requested_test_count = math.ceil(len(labels) * args.test_size)
        test_count = max(requested_test_count, len(label_names))
        if len(labels) - test_count < len(label_names):
            raise SystemExit(
                "Not enough samples to place every class in both train and test sets. "
                "Extract more balanced samples."
            )
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_count,
            random_state=args.seed,
            stratify=labels,
        )
        train_activations = activations[train_indices]
        train_labels = labels[train_indices]
        evaluation_activations = activations[test_indices]
        evaluation_labels = labels[test_indices]
        evaluation_mode = "random_stratified_split"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    coefficients = []
    for layer in range(activations.shape[1]):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=args.seed),
        )
        probe.fit(train_activations[:, layer], train_labels)
        predicted = probe.predict(evaluation_activations[:, layer])
        results.append(
            {
                "layer": layer,
                "accuracy": float(accuracy_score(evaluation_labels, predicted)),
                "macro_f1": float(
                    f1_score(evaluation_labels, predicted, average="macro")
                ),
                "confusion_matrix": confusion_matrix(
                    evaluation_labels,
                    predicted,
                    labels=np.arange(len(label_names)),
                ).tolist(),
            }
        )
        coefficients.append(probe[-1].coef_)

    (args.output_dir / f"{args.activation}_probe_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.save(
        args.output_dir / f"{args.activation}_probe_coefficients.npy",
        np.stack(coefficients),
    )
    best = max(results, key=lambda row: row["macro_f1"])
    print(
        json.dumps(
            {
                "evaluation_mode": evaluation_mode,
                "train_samples": int(len(train_labels)),
                "test_samples": int(len(evaluation_labels)),
                "best": best,
                "all_layers": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
