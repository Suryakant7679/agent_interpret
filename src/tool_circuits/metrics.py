from __future__ import annotations

from collections import Counter

from . import LABELS


def classification_metrics(gold: list[str], predicted: list[str | None]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    confusion = {actual: {guess: 0 for guess in (*LABELS, "invalid")} for actual in LABELS}
    for actual, guess in zip(gold, predicted, strict=True):
        confusion[actual][guess if guess in LABELS else "invalid"] += 1

    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in (*LABELS, "invalid") if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
        f1_values.append(f1)

    correct = sum(actual == guess for actual, guess in zip(gold, predicted, strict=True))
    return {
        "samples": len(gold),
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "invalid_outputs": Counter(predicted)[None],
        "per_label": per_label,
        "confusion_matrix": confusion,
    }
