from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from . import LABELS


def balanced_subset(rows: Sequence[dict], limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit % len(LABELS):
        raise ValueError(
            f"Balanced limit must be divisible by {len(LABELS)}; received {limit}"
        )

    per_label = limit // len(LABELS)
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for row in rows:
        label = row["label"]
        if label not in LABELS or counts[label] >= per_label:
            continue
        selected.append(row)
        counts[label] += 1
        if all(counts[label_name] == per_label for label_name in LABELS):
            break

    missing = {
        label: per_label - counts[label]
        for label in LABELS
        if counts[label] < per_label
    }
    if missing:
        raise ValueError(
            f"Input does not contain enough rows for balanced limit {limit}: {missing}"
        )
    return selected
