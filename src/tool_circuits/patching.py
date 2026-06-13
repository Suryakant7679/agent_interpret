from __future__ import annotations

import random
from collections.abc import Iterable


def select_patching_rows(
    source_rows: Iterable[dict],
    target_rows: Iterable[dict],
    source_label: str,
    positive_label: str,
    negative_label: str,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    sources = [row for row in source_rows if row["label"] == source_label]
    targets = [
        row
        for row in target_rows
        if row["label"] == positive_label
        and row.get("prediction", negative_label) == negative_label
    ]
    rng = random.Random(seed)
    rng.shuffle(sources)
    rng.shuffle(targets)
    return sources, targets
