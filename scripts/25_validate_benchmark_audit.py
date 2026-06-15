#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the completed manual audit")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper/benchmark_audit.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/benchmark_audit.summary.json"),
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = {"yes", "no"}
    reviewed = [
        row
        for row in rows
        if row["label_correct"] in valid
        and row["natural_wording"] in valid
        and row["ambiguous"] in valid
        and row["duplicate"] in valid
    ]
    counts = Counter(row["label"] for row in reviewed)
    result = {
        "total_rows": len(rows),
        "reviewed_rows": len(reviewed),
        "reviewed_per_label": dict(sorted(counts.items())),
        "incorrect_labels": sum(row["label_correct"] == "no" for row in reviewed),
        "unnatural_wording": sum(row["natural_wording"] == "no" for row in reviewed),
        "ambiguous": sum(row["ambiguous"] == "yes" for row in reviewed),
        "duplicates": sum(row["duplicate"] == "yes" for row in reviewed),
        "complete": len(reviewed) == 400
        and all(
            counts[label] >= 100
            for label in ("web_search", "calculator", "python", "none")
        ),
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit("Manual benchmark audit is incomplete.")


if __name__ == "__main__":
    main()
