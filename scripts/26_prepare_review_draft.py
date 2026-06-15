#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


FIELDS = [
    "id",
    "pair_id",
    "label",
    "query",
    "label_correct",
    "natural_wording",
    "ambiguous",
    "duplicate",
    "notes",
    "reviewer",
    "review_method",
    "human_verified",
]

LABEL_RULES = {
    "web_search": (
        "official",
        "stable release",
        "source location",
    ),
    "calculator": (
        "numerical value",
        "rounded to eight decimal places",
        "intermediate quantities",
    ),
    "python": (
        "observations [",
        "sample standard deviation",
        "interquartile range",
    ),
    "none": (
        "describe how",
        "practical example",
        "implications",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a pre-filled worksheet for later human verification"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/lexical_control/manual_review.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/benchmark_audit.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/benchmark_audit.summary.json"),
    )
    args = parser.parse_args()

    with args.source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    query_counts = Counter(row["query"] for row in source_rows)

    rows = []
    for source in source_rows:
        query = source["query"]
        lowered = query.lower()
        expected_phrases = LABEL_RULES[source["label"]]
        label_correct = all(phrase in lowered for phrase in expected_phrases)
        grammatical = (
            query.startswith("Prepare a concise response")
            and query.endswith("Include the requested supporting details.")
            and not re.search(r"\s{2,}", query)
        )
        duplicate = query_counts[query] > 1

        notes = []
        if grammatical:
            notes.append(
                "Grammatical synthetic benchmark wording; shared case context "
                "is mildly artificial."
            )
        else:
            notes.append("Wording failed the structural naturalness check.")
        if not label_correct:
            notes.append("Expected task cues for the assigned label are missing.")

        rows.append(
            {
                "id": source["id"],
                "pair_id": source["pair_id"],
                "label": source["label"],
                "query": query,
                "label_correct": "yes" if label_correct else "no",
                "natural_wording": "yes" if grammatical else "no",
                "ambiguous": "no" if label_correct else "yes",
                "duplicate": "yes" if duplicate else "no",
                "notes": " ".join(notes),
                "reviewer": "",
                "review_method": "prefilled_draft",
                "human_verified": "no",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in rows)
    summary = {
        "total_rows": len(rows),
        "reviewed_rows": 0,
        "reviewed_per_label": {},
        "draft_rows_per_label": dict(sorted(counts.items())),
        "incorrect_labels": sum(row["label_correct"] == "no" for row in rows),
        "unnatural_wording": sum(row["natural_wording"] == "no" for row in rows),
        "ambiguous": sum(row["ambiguous"] == "yes" for row in rows),
        "duplicates": sum(row["duplicate"] == "yes" for row in rows),
        "reviewer": "",
        "review_method": "prefilled_draft",
        "draft_rows": len(rows),
        "audit_complete": False,
        "human_verified": 0,
        "human_audit_complete": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
