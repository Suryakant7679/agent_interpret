#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool_circuits import LABELS
from tool_circuits.io import read_jsonl


BANNED_PHRASES = ("today", "exact result", "each value")
REQUIRED_PREFIX = "Prepare a concise response for the following request: "
REQUIRED_SUFFIX = " Include the requested supporting details."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit balance and framing of the lexical-control benchmark"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    counts = Counter(row["label"] for row in rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)

    queries = [row["query"] for row in rows]
    banned_hits = {
        phrase: [row["id"] for row in rows if phrase in row["query"].lower()]
        for phrase in BANNED_PHRASES
    }
    malformed_groups = {
        pair_id: sorted(row["label"] for row in group)
        for pair_id, group in groups.items()
        if len(group) != len(LABELS)
        or {row["label"] for row in group} != set(LABELS)
    }
    framing_failures = [
        row["id"]
        for row in rows
        if not row["query"].startswith(REQUIRED_PREFIX)
        or not row["query"].endswith(REQUIRED_SUFFIX)
    ]

    word_counts: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        word_counts[row["label"]].append(
            len(re.findall(r"[A-Za-z0-9']+", row["query"]))
        )
    length_summary = {
        label: {
            "mean": statistics.mean(word_counts[label]),
            "min": min(word_counts[label]),
            "max": max(word_counts[label]),
        }
        for label in LABELS
    }

    passed = (
        len(rows) > 0
        and set(counts) == set(LABELS)
        and len(set(counts.values())) == 1
        and len(set(queries)) == len(queries)
        and not any(banned_hits.values())
        and not malformed_groups
        and not framing_failures
    )
    report = {
        "samples": len(rows),
        "groups": len(groups),
        "label_counts": dict(sorted(counts.items())),
        "unique_queries": len(set(queries)),
        "banned_phrase_hits": banned_hits,
        "malformed_groups": malformed_groups,
        "framing_failures": framing_failures,
        "word_count_by_label": length_summary,
        "automatic_checks_passed": passed,
        "manual_review_required": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "pair_id",
                "label",
                "query",
                "label_correct",
                "natural_wording",
                "ambiguous",
                "notes",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["pair_id"], item["label"])):
            writer.writerow(
                {
                    "id": row["id"],
                    "pair_id": row["pair_id"],
                    "label": row["label"],
                    "query": row["query"],
                    "label_correct": "",
                    "natural_wording": "",
                    "ambiguous": "",
                    "notes": "",
                }
            )

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Lexical-control automatic audit failed.")


if __name__ == "__main__":
    main()
