#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
VALID = {"yes", "no"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def initialize(source: Path) -> list[dict[str, str]]:
    source_rows = load_rows(source)
    query_counts = Counter(row["query"] for row in source_rows)
    return [
        {
            "id": row["id"],
            "pair_id": row["pair_id"],
            "label": row["label"],
            "query": row["query"],
            "label_correct": "",
            "natural_wording": "",
            "ambiguous": "",
            "duplicate": "yes" if query_counts[row["query"]] > 1 else "no",
            "notes": "",
            "reviewer": "",
            "review_method": "",
            "human_verified": "no",
        }
        for row in source_rows
    ]


def ask(prompt: str) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value in {"y", "yes"}:
            return "yes"
        if value in {"n", "no"}:
            return "no"
        if value in {"q", "quit"}:
            raise KeyboardInterrupt
        print("Enter y, n, or q.")


def summary(rows: list[dict[str, str]]) -> dict:
    reviewed = [
        row
        for row in rows
        if row.get("human_verified") == "yes"
        and row["label_correct"] in VALID
        and row["natural_wording"] in VALID
        and row["ambiguous"] in VALID
    ]
    per_label = Counter(row["label"] for row in reviewed)
    issues = {
        "incorrect_labels": sum(row["label_correct"] == "no" for row in reviewed),
        "unnatural_wording": sum(row["natural_wording"] == "no" for row in reviewed),
        "ambiguous": sum(row["ambiguous"] == "yes" for row in reviewed),
        "duplicates": sum(row["duplicate"] == "yes" for row in rows),
    }
    result = {
        "total_rows": len(rows),
        "reviewed_rows": len(reviewed),
        "reviewed_per_label": dict(sorted(per_label.items())),
        "issues": issues,
        "draft_rows": sum(
            row.get("review_method") == "prefilled_draft" for row in rows
        ),
        "human_verified": sum(
            row.get("human_verified") == "yes" for row in reviewed
        ),
        "audit_complete": len(reviewed) == len(rows) and all(
            per_label[label] >= 100
            for label in ("web_search", "calculator", "python", "none")
        ),
    }
    result["human_audit_complete"] = (
        result["audit_complete"] and result["human_verified"] == len(rows)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable manual review for the lexical-control benchmark"
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
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Validate current progress without starting the interactive review.",
    )
    args = parser.parse_args()

    rows = load_rows(args.output) if args.output.exists() else initialize(args.source)
    if args.status_only:
        result = summary(rows)
        args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    try:
        for index, row in enumerate(rows):
            if row.get("human_verified") == "yes":
                continue
            print("\n" + "=" * 78)
            print(f"Prompt {index + 1}/{len(rows)} | label: {row['label']}")
            print(row["query"])
            print(f"Automatic duplicate check: {row['duplicate']}")
            row["label_correct"] = ask("Is the assigned tool label correct? [y/n/q] ")
            row["natural_wording"] = ask("Is the wording natural enough? [y/n/q] ")
            row["ambiguous"] = ask("Is the tool choice ambiguous? [y/n/q] ")
            row["notes"] = input("Notes (optional): ").strip()
            row["reviewer"] = input("Reviewer name (optional): ").strip() or "human"
            row["review_method"] = "manual"
            row["human_verified"] = "yes"
            save_rows(args.output, rows)
            result = summary(rows)
            args.summary.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved. Reviewed {result['reviewed_rows']}/{len(rows)}.")
    except (KeyboardInterrupt, EOFError):
        save_rows(args.output, rows)
        result = summary(rows)
        args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nReview paused at {result['reviewed_rows']}/{len(rows)}.")
        return

    result = summary(rows)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
