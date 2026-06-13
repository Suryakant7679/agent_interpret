#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool_circuits.evaluation import OllamaBackend, OracleBackend, TransformersBackend
from tool_circuits.io import read_jsonl, write_jsonl
from tool_circuits.metrics import classification_metrics
from tool_circuits.prompting import format_tool_prompt, normalize_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool selection behavior")
    parser.add_argument("--backend", choices=["ollama", "transformers", "oracle"], required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-cli", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    if args.backend == "ollama":
        backend = OllamaBackend(args.model, args.ollama_host, use_cli=args.ollama_cli)
    elif args.backend == "transformers":
        backend = TransformersBackend(args.model, load_in_4bit=args.load_in_4bit)
    else:
        backend = OracleBackend()

    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    results = []
    predictions: list[str | None] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(backend, OracleBackend):
            backend.current_label = row["label"]
        raw = backend.generate(format_tool_prompt(row["query"]))
        prediction = normalize_label(raw)
        predictions.append(prediction)
        results.append(
            {
                **row,
                "model": args.model,
                "backend": args.backend,
                "raw_output": raw,
                "prediction": prediction,
                "correct": prediction == row["label"],
            }
        )
        print(f"\rEvaluated {index}/{len(rows)}", end="", flush=True)
    print()

    write_jsonl(args.output, results)
    metrics = classification_metrics([row["label"] for row in rows], predictions)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
