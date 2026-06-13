#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool_circuits.dataset import PROFILES, build_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ToolUseCircuitBench")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "processed"
    )
    args = parser.parse_args()
    counts = build_benchmark(args.output_dir, args.profile, args.seed)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
