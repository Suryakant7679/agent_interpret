#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/28_make_local_report.py

echo "Local report complete."
echo "Figures: paper/figures/"
echo "Tables: paper/tables/"
echo "Summary: paper/local_report.summary.json"
