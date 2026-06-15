#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_GROUPS="${NUM_GROUPS:-100}"
SEED="${SEED:-2026}"
OUTPUT_DIR="${OUTPUT_DIR:-data/lexical_control}"

python scripts/21_generate_lexical_control.py \
  --groups "$NUM_GROUPS" \
  --seed "$SEED" \
  --output "$OUTPUT_DIR/test.jsonl"

python scripts/22_audit_lexical_control.py \
  --input "$OUTPUT_DIR/test.jsonl" \
  --report "$OUTPUT_DIR/audit.json" \
  --review-csv "$OUTPUT_DIR/manual_review.csv"

echo "Lexical-control benchmark built: $OUTPUT_DIR/test.jsonl"
echo "Complete the review columns in: $OUTPUT_DIR/manual_review.csv"
