#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"

python scripts/16_validate_head.py \
  --model "$MODEL" \
  --input data/full/val.jsonl \
  --output outputs/ablations/l26h4_validation.json \
  --layer 26 \
  --head 4 \
  --samples-per-label 32 \
  --random-heads 20 \
  --seed 42 \
  --load-in-4bit

python scripts/16_validate_head.py \
  --model "$MODEL" \
  --input data/full/challenge_test.jsonl \
  --output outputs/ablations/l26h4_challenge.json \
  --layer 26 \
  --head 4 \
  --samples-per-label 32 \
  --random-heads 20 \
  --seed 42 \
  --load-in-4bit

echo "L26H4 confirmation complete."
