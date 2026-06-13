#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

python scripts/15_screen_heads.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --rankings outputs/neuron_rankings/qwen2_5_7b_components_256.json \
  --input data/full/test.jsonl \
  --output outputs/ablations/calculator_head_screen.json \
  --tool calculator \
  --candidates 20 \
  --samples-per-label 8 \
  --load-in-4bit
