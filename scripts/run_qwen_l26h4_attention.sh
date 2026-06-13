#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

python scripts/17_analyze_head_attention.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/full/val.jsonl \
  --output outputs/ablations/l26h4_attention.json \
  --layer 26 \
  --head 4 \
  --samples-per-label 16 \
  --top-tokens 20 \
  --load-in-4bit
