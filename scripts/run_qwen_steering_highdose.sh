#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
DIRECTIONS="outputs/directions/qwen2_5_7b_train_256/residual_tool_directions.npy"
INPUT="outputs/behavior/qwen2_5_7b_ood.jsonl"

python scripts/10_activation_steering.py \
  --model "$MODEL" \
  --directions "$DIRECTIONS" \
  --input "$INPUT" \
  --output outputs/patching/steering_calculator_highdose_layer26.json \
  --tool calculator \
  --contrast python \
  --direction-tool calculator \
  --direction-scale 1 \
  --layer 26 \
  --alphas 24 32 48 64 \
  --limit 32 \
  --load-in-4bit

python scripts/10_activation_steering.py \
  --model "$MODEL" \
  --directions "$DIRECTIONS" \
  --input "$INPUT" \
  --output outputs/patching/steering_negative_calculator_highdose_layer26.json \
  --tool calculator \
  --contrast python \
  --direction-tool calculator \
  --direction-scale -1 \
  --layer 26 \
  --alphas 24 32 48 64 \
  --limit 32 \
  --load-in-4bit

echo "High-dose steering confirmation complete."
