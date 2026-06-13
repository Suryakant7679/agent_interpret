#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
DIRECTIONS="outputs/directions/qwen2_5_7b_train_256/residual_tool_directions.npy"
INPUT="outputs/behavior/qwen2_5_7b_ood.jsonl"

run_steering() {
  local direction_tool="$1"
  local scale="$2"
  local name="$3"
  python scripts/10_activation_steering.py \
    --model "$MODEL" \
    --directions "$DIRECTIONS" \
    --input "$INPUT" \
    --output "outputs/patching/steering_${name}_layer26.json" \
    --tool calculator \
    --contrast python \
    --direction-tool "$direction_tool" \
    --direction-scale "$scale" \
    --layer 26 \
    --alphas 0.5 1 2 4 8 16 \
    --limit 32 \
    --load-in-4bit
}

run_steering calculator 1 calculator
run_steering calculator -1 negative_calculator
run_steering python 1 python_control

echo "Steering controls complete."
