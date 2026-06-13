#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
SOURCE="data/full/train.jsonl"
TARGET="outputs/behavior/qwen2_5_7b_ood.jsonl"
PAIRS="${PAIRS:-32}"
SEED="${SEED:-42}"

run_condition() {
  local source_label="$1"
  local output="$2"
  python scripts/07_activation_patching.py \
    --model "$MODEL" \
    --source-input "$SOURCE" \
    --target-input "$TARGET" \
    --output "$output" \
    --positive-label calculator \
    --negative-label python \
    --source-label "$source_label" \
    --pairs "$PAIRS" \
    --seed "$SEED" \
    --load-in-4bit
}

run_condition calculator outputs/patching/calculator_source_32.json
run_condition python outputs/patching/python_control_32.json
run_condition none outputs/patching/none_control_32.json

python scripts/12_compare_patching_controls.py \
  --treatment outputs/patching/calculator_source_32.json \
  --controls outputs/patching/python_control_32.json \
             outputs/patching/none_control_32.json \
  --output-dir outputs/patching/controlled_comparison \
  --seed "$SEED"

echo "Stage 1 complete: outputs/patching/controlled_comparison"
