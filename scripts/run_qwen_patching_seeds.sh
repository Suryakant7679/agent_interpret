#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SOURCE="${SOURCE:-data/full/train.jsonl}"
TARGET="${TARGET:-outputs/behavior/qwen2_5_7b_ood.jsonl}"
PAIRS="${PAIRS:-32}"
SEEDS="${SEEDS:-7 21 84 123}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/patching/seeds}"

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing source dataset: $SOURCE" >&2
  exit 1
fi

if [[ ! -f "$TARGET" ]]; then
  echo "Missing target behavior file: $TARGET" >&2
  exit 1
fi

run_condition() {
  local seed="$1"
  local source_label="$2"
  local output="$3"

  python scripts/07_activation_patching.py \
    --model "$MODEL" \
    --source-input "$SOURCE" \
    --target-input "$TARGET" \
    --output "$output" \
    --positive-label calculator \
    --negative-label python \
    --source-label "$source_label" \
    --pairs "$PAIRS" \
    --seed "$seed" \
    --load-in-4bit
}

for seed in $SEEDS; do
  seed_dir="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$seed_dir"

  echo "Running controlled patching for seed $seed"
  run_condition "$seed" calculator "$seed_dir/calculator_source_32.json"
  run_condition "$seed" python "$seed_dir/python_control_32.json"
  run_condition "$seed" none "$seed_dir/none_control_32.json"

  python scripts/12_compare_patching_controls.py \
    --treatment "$seed_dir/calculator_source_32.json" \
    --controls "$seed_dir/python_control_32.json" \
               "$seed_dir/none_control_32.json" \
    --output-dir "$seed_dir/controlled_comparison" \
    --seed "$seed"
done

python scripts/19_summarize_patching_seeds.py \
  --input-root "$OUTPUT_ROOT" \
  --seeds $SEEDS \
  --output "$OUTPUT_ROOT/summary.json"

echo "Multi-seed patching complete: $OUTPUT_ROOT/summary.json"
