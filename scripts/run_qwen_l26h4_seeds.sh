#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INPUT="${INPUT:-data/full/val.jsonl}"
SAMPLES_PER_LABEL="${SAMPLES_PER_LABEL:-32}"
RANDOM_HEADS="${RANDOM_HEADS:-20}"
SEEDS="${SEEDS:-7 21 84 123}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations/l26h4_seeds}"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing validation dataset: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

for seed in $SEEDS; do
  echo "Validating L26H4 with prompt-sampling seed $seed"
  python scripts/16_validate_head.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTPUT_ROOT/seed_${seed}.json" \
    --layer 26 \
    --head 4 \
    --samples-per-label "$SAMPLES_PER_LABEL" \
    --random-heads "$RANDOM_HEADS" \
    --seed "$seed" \
    --shuffle-prompts \
    --load-in-4bit
done

python scripts/20_summarize_l26h4_seeds.py \
  --input-root "$OUTPUT_ROOT" \
  --seeds $SEEDS \
  --output "$OUTPUT_ROOT/summary.json"

echo "L26H4 multi-seed validation complete: $OUTPUT_ROOT/summary.json"
