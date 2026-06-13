#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
RANKINGS="outputs/neuron_rankings/qwen2_5_7b_components_256.json"

for sign in positive negative; do
  python scripts/11_component_ablation.py \
    --model "$MODEL" \
    --rankings "$RANKINGS" \
    --input data/full/test.jsonl \
    --output "outputs/ablations/heads_${sign}_k10_calculator.json" \
    --component heads \
    --tool calculator \
    --contrast python \
    --evaluation-label calculator \
    --effect-sign "$sign" \
    --top-k 10 \
    --limit 32 \
    --random-trials 10 \
    --seed 42 \
    --load-in-4bit
done

echo "Head sign validation complete."
