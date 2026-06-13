#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
RANKINGS="outputs/neuron_rankings/qwen2_5_7b_components_256.json"

run_ablation() {
  local sign="$1"
  local label="$2"
  local output="$3"
  python scripts/11_component_ablation.py \
    --model "$MODEL" \
    --rankings "$RANKINGS" \
    --input data/full/test.jsonl \
    --output "$output" \
    --component heads \
    --tool calculator \
    --contrast python \
    --evaluation-label "$label" \
    --effect-sign "$sign" \
    --top-k 10 \
    --limit 32 \
    --random-trials 10 \
    --seed 42 \
    --load-in-4bit
}

run_ablation positive calculator outputs/ablations/heads_positive_k10_calculator.json
run_ablation negative calculator outputs/ablations/heads_negative_k10_calculator.json
run_ablation positive python outputs/ablations/heads_positive_k10_python.json
run_ablation positive web_search outputs/ablations/heads_positive_k10_web.json
run_ablation positive none outputs/ablations/heads_positive_k10_none.json

echo "Head validation complete: outputs/ablations/heads_*_k10_*.json"
