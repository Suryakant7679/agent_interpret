#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"
COMPONENTS="outputs/activations/qwen2_5_7b_components_256.npz"
RANKINGS="outputs/neuron_rankings/qwen2_5_7b_components_256.json"

python scripts/08_extract_components.py \
  --model "$MODEL" \
  --input data/full/train.jsonl \
  --output "$COMPONENTS" \
  --limit 256 \
  --load-in-4bit

python scripts/09_rank_components.py \
  --input "$COMPONENTS" \
  --output "$RANKINGS"

python scripts/11_component_ablation.py \
  --model "$MODEL" \
  --rankings "$RANKINGS" \
  --input data/full/test.jsonl \
  --output outputs/ablations/calculator_neurons_k20.json \
  --component neurons \
  --tool calculator \
  --contrast python \
  --top-k 20 \
  --limit 32 \
  --seed 42 \
  --load-in-4bit

python scripts/11_component_ablation.py \
  --model "$MODEL" \
  --rankings "$RANKINGS" \
  --input data/full/test.jsonl \
  --output outputs/ablations/calculator_heads_k20.json \
  --component heads \
  --tool calculator \
  --contrast python \
  --top-k 20 \
  --limit 32 \
  --seed 42 \
  --load-in-4bit

echo "Stage 3 complete: outputs/ablations/"
