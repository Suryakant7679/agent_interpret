#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

: "${MODEL:?Set MODEL to the Hugging Face model id}"
: "${PREFIX:?Set PREFIX to the output filename prefix}"

LIMIT="${COMPONENT_LIMIT:-256}"
ABLATION_LIMIT="${ABLATION_LIMIT:-32}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"

COMPONENTS="outputs/activations/${PREFIX}_components_256.npz"
RANKINGS="outputs/neuron_rankings/${PREFIX}_components_256.json"
NEURONS="outputs/ablations/${PREFIX}_calculator_neurons_k${TOP_K}.json"
HEADS="outputs/ablations/${PREFIX}_calculator_heads_k${TOP_K}.json"
SUMMARY="outputs/${PREFIX}_components_summary.json"

python scripts/08_extract_components.py \
  --model "$MODEL" \
  --input data/full/train.jsonl \
  --output "$COMPONENTS" \
  --limit "$LIMIT" \
  --load-in-4bit

python scripts/09_rank_components.py \
  --input "$COMPONENTS" \
  --output "$RANKINGS"

python scripts/11_component_ablation.py \
  --model "$MODEL" \
  --rankings "$RANKINGS" \
  --input data/full/test.jsonl \
  --output "$NEURONS" \
  --component neurons \
  --tool calculator \
  --contrast python \
  --top-k "$TOP_K" \
  --limit "$ABLATION_LIMIT" \
  --seed "$SEED" \
  --load-in-4bit

python scripts/11_component_ablation.py \
  --model "$MODEL" \
  --rankings "$RANKINGS" \
  --input data/full/test.jsonl \
  --output "$HEADS" \
  --component heads \
  --tool calculator \
  --contrast python \
  --top-k "$TOP_K" \
  --limit "$ABLATION_LIMIT" \
  --seed "$SEED" \
  --load-in-4bit

python scripts/29_summarize_component_replication.py \
  --model "$MODEL" \
  --neurons "$NEURONS" \
  --heads "$HEADS" \
  --output "$SUMMARY"

echo "Component replication complete for $MODEL."
echo "Summary: $SUMMARY"
