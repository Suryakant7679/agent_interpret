#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INPUT="${INPUT:-data/lexical_control/test.jsonl}"
TRAIN_ACTIVATIONS="${TRAIN_ACTIVATIONS:-outputs/activations/qwen2_5_7b_train_256.npz}"
LIMIT="${LIMIT:-256}"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing lexical-control dataset: $INPUT" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_ACTIVATIONS" ]]; then
  echo "Missing training activation archive: $TRAIN_ACTIVATIONS" >&2
  exit 1
fi

python scripts/02_run_behavior_eval.py \
  --backend transformers \
  --model "$MODEL" \
  --input "$INPUT" \
  --output outputs/behavior/qwen2_5_7b_lexical_control.jsonl \
  --load-in-4bit

python scripts/03_extract_activations.py \
  --model "$MODEL" \
  --input "$INPUT" \
  --output outputs/activations/qwen2_5_7b_lexical_control_256.npz \
  --limit "$LIMIT" \
  --load-in-4bit

python scripts/04_train_linear_probes.py \
  --input "$TRAIN_ACTIVATIONS" \
  --test-input outputs/activations/qwen2_5_7b_lexical_control_256.npz \
  --output-dir outputs/probes/qwen2_5_7b_lexical_control

python scripts/04_train_linear_probes.py \
  --input "$TRAIN_ACTIVATIONS" \
  --test-input outputs/activations/qwen2_5_7b_lexical_control_256.npz \
  --activation mlp \
  --output-dir outputs/probes/qwen2_5_7b_lexical_control_mlp

python scripts/23_summarize_lexical_control.py \
  --behavior outputs/behavior/qwen2_5_7b_lexical_control.metrics.json \
  --residual outputs/probes/qwen2_5_7b_lexical_control/residual_probe_metrics.json \
  --mlp outputs/probes/qwen2_5_7b_lexical_control_mlp/mlp_probe_metrics.json \
  --output outputs/probes/qwen2_5_7b_lexical_control/summary.json

echo "Lexical-control evaluation complete."
echo "Summary: outputs/probes/qwen2_5_7b_lexical_control/summary.json"
