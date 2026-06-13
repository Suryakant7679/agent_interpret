#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-7B-Instruct"

python scripts/14_generate_challenge_set.py --groups 100 --seed 2026

python scripts/02_run_behavior_eval.py \
  --backend transformers \
  --model "$MODEL" \
  --input data/full/challenge_test.jsonl \
  --output outputs/behavior/qwen2_5_7b_challenge.jsonl \
  --load-in-4bit

python scripts/03_extract_activations.py \
  --model "$MODEL" \
  --input data/full/challenge_test.jsonl \
  --output outputs/activations/qwen2_5_7b_challenge_256.npz \
  --limit 256 \
  --load-in-4bit

python scripts/04_train_linear_probes.py \
  --input outputs/activations/qwen2_5_7b_train_256.npz \
  --test-input outputs/activations/qwen2_5_7b_challenge_256.npz \
  --output-dir outputs/probes/qwen2_5_7b_challenge

python scripts/04_train_linear_probes.py \
  --input outputs/activations/qwen2_5_7b_train_256.npz \
  --test-input outputs/activations/qwen2_5_7b_challenge_256.npz \
  --activation mlp \
  --output-dir outputs/probes/qwen2_5_7b_challenge_mlp

echo "Stage 2 complete."
echo "Behavior: outputs/behavior/qwen2_5_7b_challenge.metrics.json"
echo "Residual probe: outputs/probes/qwen2_5_7b_challenge/residual_probe_metrics.json"
echo "MLP probe: outputs/probes/qwen2_5_7b_challenge_mlp/mlp_probe_metrics.json"
