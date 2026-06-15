#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PREFIX="${PREFIX:-qwen2_5_coder_7b}"
ACTIVATION_LIMIT="${ACTIVATION_LIMIT:-256}"
PATCH_PAIRS="${PATCH_PAIRS:-16}"
SEED="${SEED:-42}"

TRAIN="data/full/train.jsonl"
TEST="data/full/test.jsonl"
OOD="data/full/ood_test.jsonl"
LEXICAL="data/lexical_control/test.jsonl"

for path in "$TRAIN" "$TEST" "$OOD" "$LEXICAL"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required dataset: $path" >&2
    exit 1
  fi
done

mkdir -p \
  outputs/behavior \
  outputs/activations \
  "outputs/probes/${PREFIX}_lexical_control" \
  "outputs/probes/${PREFIX}_lexical_control_mlp" \
  "outputs/directions/${PREFIX}_train_256" \
  "outputs/patching/${PREFIX}_controlled" \
  outputs/logs

run_behavior() {
  local input="$1"
  local name="$2"
  python scripts/02_run_behavior_eval.py \
    --backend transformers \
    --model "$MODEL" \
    --input "$input" \
    --output "outputs/behavior/${PREFIX}_${name}.jsonl" \
    --load-in-4bit
}

echo "Stage 1/5: behavior evaluation"
run_behavior "$TEST" test
run_behavior "$OOD" ood
run_behavior "$LEXICAL" lexical_control

echo "Stage 2/5: activation extraction"
python scripts/03_extract_activations.py \
  --model "$MODEL" \
  --input "$TRAIN" \
  --output "outputs/activations/${PREFIX}_train_256.npz" \
  --limit "$ACTIVATION_LIMIT" \
  --load-in-4bit

python scripts/03_extract_activations.py \
  --model "$MODEL" \
  --input "$LEXICAL" \
  --output "outputs/activations/${PREFIX}_lexical_control_256.npz" \
  --limit "$ACTIVATION_LIMIT" \
  --load-in-4bit

echo "Stage 3/5: held-out probes and directions"
python scripts/04_train_linear_probes.py \
  --input "outputs/activations/${PREFIX}_train_256.npz" \
  --test-input "outputs/activations/${PREFIX}_lexical_control_256.npz" \
  --output-dir "outputs/probes/${PREFIX}_lexical_control"

python scripts/04_train_linear_probes.py \
  --input "outputs/activations/${PREFIX}_train_256.npz" \
  --test-input "outputs/activations/${PREFIX}_lexical_control_256.npz" \
  --activation mlp \
  --output-dir "outputs/probes/${PREFIX}_lexical_control_mlp"

python scripts/05_compute_tool_directions.py \
  --input "outputs/activations/${PREFIX}_train_256.npz" \
  --output-dir "outputs/directions/${PREFIX}_train_256"

echo "Stage 4/5: controlled residual patching"
run_patching() {
  local source_label="$1"
  local output="$2"
  python scripts/07_activation_patching.py \
    --model "$MODEL" \
    --source-input "$TRAIN" \
    --target-input "outputs/behavior/${PREFIX}_ood.jsonl" \
    --output "$output" \
    --positive-label calculator \
    --negative-label python \
    --source-label "$source_label" \
    --pairs "$PATCH_PAIRS" \
    --seed "$SEED" \
    --load-in-4bit
}

run_patching calculator "outputs/patching/${PREFIX}_calculator_source.json"
run_patching python "outputs/patching/${PREFIX}_python_control.json"
run_patching none "outputs/patching/${PREFIX}_none_control.json"

python scripts/12_compare_patching_controls.py \
  --treatment "outputs/patching/${PREFIX}_calculator_source.json" \
  --controls "outputs/patching/${PREFIX}_python_control.json" \
             "outputs/patching/${PREFIX}_none_control.json" \
  --output-dir "outputs/patching/${PREFIX}_controlled" \
  --seed "$SEED"

echo "Stage 5/5: final summary"
python scripts/27_summarize_cross_model.py \
  --model "$MODEL" \
  --test-behavior "outputs/behavior/${PREFIX}_test.metrics.json" \
  --ood-behavior "outputs/behavior/${PREFIX}_ood.metrics.json" \
  --lexical-behavior "outputs/behavior/${PREFIX}_lexical_control.metrics.json" \
  --residual-probe \
    "outputs/probes/${PREFIX}_lexical_control/residual_probe_metrics.json" \
  --mlp-probe \
    "outputs/probes/${PREFIX}_lexical_control_mlp/mlp_probe_metrics.json" \
  --patching-comparison \
    "outputs/patching/${PREFIX}_controlled/patching_control_comparison.csv" \
  --output "outputs/${PREFIX}_summary.json"

echo "Qwen2.5-Coder-7B reduced pipeline complete."
echo "Summary: outputs/${PREFIX}_summary.json"
