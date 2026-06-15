#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/mnt/DATADRIVE0/hf_cache/tool_circuits}"
export TORCH_HOME="${TORCH_HOME:-/mnt/DATADRIVE0/torch_cache}"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tool-circuits}"

mkdir -p outputs/logs "$MPLCONFIGDIR"

run_if_missing() {
  local summary="$1"
  local runner="$2"
  local log="$3"
  local name="$4"

  if [[ -f "$summary" ]]; then
    echo "Skipping $name: found $summary"
  elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Would run $name via $runner"
  else
    echo "Running $name"
    bash "$runner" 2>&1 | tee "$log"
  fi
}

run_if_missing \
  outputs/qwen2_5_coder_7b_summary.json \
  scripts/run_qwen_coder_7b.sh \
  outputs/logs/qwen2_5_coder_7b.log \
  "Qwen2.5-Coder-7B"

run_if_missing \
  outputs/qwen2_5_coder_7b_components_summary.json \
  scripts/run_qwen_coder_components.sh \
  outputs/logs/qwen2_5_coder_7b_components.log \
  "Qwen2.5-Coder-7B component ablations"

run_if_missing \
  outputs/mistral_7b_instruct_v0_3_summary.json \
  scripts/run_mistral_7b.sh \
  outputs/logs/mistral_7b_instruct_v0_3.log \
  "Mistral-7B-Instruct-v0.3"

run_if_missing \
  outputs/mistral_7b_instruct_v0_3_components_summary.json \
  scripts/run_mistral_components.sh \
  outputs/logs/mistral_7b_instruct_v0_3_components.log \
  "Mistral-7B-Instruct-v0.3 component ablations"

bash scripts/run_local_report.sh

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete; no missing model stage was executed."
else
  echo "All RTX 3060 model runs and local figures are complete."
fi
