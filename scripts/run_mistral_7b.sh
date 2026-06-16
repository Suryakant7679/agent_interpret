#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export PREFIX="mistral_7b_instruct_v0_3"
export ALLOW_PATCHING_SKIP=1

bash scripts/run_reduced_7b_pipeline.sh
