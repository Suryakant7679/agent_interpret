#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export PREFIX="mistral_7b_instruct_v0_3"

bash scripts/run_reduced_7b_components.sh
