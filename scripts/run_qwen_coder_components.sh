#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
export PREFIX="qwen2_5_coder_7b"

bash scripts/run_reduced_7b_components.sh
