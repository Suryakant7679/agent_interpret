# Large-GPU Quickstart

This document is the handoff for tasks marked **LARGE GPU** in
[`PENDING_TASKS.md`](PENDING_TASKS.md). The current RTX 3060 12 GB machine
should continue only with tasks marked **LOCAL**.

## Clone and create the environment

```bash
git clone https://github.com/Suryakant7679/agent_interpret.git
cd agent_interpret

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[research]'
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Configure caches

Choose a disk with enough space:

```bash
export HF_HOME=/path/to/large-disk/hf_cache/tool_circuits
export TORCH_HOME=/path/to/large-disk/torch_cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME"
```

Hugging Face authentication is optional for public models but avoids anonymous
rate limits:

```bash
hf auth login
```

Never commit an access token.

## Recreate datasets

Generated datasets and experiment outputs are intentionally excluded from Git.

```bash
python scripts/01_generate_dataset.py \
  --profile full \
  --seed 42 \
  --output-dir data/full

python scripts/14_generate_challenge_set.py \
  --groups 100 \
  --seed 2026
```

## Reproduce Qwen2.5-7B

Run the stages in order:

```bash
# Behavioral evaluation and activation extraction:
# Follow README.md and EXPERIMENTS.md first.

bash scripts/run_qwen_tier1_stage1.sh
bash scripts/run_qwen_tier1_stage2.sh
bash scripts/run_qwen_tier1_stage3.sh
bash scripts/run_qwen_head_sign_validation.sh
bash scripts/run_qwen_head_specificity.sh
bash scripts/run_qwen_head_screen.sh
bash scripts/run_qwen_l26h4_confirmation.sh
bash scripts/run_qwen_steering_controls.sh
bash scripts/run_qwen_steering_highdose.sh
bash scripts/run_qwen_l26h4_attention.sh
bash scripts/run_qwen_l26h4_order.sh
```

Some later stages depend on outputs from earlier stages. The authoritative
sequence and expected files are documented in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Deferred large-GPU work

Run these only on the larger machine:

1. Full-precision L26H4 validation and calculator-direction steering.
2. Qwen2.5-14B behavioral evaluation and reduced causal pipeline.
3. Cross-model component comparison after the 14B outputs exist.

Recommended capacity:

- 24 GB VRAM: minimum target for full-precision 7B checks and quantized 14B.
- 48 GB or more: preferred for 14B activation extraction, patching, and
  ablation without aggressive offload.

The RTX 3060 remains suitable for Qwen2.5-Coder-7B and Mistral-7B-Instruct in
4-bit, plus all statistics, figures, and paper work.

## Progress tracking

- [`PENDING_TASKS.md`](PENDING_TASKS.md): project checklist
- [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md): evidence and limitations
- [`EXPERIMENTS.md`](EXPERIMENTS.md): exact experiment commands
