# Large-GPU Quickstart

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

## Progress tracking

- [`PENDING_TASKS.md`](PENDING_TASKS.md): project checklist
- [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md): evidence and limitations
- [`EXPERIMENTS.md`](EXPERIMENTS.md): exact experiment commands
