# Tool Circuit Interpretability

This repository implements the first runnable stages of ToolUseCircuitBench:

1. Build a controlled benchmark for `web_search`, `calculator`, `python`, and
   `none`.
2. Evaluate tool-selection behavior with Ollama or Hugging Face models.
3. Extract residual-stream and MLP activations from Hugging Face causal LMs.
4. Train layer-wise linear probes and compute tool directions.

The causal experiments from the research plan (activation patching, neuron
ablation, and head ablation) should be run after selecting a model that has
strong baseline tool-selection accuracy.

The complete staged execution plan is in [EXPERIMENTS.md](EXPERIMENTS.md).
Current progress and remaining checkpoints are tracked in
[PENDING_TASKS.md](PENDING_TASKS.md).
Instructions for cloning onto a larger GPU machine are in
[GPU_QUICKSTART.md](GPU_QUICKSTART.md).

## Quick start with local Ollama Qwen

The machine currently has `qwen3.5:9b` in
`/mnt/DATADRIVE0/ollama-models`. Start Ollama with that model directory:

```bash
export OLLAMA_MODELS=/mnt/DATADRIVE0/ollama-models
ollama serve
```

In another terminal:

```bash
cd /home/surya/AGENTS/tool_circuit_interpretability
python3 scripts/01_generate_dataset.py --profile smoke
python3 scripts/02_run_behavior_eval.py \
  --backend ollama \
  --model qwen3.5:9b \
  --input data/processed/test.jsonl \
  --output outputs/behavior/qwen3_5_9b_test.jsonl
```

The `9b-docs` tag adds a document-oriented system prompt, so use the base
`qwen3.5:9b` tag for controlled experiments.

## Full benchmark

```bash
python3 scripts/01_generate_dataset.py --profile full --seed 42
```

The full profile creates:

| Split | Samples |
|---|---:|
| train | 4,000 |
| validation | 1,000 |
| test | 1,000 |
| OOD test | 1,000 |
| adversarial test | 500 |

Each standard split is balanced across the four labels. Prompt generation is
deterministic for a fixed seed, and duplicate queries are rejected.

## Hugging Face experiments

Install the optional research dependencies in a CUDA-enabled environment:

```bash
python3 -m pip install -e '.[research]'
```

Then run:

```bash
python3 scripts/02_run_behavior_eval.py \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/processed/test.jsonl \
  --output outputs/behavior/qwen2_5_7b_test.jsonl \
  --load-in-4bit

python3 scripts/03_extract_activations.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/processed/train.jsonl \
  --output outputs/activations/qwen2_5_7b_train.npz \
  --load-in-4bit

python3 scripts/04_train_linear_probes.py \
  --input outputs/activations/qwen2_5_7b_train.npz \
  --output-dir outputs/probes/qwen2_5_7b

python3 scripts/05_compute_tool_directions.py \
  --input outputs/activations/qwen2_5_7b_train.npz \
  --output-dir outputs/directions/qwen2_5_7b

python3 scripts/06_rank_neurons.py \
  --input outputs/activations/qwen2_5_7b_train.npz \
  --output outputs/neuron_rankings/qwen2_5_7b.json

python3 scripts/07_activation_patching.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --source-input data/full/train.jsonl \
  --target-input outputs/behavior/qwen2_5_7b_ood.jsonl \
  --output outputs/patching/qwen2_5_7b_calculator_to_python.json \
  --positive-label calculator \
  --negative-label python \
  --pairs 8 \
  --load-in-4bit
```

Activation extraction records the final prompt token, immediately before the
model predicts the tool label. It caches hidden states and MLP outputs for each
layer. `--limit` selects a balanced subset and must be divisible by four. Start
with a small limit because uncompressed activations are large.
Neuron rankings are explicitly candidate features until an ablation experiment
demonstrates a causal effect.

## Experimental discipline

- Freeze benchmark files before comparing models.
- Report behavioral competence before interpreting activations.
- Fit probes on training examples and report on held-out examples.
- Use same-layer random, bottom-ranked, and frequency-matched controls.
- Do not call a component a circuit based only on probe weights or activation
  selectivity.
- Reserve causal claims for patching, steering, or ablation results.

## Research model order

1. `Qwen/Qwen2.5-7B-Instruct`: full primary analysis.
2. `Qwen/Qwen2.5-14B-Instruct`: same-family scaling analysis.
3. `mistralai/Mistral-7B-Instruct-v0.3`: cross-family comparison.
4. `Qwen/Qwen2.5-Coder-7B-Instruct`: focused Python/code analysis.

The locally available Qwen 3.5 model is useful for benchmark and behavioral
pipeline validation, but it does not replace the paper's primary model.
