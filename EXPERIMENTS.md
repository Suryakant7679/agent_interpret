# Tier-1 Experiment Sequence

This document is the execution order for the main Qwen 2.5 7B study. Do not
skip controls or describe correlational component rankings as circuits.

## Environment

```bash
cd /home/surya/AGENTS/tool_circuit_interpretability
source .venv/bin/activate
export HF_HOME=/mnt/DATADRIVE0/hf_cache/tool_circuits
export TORCH_HOME=/mnt/DATADRIVE0/torch_cache
export TOKENIZERS_PARALLELISM=false
```

## 1. Controlled residual patching

Use the same seed and pair count for every source condition so target errors
are paired exactly.

Run all Stage 1 conditions with:

```bash
bash scripts/run_qwen_tier1_stage1.sh
```

Equivalent individual commands:

```bash
python scripts/07_activation_patching.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --source-input data/full/train.jsonl \
  --target-input outputs/behavior/qwen2_5_7b_ood.jsonl \
  --output outputs/patching/calculator_source_32.json \
  --positive-label calculator --negative-label python \
  --source-label calculator --pairs 32 --seed 42 --load-in-4bit

python scripts/07_activation_patching.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --source-input data/full/train.jsonl \
  --target-input outputs/behavior/qwen2_5_7b_ood.jsonl \
  --output outputs/patching/python_control_32.json \
  --positive-label calculator --negative-label python \
  --source-label python --pairs 32 --seed 42 --load-in-4bit

python scripts/07_activation_patching.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --source-input data/full/train.jsonl \
  --target-input outputs/behavior/qwen2_5_7b_ood.jsonl \
  --output outputs/patching/none_control_32.json \
  --positive-label calculator --negative-label python \
  --source-label none --pairs 32 --seed 42 --load-in-4bit

python scripts/12_compare_patching_controls.py \
  --treatment outputs/patching/calculator_source_32.json \
  --controls outputs/patching/python_control_32.json \
             outputs/patching/none_control_32.json \
  --output-dir outputs/patching/controlled_comparison
```

Seed `42` is produced by Stage 1. Run the four remaining seeds and create an
automatic checkpoint summary with:

```bash
mkdir -p outputs/patching/seeds
bash scripts/run_qwen_patching_seeds.sh 2>&1 \
  | tee outputs/patching/seeds/run.log
```

The checkpoint passes only when
`outputs/patching/seeds/summary.json` reports `"all_seeds_passed": true`.

## 1b. Counterfactual challenge set

Run all Stage 2 conditions with:

```bash
bash scripts/run_qwen_tier1_stage2.sh
```

Equivalent individual commands:

```bash
python scripts/14_generate_challenge_set.py --groups 100 --seed 2026

python scripts/02_run_behavior_eval.py \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/full/challenge_test.jsonl \
  --output outputs/behavior/qwen2_5_7b_challenge.jsonl \
  --load-in-4bit

python scripts/03_extract_activations.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/full/challenge_test.jsonl \
  --output outputs/activations/qwen2_5_7b_challenge_256.npz \
  --limit 256 --load-in-4bit

python scripts/04_train_linear_probes.py \
  --input outputs/activations/qwen2_5_7b_train_256.npz \
  --test-input outputs/activations/qwen2_5_7b_challenge_256.npz \
  --output-dir outputs/probes/qwen2_5_7b_challenge
```

## 2. Steering sufficiency

Run at the strongest controlled patching layers, initially 22, 26, and 27.

```bash
python scripts/10_activation_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --directions outputs/directions/qwen2_5_7b_train_256/residual_tool_directions.npy \
  --input outputs/behavior/qwen2_5_7b_ood.jsonl \
  --output outputs/patching/calculator_steering_layer26.json \
  --tool calculator --contrast python --layer 26 \
  --alphas 0.5 1 2 4 8 16 --limit 32 --load-in-4bit
```

Run the negative-direction control by adding support for or using a separately
negated direction archive. A valid sufficiency result should show a monotonic
dose-response and should not indiscriminately increase unrelated tool logits.

## 3. True neuron and head discovery

Run extraction, ranking, and the initial ablation sweep with:

```bash
bash scripts/run_qwen_tier1_stage3.sh
```

Equivalent individual commands:

```bash
python scripts/08_extract_components.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data/full/train.jsonl \
  --output outputs/activations/qwen2_5_7b_components_256.npz \
  --limit 256 --load-in-4bit

python scripts/09_rank_components.py \
  --input outputs/activations/qwen2_5_7b_components_256.npz \
  --output outputs/neuron_rankings/qwen2_5_7b_components_256.json
```

## 4. Necessity with matched controls

Use correctly classified calculator prompts. Test multiple `top-k` values:
`5`, `10`, `20`, `50`, and `100`.

```bash
python scripts/11_component_ablation.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --rankings outputs/neuron_rankings/qwen2_5_7b_components_256.json \
  --input data/full/test.jsonl \
  --output outputs/ablations/calculator_neurons_k20.json \
  --component neurons --tool calculator --contrast python \
  --top-k 20 --limit 32 --seed 42 --load-in-4bit

python scripts/11_component_ablation.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --rankings outputs/neuron_rankings/qwen2_5_7b_components_256.json \
  --input data/full/test.jsonl \
  --output outputs/ablations/calculator_heads_k20.json \
  --component heads --tool calculator --contrast python \
  --top-k 20 --limit 32 --seed 42 --load-in-4bit
```

Repeat across seeds and report bootstrap confidence intervals and paired
permutation tests against same-layer random controls.

## 5. Figures

```bash
pip install -e '.[research]'

python scripts/13_make_figures.py \
  --residual-probe outputs/probes/qwen2_5_7b_ood/residual_probe_metrics.json \
  --mlp-probe outputs/probes/qwen2_5_7b_ood_mlp/mlp_probe_metrics.json \
  --patching outputs/patching/calculator_source_32.json \
  --directions outputs/directions/qwen2_5_7b_train_256/residual_direction_cosines.npy \
  --output-dir paper/figures
```

## 6. Required replication

The main-model study is not enough for the intended paper. Repeat the reduced
pipeline on:

1. `Qwen/Qwen2.5-14B-Instruct`
2. `mistralai/Mistral-7B-Instruct-v0.3`
3. `Qwen/Qwen2.5-Coder-7B-Instruct`

The RTX 3060 may run 7B models in 4-bit. The 14B analysis will likely require
CPU offload, a larger GPU, or rented compute.
