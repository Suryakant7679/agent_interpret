# Results Index

Last updated: 2026-06-16

This file is the quick map for inspecting the current workspace and GitHub
state. Use `PENDING_TASKS.md` as the authoritative checklist and
`RESULTS_LEDGER.md` as the authoritative evidence ledger.

## Current Status

| Area | Status | Hardware used |
|---|---|---|
| Qwen2.5-7B primary analysis | Complete for local 4-bit evidence | RTX 3060 12 GB |
| Qwen2.5-Coder-7B local replication | Complete | RTX 3060 12 GB |
| Mistral-7B local replication | Complete for behavior, probes, and component ablations | RTX 3060 12 GB |
| Local report tables and figures | Complete and refreshed for all three local models | CPU/local |
| Full-precision 7B replication | Pending | Large GPU, 24 GB minimum |
| Qwen2.5-14B replication | Pending | Large GPU, 24 GB minimum; 48 GB preferred |
| Final paper writing and claim audit | Pending | CPU/local after result scope is frozen |

## Where To Inspect Results

| What to inspect | Path |
|---|---|
| Project checklist and hardware split | `PENDING_TASKS.md` |
| Completed observations and limitations | `RESULTS_LEDGER.md` |
| Large-GPU handoff commands | `GPU_QUICKSTART.md` |
| Local report summary | `paper/local_report.summary.json` |
| Cross-model behavior/probe/patching table | `paper/tables/model_comparison.csv` |
| Cross-model component ablation table | `paper/tables/component_ablation_comparison.csv` |
| Statistical tests table | `paper/tables/statistical_tests.csv` |
| Paper figures | `paper/figures/` |
| Human benchmark audit | `paper/benchmark_audit.csv` |

## Figures Included

All figures are saved as PNG for quick viewing and PDF for paper use.

| Figure | Scope |
|---|---|
| `behavior_confusion_matrices` | Lexical-control behavior for Qwen2.5-7B, Qwen2.5-Coder-7B, and Mistral-7B |
| `layerwise_probe_comparison` | Residual-stream and MLP probe curves for all three local models |
| `controlled_patching_comparison` | Controlled residual patching for models with usable calculator-to-Python target errors |
| `component_ablation_comparison` | Neuron/head ablations for all three local models |
| `multiseed_replication` | Qwen2.5-7B multi-seed patching and L26H4 checks |
| `steering_dose_response` | Qwen2.5-7B calculator-direction steering controls |
| `l26h4_specificity` | Qwen2.5-7B L26H4 validation/challenge specificity |
| `l26h4_attention_tokens` | Qwen2.5-7B L26H4 attention-token summary |

## Important Interpretation Notes

- Mistral controlled calculator-to-Python patching is marked not available
  because its OOD run produced no calculator-to-Python target errors.
- Mistral still contributes useful cross-family evidence through component
  ablations: top calculator heads strongly reduce the calculator margin when
  ablated.
- All local model runs used 4-bit loading. Full-precision confirmation remains
  a large-GPU task.
- Generated raw outputs are intentionally excluded from Git where they are too
  large or machine-specific. The tracked report tables, figures, scripts, and
  ledgers are the GitHub inspection surface.
