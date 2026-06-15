# Paper Checkpoint

Last updated: 2026-06-15

Use this file as the single progress tracker. Check an item only after its
expected output exists and has been inspected.

## Compute Policy

- **LOCAL**: safe to run on the current RTX 3060 12 GB with 4-bit loading, or
  on CPU without substantial GPU memory.
- **LARGE GPU**: defer until a machine with at least 24 GB VRAM is available;
  48 GB or more is preferred for activation interventions.
- Do not start a **LARGE GPU** item on the RTX 3060 merely to test whether it
  fits. Preserve the local machine for the listed **LOCAL** work.

## Completed: Qwen2.5-7B Core

- [x] Generate balanced ToolUseCircuitBench datasets.
- [x] Run smoke, OOD, adversarial, and challenge behavioral evaluations.
- [x] Extract residual-stream and MLP-output activations.
- [x] Train held-out residual and MLP probes.
- [x] Compute tool directions and direction geometry.
- [x] Run controlled residual patching with calculator, Python, and none sources.
- [x] Demonstrate signed calculator-direction steering.
- [x] Demonstrate high-dose behavioral rescue with steering.
- [x] Extract and rank true MLP neurons and attention heads.
- [x] Run neuron and head ablations with matched random controls.
- [x] Discover L26H4 as a calculator-specific candidate.
- [x] Replicate L26H4 on validation and challenge datasets.
- [x] Test L26H4 specificity across all four tool labels.
- [x] Analyze L26H4 attention to prompt tokens.
- [x] Verify L26H4 robustness across prompt-order permutations.
- [x] Maintain verified findings in `RESULTS_LEDGER.md`.

## Pending: Qwen2.5-7B Robustness

- [x] Repeat controlled 32-pair patching with seeds `7`, `21`, `84`, and `123`.
  - Checkpoint: treatment remains stronger than Python and none controls.
  - Output: `outputs/patching/seeds/`

- [x] Repeat L26H4 validation with multiple prompt-sampling seeds.
  - Checkpoint: calculator margin remains strongly negative after ablation.
  - Checkpoint: unrelated tool decisions remain stable.
  - Output: `outputs/ablations/l26h4_seeds/`

- [x] Build a stricter lexically controlled benchmark.
  - Remove class-specific boilerplate such as `today`, `exact result`, and
    `each value`.
  - Checkpoint: labels remain balanced and templates pass structural and
    semantic review.
  - Output: `data/lexical_control/`

- [x] Evaluate behavior and probes on the lexical-control benchmark.
  - Checkpoint: distinguish early lexical decoding from late tool decisions.
  - Output: `outputs/behavior/qwen2_5_7b_lexical_control.*`
  - Output: `outputs/probes/qwen2_5_7b_lexical_control/`

- [ ] **LOCAL / HUMAN** Manually audit benchmark labels and prompt quality.
  - Review at least 100 prompts per class.
  - Record ambiguity, incorrect labels, unnatural wording, and duplicates.
  - Output: `paper/benchmark_audit.csv`
  - Status: deferred until a human reviewer is available; an incomplete local
    audit file must not be treated as evidence.

- [ ] **LARGE GPU** Replicate key L26H4 and steering effects without 4-bit
  quantization.
  - Target: at least 24 GB VRAM; 48 GB preferred.
  - Checkpoint: effect direction and approximate magnitude survive.

## Pending: Local Cross-Model Replication

- [ ] **LOCAL** Run Qwen2.5-Coder-7B behavioral evaluation and reduced causal
  analysis in 4-bit.
  - Focus: calculator versus Python decisions.
  - Checkpoint: compare Python-tool bias against general Qwen2.5-7B.

- [ ] **LOCAL** Run Mistral-7B-Instruct-v0.3 behavioral evaluation and reduced
  pipeline in 4-bit.
  - Focus: probes, directions, patching, and head ablation.
  - Checkpoint: determine whether the mechanism is cross-family.

## Pending: Large-GPU Replication

- [ ] **LARGE GPU** Run Qwen2.5-14B behavioral evaluation and reduced pipeline.
  - Target: at least 24 GB VRAM; 48 GB preferred for causal interventions.
  - Checkpoint: measure whether scale sharpens or distributes tool circuits.

- [ ] **LARGE GPU / ANALYSIS** Compare components across all completed models.
  - Layer-relative location.
  - Tool-direction geometry.
  - Head/neuron concentration.
  - Causal effect size.
  - Error patterns.

## Pending: Local Statistics And Figures

- [ ] **LOCAL / CPU** Aggregate completed multi-seed experiments.
- [ ] **LOCAL / CPU** Apply multiple-comparison corrections where relevant.
- [ ] **LOCAL / CPU** Report bootstrap confidence intervals and paired tests
  consistently.
- [ ] **LOCAL / CPU** Produce behavioral confusion matrices.
- [ ] **LOCAL / CPU** Produce layer-wise probe figures.
- [ ] **LOCAL / CPU** Produce controlled patching curves.
- [ ] **LOCAL / CPU** Produce steering dose-response curves.
- [ ] **LOCAL / CPU** Produce L26H4 necessity and specificity figures.
- [ ] **LOCAL / CPU** Produce attention-pattern visualization.
- [ ] **LARGE GPU / ANALYSIS** Produce cross-model comparison tables after all
  required model runs finish.
- [ ] **LOCAL / CPU** Copy completed figures into `paper/figures/`.

## Pending: Local Paper Work

- [ ] **LOCAL / CPU** Finalize the paper title and provisional central claims.
- [ ] **LOCAL / CPU** Write the abstract after all replications finish.
- [ ] **LOCAL / CPU** Write Introduction.
- [ ] **LOCAL / CPU** Write and verify Related Work.
- [ ] **LOCAL / CPU** Write Benchmark section.
- [ ] **LOCAL / CPU** Write Methods section.
- [ ] **LOCAL / CPU** Write Behavioral Results.
- [ ] **LOCAL / CPU** Write Representation Results.
- [ ] **LOCAL / CPU** Write Causal Results.
- [ ] **LOCAL / CPU** Write Error Analysis.
- [ ] **LARGE GPU / ANALYSIS** Write Cross-Model Results after replications.
- [ ] **LOCAL / CPU** Write Discussion, Limitations, and Broader Impacts.
- [ ] **LOCAL / CPU** Add verified bibliography entries to
  `paper/references.bib`.
- [ ] **LOCAL / CPU** Compile `paper/main.tex`.
- [ ] **LOCAL / CPU** Perform internal claim-to-evidence audit.
- [ ] **LOCAL / CPU** Release code, configs, seeds, and benchmark documentation.

## Submission Gate

Do not mark the project submission-ready until all are checked:

- [ ] Main-model results replicate across seeds.
- [ ] Lexical-control benchmark passes.
- [ ] At least one cross-family model replicates the core causal result.
- [ ] Same-family scale or specialization comparison is complete.
- [ ] Full-precision check is complete.
- [ ] Figures and statistical tables are final.
- [ ] Every paper claim maps to a saved result file.
- [ ] Limitations clearly disclose synthetic prompts and quantized inference.
