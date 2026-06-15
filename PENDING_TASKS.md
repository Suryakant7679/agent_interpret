# Paper Checkpoint

Last updated: 2026-06-15

Use this file as the single progress tracker. Check an item only after its
expected output exists and has been inspected.

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
  - Checkpoint: labels remain balanced and prompts pass manual review.
  - Output: `data/lexical_control/`

- [ ] Evaluate behavior and probes on the lexical-control benchmark.
  - Checkpoint: distinguish early lexical decoding from late tool decisions.
  - Output: `outputs/behavior/qwen2_5_7b_lexical_control.*`
  - Output: `outputs/probes/qwen2_5_7b_lexical_control/`

- [ ] Manually audit benchmark labels and prompt quality.
  - Review at least 100 prompts per class.
  - Record ambiguity, incorrect labels, unnatural wording, and duplicates.
  - Output: `paper/benchmark_audit.csv`

- [ ] Replicate the key L26H4 and steering effects without 4-bit quantization.
  - Requires a larger GPU or rented compute.
  - Checkpoint: effect direction and approximate magnitude survive.

## Pending: Cross-Model Replication

- [ ] Run Qwen2.5-Coder-7B behavioral evaluation and core causal analysis.
  - Focus: calculator versus Python decisions.
  - Checkpoint: compare Python-tool bias against general Qwen2.5-7B.

- [ ] Run Mistral-7B-Instruct-v0.3 behavioral evaluation and reduced pipeline.
  - Focus: probes, directions, patching, and head ablation.
  - Checkpoint: determine whether the mechanism is cross-family.

- [ ] Run Qwen2.5-14B behavioral evaluation and reduced pipeline.
  - Requires larger GPU, CPU offload, or rented compute.
  - Checkpoint: measure whether scale sharpens or distributes tool circuits.

- [ ] Compare components across models.
  - Layer-relative location.
  - Tool-direction geometry.
  - Head/neuron concentration.
  - Causal effect size.
  - Error patterns.

## Pending: Statistics And Figures

- [ ] Aggregate all multi-seed experiments.
- [ ] Apply multiple-comparison corrections where relevant.
- [ ] Report bootstrap confidence intervals and paired tests consistently.
- [ ] Produce behavioral confusion matrices.
- [ ] Produce layer-wise probe figures.
- [ ] Produce controlled patching curves.
- [ ] Produce steering dose-response curves.
- [ ] Produce L26H4 necessity and specificity figures.
- [ ] Produce attention-pattern visualization.
- [ ] Produce cross-model comparison tables.
- [ ] Copy final figures into `paper/figures/`.

## Pending: Paper

- [ ] Finalize the paper title and central claims.
- [ ] Write the abstract after all replications finish.
- [ ] Write Introduction.
- [ ] Write and verify Related Work.
- [ ] Write Benchmark section.
- [ ] Write Methods section.
- [ ] Write Behavioral Results.
- [ ] Write Representation Results.
- [ ] Write Causal Results.
- [ ] Write Error Analysis.
- [ ] Write Cross-Model Results.
- [ ] Write Discussion, Limitations, and Broader Impacts.
- [ ] Add verified bibliography entries to `paper/references.bib`.
- [ ] Compile `paper/main.tex`.
- [ ] Perform internal claim-to-evidence audit.
- [ ] Release code, configs, seeds, and benchmark documentation.

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
