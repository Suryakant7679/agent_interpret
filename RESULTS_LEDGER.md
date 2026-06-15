# Results Ledger

This file separates completed observations from claims that still require
controls or replication.

## Completed

- Qwen2.5-7B-Instruct smoke behavior: accuracy 0.90, macro-F1 0.896.
- Smoke errors: all four large-multiplication prompts selected Python instead
  of calculator.
- OOD behavior: accuracy 0.758, macro-F1 0.684.
- OOD calculator recall: 0.032; 242/250 calculator prompts selected Python.
- Adversarial behavior: accuracy 0.47, macro-F1 0.319.
- Held-out OOD residual probe reaches macro-F1 1.0 at layer 19.
- Held-out OOD MLP-output probe reaches macro-F1 1.0 at layer 18.
- Pilot residual patching with eight pairs flips all targets at layers 22, 26,
  and 27.
- Controlled 32-pair residual patching uses identical ordered
  calculator-to-Python target errors across calculator, Python, and none source
  conditions.
- Calculator-source patching at layer 27 increases the calculator-minus-Python
  logit difference by 4.707 (95% bootstrap CI [4.266, 5.152]) and flips all
  32 targets to calculator.
- At layer 27, calculator-source patching exceeds Python-source patching by
  9.184 logits and none-source patching by 6.844 logits; paired permutation
  p-values are below 0.00001 for both comparisons.
- Controlled patching replicates across independently sampled pairs at seeds
  7, 21, 42, 84, and 123. Layer 27 is the strongest treatment layer for every
  seed, with calculator-source mean effects from 4.641 to 5.352 logits and a
  100% target flip rate in every run.
- Across the four additional seeds, the layer-27 treatment effect averages
  5.015 logits. It exceeds Python controls by an average of 9.341 logits and
  none controls by 6.721 logits; every seed passes both comparisons.
- Calculator-source patching reaches flip rates of 0.969 at layers 22 and 26,
  while both source controls have zero flips at those layers.
- Matched challenge behavior reaches accuracy 0.9825 and macro-F1 0.9825 over
  400 prompts. Calculator recall is 0.93; all seven errors are
  calculator-to-Python confusions.
- Challenge behavior therefore replicates the calculator-to-Python error
  direction under different wording while substantially reducing its
  frequency relative to the original OOD arithmetic template.
- Ablating the 20 highest absolute-selectivity calculator attention heads
  reduces the calculator-minus-Python logit difference by 3.738 on 32
  correctly classified calculator prompts (95% bootstrap CI
  [-4.102, -3.391]) and flips all 32 decisions to Python.
- One same-layer random-head control increases the logit difference by 0.500
  (95% CI [0.313, 0.691]); the paired permutation p-value is below 0.00001.
- Ablating the 20 highest absolute-selectivity intermediate MLP neurons does
  not support calculator necessity: it increases the calculator-minus-Python
  margin by 0.238 and produces no decision flips.
- Ablating the top 10 positively calculator-selective heads reduces the
  calculator-minus-Python margin by 3.059 (95% CI [-3.480, -2.645]) and flips
  81.25% of 32 calculator decisions. Ten same-layer random-control draws have
  mean effect +0.415 and zero flips.
- Ablating the top 10 negatively calculator-selective heads has a smaller
  effect of -0.844 (95% CI [-1.070, -0.609]) and flips 43.75% of decisions.
- On identical prompts, the positive-head effect exceeds the negative-head
  effect by 2.215 logits (paired 95% bootstrap CI [1.617, 2.828],
  permutation p < 0.00001).
- Corrected prompt-class specificity shows that ablating the positive
  calculator-selective head set changes each class's own label margin by:
  calculator -3.059, Python +1.484, web search -6.371, and none +0.805.
  No Python, web-search, or none decisions flip because their baseline margins
  remain large.
- The exploratory individual-head screen nominates L26H4: calculator margin
  effect -2.979 over six correctly classified prompts, with effects +0.172 on
  web search, +4.063 on Python, and -0.219 on none.
- L26H4 replicates on 32 held-out validation prompts per class. Ablation
  changes label margins by web search +0.129, calculator -2.977, Python
  +3.875, and none -0.309. It flips 78.125% of calculator decisions and no
  decisions from the other three classes.
- L26H4 independently replicates on the challenge set: web search -0.289,
  calculator -2.984, Python +4.137, and none -0.133. It flips 100% of
  calculator decisions and no decisions from the other classes.
- Against 20 same-layer random-head controls, the L26H4 calculator effect is
  significant on both held-out sets (paired permutation p < 0.00001).
  Specificity scores are 2.668 on validation and 2.695 on challenge.
- L26H4 necessity and specificity replicate across independently shuffled
  validation samples at seeds 7, 21, 84, and 123. Calculator-margin effects
  range from -2.953 to -2.840 logits, with every 95% bootstrap interval fully
  below zero.
- Across those four prompt-sampling seeds, the mean calculator effect is
  -2.904 logits and the mean calculator flip rate is 69.5%. Ablation causes
  zero web-search, Python, or none decision flips across all four runs.
- High-dose calculator-direction steering at layer 26 produces monotonic
  behavioral rescue on every one of 32 paired calculator-to-Python errors:
  flip rates are 21.875%, 50%, 84.375%, and 100% at alpha 24, 32, 48, and 64.
- The sign-reversed calculator direction monotonically worsens every example,
  with effects -1.414, -1.836, -2.891, and -4.012 logits at the same doses
  and zero rescue flips.
- L26H4 attention concentrates on explicit calculator tokens in the prompt
  scaffold rather than arithmetic operands. On validation prompts, mean
  attention to the calculator token is 0.289 in the tool description and
  0.305 in the final allowed-label list for calculator examples. Combined
  calculator-token attention is 0.594 for calculator prompts versus 0.440
  for Python, 0.442 for none, and 0.160 for web search.
- L26H4 necessity is robust to eight permutations of tool descriptions and
  allowed-label order. Mean calculator-margin effects range from -2.332 to
  -2.977 logits (overall mean -2.697, standard deviation 0.210).
- Flip-rate variation across prompt orders is explained by changed baseline
  calculator margins (correlation -0.843), not disappearance of the L26H4
  effect. Calculator position 0--3 all retain substantial negative effects.

## Preliminary interpretation

- Calculator and Python tool decisions are behaviorally confusable.
- Late-layer states carry information sufficient to alter this decision under
  full residual replacement.
- The controlled source-class comparison supports tool-specific causal
  mediation in late residual states rather than a generic patching effect.
- Calculator/Python confusion is robust across large multiplication,
  exponent/division, and differently phrased exact-arithmetic prompts.
- Attention-head outputs appear more causally important than the current
  selectivity-ranked MLP-neuron set for calculator choice.
- Positively calculator-selective heads provide the strongest necessity
  signal, concentrated in layers 13, 16, 18, 19, 25, and 26.
- The positive-head set is not calculator-specific. Its opposite effects on
  calculator/web-search versus Python/none are more consistent with shared
  tool-routing or tool-family mediation.
- L26H4 is a replicated calculator-specific causal component. Its ablation
  consistently suppresses calculator choice, strengthens Python choice, and
  leaves web-search and none decisions intact.
- L26H4 is best interpreted as a late tool-label routing head that reads the
  explicit calculator option from the agent scaffold, not as an arithmetic
  detector.
- Layer-26 calculator-direction steering shows a signed monotonic
  dose-response over alpha 0.5--16. The calculator-minus-Python effect reaches
  +0.797 logits at alpha 16, while the sign-reversed calculator direction
  reaches -0.875 and the Python-direction control reaches -0.984.
- Low-dose steering demonstrates directional sufficiency at the logit level
  and high-dose steering demonstrates behavioral sufficiency with a signed,
  per-example monotonic dose-response.
- Middle-layer OOD probe errors frequently map web-search exchange-rate
  prompts to calculator before late layers separate them.

## Not yet established

- No individual MLP neuron has yet shown replicated causal necessity.
- The representation generalizes beyond distinctive synthetic templates.
- The result has not yet replicated across scales or model families.
- Quantized and full-precision interventions have equivalent effects.
- The challenge probe is not valid evidence for late emergence: residual
  macro-F1 is already 1.0 at layer 1 and MLP-output macro-F1 is 1.0 at layer
  0, consistent with class-specific lexical frames in the challenge prompts.
- L26H4 is one component, not yet a complete circuit. Its attended token
  patterns, upstream inputs, and downstream interactions remain unknown.
- The first specificity run on Python, web-search, and none prompts used the
  calculator-minus-Python margin for every class, making its reported flip
  rates invalid. Those files are superseded and must be regenerated with each
  class's own label margin.

## Required before submission

- Counterfactual challenge-set behavior and probing.
- Replace the current challenge probe with stricter lexical/template controls
  before using it to make claims about layer-wise emergence.
- True intermediate-neuron and attention-head ablation with random controls.
- Analyze L26H4 attention patterns and interactions with late residual
  calculator directions.
- Steering dose-response and negative/mismatched direction controls.
- Same-family scale and cross-family replication.
- Human audit of benchmark labels and prompt naturalness.
