# Phase 2 gate, Fashion-MNIST readable architecture: current state (2026-11-12)

Gate items measured on the shipped configuration - **12 named concepts -> 32 linear rules (pruned to 3 connections each) ->
answer, concept weight alpha 3, per-rule output-variance floor 0.10, seeds 0/1/2** - plus the two items that are not clean.

## Item 1: accuracy against the baseline
```
readable model, after rule pruning      81.73% +/- 0.54     (82.35 / 81.33 / 81.51 across three seeds)
same weights, before rule pruning       76.26% +/- 4.62     (81.59 / 73.35 / 73.84)
PHASE 1 BASELINE (plain MLP, no rules)  86.69% +/- 0.59
earlier sigmoid-rule readable model     76.10% +/- 2.40
```
**The readable model does not beat the Phase 1 baseline: it is 4.96 points below it.** That is the honest headline and it has not
changed all day. What did change: within the readable family, linear rules and rule pruning moved this architecture from 76.10 to
81.73 (+5.63), and rule pruning alone is worth ~5 points because it deletes rules that actively hurt borderline predictions.

## Item 2: connections per rule
**3.00, exactly** - min 3, max 3, on every kept rule in all three seeds (measured on the pruned masks). The sparse-rule
requirement is met by construction, not on average.

## Item 3: rules a human must read
```
seed 0:  27 rules kept (5 of 32 removed)
seed 1:  26 rules kept (6 of 32 removed)
seed 2:  21 rules kept (11 of 32 removed)
mean     24.7 rules
```
Each rule is a sentence of the form `concept_a AND concept_b AND concept_c`, printed by `tools/explain_image.py`.

## Item 4: does each concept mean what its name says, and the sanity test
```
mean |spearman| with own label     0.625 +/- 0.034
concepts below 0.50                1 / 4 / 2 of 12  (per seed)
tracking ANOTHER named feature     2 / 2 / 3 of 12  (mean 2.3)  <- the sanity test
```
The sanity test is the one that matters: a neuron whose activation correlates better with a *different* named feature than its own
label is misnamed, and on average **2.3 of 12 concepts per seed are**. Two to four more sit below 0.50 rank agreement.

**And the caveat measured two wakes ago still stands:** minimum causal influence is **0.00% in two of three seeds**. A concept can
pass this item completely - `top_heavy` was 0.968 honest - and still change no prediction. Honesty is necessary for readability
and demonstrably not sufficient.

## Item 5: ablation, removing a layer
**Not yet measured on this configuration.** The rule-layer ablation (`no_rules`: concepts read straight into the answer layer) needs
a training run for each seed, and a game was running during this measurement window, so it was deliberately not launched rather
than run at the cost of the machine. It is the one open gate item, and the runner support exists (`no_rules` in the model,
`--rule_linear`/`--rfloor`/`--varfloor` in the trainer).

What can be said without it: the concept-layer ablation was measured on the sigmoid architecture earlier - deleting the rule layer
*raised* accuracy from 13.47% to 17.18% on the chat task, i.e. the readable middle layer cost 3.71 points there. Nothing measured
today suggests the image task will behave differently, and that remains an expectation, not a result.

## Everything in this report was measured in this run or earlier in the same session
- accuracy / connections / rule counts: `runs/phase2_sweep/VF0.10_s*/` + their `pruned_final_seed*.json`
- honesty: `tools/phase2_reaudit.py` on the same checkpoints (unpruned pass, which is what the audit measures)
- causal influence: `tools/explain_image.py <ckpt> 0 --pruned`, `docs/VARIANCE-FLOOR.md`
- baseline: `runs/phase1/BASELINE.md` (86.69% +/- 0.59, 784->128 ReLU->10, own SGD lr 0.1, 10 epochs, 3 seeds)

## Reproduction, single core, for a machine that is being used for something else
```
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u tools/phase2_reaudit.py runs/phase2_sweep/VF0.10_s0/final_seed0.npz
```
