# Re-audit of the original Phase 2 Fashion-MNIST model, with today's instruments (2026-11-12)

The Phase 2 gate (2026-10-01) reported 80.75% accuracy, concept fidelity as AUROC >= 0.55 for 12 of 14 concepts, 3.00
connections per rule, 30 rules, and "0 dead rules of 30". Today's instruments - built while working on the chat-corpus rule
layer - are applied to that same checkpoint (`runs/phase2/final_seed0.npz`).

The method is validated first: the reconstruction reproduces the recorded accuracy **exactly**.

```
rule mode: sigmoid reproduces 80.75%   (linear would give 75.01%)   recorded post_prune_acc: 80.75%
held-out accuracy on all 10,000 test images: 80.75%
```

## 1. Concept honesty, with a cross-firing test the original did not have
The concept targets are continuous features, so the honest measure is rank agreement (Spearman), not a 0.5 threshold.

```
concept         feature mean   spearman vs own label
ink_amount          0.365            0.876
symmetry            0.821            0.345
top_heavy           0.537            0.331
horizontal_edges    0.244            0.561
vertical_edges      0.155            0.643
quad_UL_ink         0.298            0.788
quad_UR_ink         0.360            0.311
quad_LL_ink         0.344            0.652
quad_LR_ink         0.388           -0.084
center_ink          0.515            0.930
left_heavy          0.579            0.605
border_ink          0.114            0.532
tall_shape          0.841            0.725
wide_shape          0.755            0.898

mean |spearman| 0.591      concepts below 0.50: 4 of 14
```

Four neurons are better explained by a DIFFERENT named feature than their own:

```
top_heavy         own +0.331   tracks quad_LL_ink  +0.543
horizontal_edges  own +0.561   tracks wide_shape   +0.753
vertical_edges    own +0.643   tracks wide_shape   +0.723
quad_LR_ink       own -0.084   tracks symmetry     +0.282
```

`quad_LR_ink` is the worst case: its correlation with its own label is **-0.084**, i.e. it does not track the feature it is
named after at all. The original gate's "12/14 AUROC >= 0.55" is not wrong - AUROC measures rank separation - but it did not
ask the question that matters for a readable system: *is this neuron better explained by another thing we named?* For 4 of
14, the answer is yes.

## 2. "0 dead rules of 30" does not survive a necessity test
```
per-rule ablation (zero the rule's input weights): min -0.19  median +0.01  max +1.47 accuracy points
rules worth less than 0.01 points: 6 of 30
```
The original claim counted rules that were still *connected* and non-zero. By necessity, the **median rule is worth 0.01
accuracy points** - so half the layer is inert in the same way the 96-rule chat model was. One rule is actively negative
(-0.19).

## 3. Concept interaction: 5.3%, so explanations are ~95% complete
```
additivity: sum of individual concept effects -4.144   all concepts abandoned -3.934   interaction 5.3%
```
Worse than the chat model (0.4-3.7%) but the same order: a per-concept explanation of a Fashion-MNIST decision accounts for
about 95% of the effect on the predicted class score.

## 4. Rule plausibility
```
rule-pair lift (median split of each feature): median 1.21   rules joining features that do not co-occur: 10 of 30
```
A third of the rules AND together features that are statistically unrelated in the data - the same defect the chat model has,
at a similar rate.

## What this changes about the Phase 2 record
```
claim in the gate                        status after re-audit
accuracy 80.75%                          CONFIRMED exactly
3.00 connections per rule                unchanged (structural)
30 rules a human must read               TRUE, but the median rule is worth 0.01 points - read 24, not 30
12/14 concepts pass AUROC >= 0.55        does not establish label honesty: 4 of 14 track another feature better
0 dead rules                             FALSE by necessity measurement: 6 of 30 are worth nothing
```
The readable middle layer is real and the accuracy is real. What the original gate did not measure - and what the new
instruments show - is that **the layer is legible but loosely attached**: nearly half its rules are individually inert, a
third join unrelated features, and four of fourteen named neurons are better described by a different name.

## Reproduction
`tools/phase2_reaudit.py` (read-only, loads `runs/phase2/final_seed0.npz` and the cached Fashion-MNIST split). It prints the
reproduced accuracy beside the recorded one first, so a wrong reconstruction is visible before any of its other numbers are
believed - the rule this project keeps relearning.
