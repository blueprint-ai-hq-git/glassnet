# Concept dropout tested (and my own flaw in it), then the actual mechanism found (2026-11-12)

Fourth lever at the dead-concept defect, plus the diagnosis that explains all four.

## Concept dropout: prediction failed, and the first run was my bug
`--cdrop p` drops each concept with probability p during training only (evaluation always sees the full concept vector; the
trainer sets and clears `net.training` explicitly and prints the flag so a silent no-op would be visible).

**First run was invalid**: I implemented standard (non-inverted) dropout, which leaves training batches and evaluation passes
with different expected activation magnitude - a train/test mismatch. It scored 54.80% +/- 5.04, and most of that loss was my
error, not the idea. Corrected to **inverted** dropout (survivors scaled by 1/(1-p)) and re-run:

```
setting                                accuracy (3 seeds)   min influence   mean influence   concepts <1%
no dropout, pruned (reference)         76.10 +/- 2.40         0.00%            20.90%          3 of 12
non-inverted dropout 0.3               54.80 +/- 5.04         0.00%            18.80%          4-5 of 12
INVERTED dropout 0.3                   65.58 +/- 1.83         0.00%            17.72%          1-4 of 12
dropout 0.6                            16.67 +/- 5.70         0.00%            11.10%          7-9 of 12
```
The correction recovered **10.8 points** and tightened the spread from +/-5.04 to +/-1.83 - so my implementation flaw was worth
more than the lever. Even corrected, dropout costs **10.52 points** and the minimum concept influence is **still 0.00%**, with
mean influence *lower* (17.72% against 20.90%). **Prediction (min influence above 2%, cost under 3 points) failed on both halves.**

Dropout assumes redundancy - in a wide hidden layer other units carry the same information. A concept layer has none: each
concept is a distinct measurement of the image, so dropping one destroys information instead of forcing robustness.

## The mechanism: a concept needs a surviving, load-bearing, unsaturated rule
Per-concept cause, measured on `A3R32_s0` (pruned, 29 rules kept), with each rule's own ablation influence:

```
concept        causal inf   best rule    rules reading it (kept)
ink_amount        46.79%        7.76        11
row_ink_std       38.15%       15.60         9
symmetry_lr       37.85%       10.04        17
grad_ratio        47.87%       15.60        11
symmetry_ud       15.79%        7.76        10
left_heavy        29.23%        7.76        11
col_ink_std       30.49%        2.75         8
corner_ink         1.49%       10.04         2
grad_x_mean        3.14%       15.60         3
top_heavy          0.01%        0.00         2
edge_balance       0.00%        0.00         3
ring_ink           0.00%        0.00         0     <- every connection pruned away
```
Three distinct sub-causes, all measurable:
1. **Connections pruned away.** `ring_ink` has **0** readers among the kept rules - the L1 penalty plus top-3-per-rule pruning
   deleted its last connection. No amount of training-time pressure can help a channel that has no edge.
2. **Readers that are inert.** `top_heavy` and `edge_balance` have 2 and 3 readers, and the best of those rules is worth
   **0.00** points. The concept is wired into rules that do nothing.
3. **Saturation.** `corner_ink` is read by a rule worth **10.04** points and still carries only 1.49% causal influence;
   `grad_x_mean` is read by a rule worth 15.60 and carries 3.14%. The rule's sigmoid is already saturated, so moving that
   concept changes its output by almost nothing.

## Why the four levers all missed
```
lever                        targeted                 missed
delete dead concepts         the vocabulary           removes channels; -8.62 points
weight-coverage penalty      how much weight is read  a wired concept can still sit in an inert or saturated rule
dispersion selection         the candidate pool       deadness is created by training, not inherited
concept dropout              what training may rely on  cannot restore a pruned edge; cannot un-saturate a rule
```
Every one of them attacked *the concept's usage*. The measurement says influence is a property of the **concept-to-rule edge**
(does an edge exist, is the rule it feeds worth anything, and is that rule's output still sensitive). That is where the next
lever has to act:
1. **Coverage-aware pruning**: never prune a concept's last surviving connection.
2. **Linear rules** (`--rule_linear`, already in the model): a linear rule cannot saturate, which removes sub-cause 3 outright.
3. Rule-level necessity is already measured (`tools/prune_rules.py`), so 2 is detectable and fixable at training time.

## Reproduction
`--cdrop {0.3,0.6}` (inverted scaling in `glassnet/phase2_model.py`), three seeds each, checkpoints under
`runs/phase2_sweep/CD*_s*/` and `CDI0.3_s*/`; per-concept cause table from the rule-ablation path in this document's method.
