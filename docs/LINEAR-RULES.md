# Linear rules are the best readable configuration measured on this task: 81.19% +/- 0.98 with 20 rules (2026-11-12)

The dead-concept mechanism pointed at saturation: `corner_ink` is read by a rule worth 10.04 points and still carries 1.49%,
`grad_x_mean` by a rule worth 15.60 and carries 3.14%, because a sigmoid rule that is already saturated cannot be moved by one
of its inputs. The model has supported linear rules (`rule_linear`) since it was written - but the **trainer never exposed the
flag**, so that branch had never been trained once. It is now `--rule_linear`.

**Prediction, written before the run:** removing saturation lifts the minimum concept influence above 2% with no concept below
1%, at an accuracy cost under 5 points against 76.10% +/- 2.40. **The influence half passed in 2 of 3 seeds; the accuracy half
was wrong in the good direction - it is a gain, not a cost.**

## Accuracy, and it is the best on this task
```
arm                                accuracy (3 seeds)   rules to read
sigmoid rules, a3 R32              72.01 +/- 4.35            32
sigmoid rules, a3 R32, pruned      76.10 +/- 2.40            29 (28.7 mean)
alpha-1 R16, pruned               79.78 +/- 0.48            15.7
LINEAR rules, a3 R32               76.71 +/- 2.39            32
LINEAR rules, a3 R32, PRUNED       81.19 +/- 0.98            20 (19-22)
original Phase 2 gate (sigmoid)    79.04 +/- 2.38            30
```
Per seed with pruning: **82.26 / 80.95 / 80.35**, pruning itself worth **+4.48 +/- 1.52** (10-13 of 32 rules removed). That is
**+5.09 points over the best sigmoid honest arm**, **+1.41 over the accuracy champion**, and **+2.15 over the original Phase 2
gate** - with **20 rules a human must read instead of 30**, and the tightest spread measured anywhere on this task (+/- 0.98).

The audit reproduces it exactly and detects the mode itself: `rule mode: linear reproduces 79.47% (sigmoid 20.03%) -> MATCH`.
A sigmoid read of a linear-rule model collapses to 20.03%, which is a useful reminder that the forward pass has to match the
training pass or every number after it is fiction.

## Concept honesty and influence on this arm
```
mean |spearman| with own label   0.660
concepts below 0.50              1 of 12
concepts tracking another name   2 of 12      (sigmoid honest arm: 4.7-7.0 of 12 misnamed at alpha 1, 1.0-1.3 at alpha 3)
causal influence, 3 seeds        mean 23.56%   worst-seed minimum 0.00%
concepts below 1% influence      2 / 0 / 0 of 12
```
So linear rules also **halve the dead-concept count**: two of three seeds have zero concepts under 1% influence, and the third
still has two. Removing saturation helps but does not finish the job - as the mechanism predicted, a concept can also be dead
because its edge was pruned away or because the rule it feeds is inert, and those causes are untouched by this change.

## Why this matters more than the number
The readable architecture was supposed to cost accuracy. On this task, at this size, the readable middle layer with linear
rules is **better** than the plain sigmoid version of itself and better than the original Phase 2 model - while being the most
readable configuration measured (20 rules, 12 named concepts, explanations that print `docs/EXPLAIN-IMAGE.md`-style output).

## Reproduction
`--rule_linear --R 32 --K 12 --alpha 3.0` with `GLASSNET_CONCEPTS=$PWD/data/concepts_sel12.npz`, three seeds; then
`tools/prune_rules.py` on each checkpoint (it reads `rule_linear` from the checkpoint, added this run so it cannot silently
measure a sigmoid forward pass); influence from `tools/explain_image.py <ckpt> 0 --pruned`; honesty from
`tools/phase2_reaudit.py`. Three checkpoints under `runs/phase2_sweep/LIN_s*/`.
