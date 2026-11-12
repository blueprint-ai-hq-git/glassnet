# Output-variance floor: the right quantity, a better ship, and dead concepts still survive (2026-11-12)

Seventh lever at the dead-concept defect, and this one targets the quantity the mechanism actually named.

## A hypothesis killed before it cost a training run
The obvious training-time pressure is "make every rule carry decision weight", i.e. a floor on each rule's answer row
(`--rfloor`, implemented). I checked the premise first, on the shipped arm:

```
per-rule answer-row L1 magnitude:  min 1.747   median 5.648   max 12.283   rules below 0.5: 0 of 32
```
**Every rule already carries answer weight**, yet some rules are worth exactly 0.000 points. So answer weight is not the quantity
that decides whether a rule does anything - the same error the coverage penalty made one level down (weight instead of
behaviour). The floor was never run.

## The right quantity: a rule that emits a constant makes no decisions
`--varfloor F` penalises any rule whose output variance across the batch is below F. Gradient checked first: **exact
(0.00e+00)** wherever no rule sits at the relu kink, and a 1.17e-04 discrepancy only when a rule's variance lies exactly on the
floor - a property of the penalty's kink, not a bug.

**Prediction:** per-rule ablation values rise, minimum concept influence rises above 1%, accuracy cost under 3 points.
**The influence half failed; the accuracy half came out better than predicted.**

```
setting                          pruned accuracy (3 seeds)   worst min influence   mean influence   concepts <1%
ship config, no floor            81.19 +/- 0.98                  0.00%               23.56%          3 of 12
varfloor 0.02                    80.64 +/- 1.63                  0.00%               25.27%          1-2 of 12
varfloor 0.10                    81.73 +/- 0.54                  0.00%               24.49%          0-2 of 12
```
Per seed at varfloor 0.10: **82.35 / 81.33 / 81.51**, rules to read 27 / 26 / 21.

## What this buys, honestly
- **A better ship configuration**: `alpha 3 / 32 rules / linear / varfloor 0.10 / pruned` = **81.73% +/- 0.54**, which is +0.54
  points over the previous ship and, more usefully, **half the seed spread** (+/- 0.54 against +/- 0.98). Reproducibility is the
  part that matters for a claim.
- **Fewer dead concepts**: 3 of 12 drop to 0-2 of 12, and mean influence rises slightly.
- **But the minimum is still 0.00%** in two of three seeds. The prediction failed, and for the seventh time a lever aimed at
  dead concepts did not remove them.

## Seven levers, and the one that worked
```
lever                                accuracy effect   dead concepts
delete the dead concepts             -8.62             gone, model worse
weight-coverage penalty              none              none
dispersion-based selection           -5.97 (confounded) none
concept dropout                      -10.52            none
linear rules                         +5.09             halved (3 -> 0-2 of 12)
coverage-aware rule pruning          -0.27             none
output-variance floor                +0.54 (and half the spread)   0-2 of 12, minimum still 0.00%
```
Two levers that changed the right thing (linear rules, variance floor) both helped accuracy as well. Every lever that aimed at
*weights* did nothing. That pattern is the useful result of this whole thread: on this architecture, weight magnitude is not a
proxy for function, at either level (concept-to-rule or rule-to-answer).

## Status of the defect
**Unsolved.** Some concept in every configuration measured changes 0.00% of predictions while passing the label-honesty gate.
The honest description of the shipped model is therefore: 12 named concepts, 22-27 load-bearing rules, **one concept that means
what its name says and decides nothing** - and that caveat now belongs in the public claim next to the accuracy, not in a
footnote.

## Reproduction
`--varfloor {0.02,0.10}` with `--rule_linear --R 32 --K 12 --alpha 3.0`, three seeds each; checkpoints under
`runs/phase2_sweep/VF0.02_s*/` and `VF0.10_s*/`; gradient harness in the method section above (float64, eps 1e-5).
