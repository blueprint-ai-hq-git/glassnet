# The concept readout is degenerate: it predicts `<unk>` for every input

Date: 2026-11-14. Tool `tools/concept_path_collapse.py`, runner `lens/scripts/run_concept_path_collapse.sh`, predictions
`lens/runs/CONCEPT-PATH-COLLAPSE-PREDICTIONS.txt`, output `lens/runs/CONCEPT-PATH-COLLAPSE.txt` and per-checkpoint JSON.

## What started this

In the direct-path cost run (`docs/history/DIRECT-PATH-COST.md`) the concept-only accuracy was **5.25% at d512 AND at d1024** -
identical to two decimals across two very different models. Identical numbers across models usually mean the measurement is
reading something that does not depend on the model, so it was checked rather than quoted.

## Result: the readout emits one token, always

```
checkpoint        concept-only acc   bias-only acc   distinct tokens predicted   top-1 share
d512  e9 s0            5.25%             5.25%                 1                    100.0%
d1024 e9 s0            5.25%             5.25%                 1                    100.0%
bias top token: <unk>   predictions: <unk> on 2000 of 2000 positions
```

**The concept path predicts `<unk>` for every one of 2,000 held-out positions, at both widths.** Predicting the bias top token
reproduces its accuracy exactly (gap 0.000 points).

## The coincidence explained

```
targets that are <unk> : 105 of 2000 = 5.2500%
concept-only accuracy  :              5.2500%
```

The concept path's "accuracy" **is exactly the `<unk>` base rate**, because it always answers `<unk>`. It is not 18% of the model's
skill; it is **0% skill and a constant answer**.

## Why the readout collapses - two reinforcing causes, both measured

```
bias[<unk>] = 1.2886, next best token = 0.3276        the bias puts <unk> 3.9x ahead of anything else
mean |W[:, <unk>]| = 0.3197  vs  mean |W| = 0.1879     the concept->answer column for <unk> is 1.7x the average
```

So both the bias and the concept-to-answer weights favour `<unk>` strongly enough that no concept activation in 36 dimensions can
outvote it. Median per-row concept contribution magnitude is **2.618** (d512) / **1.496** (d1024) against a median bias magnitude of
**1.289** / **1.408** - the concepts are NOT too small; they are simply all funnelled into the same winning column.

Predictions: **P1 HOLDS** (bias-only within 1 point - exactly equal), **P2 HOLDS** (1 distinct token, bound was 20), **P3 FAILED**
(I predicted concept contributions would be smaller than the bias; they are larger, 2.618 vs 1.289, so my magnitude explanation was
wrong and the real mechanism is the `<unk>` column).

## This corrects the previous number

`docs/history/DIRECT-PATH-COST.md` reported that the concept path "keeps 18.58% / 17.92% of the accuracy". That is arithmetically
true and misleading: the retained accuracy is the `<unk>` base rate, not retained skill. The honest statement is:

> **With the direct pathway removed, the concept layer scores exactly the `<unk>` base rate because it answers `<unk>` unconditionally.
> Its task skill, as currently read out, is zero.**

## What it means for the readability goal

There are now **two separate defects**, and fixing one without the other changes nothing:

1. **Coverage** - the direct pathway carries 82-91% of the decision (measured, docs/history/CHAT-EXPLAIN-CAUSAL.md).
2. **Readout degeneracy** - the concept-to-answer path emits a constant `<unk>` (this document).

So "constrain the direct pathway and let the concepts carry the work" cannot succeed as a single change: with the readout
degenerate, a constrained model would simply become a `<unk>` machine. Any constrained-path arm must **also** fix the readout -
`<unk>` in targets is a known corpus issue here (the trainer logs "held-out assistant targets that are `<unk>`: 6,929 of 119,436
(5.801%)"), and the earlier `<unk>` audit found 86% of `<unk>` predictions were wrong.

The dependency is now explicit and it changes the plan: **fix the readout first (or at least alongside), then test constraint.**
