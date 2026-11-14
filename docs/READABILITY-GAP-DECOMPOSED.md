# The 4.96-point readability gap, decomposed: 2.32 points capacity + 2.64 points readability

Date: 2026-11-14. Tool: `tools/matched_plain_p2.py`, output `runs/matched_plain_p2/matched_plain_summary.json`.
Corrects and completes `docs/history/READABILITY-GAP-CORRECTION.md`.

## First, a correction to my own correction

That document said the phase-2 doc's "+5.63 from pruning" implied the unpruned readable model scored about **76.1%**, and called
that an unexplained discrepancy. **That inference was mine and it was wrong.** The artifacts say:

```
runs/phase2_sweep/VF0.10_s0/PHASE2-BASELINE.md:
  test pre-prune  81.09%
  test post-prune 81.59%      (the three pruned_final jsons give 82.35 / 81.33 / 81.51, mean 81.73)
```

So pruning is worth about **+0.5 to +0.6 points**, not +5.63, and the "~76.1%" number was my own bad arithmetic on a figure
from the phase-2 document whose original meaning I did not check. There is no missing readable implementation and no
discrepancy: the phase-2 readable model has always been an ~81% model. The ~76.1% figure is withdrawn here and appears in no
document.

## Second, the missing half of the comparison, now measured

`gates/phase2_train.py` does not train a plain control. It compares against `runs/phase1/baseline_summary.json` - a
**784 -> 128 -> 10** net with **101,770 parameters** - while the readable model has **10,173**. Ten times the capacity sat on
the other side of the published comparison.

The control reuses `gates/phase1_fmnist.py` itself (`load_fmnist` and `run_seed`), so the loop, normalisation, shuffle stream,
init stream and evaluation are identical to the Phase-1 baseline. The only change is `--hidden 13`, giving **10,345 parameters
(+1.7% vs 10,173)**, at the same 10 epochs / batch 128 / lr 0.1. Three seeds:

```
seed 0  82.94%      seed 1  84.80%      seed 2  85.38%
MATCHED PLAIN CONTROL: 84.37% +/- 1.04
```

## The decomposition

```
101,770-param plain (Phase-1 baseline)     86.69%
 10,345-param plain (matched control)      84.37%     -2.32 points  <- CAPACITY
  10,173-param readable, post-prune        81.73%     -2.64 points  <- READABILITY
  10,173-param readable, pre-prune         81.09%     -0.64 from pruning
```

**Of the published 4.96-point gap, 2.32 points (47%) is capacity and 2.64 points (53%) is the readable architecture itself.**
The readable model still loses to a plain net of its own size - that part of the published claim was right - but half of the
quoted penalty was never readability's.

## Predictions, scored honestly

- **P1 the matched plain control scores at most 85.5% - HOLDS** at **84.37%**. Shrinking the plain net 10x costs **2.32 points**
  on Fashion-MNIST, so capacity genuinely matters at this scale.
- **P2 the readable model lands within 2 points of the matched control - FAILED** at **-2.64**. This is the prediction I wanted
  to pass, and it did not. My earlier suggestion that "most of the gap is capacity" is therefore **not supported**: capacity is
  47% of it, not most of it.
- **P3 the matched gap stays negative - HOLDS**. Readability costs something at equal size; the measurement says 2.64 points.

Falsifier (matched plain >= 85.5%, which would collapse the capacity story) was **not** triggered.

## What the readable architecture's price actually is

Two readable models now measured against matched plain controls, same data, same schedule, same seeds:

```
rule-less readable (train_image.py, 16,665 params)    -1.53 +/- 0.44
Phase-2 readable with the 32-rule layer (10,173)      -2.64   (vs a 3-seed control, sd 1.04)
```

Both lose. The one with the sparse rule layer loses **about a point more**, which is the price of the 32 rules and 3 connections
per rule - the mechanism that makes `explain()` possible. That is the honest trade the project is making, stated as a
measurement rather than a hope.

## Reproduce

```
python3 tools/matched_plain_p2.py          # writes runs/matched_plain_p2/, never touches gates/runs/phase1/
```
