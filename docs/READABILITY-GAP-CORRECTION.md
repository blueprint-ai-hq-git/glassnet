# Correction: the "readable model is 4.96 points worse" headline compares a 24k model to a 102k model

Date: 2026-11-14. Sources: `docs/history/PHASE2-IMAGE-GATE.md` (the published claim), `runs/phase1/BASELINE.md` (the plain
baseline), and the 8-arm sweep in `docs/history/READABILITY-SWEEP.md` (this run's matched measurement). Same dataset throughout:
Fashion-MNIST, MIT.

## The defect

`docs/history/PHASE2-IMAGE-GATE.md` states:

```
readable model, after rule pruning      81.73% +/- 0.54   (82.35 / 81.33 / 81.51 across three seeds)
PHASE 1 BASELINE (plain MLP, no rules)  86.69% +/- 0.59
**The readable model does not beat the Phase 1 baseline: it is 4.96 points below it.**
```

Both numbers are correct. **The comparison is not matched.** Parameter arithmetic, done here rather than recalled:

```
plain Phase-1 baseline  784 -> 128 -> 10    100,480 + 1,290 = 101,770 params, 10 epochs, lr 0.1
readable model (this config)                            24,458 params, 24 epochs
ratio                                                    4.2x bigger plain model
```

So the headline sentence - "the readable model does not beat the baseline" - attributes to **readability** a gap that is at
least partly **size**. A 24k-parameter readable model scoring 81.73 against a 102k-parameter plain model scoring 86.69 is not a
measurement of what readability costs; it is a measurement of what 4.2x more parameters buys.

## What a matched comparison measures

Every arm of the 8-arm sweep prints READABLE and PLAIN at **matched parameter counts inside the same run** (A: readable
24,458 vs plain 24,467, a 0.04% difference), same data, same epochs, same seed:

```
arm                     gap, 2 seeds      concept honesty
shipped  d64 patch 7    -2.06 / -1.54     100.0% / 92.3%
wide     d96 patch 7    -2.23 / -1.91     100.0% / 100.0%
softlam  lam 0.15       -1.63 / -1.36      84.6% / 92.3%
finer    patch 4        -3.81 / -3.57      (concept granularity, not the shipped layout)
```

**At matched parameters the readable architecture costs 1.4 to 2.2 points, not 4.96.** The direction is the same - readable is
still behind - but the size of the penalty the project has been quoting is roughly double what matching supports.

## What is still unexplained, and is not being papered over

The phase-2 document also says pruning is worth **+5.63** to reach 81.73, which implies its unpruned readable model scored about
**76.1%**. The model in `lens/scripts/train_image.py` scores **84.36%** unpruned on the shipped configuration. Those cannot both
describe the same readable variant:

```
phase-2 readable, unpruned (implied by +5.63)   ~76.1%
train_image.py readable, unpruned (measured)     84.36%
```

So the phase-2 sweep and the current trainer are **not the same readable implementation** - they differ in at least concept count
(12 vs 13 kept) and the pruning pipeline, and possibly in the sweep variant (VF0.10). Until that difference is located, the honest
statement is:

- the 4.96 headline is an **unmatched** comparison and cannot be read as the cost of readability;
- a matched comparison of the current implementation puts that cost at **1.4-2.2 points**;
- the ~76% implied by the phase-2 document is **not reproduced** by the current trainer and remains an open discrepancy, not a
  number to quote.

## The next experiment this implies

Run the phase-2 pruned configuration (`VF0.10` sweep, 3 seeds, pruning applied) with the same matched plain control inside the
same run, so one run answers both questions at once: what pruning does, and what readability costs at equal size. Predictions
for it will be written into the script before it runs, as always.

## Confirmed at three seeds: the matched gap is -1.53 +/- 0.44 points

Runner `lens/scripts/run_matched_gap_3seed.sh`, shipped configuration (d 64, patch 7, lam-concept 0.3, 24 epochs, batch 256,
lr 0.3, `--extra none`), Fashion-MNIST, each seed training its own PLAIN control at a matched parameter count in the same run
(readable 16,665 vs plain 16,705, 0.24% apart):

```
seed   readable    plain     gap      concept honesty
0      84.36%      86.42%    -2.06     100.0%  (13/13)
1      84.27%      85.81%    -1.54      92.3%  (12/13)
2      83.69%      84.67%    -0.98     100.0%  (13/13)

MEAN MATCHED GAP   -1.53 points   sd 0.44
```

Predictions, written before the run, all three **HOLD**:
- **P1** mean gap between -1.0 and -2.6 -> **-1.53 HOLDS**;
- **P2** mean above -3.0, i.e. materially smaller than the published 4.96 -> **-1.53 HOLDS**, so the correction stands and the
  size mismatch is the explanation;
- **P3** concept honesty at or above 85% mean -> **97.4% HOLDS**.

**So the honest headline for the readable image model is: 1.53 +/- 0.44 points behind a plain MLP of the same size, with 97.4%
concept honesty.** Not 4.96. The readable architecture still does not win, but the penalty is roughly a third of what the
project has been quoting, and it now has a standard deviation attached.

## Third harness bug found and fixed in this run

The trainer prepends its own `lens/runs/image/` to `--out`, so passing an already-prefixed path writes to a **nested**
`lens/runs/image/lens/runs/image/` directory. Every arm whose numbers appear above therefore "disappeared" from the summary
that globbed the flat directory and reported **"NO RESULTS - every arm failed"** while three arms had in fact succeeded. The
files were recovered by copying (never moving) to `lens/runs/image/recovered/`, and the runner now passes a bare output name.
That is the third time today that my own harness hid a real result - the grep that filtered a traceback, the 3-tuple unpack,
and now a path prefix. Each one reported success or emptiness instead of the truth.
