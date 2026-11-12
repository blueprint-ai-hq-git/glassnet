# A de-duplicated concept set on Fashion-MNIST: best readable accuracy so far, but honesty needs pressure (2026-11-12)

The re-audit found the original concept set has near-duplicate NAMES (`ink_amount`/`quad_LL_ink` at 0.926,
`ink_amount`/`center_ink` at 0.922) and 4-6 of 14 neurons better explained by a different name. This builds a wider candidate
pool of 26 measurable pixel features and selects a set that is both learnable and mutually distinguishable, then measures what
that buys.

## How the set was chosen (train split only, test never looked at)
1. **Learnability**: a ridge probe from the 784 pixels to each feature, scored by rank agreement on a held-out slice of train.
   A concept a linear read of the pixels cannot recover is a concept no concept neuron can honestly claim.
2. **Diversity**: greedy selection by probe score, refusing any candidate correlating |Spearman| >= 0.75 with one already
   chosen - two names a human cannot tell apart cannot be audited separately.

The diversity ceiling turned out to be the binding constraint:

```
|spearman| limit    concepts available from the 26-candidate pool
     0.60                       6
     0.70                      11
     0.75                      12   <- chosen
     0.80                      15
     0.85                      18
```

One candidate failed learnability outright: `bright_core` (probe **-0.004**, i.e. not linearly recoverable at all) and is
excluded rather than shipped with a name it cannot earn.

The 12 selected, with probe scores: `ink_amount` +1.000, `top_heavy` +1.000, `left_heavy` +1.000, `ring_ink` +1.000,
`col_ink_std` +0.988, `corner_ink` +0.953, `row_ink_std` +0.951, `grad_ratio` +0.905, `grad_x_mean` +0.828, `symmetry_lr`
+0.819, `symmetry_ud` +0.777, `edge_balance` +0.669. Every one has a plain-English definition in
`glassnet/concept_pool.py`.

## Result at 16 rules, three seeds each
```
| alpha | accuracy (3 seeds) | mean |rho| | misnamed concepts | median per-rule | dead | interaction |
|---|---|---|---|---|---|---|
| 1.0 | 79.88 +/- 0.31 | 0.478 | 7.0 of 12 | +0.35 | 1.0 | 9.5% |
| 3.0 | 70.00 +/- 3.41 | 0.698 | 0.7 of 12 | +1.11 | 1.7 | 7.8% |
```

## Against the original 14-concept set at the same 16 rules
```
setting                          accuracy           mean |rho|   misnamed   dead rules
original 14, alpha 1            77.93 +/- 1.40        0.587      4.7 / 14      0.7
de-duplicated 12, alpha 1       79.88 +/- 0.32        0.478      7.0 / 12      1.0
original 14, alpha 3            73.65 +/- 3.04        0.759      2.3 / 14      2.0
de-duplicated 12, alpha 3       70.00 +/- 3.54        0.698      0.7 / 12      1.7
```

**The accuracy gain is real**: 79.88% +/- 0.32 is the best three-seed readable-architecture accuracy measured on this task - it
beats the original gate's 30-rule 79.04% +/- 2.38 by 0.84 points **with 16 rules instead of 30 and a spread seven times
tighter** (0.32 against 2.38).

**My prediction failed.** I predicted that removing near-duplicate names would reduce misnaming. At alpha 1 it got **worse**:
7.0 of 12 misnamed against 4.7 of 14, and mean rank honesty fell 0.587 -> 0.478. Removing confusable names did not stop the
neurons from learning something else - consistent with the earlier finding that misnaming is not label confusion.

**But the de-duplicated set responds far better to concept pressure.** At alpha 3 it reaches **0.7 of 12 misnamed** - the
lowest misnaming of any configuration measured on this task - against 2.3 of 14 for the original set at the same weight, at a
cost of 3.65 accuracy points.

## What I would ship from these numbers
- **If accuracy is the goal**: de-duplicated 12 concepts, 16 rules, alpha 1 - 79.88% +/- 0.32. Say plainly that 7 of its 12
  concept names do not describe their neurons.
- **If readable honesty is the goal**: de-duplicated 12 concepts, 16 rules, alpha 3 - 70.00% +/- 3.54 with 0.7 of 12
  misnamed, and every rule worth +0.71 on median.
- Neither is a free win. On this task accuracy and honest names still pull against each other, and the de-duplicated set makes
  both ends better but does not remove the trade.

## Reproduction
`tools/select_concepts.py <K> <limit>` (writes `data/concepts_selected.npz` + `.json`), then the trainer with
`GLASSNET_CONCEPTS=$PWD/data/concepts_selected.npz --K 12 --R 16 --alpha <A> --tag <name>`, audited by
`tools/phase2_reaudit.py` with the same environment variable set so the honesty labels come from the same pool as the training
targets. Six checkpoints kept under `runs/phase2_sweep/SEL12_A*_s*/`; `runs/phase2/` untouched.
