# Concept-set size is not the lever; concept pressure is (2026-11-12)

The best honest configuration on the original task was 12 de-duplicated concepts at alpha 3: **70.00% +/- 3.54** with **0.7 of
12** concepts misnamed. Its obvious weakness was capacity - only 12 concepts, and an accuracy bill of 3.65 points against the
alpha 1 arm. So the next single change was to give that configuration more concepts: **K=18** selected from the same 26-feature
pool at a looser diversity limit (0.85 instead of 0.75).

**Prediction, written before the run:** at alpha 3, 18 concepts recovers >=3 accuracy points while keeping misnamed <=2 of 18.
**It failed on both counts.**

```
concepts  alpha  accuracy (3 seeds)   mean |rho|   misnamed          median per-rule   dead
   12      1.0    79.88 +/- 0.32        0.478      7.0 of 12 (58%)     +0.26            1.0
   12      3.0    70.00 +/- 3.54        0.698      0.7 of 12 ( 6%)     +0.71            1.7
   18      1.0    79.68 +/- 0.58        0.485     11.0 of 18 (61%)     +0.65            1.0
   18      3.0    72.17 +/- 6.59        0.684      5.7 of 18 (32%)     +0.81            2.0
```

- **Accuracy**: 79.68% at alpha 1 against 79.88% for 12 concepts - no gain, slightly worse. At alpha 3 it went to 72.17%, which
  is +2.17 points but well short of the +3 predicted, and the seed spread exploded from +/- 3.54 to **+/- 6.59** (individual
  seeds 75.35 / 76.67 / 64.48). Trading a stable 70.00 for a volatile 72.17 is not a win.
- **Honesty**: misnamed went to **5.7 of 18**, not <=2. Per-concept rank agreement is unchanged (0.684 against 0.698). More
  concepts bought nothing in honesty terms.
- **The useful part is the RATE.** At low concept pressure, **58% of concepts are misnamed with 12 concepts and 61% with 18** -
  the same fraction at either set size. The failure is not that the set is too small or too large; it is that the training
  objective does not make concept neurons match their names unless the concept loss is weighted up. Raising the weight fixes
  it (7.0 -> 0.7 misnamed at K=12) and raising the set size does not.

This is the third independent check pointing at the same conclusion: the concept set keeps getting blamed for a property that
belongs to the training objective. The set determines *what could* be named; the objective determines *whether it is*.

## What stands after this run
```
task                        best accuracy          best honesty
original 14 concepts        77.93 +/- 1.40 (a1)    2.3 of 14 misnamed (a3, 73.65%)
de-duplicated 12            79.88 +/- 0.32 (a1)    0.7 of 12 misnamed (a3, 70.00%)  <- both bests
de-duplicated 18            79.68 +/- 0.58 (a1)    5.7 of 18 misnamed (a3, 72.17%)
```
**The 12-concept set holds both records** and is the one to ship in either mode. The 18-concept arm is kept as the negative
result it is.

## Reproduction
`tools/select_concepts.py 18 0.85` then the trainer with `GLASSNET_CONCEPTS=$PWD/data/concepts_selected.npz --K 18 --R 16
--alpha {1.0,3.0} --tag SEL18_A*_s*`, audited by `tools/phase2_reaudit.py`. The 12-concept cache is preserved as
`data/concepts_sel12.npz` / `.json` (never overwritten). Six more checkpoints under `runs/phase2_sweep/SEL18_A*_s*/`.
