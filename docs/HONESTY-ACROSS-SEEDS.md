# Concept honesty across 8 seeds: one unreliable concept, no others (2026-11-18)

The audit that found a dishonest concept was run on **one** checkpoint. A single seed is not evidence that a concept is dishonestly learned - it may be an unlucky
initialisation. So the same audit was run on **all eight** parity checkpoints, 4,000 held-out images each (`tools/honesty_all_seeds.py`).

```
seed       pass    centre_heavier AUROC   failing concepts
0         20/21          0.2889         centre_heavier(0.29)
1         21/21          0.6336         none
2         20/21          0.5095         centre_heavier(0.51)
3         20/21          0.2903         centre_heavier(0.29)
4         21/21          0.5999         none
5         21/21          0.7741         none
6         20/21          0.4643         centre_heavier(0.46)
7         20/21          0.4282         centre_heavier(0.43)

failures counted across the 8 seeds:
  centre_heavier   in 5 of 8 seeds
```

## What this establishes

1. **`centre_heavier` is the only concept that ever fails** - across eight seeds and 21 concepts (168 concept-seed pairs), no other concept fails in any seed. The other 20 are
   solid.
2. **It is unreliable rather than always inverted.** It fails in **5 of 8** seeds and passes in 3, with AUROC ranging from **0.2889 to 0.7741** - a spread of 0.49. At seed 5 it
   learns something real (0.77); at seeds 0 and 3 it inverts (0.29).
3. **A concept whose honesty depends on the seed cannot be shipped as readable**, whatever its average is.

## A correction to my own earlier claim

The first report said `centre_heavier` scores **AUROC 0.3196 - below chance - so the neuron fires more often where the claim is false.** That was measured at **seed 0** and stated
as if it were a property of the concept. Across eight seeds it is true at seeds 0 and 3, and false at seeds 1, 4 and 5 where the concept passes. The honest statement is
**unreliable, seed-dependent**, not "always inverted". The stronger claim came from one checkpoint, which is precisely the mistake the three-seed rule exists to prevent, and it
was caught by applying that rule to my own result rather than only to experiments.

## The fix, now tested at three seeds

Same formula, sharper claim: threshold **0.05 -> 0.40** (~1 sd above the training mean 0.2993), support **96.9% -> 23.45%**, still 21 concepts at the 0.002 floor.
`lens/scripts/run_fix_centre.sh` was rewritten to run **three seeds** (it had not started training, so nothing was lost) and to re-audit each one:

- **P1** `centre_heavier` passes in **3 of 3** seeds, against 3 of 8 at the old threshold
- **P2** its AUROC range across the three seeds is **below 0.25** (it was 0.49)
- **P3** overall accuracy stays within **0.5 points** of 84.62%
- **FALSIFIER** it still fails in **2 or more** of the 3 seeds: the threshold is not the problem and the concept is **dropped** per doctrine rather than kept as a name whose meaning
  depends on the initialisation.
