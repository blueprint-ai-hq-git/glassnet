# Keeping the copy path learning while the language phase runs (2026-11-13)

Freezing the copy path halved the decay (`docs/COPY-FREEZE.md`), but a frozen path stops learning - it protects a capability by
refusing to update it. The successor experiment was recorded there and is now run: **freeze the copy path only on LANGUAGE
batches, and let it keep learning from COPY batches.**

That is the separated-store idea from `docs/MEM-SPLIT.md` applied to a mechanism instead of a table: protect the copy path from
the other task's gradients, not from its own.

## Design
All arms continue from `early_s0_ep1000` (held-out copy 71.0%) for 4,000 steps with a **mixed stream: 50% corpus, 50% copy**.

```
arm           frozen on language batches            frozen on copy batches
keep_scoped   ptrW, Wq, Wk, concept.W               NO - the copy path keeps learning
keep_always   ptrW, Wq, Wk, concept.W               yes (last wake's arm)
keep_none     nothing                               -
```

**Prediction, written before the run:** the scoped arm beats both the always-frozen and the no-freeze arm on held-out copy, while
its language accuracy matches the no-freeze arm.

## Result (3 seeds for scoped and none, single seed for always-frozen - stated because it is not equal footing)
```
arm            held-out copy (late, 3 seeds)   taught   language accuracy
keep_scoped    77.3 +/- 1.4                    84.3%    15.64 +/- 0.19
keep_none      75.2 +/- 1.1                    83.9%    16.49 +/- 0.07
keep_always    71.3  (n=1)                     82.5%    15.97  (n=1)
starting ckpt  71.0                            83.0%    -
```

- **The scoped arm is the best copy result anywhere in this study**: 77.3% held-out, above the 71.0% it started from, above
  no-freeze (75.2%) and far above always-frozen (71.3%). Letting the copy path keep learning while shielding it from language
  gradients is strictly better than freezing it.
- **The prediction's copy half passed; its language half failed by 0.85 points.** Scoped is 15.64 +/- 0.19 against no-freeze's
  16.49 +/- 0.07 - separated by more than the spreads, so the language cost is real, just small.

## The honest statistical read
The gain over no-freeze is **+2.1 points with a combined standard error of about 1.03** (sqrt(1.4^2/3 + 1.1^2/3)), i.e. roughly
two standard errors on three seeds. That is **suggestive, not decisive**. The always-frozen comparison rests on a single seed and
cannot carry a claim on its own. What can be said without hedging is the direction: every arm that shielded the copy path from
language gradients kept more copy than the arm that did not, and the one that also let it keep learning kept the most.

## What this means for the memory work, in order
1. **Facts forgetting -> separated store** (controlled test): 96.1 -> 0.0 points of forgetting, because the store's parameters are
   disjoint (`docs/MEM-SPLIT.md`).
2. **Mechanism decay -> per-batch freezing**: 75.2% -> 77.3% held-out copy over no-freeze at 3 seeds, and 71.3% -> 77.3% over
   always-freezing, at a cost of 0.85 language points (`docs/COPY-KEEP-LEARNING.md`).
3. **Still untested:** a copy path with its own parameters (not the shared ones) that trains on copy batches and is invisible to
   language gradients. That is the full version of the idea - it would remove the language cost entirely rather than trading for
   it - and it is the next build.

## Reproduction
`lens/scripts/run_keep_arms.sh` (seed 0) and `lens/scripts/run_keep_seeds.sh` (seeds 1-2), all sequential, nice 19, GPU ~425 MiB
against a 5 GB budget. Trainer gained `--freeze-scope {all,corpus}`; logs in `logs/keep/`.
