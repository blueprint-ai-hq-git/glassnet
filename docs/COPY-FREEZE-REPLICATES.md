# The copy-freeze win replicates from a different starting checkpoint (2026-11-13)

`docs/COPY-FREEZE-SEEDS5.md` measured the per-batch copy freeze at **+1.98 points** of held-out copy over five seeds (paired,
t = 6.2, 5 of 5 seeds positive). Every one of those arms started from `early_s0_ep1000`, so the result could have been a property of
that one starting point rather than of the method. This repeats the pair from `early_s0_ep2000` - a weaker start, already decayed
to 64.3 percent held-out copy.

## Result - the falsifier did not trigger
```
seed   keep_scoped   keep_none   difference
0         72.0         69.2         +2.9
1         72.4         70.5         +1.9
2         72.7         71.4         +1.3
mean   72.37          70.37        +2.00
```

Paired on seed: mean +2.00, sd 0.75, paired SE 0.44, **t = 4.6**, positive on **3 of 3** seeds.

**+2.00 points from a different starting checkpoint, positive on every seed** - the same effect and the same size from a start that
begins about 7 points lower. The pre-registered falsifier was "if the paired gap falls below 1.0 point from this starting point, the
effect is specific to ep1000". It did not: the gap is **2.00**.

The two experiments together:
```
starting checkpoint     paired gap   seeds positive
early_s0_ep1000          +1.98        5 of 5   (docs/COPY-FREEZE-SEEDS5.md)
early_s0_ep2000          +2.00        3 of 3   (this doc)
```

## What this does and does not establish
- **Establishes:** freezing the copy path on language batches, so it keeps learning from copy batches, protects held-out copy by
  about **2 points**, and the effect does not depend on the checkpoint it was discovered on.
- **Does not establish:** that it is free. The language cost over five seeds is **-0.73 points** (15.56 against 16.30), unchanged by
  this. The successor meant to remove that trade - a copy path with its **own** pointer matrices, invisible to language gradients -
  is built, its mechanism verified at **0.000e+00** drift, and its arms are running.

## Reproduction
`lens/scripts/run_keep3_startpoint.sh` (prediction and falsifier in the header), logs `logs/keep3/{scoped,none}_s{0,1,2}.log`,
checkpoints `k2_scoped_s*` / `k2_none_s*` in `lens/runs/chat/`. Late held-out copy = mean of the last three evaluations.
