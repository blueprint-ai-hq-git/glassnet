

---

# THE DIAGNOSIS SURVIVES THE INSTRUMENT FIX: EVERYTHING SELECTS AT 100%, ARITHMETIC STILL FAILS AT 2% (2026-11-21 14:30)

The batch-leak fix invalidated every probe number of the day, so the ladder was re-run on the corrected model with the prediction written
first. **The prediction held, and it is stronger than before.**

```
rung        what it requires                                     leaky instrument   FIXED instrument
copy1       find the ONE digit among 7 decoys, copy it                 98.0%            100.0%
copylast    digit at the LAST position, copy it                       100.0%            100.0%
copyfirst   the same digit at position 0 (7 decoys after it)        80.0/100.0%         100.0%
max2        find TWO digits among 8, output the larger                 92.0%            100.0%
sum2fixed   two digits handed at positions 0 and 1, ADD                 2.0%              2.0%
sum2        find two digits, ADD mod 10                                 8.0%           (running)
```

- **P1, P2, P3, P4 all MET:** with a clean instrument, every selection and comparison task is **100.0%**.
- **The arithmetic rung is unchanged at 2.0%** - below the 10% chance line, on a model that scores 100% on every other rung.
- **The falsifier (either arithmetic rung >= 80%) did not fire.**

## Why this is stronger evidence than before the fix

Before the fix, someone could reasonably object that the model was handicapped by a broken instrument. It was not - it was helped, since
the leak gave it up to 256 tokens of context instead of 8. On the fixed instrument it does **better on every non-arithmetic rung (four at
100.0%)** and **exactly as badly on arithmetic (2.0%)**. The contrast is now clean in both directions:

> this model can select, copy, and compare perfectly, and cannot add.

That is a precise, reproducible statement about the architecture, and it is the reason the missing layer is a **carrier for computation**
rather than memory, attention, or more data.

## L3 salience: FAILED as specified, on nine arms

With the better-trained head (1500 steps on 3,200 points) the certificate was re-run on copy1:

```
seed   AUROC    free norm   margin    Spearman   probe acc   verdict
 0     0.7599    0.7032    +0.0567    0.3786      94.0%      SALIENCE_FAILED
 1     0.8058    0.6351    +0.1707    0.3425     100.0%      SALIENCE_FAILED
 2     0.7755    0.6062    +0.1693    0.3793     100.0%      SALIENCE_FAILED
```

- **P1 (AUROC >= 0.90) missed on all three seeds.** P2 is met consistently - the layer does carry information the hidden norm does not.
- **A better-fit head made the margin bigger and the ranking worse** (+0.17/+0.17 vs +0.014/+0.305) while the free baseline's own AUROC
  fell to 0.61-0.70. So part of the "bigger margin" is a weaker baseline, not a better layer, and the absolute numbers are what the gate
  is judged on.
- Across nine arms over three attempts, L3 has never reliably cleared its own bar. **L3 is FAILED as specified.**

One honest caveat, stated because it limits the claim: all nine of those arms ran on the **leaky** instrument. A re-run on the fixed
instrument is running now. But the outcome is unlikely to be saved by it - the leak could only have *helped* the layer, so a layer that
missed 0.90 while leaking is unlikely to clear it once the extra context is removed.
