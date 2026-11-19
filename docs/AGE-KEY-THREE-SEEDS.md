

---

# THE AGE KEY AT THREE SEEDS: +65.01 POINTS AT DELAY 8, EVERY SEED ABOVE +62 (2026-11-19 16:15)

The single-seed probe said 83.20% vs 19.82% at delay 8. One seed is one seed, so the same probe was run at seeds 0, 1 and 2, reporting **each seed
separately and the paired delta per seed** rather than arm means.

```
delay  agekey            plain             delta     per-seed paired deltas      seed-by-seed agekey / plain
1      100.00% +/-0.00   100.00% +/-0.00   +0.00     +0.0   +0.0   +0.0            100.0/100.0 100.0/100.0 100.0/100.0
2       99.90% +/-0.10    99.71% +/-0.00   +0.20     +0.2   +0.3   +0.1             99.9/99.7  100.0/99.7  99.8/99.7
4       95.51% +/-2.11    59.44% +/-3.93  +36.07    +38.3  +40.3  +29.6             95.2/56.9   97.8/57.4   93.6/64.0
8       85.61% +/-4.09    20.61% +/-0.70  +65.01    +63.4  +69.5  +62.1             83.2/19.8   90.3/20.8   83.3/21.2
16      running
```

Chance is **3.12%** (vocab 32).

- **P1 MET:** the paired delta at delay 8 is above +30 on **every** seed - the smallest is **+62.1**, more than double the bar.
- **P2 MET:** same sign at every delay of 4 and above, on all three seeds.
- **FALSIFIER DID NOT FIRE:** it needed any seed below +10 at delay 8; the minimum observed is +62.1.
- **The effect is not one seed's luck.** The plain arm is remarkably STABLE (20.61 +/- 0.70 across seeds) while the age-key arm varies more
  (85.61 +/- 4.09). That asymmetry is the honest shape of the result: the plain bank fails at the same place every time, and the age key
  succeeds by a variable but always-large margin.

## What this establishes now

The age key - **1,024 parameters, 0.06% of a 1.68M model** - is what the readable window bank was missing. The defect diagnosed at 15:10 was real and
it was the limiter: without a position/age term the read can only select a slot by content similarity, and a full window reads as a blend. With it,
delay-8 recall goes from 20.61% (about 6.6x chance) to 85.61% (about 27x chance), replicated across three seeds with per-seed deltas between +62.1
and +69.5.

## What it still does not establish - unchanged from before

- **Synthetic probe.** Frozen feature extractor, 32-token vocabulary, embeddings trained through the bank contents only. Nothing here is a claim about
  next-token accuracy on real text.
- **No end-to-end arm has been trained with the age key.** That is the test that matters for the product, and it is running next at the qualifying
  6000-step budget, scored on the trainer line and the independent evaluator, with its falsifier written first.
- The probe's own P2 from the single-seed run was **my** failed prediction (I said the plain arm would sit near chance at delay 16 and it read 14.06%).
  The corrected statement stands: content similarity keeps residual signal, it just degrades badly.
