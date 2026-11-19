

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
16      46.68% +/-14.78   13.44% +/-0.64  +33.24    +43.2  +16.3  +40.2             57.2/14.1   29.8/13.5   53.0/12.8
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


## Delay 16 is the boundary case, and the single seed flattered it

```
delay 16:  agekey 46.68% +/-14.78 (seeds 57.2 / 29.8 / 53.0)   plain 13.44% +/-0.64 (14.1 / 13.5 / 12.8)
           paired deltas +43.2 / +16.3 / +40.2
```

Three facts worth recording:

1. **The advantage survives at delay 16 on every seed** (+16.3 to +43.2), so the sign is not in question - but the SIZE is seed-sensitive in a way delay 8 is
   not (delay 8: sd 4.09 on the arm; delay 16: sd 14.78).
2. **This is exactly the boundary of the buffer.** Slot `j = t % K` is written AFTER the read at each step, so at `t = K = 16` the read still sees the fact
   that step 0 wrote, and the very next step overwrites it. Delay 16 is the last delay the architecture can serve at K=16, which is precisely where a small
   difference in how far training pushed the age key decides success or failure - hence the spread.
3. **The single-seed number OVERSTATED the typical case.** The one-seed run read 57.23%; the three-seed mean is 46.68%, because that seed happened to be the
   best of the three (57.2 / 29.8 / 53.0). This is the clearest local example of why the three-seed rule exists, on my own result, in the same day.

So the durable claim is at delay 8 (+65.01, per-seed deltas +62.1 to +69.5, sd 4.09); delay 16 is **positive but variable at the buffer's edge**.
