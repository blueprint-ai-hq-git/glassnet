

---

# THE AGE KEY IS DECISIVE ON THE SHARP INSTRUMENT: 83.20% AGAINST 19.82% AT DELAY 8 (2026-11-19 15:45)

The first measurement all day that is both large and on the right instrument. Delayed recall: remember one token from position 0 across D filler steps and
output it. Chance is 3.12% (vocab 32). Both arms are the same readable window bank, same data stream, same frozen feature extractor, same parameters apart
from `age_emb` (**K*H = 1024 params**).

```
delay   agekey      plain        diff        (chance 3.12%)
1       100.00%     100.00%      +0.00
2        99.90%      99.71%      +0.20
4        95.21%      56.93%     +38.28
8        83.20%      19.82%     +63.38
16       running
```

- **P2 MET:** the plain arm decays to **19.82%** by delay 8 - content-similarity addressing works while the window is short and collapses as it fills.
- **P3 MET:** the age-key arm clears 50% at delay 8 by a wide margin (**83.20%**).
- **FALSIFIER DID NOT FIRE:** it required agekey to beat plain by 5 points at delay 8; the measured gain is **+63.38**.
- **P1 was corrected by the smoke run before the full run:** both arms solve delay 1 at 100% (the window holds two contents, and the mean of two token
  embeddings still identifies one 100% of the time at K=2 - measured). Not a leak: delay 4 shows the decay.

## What this confirms

The defect diagnosed at 15:10 was real and it was **the** limiter: the read had no position or age term, so slot selection depended entirely on content
similarity, and a full window read as a blend whose nearest real token is one of the K only 5% of the time at K=16. Adding a learned per-age key - a
**1024-parameter** addition, 0.06% of a 1.68M model - takes delay-8 recall from 19.82% to 83.20%.

This also retro-explains the whole memory line: the earlier slot bank and the window bank both failed to show benefit at 16 steps, because neither could
address a slot by age. The mechanism was never the problem; the missing key was.

## What this does NOT establish, stated before anyone reads it as more

- **It is a synthetic probe with a frozen feature extractor**, a 32-token vocabulary, and training that flows to the embeddings through the bank contents
  only. It says nothing yet about next-token accuracy on real text.
- **It is one seed.** The seed-noise measurement earlier today (sd 2.89 on the 3000-step accuracy metric) applies to that metric, not this probe's, but a
  single seed is still a single seed and the probe becomes a result at 3 seeds.
- **No end-to-end arm has been trained with the age key.** That is the next step, and it must be scored on both instruments with the falsifier written first.

## Next

1. Three seeds of this probe (cheap: the whole 5-delay pair runs in minutes).
2. An end-to-end arm at the qualifying budget (6000 steps) with `age_emb` on the window bank, scored on the trainer line AND the independent evaluator,
   with `BEATS/BELOW constant` printed by the trainer itself.
