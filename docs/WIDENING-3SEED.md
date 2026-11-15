# The concept-widening gain is real: +0.57 VAL across three paired seeds (2026-11-15)

Predictions were written in `lens/runs/WIDENING-3SEED-PREDICTIONS.txt` before the four confirmation arms ran. The question: the
widened arm beat the 36-concept baseline by +0.56 at seed 0 - was that seed noise? This project has twice seen a width-style verdict
reverse on a later seed, so the pair was retrained at seeds 1 and 2.

## Every seed, both arms

```
seed  arm      final VAL   last-3   held-out best (ep)   final ppl
 0    base36     21.48%    21.41%     23.67% (18)        253.02
 0    wide60     22.04%    22.02%     24.22% (18)        240.55
 1    base36     21.37%    21.41%     23.54% (16)        251.90
 1    wide60     21.98%    21.92%     24.14% (18)        240.91
 2    base36     21.46%    21.48%     23.59% (18)        253.10
 2    wide60     21.99%    21.98%     24.16% (18)        235.37
```

## Paired deltas (wide60 minus base36)

```
seed 0: final +0.56   last-3 +0.61   held-out best +0.55
seed 1: final +0.61   last-3 +0.51   held-out best +0.60
seed 2: final +0.53   last-3 +0.49   held-out best +0.57

final VAL      : mean +0.5667, sd 0.0404
last-3 mean    : mean +0.5378
held-out best  : mean +0.5733
```

**The widened concept set wins on all three seeds and on all three endpoints, with a spread of 0.08 points.** This is the first
seed-robust accuracy win in the readable chat line, and it survived exactly the test that killed the earlier width claim.

## Predictions

- **S1 HOLDS**: mean paired delta +0.5667, above zero.
- **S2 HOLDS**: the widened arm wins on **3 of 3** seeds, not 2.
- **S3 HOLDS**: the widened arms keep 53 concepts, above the 45 floor, against 36 for the baseline.
- **FALSIFIER NOT TRIGGERED**: the mean delta is positive and the widened arm never lost a seed.

## What is established, and what is not

- **Established**: adding 17 measured concepts (36 -> 53 kept) improves final validation accuracy by **+0.57 +/- 0.04** across three
  paired seeds, improves held-out best by **+0.57**, and improves validation perplexity on every seed (235.37-240.91 vs 251.90-253.10).
  It also raises the share of decisions that flow through named concepts from 83.4% to 86.9%.
- **Not established**: that widening fixes vocabulary breadth. It bought only **+8 distinct tokens (78 -> 86, +10%)** for +47% more
  concepts, which is a sharply diminishing return. On that axis the widening is a poor lever and the honest reading is that named
  concepts are a route to accuracy, not to breadth.
- **Still open**: concept-name honesty remains the weakest measured property at **0.5884** on held-out synonyms (0.60 was the target,
  missed by 0.0116), with 11 of 22 concepts below 0.60.
