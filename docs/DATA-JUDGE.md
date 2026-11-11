# The data judge: built, gated, and it FAILS its gate (2026-11-11)

Board item A4, "a small data judge model that scores data as good or bad", was listed as designed-only for weeks. It now
exists, trains in four seconds, and **fails its own gate**, which is the useful part.

## What it is
`lens/judge.py` + `lens/scripts/train_judge.py`. Fifteen features per 32-token window, every one recomputable by hand
from the token ids: unknown rate, distinct ratio, symbol rate, digit rate, mean and median token length, tag imbalance,
normalised entropy, longest same-token run, top repeated 3-gram, single-char rate, modal share, plus three order-sensitive
features (mean bigram log-probability, unseen bigram rate, worst bigram) built from **clean training windows only**.

Trained on real windows as GOOD and five kinds of damage as BAD - symbol soup, shuffled order, one-token loops, repeated
chunks, unknown floods - judged on held-out CONVERSATIONS the model never saw.

## Gate 1: no order features
AUC **0.9220**, false alarm **1.25%**, and per corruption kind:

```
symbol soup      100.0%
shuffled           1.0%     <- the judge was blind to destroyed word order
one-token loop   100.0%
chunk repeat     100.0%
unknown flood     98.7%
```

Every feature was a bag-of-tokens statistic, so scrambling order changed nothing. That is exactly the "does it detect
what its label says?" trap, caught by a per-kind gate rather than by a single accuracy number.

## Gate 2: with order features
AUC rose to **0.98** and all five kinds were caught above 99%, but clean windows from held-out conversations were flagged
**29.4%** of the time. The judge had learned to detect DOMAIN, not damage.

## Gate 3: calibrated threshold on one split, reported on a disjoint split
```
calibrated threshold 0.002278  (5th percentile of 1000 clean calibration windows)
calib  n=2000  AUC 0.9687  false-alarm 0.0500  corrupt caught 0.8730
TEST   n=2000  AUC 0.9597  false-alarm 0.0730  corrupt caught 0.8580

per-kind on TEST:  soup 100.0%  shuffled 86.9%  one-token loop 69.5%  chunk repeat 80.0%  unknown flood 92.6%
```

**G1 AUC >= 0.90: PASS (0.9597). G2 false alarm <= 5%: FAIL (0.0730). G3 every kind >= 80%: FAIL (one-token loop
69.5%).** The threshold that hit a 5% budget on one split produced 7.3% on another, which means the score is not
calibrated well enough to hold a promise of "5% false alarms".

## Why it fails, named
The clean windows in a held-out split come from conversations with their own vocabulary and topic mix, so their bigram
statistics sit further from the training distribution than damaged copies of training windows do. The model separates the
two populations it was trained on, and generalises the part that matters less.

## The next design, recorded before it is built
Make the decision RELATIVE rather than absolute: score a window against its own neighbours (its conversation, or a
bootstrap sample drawn from the same source) instead of against a global threshold. Damage is a property of a window
relative to the text around it. Until a relative judge passes all three gates, this one is not wired into training - a
data filter that quietly throws away 7% of good data would cost more than the damage it removes.
