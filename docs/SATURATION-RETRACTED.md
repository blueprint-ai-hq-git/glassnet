
## RETRACTION: the saturation numbers were wrong. The readout is not one-hot (2026-11-17)

Two documents published today - `SATURATION.md` and `SATURATION-FROM-START.md` - claimed the readout is one-hot: **entropy 0.0079 nats = 0.08% of uniform**, top-1 probability effectively 1,
"born saturated". **Those numbers are wrong and the conclusion is withdrawn.**

## How the contradiction surfaced

A temperature sweep produced two numbers that cannot coexist: entropy **0.0079 nats** (which implies one-hot) beside **mean top-1 probability 0.00%** (which implies the opposite, and is
impossible for a softmax - the top probability can never be below 1/12002).

`tools/temp_debug.py` recomputed the same quantities by direct normalisation and **asserted the distribution sums to 1**:

```
logits: min -5.253  max 11.750  mean 0.011   row range mean 11.89
T=1.0: top1 prob mean 0.110566  entropy 6.954274 nats   sum(pr) = 1.000000000000
T=4.0: top1 prob mean 0.000608  entropy 9.353526 nats   sum(pr) = 1.000000000000

same rows, my sweep's expression: entropy 0.009641  top1 0.000029     <- wrong
```

**The correct numbers: entropy 6.954 nats (74% of uniform), top-1 probability 0.1106.** That is an ordinary, moderately-peaked distribution - not one-hot, not saturated, and perfectly
consistent with 17.97% next-token accuracy.

## What is withdrawn

- "entropy 0.0079 nats = 0.08% of uniform" - **wrong**, correct value is **6.954 nats = 74% of uniform**.
- "the softmax is effectively one-hot" - **wrong**.
- "the model is confident and wrong 83% of the time" - **wrong**; mean top-1 confidence is 0.11, so it is *un*confident and usually wrong, which is a completely different condition.
- "the readout is born saturated / saturation present from ep250" - **withdrawn**: it rested entirely on the bad entropy figure. The logit-range measurement (11.6 -> 12.1) is unaffected and
  still true, but a logit range of ~12 with these weights produces a top probability of 0.11, not 0.99, so it never implied saturation.
- "the looping generation is explained by a one-hot distribution" - **withdrawn**; that explanation is gone.

## What survives from the last two entries

- Logit range across checkpoints: **11.6 -> 12.1**, a factor of 1.05, while accuracy rises 7.10% -> 17.97%. Unaffected - a direct measurement of logits.
- Accuracy **17.97%** at ep1500, entropy on real text **6.954 nats**, top-1 **0.1106**.
- Temperature behaviour: raising T *does* raise entropy monotonically (9.353 nats at T=4) with accuracy untouched, which is a true and unremarkable property of any softmax.

## The failure mode, named

**Two of my own implementations of the same quantity disagreed by a factor of 700, and I published the figure from the one I had not verified.** The check that exposed it took one line -
*does the distribution sum to 1?* - and it was not in either script.

Standing rule from here: **any entropy or probability figure must print `sum(probs)` beside it**, so a broken normalisation cannot pass silently. That is the same instrument-first rule the
project already applies to gradients and gates, applied to the one quantity I had been quoting without it.
