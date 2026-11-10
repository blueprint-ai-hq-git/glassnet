# Correction: the code tail is the model's BEST region, not a handicap (2026-11-10)

An hour ago I published `DATA-SPLIT-SHIFT.md`, which established that the validation tail is Python code while
the training corpus is mostly conversation, and said that this "partly explains the near-baseline held-out
accuracy" because the held-out tokens come from text the model was never shown anything like.

I then measured accuracy by region (`tools/matched_eval.py`, 300 samples per region, `gpu_d512_s0_best.npz`).
The direction is the opposite of what I implied:

```
frequency baseline (always predict ',') over the training region: 4.23%

region                          acc      ppl   tgt sym%  pred sym%
A. first quarter (seen)      24.33%    269.9      24.3%      47.3%
A. middle (seen)             17.33%    635.4      22.3%      49.3%
A. last quarter before tail  25.00%    353.7      36.0%      59.3%
B. held-out code tail        33.00%    107.6      47.7%      71.3%
```

## The correction
The held-out code tail is where this model performs **best** - 33.00% accuracy and perplexity 107.6, against
17-25% accuracy and perplexity 270-635 in the conversational regions it was trained on.

So the split mismatch does not depress the published headline; it **flatters** it. Code is more predictable
than conversation, and the model gets both its best accuracy and its lowest perplexity there. The
compositional difference is real and still worth fixing, but my stated direction was wrong and the numbers
here are the correction.

## What survives from the earlier document
- The split IS compositionally different: 13.29% symbols in the first 200k, 44.04% in the held-out tail.
- The symbol attractor IS partly the code tail: 47.7% of targets there are symbols, the model predicts symbols
  71.3% of the time.
- Per-source reporting IS needed, because a single average hides a 17% - 33% spread.
- What does NOT survive is the implication that the tail is a harder, unfair test. It is an easier one.

## What this makes sharper
The talk core scored **4.69%** on this same easy region, against a 3.99% baseline, while the readable
architecture scores **33.00%** on it. Measured on identical ground, the recurrent core is 7x worse than the
readable model, which strengthens the earlier conclusion that talk should be built on the concept/answer
family rather than the recurrent core.

## Standing correction rule
A directional claim made before the measurement is not allowed to stand once the measurement disagrees with
it. This document exists because the first claim was published and then contradicted; leaving only the first
version on the record would be the dishonest option.
