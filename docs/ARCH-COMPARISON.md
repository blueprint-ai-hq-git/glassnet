# The readable architecture has real signal; the recurrent talk core does not (2026-11-10)

Two different architectures, same corpus, same 12,002-token vocabulary, same held-out tail, both measured
today against the same frequency baseline of **3.99%** (`docs/MARGINAL-BASELINE.md`).

## The readable concept/answer model (d=512, published as 37.37% held-out)
Re-measured in this session with `lens/scripts/pos_accuracy.py`, 200 samples per context length, held-out
range 293,725,090..294,225,026:

```
  ctx   acc      ppl      vs random
    1    11.00%   1348.02   1320.2x chance
    2    21.00%    613.55   2520.4x chance
    4    22.50%    428.46   2700.4x chance
    8    23.50%    825.69   2820.5x chance
   16    25.00%    307.77   3000.5x chance
   24    31.50%    253.34   3780.6x chance
   32    25.50%    167.37   3060.5x chance

VERDICT: context is used (accuracy rises with length)
```

- With only **one** token of context it scores **11.00%**, which is already **2.8x** the 3.99% frequency
  baseline. It is not a bigram lookup, but it is well past the base rate even there.
- It rises to **31.50%** at 24 tokens of context. The 32-token point reading lower (25.50%) is within the
  sampling noise of 200 samples per length and is not evidence of a decline.

## The recurrent talk core (runs 5-10, `scan8_ep1500`)
```
model held-out accuracy : 4.69%
frequency baseline      : 3.99%
advantage               : +0.70 pp  (1.18x the baseline)
```

## The comparison that matters
| model | accuracy | frequency baseline | ratio |
|---|---:|---:|---:|
| readable concept/answer, d=512 | 11.00% at 1 token, 31.50% at 24 tokens | 3.99% | 2.8x .. 7.9x |
| recurrent talk core (run 8) | 4.69% | 3.99% | 1.18x |

The readable architecture learns real conditional structure. The recurrent talk core, with four verified
hand-written gradient paths, decorrelation, a direct path and its own readout normalisation, is within noise
of guessing `,`. That is not a tuning difference, it is a difference in whether anything is being learned.

## What follows
1. **Talk should be built on the concept/answer family, not the scan core.** The core has had six arms of
   attention; the architecture that actually learns has been sitting in the repository the whole time.
2. The context-length curve is the tool to develop it with: every change must show up as a higher curve at a
   fixed sample count, or it does not count.
3. The recurrent core stays in the repository as a documented negative result rather than being deleted, so
   the same ground is not re-covered.
