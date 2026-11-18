# The published "32-token context" is wrong by measurement: this model uses 2 tokens (2026-11-18)

Both published chat cards said **context window 32 tokens**. That figure was inherited from an older README, never measured. `tools/context_curve.py` now measures it: feed
exactly L tokens, score the prediction for the next one at the final position, over the same 1,500 held-out chunks at every length.

```
     L   accuracy   vs L=1
     1      9.27%   +0.00     <- the model sees only the current token
     2     16.73%   +7.47
     4     17.40%   +8.13
     8     17.40%   +8.13
    16     17.40%   +8.13
    32     16.73%   +7.47     <- the published "context window"
    64     17.93%   +8.67
   128     16.40%   +7.13
```

## What is resolvable and what is not

With 1,500 samples per length and p ~ 0.174, the standard error is **0.98 points**, so a difference needs ~1.9 points to be resolvable at 95%.

- **Gain from 1 to 2 tokens: +7.47 points = 7.6 standard errors. RESOLVABLE.** Context genuinely helps - the model goes from a unigram-level 9.27% to ~17%.
- **Gain from 2 to 128 tokens: -0.33 points = -0.3 standard errors. NOT RESOLVABLE.**
- **Spread across every length from 2 to 128: 1.53 points = 1.6 standard errors. NOT RESOLVABLE.**

So the honest statement is: **the effective context is 2 tokens.** Everything from 2 to 128 is the same measurement within noise. The model is a bigram-plus-recurrence, and the "32"
on the card overstated what it can use by a factor of sixteen.

## This is the same finding as the memory measurements, seen from the outside

Earlier, measured from inside the state: a token 16 steps back moves the state by **0.878%** (radius 0.9) or **4.730%** (radius 1.0), against a 10% bar. Now measured from the
outside, on the output: nothing beyond two tokens changes the prediction. The two measurements agree, and the external one is the one a user would notice.

## Corrected on the artifact

`glassnet-scan-r9-recurrent`'s card no longer claims a 32-token window. It now reports **effective context: 2 tokens**, and `config.json` carries both the measured
`effective_context_tokens` and the `trainer_context_length` separately, so the trained length can never again be mistaken for what the model uses.
