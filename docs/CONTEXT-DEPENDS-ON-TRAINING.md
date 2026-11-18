# Training length did teach the model to use context: ~2 tokens -> ~16 (2026-11-18)

The published figure **"effective context = 2 tokens"** was measured on `scan9_ep1500` and has been carried in the config as a property of the architecture. Re-measured on the
**6000-step** model (`scan12_r090_6k_ep6000`, the strongest this project has), the conclusion does not survive.

```
L       6000-step    old 1500-step
1        13.07%          9.27%
2        20.33%         16.73%
4        21.40%         17.40%
8        22.47%         17.40%
16       24.13%         17.40%
32       22.47%         16.73%
64       25.00%         17.93%
128      22.07%         16.40%
```

n = 1,500 chunks per length, SE ~1.05 points at these accuracies.

## What is resolvable

```
6000-step model         delta        SE      verdict
 1 -> 2 tokens        +7.26 pts     5.4      RESOLVABLE
 2 -> 16 tokens       +3.80 pts     2.5      RESOLVABLE   <-- new
 2 -> 32 tokens       +2.14 pts     1.4      not resolvable
16 -> 32 tokens       -1.66 pts    -1.1      noise
16 -> 64 tokens       +0.87 pts     0.6      noise
64 -> 128 tokens      -2.93 pts    -1.9      not resolvable

old 1500-step model     delta        SE      verdict
 1 -> 2 tokens        +7.46 pts     6.1      RESOLVABLE
 2 -> 16 tokens       +0.67 pts     0.5      not resolvable   <-- this produced "2 tokens"
 2 -> 128 tokens      -0.33 pts    -0.2      not resolvable
```

**The 6000-step model gains +3.80 points going from 2 to 16 tokens of context, at 2.5 standard errors. The 1500-step model gained +0.67 points at 0.5 standard errors.** That is the
difference between a model that cannot use context and one that can.

## What this means, stated carefully

- **"Effective context = 2 tokens" was true of the model it was measured on and is false as an architectural claim.** For `scan9_ep1500` it remains correct; for `scan12_r090_6k_ep6000`
  the usable context is **at least 16 tokens** and no further gain is resolvable past that, in a window of 32.
- **The curve is not monotone, and the dips are not signal.** L=32 reads 22.47% and L=128 reads 22.07%, both *below* L=16's 24.13%, but at -1.1 and -1.9 SE those are noise. The honest
  reading is a **plateau from ~16 tokens onward**, not a peak at 16.
- **So training length fixed one thing the plateau could not:** the model learned to use context it previously ignored. That is a real capability gain that the single accuracy number
  (23.95% -> 25.89%) badly understates.
- **And it bounds the ceiling's cause.** With a 32-token window and usable context of ~16, more steps stopped helping at a point where the model was still improving its use of context.
  The remaining limit is therefore **capacity or the mechanism**, not the schedule and not the amount of text.

## The correction to make on the artifacts

A config field `effective_context_tokens` must be **per-model**, because it demonstrably changes with training. It cannot be inherited from an earlier checkpoint and presented as an
architecture property - which is exactly how the wrong "32-token context" figure and then the over-corrected "2 tokens" figure both came into existence.
