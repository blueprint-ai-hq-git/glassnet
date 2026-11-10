# Who decides the answer: the state, and it says one of three things (2026-11-10)

Run 8 collapsed to the same generation for every prompt while its distribution sat at 99.8% of uniform entropy
on the six test prompts. Both facts pointed at "the state is ignored". This measurement says otherwise.

`tools/collapse_diagnosis.py`, 200 real held-out contexts from the corpus, checkpoint `scan8_ep1500.npz`:

```
full argmax == direct-only argmax   :   8.5%
full argmax == state-only argmax    :  87.0%
distinct argmax tokens over 200 contexts:   3
mean top-1 probability  : 0.033361   (uniform would be 0.000083)
mean top-1/top-2 gap    : 7.975e-03
mean entropy            : 8.987      (max 9.393)
```

## What this means
1. **The state decides the answer.** Zeroing the direct term leaves the argmax unchanged 87% of the time;
   zeroing the state leaves it unchanged only 8.5% of the time. The recurrent core is where the decision is
   made, which contradicts the "the readout ignores the state" reading of the KL number.
2. **But it only ever decides three things.** Across 200 genuinely different held-out contexts, the model's
   top choice takes **three distinct values**. That is functional collapse in the answer, and it is what the
   E4 repetition loop looks like from the inside.
3. **The distribution is not uniform, it is merely weak.** Mean top-1 probability is **0.0334**, about 400x
   the uniform 0.000083, with a top-1/top-2 gap of only **7.975e-03**. Confident about nothing, but not flat.
4. Mean entropy on real data is **8.987** against a maximum of 9.393, so the six-prompt figure of 9.373 (99.8%
   of uniform) was partly an artefact of scoring prompts unrelated to the corpus.

## The corrected problem statement
Not "the readout ignores the state", but "the state-to-answer map is degenerate: hundreds of different states
land on three answers". That is a different bug with different fixes - it wants capacity or structure in the
map from state to answer, or training signal that forces different contexts onto different answers, not more
state separation.

## Next measurement before any new arm
Whether the model responds to state distance at all: pair up held-out contexts, measure how far apart their
states are, and see whether far-apart states give far-apart predictions. If logit differences stay flat as
state distance grows, the map is compressing and that is the thing to fix.
