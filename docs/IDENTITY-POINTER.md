# The identity pointer: 83.3% held-out copying, a new best (2026-11-11)

Copying unseen words was stuck at a trade-off: best 70.2% held-out at step 1,000 of a mixed run, decaying to 47% by step
4,000 as language improved. Six training schedules failed to escape it (`docs/TWO-STAGE-FAILS.md`). This change is
architectural and costs nothing at inference: **delete the pointer's learned readout matrix and tie the pointer straight
to the embedding table.**

```
                 step 1000   step 2000   step 3000   step 4000   step 5000
held-out copy      83.3%       76.6%       47.5%       38.7%       50.6%
taught copy        84.7%       84.7%       84.5%       85.9%       85.1%
language           13.45%      14.93%      16.52%      17.07%      17.52%
```

**Prediction recorded before the run: held-out copying 75% or better, language 16% or better. Copy PASSED at 83.3%
(best checkpoint, step 1,000). Language FAILED at that checkpoint - 13.45% against the 16% bar - and only reached 17.52%
by step 5,000, where copying had fallen to 50.6%.**

## Why this works
The pointer used to end in a learned 512x512 matrix: `(alpha-weighted window embedding) @ ptrW @ E^T`. `ptrW` is a
parameter that language batches keep updating, so the copy operation was being edited by an objective that does not care
about it. Removing it leaves

```
(alpha-weighted window embedding) @ E^T
```

which is exactly the attention-weighted sum of one-hot token scores: "attend to position t" means "output the token at
position t", with nothing in between to corrupt. It also removes 262,144 parameters.

## The measurement that motivated it
On `early_s0_best`, the 70.2% checkpoint, evaluated on identical items (`tools/identity_pointer_eval.py`):

```
learned readout : taught 83.0%   held-out 71.5%
identity readout: taught 83.8%   held-out 85.3%
```

The identity was better *before any retraining*, which is why the arm was run.

## What also came out of the diagnostics, and what it corrects
`tools/hard_route_eval.py` measured a hard inference-time switch (take the pointer's answer when its sharpness passes a
threshold) and returned **null** on `langfirst_stage1_final`, because that checkpoint's routing had collapsed onto the
pointer path - assembled and pointer-only logits give identical answers there, so a switch cannot differ.

More importantly, `identity_pointer_eval` measured AIM mass **0.075** on that language-only checkpoint. So the earlier
finding that "the pointer keeps its aim and only loses the vote" holds **for mixed training only**; language-only training
destroys the aim itself. That is a narrower claim than the one published this morning and the earlier doc now needs to be
read with this caveat.

## Still true, and not fixed by this
The trade-off remains: copy peaks at step 1,000 and decays as language climbs. The identity readout raised the whole curve
(70.2 -> 83.3 at the peak) but did not remove the slope. The best "both" point on this run is step 2,000 at 76.6% copy with
14.93% language.
