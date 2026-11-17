
## The run-9 readout is saturated: 0.007 nats of entropy while wrong 83% of the time (2026-11-17)

G2 reported mean predictive entropy **8.835 nats = 94.1% of uniform**, which sat badly against the same checkpoint's **16.9%** next-token accuracy. Two hypotheses were written
before measuring: H1 (G2 came from near-contextless probes, which should be near-uniform) and H2 (the model really does stay flat even with context).

**Both hypotheses are wrong.** `tools/g2_entropy_check.py`:

```
held-out real text (5,120 tokens):  entropy 0.007 nats =  0.1% of uniform   |  accuracy 16.78%
six one-word probes:               entropy 0.148 nats =  1.6% of uniform
uniform reference:                 9.393 nats
```

## What this actually says

**The softmax is saturated almost everywhere: 0.007 nats means the output distribution is effectively one-hot.** The model commits to a single word with total confidence and is
**wrong 83% of the time.** That is the most damaging single fact found about run 9 so far, and it is not "a well-calibrated model that is merely weak" - it is a model that has
learned to be certain and is usually incorrect.

This also explains the generation behaviour directly: a near-one-hot distribution decoded greedily walks the same few tokens forever, which is exactly the `"the and of the"`
loop seen at the end of every scan run.

## H1 is refuted, and that matters for the eval itself

The probes are **not** near-uniform - they sit at **0.148 nats**, i.e. 1.6% of uniform, just as saturated. So the "little context explains it" story is dead.

And the reported **8.835** cannot be reproduced from **either** slice. No measurement in this script lands anywhere near it. **That means `tools/eval_run9.py`'s entropy path computes
something different from "mean predictive entropy of the final model on data"** - most plausibly a different readout call (no learned gain), a different input, or an average taken
over logits before the softmax normalisation. **G2's reported number is therefore not usable until that path is audited**, and it should not be quoted again in the meantime.

This is now the **second** metric defect inside the same evaluation script, after F4's blind KL. Two of its five gates do not measure what their names claim.

## The honest standing summary of run 9

**Established:**
- held-out accuracy **16.93%** (log) / **16.78%** (independent, different slice) - real, and 3.2x the 5.34% bar
- state cosine **0.1736** - states are decorrelated
- trained readout beats a random-noise readout **859x** on next-token accuracy (16.7773% vs 0.0000%) - the answer layer is doing real work
- **the readout is saturated: 0.007 nats of predictive entropy, wrong 83% of the time** - confident and mostly incorrect

**Not usable:**
- F4 as a measure of input-sensitivity (blind to predictive competence; see the correction)
- G2 as a measure of entropy (8.835 unreproducible from either slice)

**Not established:** anything about whether the *architecture* is the limit, because the two gates that were supposed to test the readout are the two that do not measure it.
