
## G2 is correct. Both of my disagreeing figures were my own bugs (2026-11-17)

The previous two entries said `tools/eval_run9.py`'s G2 reported an entropy that "cannot be reproduced from either slice" and called it the second broken gate in that script.
**That was wrong. G2 is reproducible and correct. My measurements were the broken ones.**

`tools/path_reconcile.py` re-derives the six-probe mean using the eval's own functions and prints `sum(probs)` as the standing rule now requires:

```
checkpoint has Wd: True | has ba: True

case                            entropy    sum(pr)      top1
probe hello     direct=False      8.871   1.000000    0.0040
probe france    direct=False      8.849   1.000000    0.0049
probe hello     direct=True       8.870   1.000000    0.0040
probe france    direct=True       8.851   1.000000    0.0048
six-probe mean  direct=False      8.838
six-probe mean  direct=True       8.835        <- exact match to the eval's reported G2
```

**8.835 reproduces exactly**, on the six one-word probes, with the distribution verified to sum to 1. The direct `Wd` path makes a difference of 0.003 nats, not the gulf I claimed.

## My two wrong numbers, and what caused each

1. **0.148 nats** (probes, `g2_entropy_check.py`): I encoded probe words without **masking `<unk>`**. The eval's `softmax()` sets `x[UNK] = -1e9`; my copy did not. Unmasked, the model puts nearly all its mass on `<unk>`, which reads as an extremely peaked distribution. **A missing mask, not a model property.**
2. **0.0079 nats** (held-out, `saturation_trajectory.py` and the temperature sweep): a broken entropy expression that disagreed with direct normalisation by a factor of ~700. Already retracted.

So three published claims in a row were wrong for three different self-inflicted reasons: a blind proxy (F4), a missing mask, and a broken expression. **The eval script was right twice while I was wrong three times.**

## What G2 actually says, correctly

**On the six one-word probes, mean predictive entropy is 8.835 nats = 94.1% of uniform.** That is the correct meaning of G2: asked a single word with no context, this model is nearly agnostic about the next token. That is *expected* and it is not a saturation claim.

The warning G2 was written to carry - "collapse toward 0 alongside a KL rise means saturation" - is therefore **not** triggered: 94.1% of uniform is the opposite of collapse.

## What is still not verified

**Held-out entropy under the model's full forward path (with the direct `Wd` term and `<unk>` masked) has not been measured correctly yet.** The 6.954 nats figure published in the retraction came from a path that omitted the direct term and did not mask UNK. It is plausible and probably close, but by this week's standard it is **not established**, and it should be re-measured before it is quoted again.

## The rule that now covers all three failures

Every probability or entropy figure must print **`sum(probs)`** *and* state **which forward path** it used. Both of today's silent failures would have been caught by the second requirement alone: a probe measurement that omits the direct term, or forgets the UNK mask, is measuring a different model than the one being scored.
