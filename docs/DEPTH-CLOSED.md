

---

# DEPTH, CLOSED: THE RETRY MOVED IT FURTHER FROM THE BASELINE (2026-11-19 13:51)

Pre-registered single change: `LR0 0.03 -> 0.01`, everything else identical. The stated outcome for a failure was: "the claim becomes depth does not help
at 1.8M parameters and 1200 steps, and the next repair is FLAW 4 (`<unk>`), not more architecture."

```
arm                         HELD-OUT (layer 2)   layer 1 own readout   train ema   held-out loss
single layer (baseline)      6.07%                -                     7.0492      6.5013
depth2, lr 0.03              5.68%                4.90%                 9.1805      8.4180
depth2, lr 0.01              5.08%                4.37%                 9.9654      9.4252
```

**The retry made it worse, not better.** A smaller learning rate produced a worse model on every measure: accuracy down 0.60, train EMA up 0.78, held-out
loss up 1.01. So the earlier failure was **not** a learning-rate artefact, and the falsifier stands.

**One retry was the budget, and it is spent.** A second attempt after seeing this result would be fishing for a setting rather than testing a hypothesis.
The depth question is closed at this size.

## The caveat that limits what "closed" means

All three arms are **below the trivial constant** on the corrected protocol (constant is 13.14% `<unk>`-excluded, 14.70% over all positions; these arms
sit at 5-6% on the trainer's own line). The comparison is **internally valid** - same trainer, same instrument, same 104 window draws (`rng(999)` is
fixed, so these runs are paired on windows) - but it tests whether a second layer helps **a model that cannot yet beat a constant**, which is a narrow
question.

So the correct claim is: **"a second recurrent layer does not help a model that is still below trivial at 1.8M parameters"**. It is not "depth does not
help", and I will not let the distinction blur.

## The transparency result survives all of this

Layer 1's own readout scores **4.37-4.90%** - every layer makes its own next-token prediction and both are scored on every run. That is the deliverable
the two-layer model was built for, and it is independent of whether depth improves accuracy.

## Next, and it is a prerequisite rather than a hypothesis

The denominator correction means **no model in this project has yet been shown to beat the trivial constant on a common protocol.** The 6000-step base
(21.74% on the trainer's line) is the only candidate, and it has never been re-scored with matching denominators. That re-score comes before any new
architecture work: without a model that clears trivial, every mechanism test - memory, depth, normalisation - is measuring an under-trained model.
