
## Context length is not the mechanism - and the state carries almost no information (2026-11-17)

The failed rollout arm redirected to generation-context construction. The first candidate was **context length**: the trainer scores inside 128-token windows, the probe started from 64.
`tools/context_length_sweep.py`, same checkpoint, same 40 generation steps, same 8 starts, only the starting context changes:

```
  ctx   teacher-forced   free-running   repetition
    4          16.25%          5.00%        0.89
   16          15.62%          4.69%        0.91
   32          15.62%          4.38%        0.92
   64          15.62%          4.38%        0.92
   96          15.62%          4.38%        0.92
  128          15.62%          4.38%        0.92
```

## Two results, and the second is the important one

**1. Context length is refuted as the mechanism.** Free-running accuracy moves between **4.38% and 5.00%** across a 32x range of context, and the *shortest* context is marginally
the best. If the loop were a context-window mismatch, more context would have helped. It does nothing, and repetition stays at 0.89-0.92 everywhere.

**2. Teacher-forced accuracy is FLAT: 15.62% from 16 tokens to 128 tokens, to two decimal places.** Thirty-two to 128 tokens of extra context buys **0.00 points**.

That second number is the one that reframes everything. A model whose prediction accuracy does not improve when you give it **8x more context** is not using its context. Whatever the
recurrent state is doing, it is not carrying usable information forward from token t to token t+1 in a way that changes predictions.

## Why this ties the whole thread together

Every result this week is consistent with a model that is effectively **context-free at prediction time**:

- **Generation loops**: with no carried information, each step predicts from roughly the same place, so it emits nearly the same token. Repetition 0.89-0.92 is that, directly.
- **Exposure bias training failed to help**: if the state carries nothing, training the model on its own rollouts has nothing to teach it - feeding it its own tokens gives the same
  information as feeding it corpus tokens. That is exactly the "both metrics fell, repetition did not move" pattern the arm produced.
- **The readout passes while generation fails**: a context-free predictor can still be 16-17% accurate on next-token statistics (frequent continuations are predictable) while being
  incapable of producing a coherent sequence.
- The **F4 probe failure** and the **flat KL** are the same fact seen through different instruments.

## The next question, written before it is measured

**Does the recurrent state actually change predictions?** Two direct tests on this checkpoint, no training:

1. **Context shuffle**: keep the same multiset of context tokens but destroy their order. If accuracy is unchanged, the state is reading a bag, not a sequence.
2. **State ablation**: replace the carried state `h` with zeros at every step and re-measure next-token accuracy. If accuracy is unchanged, the state contributes nothing and the
   model is a position-wise predictor wearing a recurrence.

The second is the sharper one and it has an unambiguous reading: **if zeroing the state costs less than 1 point of accuracy, the recurrence is decorative.**

Whether that is an architecture defect, a training defect (the recurrence never received a gradient reason to matter), or an artefact of how `Wh` was initialised is the following
question - but it cannot be asked honestly until we know whether the state is load-bearing at all.
