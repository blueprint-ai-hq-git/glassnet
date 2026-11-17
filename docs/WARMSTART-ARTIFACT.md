
## The warm-start artifact exists and is step-0 equivalent (2026-11-17)

`tools/build_warmstart.py` produced `lens/runs/chat/warmstart_block1_d1024.npz`: the shipped model's weights plus one certified block whose output projections are
zero-initialised, so the block adds exactly nothing until the first gradient step.

```
source: V=12002 D=1024 W=32 K=53, 14,674,913 params

gate closed: max |shipped - warmstart| = 1.624e-06   IDENTICAL (float32 rounding)
gate open  : max |shipped - warmstart| = 2.748e+01   gate is real, not vacuous

saved 14 tensors: emb, concept.W, concept.b, answer.W, answer.b,
                  block.n1, block.Wq, block.Wk, block.Wv, block.Wo, block.n2, block.Wg, block.Wu, block.Wd
total parameters: 29,881,313 (29.88M)   under the 50M limit
round-trip verified: 14 keys, meta alpha=1.0, recorded step-0 diff 1.624e-06
```

**The parameter count lands exactly on the figure I corrected earlier today** - 29,881,313 - now confirmed by construction rather than by arithmetic on paper. That
is the best kind of coincidence: the correction was right, and building the thing proved it.

The file also carries its own provenance in `__meta__`: source checkpoint, shapes, head configuration, gate value, and the measured step-0 difference, so a future
reader does not have to reconstruct what it is.

## Why this artifact matters more than it looks

Training the new architecture from scratch would take GPU hours and would confound two questions: *does depth help* and *did the model simply get more training*.
Starting from the shipped 24.58% +/- 0.055 model with a block that contributes exactly zero means the **first gradient step is the only difference**. If the loss
then fails to improve, depth is not what is missing - and the falsifier written for this arm before any of this existed applies unchanged: **breadth still under 0.20
with depth in place means the concept bottleneck K is the binding constraint, not the depth.**

## What is left

One piece: the training loop. Load this checkpoint, run the certified block forward and backward alongside the concept and answer layers with the project's own SGD,
and measure on the same held-out split and the same instruments as every arm this week - VAL accuracy, perplexity, coverage, repetition, and own-word mass for the
concept rows. Nothing about the measurement protocol changes, which is the point: the new architecture gets no easier test than the old one did.
