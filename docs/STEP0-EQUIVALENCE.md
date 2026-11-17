
## Step-0 equivalence proved: adding a block changes nothing until training starts (2026-11-17)

`tools/step0_equiv.py`. This is the gate that has to pass before any new architecture path exists, because this project has already shipped one silent no-op (a flag
that reached argparse and the model signature but not the construction site) and one arm that reported COMPLETE over an empty measurement.

```
shipped checkpoint: V=12002 D=1024 W=32 K=53

GATE CLOSED  (alpha=0, zero-init output projections):
  max |old - new| logits = 2.636e-06   -> identical to float32 rounding

GATE OPEN    (alpha=1, nonzero output projections - i.e. after one training step):
  max |old - new| logits = 1.525e+01   -> the block changes the prediction

VERDICT: step-0 equivalence proved and the gate is not vacuous
```

**Why equivalence is even possible**: the shipped model's input is the concatenation of the window's embeddings. A pre-norm block whose output projection is
zero-initialised adds exactly zero to every position, so the concatenated residual stream is the same 32768 features the existing `concept.W (32768, 53)` already
consumes. The concept and answer layers transfer unchanged, and the new model starts life as the old model plus an empty block.

## The test was wrong the first time, and that is the interesting part

The first run reported **NO EFFECT** for the open gate. The cause was my test, not the model: it opened `alpha` while `Wo` and `Wd` were still zero matrices, so the
block contributed nothing in either state and both runs printed the same 2.636e-06. **A gate test that cannot distinguish open from closed proves nothing**, and
worse, it would have "passed" the closed case for the wrong reason. The fix simulates one training step properly - `alpha=1` **and** nonzero output projections -
and the difference jumps to 15.25.

This is the third time this week that auditing the instrument found the defect rather than the model, and the second time the defect was in a test I had just
written to be careful.

## What this unlocks: a warm start instead of a cold one

Because the shipped model is d=1024 and the block preserves shapes, we do not have to train a new architecture from scratch to test whether depth helps. The
candidate is **the shipped weights plus one certified block**:

```
embedding  12,290,048
block      15,096,672   (q,k,v,o with GQA 16/4 at head_dim 64, SwiGLU 1024->4096, two RMSNorms)
concept     1,736,704
answer        636,106 + 12,002
total      29,771,585   (29.8M, inside the 50M limit)
```

Training starts from a model that already reaches 24.58% +/- 0.055 instead of from noise, with the block adding exactly nothing on step 0. If depth is what the
diagnosis says it is - the ability to compute relations rather than interpolate a bag of positions - this is the cheapest possible test of that claim, and its
falsifier is already written: breadth still under 0.20 with depth in place means the concept bottleneck K is the binding constraint, not the depth.
