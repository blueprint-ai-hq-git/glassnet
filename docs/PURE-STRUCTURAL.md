
## Pure structural rows: FALSIFIER FIRED at VAL 13.74% - and the meaning is missing even with nothing to blame (2026-11-16)

Predictions in `lens/runs/PURE-STRUCTURAL-PREDICTIONS.txt`, written before the run. One change from the structural arm: **no learned
correction at all** (`--structural-rank 0`), so the answer row is exactly its own vocabulary's geometry. Numbers from
`lens/runs/PURE-STRUCTURAL.txt`.

```
final VAL accuracy        13.74%     (predicted 20.0-23.5%, falsifier at 18.0%)  -> FALSIFIER FIRED
final VAL perplexity     392.84
concepts passing          46/53      (every other arm: 52-53)
median own-word mass      0.0019     (S1 predicted above 0.10)                  -> FAILED
explain() max |residual| 2.089e-06   | concept share 1.000000 / 1.000000         -> PASSED
```

## The result, stated without softening

**Removing the correction costs 10.6 points** (24.33% -> 13.74%): the low-rank correction is load-bearing, as its growth from 46.89 to
87.42 in norm already suggested. **And removing it does not buy meaning either** - own-word mass is 0.0019, far below the 0.10 bar and only
a doubling of the free answer matrix's 0.0008.

## The uncomfortable part

The pure-structural arm has the **most coherent own-word embedding geometry of any model in this project**:

```
checkpoint                       alignment   coherence
shipped 4M (free answer.W)         0.3822      0.0034
structural, correction carried     0.3826      0.0064
pure structural, no correction     0.4893      0.0969
```

Its own-word embeddings are 25% better aligned and **28x more pairwise-coherent** than the shipped model's, because its structural rows were
the only answer path. **And its trained rows still put only 0.19% of their mass on the concepts' own words.**

**So the correction was never the villain.** Even with no correction to occupy the row, next-token training produces answer rows that do not
favour their own vocabulary. The objective itself rewards a river of evidence across thousands of plausible continuations, and the
concept's own handful of words is not where that water flows. This is now measured with the correction absent, present-but-unpriced, and
present-but-priced:

```
arm                              own-word mass   VAL
free answer matrix (shipped)        0.0008       24.58%
structural + correction             0.0008       24.33%
structural + priced correction      0.0008       (running)
pure structural, no correction      0.0019       13.74%
```

## What this rules out, and what is left

**Ruled out as routes to concept meaning**: embedding geometry (present and better, worth nothing), row magnitude (destroys accuracy),
row support restriction (meaning 0.63 but accuracy 0.0022), and removing the free correction (costs 10.6 points and buys 0.0011 of mass).

**What remains**: supervising the row directly against its own words, at a weight that the measurement supports. The term exists, its
gradient is verified to 3.7e-10, and this morning it was retired at **2000%** of the cross-entropy gradient - a weight far outside the
range where it can work. That fair test is now queued at 13% of the CE gradient.

If that fails too, the honest conclusion is that this project's transparency guarantee is **exact attribution** - which it delivers to
2.4e-06 with concept share 1.000000 - and **not concept content**, and the public documentation will say exactly that.
