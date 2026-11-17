
## The trained-gate question is NOT answered: the standalone harness does not learn (2026-11-17)

The pre-registered question after certifying the cell was: *do trained gates clear the k=16 memory gate?* `tools/cell_train.py` was built to ask it - frozen embeddings from the shipped
checkpoint, the cell's linear gated state, a fresh readout, cross entropy, no spectral clamp anywhere.

**It does not learn, so it cannot answer the question.**

```
first attempt (readout init 0.02, lr 3e-3, 100 steps):
  step 1 loss 9.4054 acc 0.27%  ->  step 100 loss 9.3936 acc 0.00%
  ln(12002) = 9.3936   <- the loss sits exactly at uniform, so the model outputs near-zero logits and nothing moves

second attempt (readout init 0.2, lr 1e-2, 60 steps):
  step 1 loss 10.0356 acc 0.27%  ->  step 50 loss 9.3083 acc 0.00%

memory after those marginal steps: k=1 10.16%, k=4 8.61%, k=8 7.85%, k=16 4.40%, k=24 3.57%
```

## What is honest to say

- **The cell is certified** (gradients pass at T = 1, 2, 3, 8, 16, 32) and that stands.
- **The harness does not learn next-token prediction.** Accuracy stays at 0.00% and the loss stays within 0.1 nats of `ln(V)`. A model that predicts nothing cannot have its trained
  gates judged.
- **Memory moved only 3.10% -> 4.40% at k=16**, still far below the 10% gate, but on a model that has not learned anything, so this is **not evidence either way**.
- **The falsifier has not fired and is not being claimed.** It says *if trained gates also fail to lift k=16 above 10%*, and nothing here was trained in any meaningful sense.

## Why this happened, and what it means

The pre-registration said the integration needs its own end-to-end check that the cell's gradient path composes correctly with the existing readout. **That check was skipped in order to get
to an answer faster, and this is what skipping it costs**: a harness that runs without error, produces a number, and measures nothing.

Two concrete suspects, both untouched:
1. **Readout scale and learning rate** - the model stays at uniform, which is what an under-driven readout looks like, and my two attempts guessed at the scale rather than measuring it.
2. **Composition** - the cell's certified backward has only ever been exercised where the loss touches the cell state *directly*. Here the loss reaches it through a readout matmul, and that
   path has never been verified against finite differences.

**The next work is the composition check, not another training run**: a single end-to-end finite-difference test of the whole cell+readout+loss stack on small dimensions, before any more
steps are spent. That is the same rule that caught the transposition and the mask and the entropy expression - verify the instrument, then measure.
