

---

# SELF-ASSESSMENT HEAD FAILS, AND THE REASON IS THE FINDING: THE MODEL ALREADY KNOWS WHEN IT IS RIGHT (2026-11-21 12:18)

Phase 6.1, three seeds, two arms each, complete. The head was supposed to give the network a way to catch itself. It works - and it
still loses, because the network was already doing this for free.

## Measured, all six arms

```
seed  arm          head AUROC   free confidence (max softmax)   margin
 0    A_headoff      0.1049            0.9423                  -0.8374
 0    B_headon       0.8517            0.9418                  -0.0901
 1    A_headoff      0.7850            0.9384                  -0.1534
 1    B_headon       0.7900            0.9140                  -0.1239
 2    A_headoff      0.8173            0.9392                  -0.1219
 2    B_headon       0.9122            0.9399                  -0.0278
                          mean 0.8513        mean 0.9319        -0.0806
```

- **P1 MET on all six arms:** the head does rank its own correct answers above its wrong ones (AUROC 0.79-0.91).
- **P2 NOT MET on all three seeds:** the margin against free confidence is **negative every time** (-0.0901, -0.1239, -0.0278; mean
  **-0.0806**). A gate of +0.02 was required.
- **Falsifier did not fire** (nothing below 0.60).
- **OUTCOME: SELF_THINKING_FAILED on all six arms.**
- **P4 was pre-registered and is UNMEASURED.** It said "thinking costs nothing: held-out language accuracy within 1.0 point". The
  script prints language loss during training but never a held-out language number, so this was never checked. It is recorded as
  unmeasured, **not** as passed.

## The finding that matters more than the verdict

**The model already knows when it is right: plain max-softmax confidence ranks its own correctness at 0.93 mean AUROC, for zero extra
parameters.** And correctness is not hidden in some deep nonlinear code - on seed 1 a **completely untrained random** linear projection
of the hidden state scored **0.7850**. The information sits openly in the final hidden state, and the output distribution exposes it.

So the honest answer to "can the network know itself" is **yes, and it already does, for free.** A 33-parameter head on top of that adds
nothing and loses a little, because it is a lossy re-learning of a number that costs nothing to compute.

## What this closes and what it opens

**Closes:** self-assessment as a *new tensor*. 6.1 is FAILED, and per the project rule it is not tuned until it passes.
**Opens:** the common-sense loop does not need it. Everything downstream - deciding what is important (L3 salience), ignoring bad data
(6.2 self-cleaning) - can run on the confidence the model already produces. 6.2 is already built that way and is running.

## Honest note on the arms

The A_headoff arm was meant to be the control with alpha 0.0 (head gradient zero, so an untrained head). Its AUROC is therefore a
*random direction* reading, which is why the three values scatter so widely (0.1049 / 0.7850 / 0.8173). That scatter is itself the
evidence for the paragraph above: it is the hidden state doing the work, not the head.
