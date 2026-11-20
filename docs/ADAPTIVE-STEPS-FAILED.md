

---

# PER-TENSOR ADAPTIVE STEP SIZES DO NOT REMOVE THE LEARNING-RATE KNOB (2026-11-20 17:56)

The first attempt at "when we feed the data it automatically aligns itself", and it did not work.

## Measured, all at emb_rank 32, seed 0, 600 steps, 20k windows

```
arm                        lr      held-out acc   outcome
momentum (one lr)          0.05      12.51%       reference
momentum (one lr)          0.50       0.00%       DIVERGED, loss nan by step 300
adaptive (per-tensor)      0.50       0.03%       did NOT diverge - it sank instead (loss 9.41 -> 27.57)
```

- **P1 CONFIRMED:** a hand-set constant 10x too high destroys the model outright.
- **P2 FAILED:** the per-tensor adaptive rule was supposed to land within 1.0 point of 12.51% at lr 0.50. It landed **12.48 points
  below** it. Dividing each gradient by its own RMS scale does **not** remove the sensitivity; it only changes the units.
- **FALSIFIER FIRED**, so the pre-registered retry is owed: the per-**element** RMS variant, now running.

## The diagnosis, because it decides what to build next

Normalising a gradient by its own magnitude controls the **direction** and removes **relative** scale differences between tensors. It
does nothing about the **absolute** effect a step has on the loss. With gradients normalised to order 1, `lr = 0.5` still means "move
half a unit in every coordinate", which is enormous. That is why the arm sank instead of diverging - a different failure, same cause.

**So gradient-magnitude normalisation is the wrong family.** What actually removes a learning-rate knob is a rule that measures the
**response** of the loss to the step it just took and grows or shrinks the step accordingly - a gain rule. That will be pre-registered
separately, after the owed per-element retry reports, and it is a genuinely different mechanism rather than another rescaling.

## Two runner defects this exposed, both mine, both fixed

1. **A diverged arm aborted the run** (exit 1 was treated as infrastructure failure) - so the first attempt threw away the two
   adaptive arms, i.e. the actual test.
2. **A completed-but-bad arm also aborted the run** (rc 1 again, this time for "below constant"). Same root cause: the trainer used
   one nonzero exit code for "bad result" and "crash", so the runner could not tell them apart.

Fix: the trainer now prints an explicit **`OUTCOME:`** line (`BEATS_CONSTANT` / `BELOW_CONSTANT` / `DIVERGED`), and the runner aborts
**only when that line is missing** - i.e. only on a real crash. A bad result is data and must not stop an experiment; that is the whole
point of pre-registering predictions you might fail.
