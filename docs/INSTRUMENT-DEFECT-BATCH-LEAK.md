

---

# INSTRUMENT DEFECT FOUND AND FIXED: THE MODEL WAS READING ACROSS ITS OWN BATCH (2026-11-21 14:25)

This is the most important entry of the day, and it is not a result - it is a defect in the measuring instrument that sat under every
probe run since this morning.

## The defect

`glass_lm.forward` computed `h = x.reshape(B*T, d)` and handed it to `forward_record`, which reads its input as **one sequence of length
T**. With a batch of B sequences, that length became **B\*T**, so the causal attention mask spanned the whole flattened batch: sequence k
could read the tokens of sequences 0..k-1.

**Measured, not inferred** (`lens/diag_batch_leak.py`): one model, batch of 4 sequences of 6 tokens, change **only sequence 0**:

```
sequence 0's own logits moved            1.183650e-01   (expected)
the LAST sequence's logits moved         4.599481e-02   (it read sequence 0)
```

Consequences: the batch was never independent, effective context was **B\*T = 256 tokens** in most probe runs instead of T, and batch
size silently changed the model.

## My first version of the leak test was wrong, and it said "no leakage"

It changed the **last** sequence and checked the **first**. A causal mask blocks that direction for free, so it printed
`0.000000e+00` - a clean pass on a test that could not fail. Changing the **first** and checking the **last** is the direction that can
actually leak. Recorded because a control that cannot fail is worse than no control.

## The fix, and how it is certified

`lens/glass_batch.py` re-derives the same block with the batch carried through the head reshape, so attention runs **within** each
sequence. `glass_grad.py` is untouched and still certified for the single-sequence case. Three checks, all passing:

```
1. B=1 equivalence with the certified single-sequence code
   forward max abs difference      0.000e+00
   every gradient max abs diff     8.882e-16
2. finite differences, B=3 (float64) - worst relative 4.573e-07 across W1 W2 Wq Wk Wv Wo ln1* ln2* dInput
3. isolation: changing sequence 0 moves the LAST sequence by 0.000e+00   (exactly isolated)
VERDICT: PASS
```

A second bug was caught by check 1 before anything ran: the shared `ln_backward` sums the scale/shift gradients over axis 0 only, which
is right for `(T,D)` and wrong for `(B,T,D)` - it returned scales of shape `(T,D)` instead of `(D,)`. The forward matched to 0.000e+00
while the gradients differed by **5.99**, which is exactly why an equivalence check is worth running.

`glass_lm` now uses the batched block. Verified after the patch: leak test **0.000000e+00**, and `glass_lm`'s own gradient check still
**PASS** (worst absolute 6.635e-10 against a 1e-7 tolerance).

## What this does to today's conclusions

| claim | effect of the leak fix |
|---|---|
| "selection and comparison work, arithmetic fails" | **survives in one direction**: the leak ADDED information (up to 256 tokens of context), so a task failed *while leaking more context* is still failed. But the specific accuracies are provisional and are being re-measured. |
| copylast 100% / copyfirst 80-100% (distance claims) | **weakened**: the flattened batch gave the model many more recent tokens than its own sequence, so "distance" was not what it seemed. Re-running. |
| L1 memory closes as "not needed at this size" | **on hold** until the fixed instrument re-measures it |
| L3 salience certificate | unchanged in form, but every arm must be re-run |

## Replication now running, with the prediction written first

Six rungs on the fixed instrument, one seed each: copy1, copylast, copyfirst, max2, sum2fixed, sum2.

- **If the diagnosis is real** (the operation is the wall): copy1 >= 90%, copylast >= 90%, copyfirst >= 70%, max2 >= 80%, and **both
  arithmetic rungs stay at or below 25%**.
- **FALSIFIER**: if either arithmetic rung now passes at >= 80%, the earlier failure was an artefact of the leak, "computation is the
  wall" is WRONG, and that conclusion is withdrawn in public.
