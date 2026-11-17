
## The gated cell is certified: the gradient bug was one transposition (2026-11-17)

The previous entry documented `lens/cell.py` as uncertified: gradients exact at T=1 (2.4e-12) and wrong from T=2 upward, with the analytic value consistently larger than the numeric one.
**That is fixed and the module is now certified at every length this project requires.**

## The bug, found by bisecting on controlled loss weights

With T=2 and only the FIRST step carrying loss weight, the check passed (2.4e-12). With only the LAST step carrying weight, it failed. That localised the fault to the last step's own
gradient rather than to the carry path.

The cause is a **row-vector transposition**:

```python
# forward computes  c_prev @ Wf          (row vector times matrix)
# so the backward is  gf @ Wf.T
dc = dc * f + gf @ P["Wf"].T + gi @ P["Wi"].T + gv @ P["Wv"].T     # correct

dc = dc * f + P["Wf"].T @ gf + ...                                  # what I wrote: column-vector form
```

**T=1 hid it completely**: that line is never reached at the final step, and at T=1 the previous state is zero anyway, so every term through it vanished. A check that only ran T=1 would have
certified a wrong gradient - which is why the project's rule is a check at *several* lengths, not one.

## Certified

```
   T    worst err   where        verdict
   1    8.977e-09   Wxi(0,2)     PASS
   2    5.651e-07   Wf(3,3)      PASS
   3    1.336e-06   Wi(1,1)      PASS
   8    1.431e-07   Wxf(0,0)     PASS
  16    1.993e-06   Wxf(3,0)     PASS
  32    3.758e-07   Wf(3,4)      PASS

CELL CERTIFIED at every required length   (relative error, finite differences, float64)
```

`tools/cell_gradcheck_all.py` runs the whole sweep in under a second, so it can be part of every future launch check rather than a one-time ceremony.

## Where the cell stands

- **Structure**: separate linear cell state `c_{t+1} = f*c_t + i*v`, never through `tanh`, with learned forget/input gates - built to answer the measurement that memory cannot live inside a
  saturating nonlinearity.
- **Gradients**: certified at T = 1, 2, 3, 8, 16, 32.
- **Memory, untrained**: k=16 shift **3.10%** against the 10% gate (the tanh carry reached 1.719%). Untrained gates have had no opportunity to learn to hold anything, so this is a property of the
  initialisation, not a verdict on the mechanism.

## The next question, unchanged and now askable

**Do trained gates clear the gate?** Train the cell on next-token prediction with the spectral clamp **off**, then re-measure the k=16 shift against the 10% bar with accuracy held at or above 16%.
The falsifier stands as written: if trained gates also fail to lift k=16 above 10%, then no recurrence of this size holds 16-step memory on this corpus and the memory question is answered
negatively for the whole family.

Nothing is launched yet - the integration needs its own end-to-end check that the cell's gradient path composes correctly with the existing readout, which is a separate verification from the
cell's own certified backward.
