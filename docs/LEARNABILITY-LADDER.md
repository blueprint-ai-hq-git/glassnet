

---

# THE LADDER: SELECTION AND COMPARISON WORK, ARITHMETIC DOES NOT (2026-11-21 12:20)

Four rungs of the same probe, easiest first, one requirement removed at each step. This was run to find out whether the salience probe
was failing because of the TASK or the MODEL. The answer is neither of the two options I offered - it is a third thing.

## Measured, 2000 steps per rung, same model, same protocol

```
rung        what it requires                                          answer accuracy   verdict
copy1       find the ONE digit among 7 decoys and copy it                  98.0%        LEARNED
sum2fixed   two digits handed at positions 0 and 1, output their sum        2.0%        NOT LEARNED
max2        find TWO digits among 8 decoys, output the larger              92.0%        LEARNED
sum2        two digits among 8 decoys, output their sum mod 10             (running)
```

## What this settles, and it is worth more than the probe

1. **The pipeline works.** copy1 at **98.0%** is the positive control. Everything downstream is measurable.
2. **Content-based selection works.** max2 needs the model to find two digits among eight decoys - no positional cue - and it reaches
   **92.0%**. So the model can locate what matters in its input.
3. **Comparison works.** max2 also requires comparing the two digits it found and emitting the larger one.
4. **Arithmetic does not.** sum2fixed puts the two digits at positions **0 and 1** - the easiest possible setup, no selection needed at
   all - and the model scores **2.0%**, BELOW the 10% chance line.

## Why this is the answer to the whole week's question

Every earlier failure on this line was attributed to data (+6.11 points from 8.3x the pairs), capacity (-0.94 or +0.27, unstable), or
the probe being too hard. **All three were wrong.** The model can select, can compare, and cannot add - and it fails to add even when
handed the operands. That is a missing **structure for computation**, exactly as the pre-registration on generalisation predicted, and it
is the first diagnosis in this project that is supported by a controlled contrast rather than a correlation.

## Consequences, applied immediately

- **L3's certificate moves to `max2`** - the layer must be certified on a rung the model actually solves, otherwise its numbers are
  noise. Running now with three seeds.
- **The self-cleaning run on `sum2fixed` was stopped**, not because it failed but because it was measuring a task the model cannot learn:
  its three arms sat at **9.0% / 10.0% / 7.5%**, all at chance, so the comparison was meaningless. It will be re-run on `max2`.
- **The structural change this justifies** is a loop over output tokens (a scratchpad), because selection and comparison already work
  and only the computation is missing.

## Recorded honestly

I predicted R3: that the ladder would be ordered, with accuracy falling from copy1 to sum2. **The actual order is not monotonic** -
sum2fixed at 2.0% is far below max2 at 92.0% even though sum2fixed removes the selection requirement entirely. The prediction was
wrong and the inversion is the finding.
