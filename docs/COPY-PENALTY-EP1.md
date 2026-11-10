# Copy-pressure penalty: first epoch, first sign of the mechanism moving (2026-11-10)

Interim result. **One seed, epoch 1 of 3, the run is still going.** Every number below is epoch 1 against the
matched-split baseline's epoch 1, same data, same recipe, one change: `--lam-copy-penalty 2.0`.

```
                                         baseline ep1     penalised ep1     change
assistant by-position accuracy               19.69%           21.75%        +2.06 pp
validation perplexity                        153.92           144.25        better
emitted token already in the window           76.4%            72.9%         -3.5 pp
emitted token == previous token               38.5%            24.3%        -14.2 pp
emitted token == modal window token           45.8%            38.5%         -7.3 pp
copy path's mass on visible tokens            36.9%            29.7%         -7.2 pp
  (frequency alone justifies about 10%)
```

Source: `lens/runs/MATCHED-COPY-D512.txt` against `lens/runs/MATCHED-D512.txt`, measured with
`tools/copy_echo.py` on `matched_copy_d512_s0_ep1.npz` and `matched_d512_s0_ep1.npz`.

## Why this is the first good sign of the day
Every other lever tried today moved one number at the cost of another, or moved a symptom while the mechanism
stayed put:

| lever | accuracy | copying |
|---|---|---|
| symbol mask | -2.21 pp | punctuation gone, loop relocated to words |
| decode guard | unchanged | repetition gone, symbol appetite untouched |
| copy-pressure penalty (ep1) | **+2.06 pp** | **copy mass -7.2 pp, previous-token echo -14.2 pp** |

The penalty is the only one that reduced the behaviour I am trying to reduce while also improving the accuracy
number. That is not proof it works - it is one seed and one epoch - but it is the first evidence that the echo is a
treatable property rather than a fixed cost of the architecture.

## What is still unproven
1. **One seed.** Today's own lesson is that single-seed gaps of 2 pp are inside the noise of this project
   (measured spread 0.26-3.07 pp on earlier arms). The final epoch, and then two more seeds, decide it.
2. **Epoch 1 only.** Copying grows with training (`COPY-ECHO.md`), so the interesting question is whether the
   penalty still holds at epoch 3, not whether it worked at the start.
3. **Copy mass is still 29.7% against about 10% justified by frequency.** The penalty moved it a fifth of the way,
   not to zero. Whether the remaining gap is what makes the text repetitive is untested.

## Honest reading
The echo fell, accuracy rose, and both are measured on the fair split. If that holds at epoch 3 across seeds, this
is the first lever worth building on; if it does not, it joins the mask and the guard as a lever that looked good
at epoch 1.
