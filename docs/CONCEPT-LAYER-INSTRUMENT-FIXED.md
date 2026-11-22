

---

# THE CONCEPT LAYER WAS BEING JUDGED BY A BROKEN INSTRUMENT - AND THE REAL ANSWER IS WORSE THAN "CANCELLATION" (2026-11-22 15:14)

The recorded status of the readable middle layer was: **"19 of 36 concepts oppose their own label; top-8 supply 68.7% against an 80%
bar."** That figure comes from single-concept ablation, and `tools/concept_rank.py` says of its own output:

> "Those numbers are not a failed prediction, they are an invalid instrument."

So it was re-run on `lens/runs/chat/d1024_s1_best.npz` with the valid path (`--fixed-runner`). **That path was broken too, in three
places, and all three had to be fixed before any number could be quoted:**

1. **It crashed** on `_pos[_ordp[:_npos] / _pos.sum()]` - dividing an INDEX ARRAY by a sum and then using the result as an index
   (`IndexError: arrays used as indices must be integer`). The intent was to select the positive suppliers, then normalise.
2. **Its own accounting check failed**: `max |sum(contribution) - margin| = 5.566e+00`. Root cause: the check compared the
   **concept part** of the margin against the **whole** margin, which also contains the bias and the direct path.
3. **A stale variable**: `direct_vec` was never computed inside the attribution loop at all - the name referred to the **previous
   loop's last iteration**, so the residual of a different window was being used. This was the real reason the accounting could not
   close.

After all three fixes:

```
windows: 60   max |sum(contribution) - margin|: 1.776e-15   (a check that CAN fail, and now closes to machine precision)
```

## The corrected numbers

```
23 of 36 concepts supply POSITIVE margin        (the recorded figure was 17)
  the concept layer as a whole accounts for 993.4% of the margin
  the remaining -893.4% is the bias plus the direct path
top-8 concepts among the positive suppliers carry 68.2% of all positive supply   (bar 80%)  -> NOT MET
supply entropy / log(n_pos) = 0.890   (1.00 = perfectly spread)
```

## What this actually says, and it is not "cancellation"

**The concept layer supplies 993.4% of the margin and the bias-plus-direct path removes 893.4% of it.** The final margin is a residue
of about **one per cent** of what the concept layer contributes. The concept layer is not weak - it is **ten times larger than the
answer it produces**, and almost all of it is cancelled by the direct path.

That single number explains three things that previously looked like separate mysteries:
- why single-concept ablation was **non-additive** and reported negative contributions for 19 concepts: it was measuring a ~1%
  residue of two enormous opposing quantities, so the runner-up class flips and the sign flips with it;
- why the recorded "19 of 36 oppose" was unstable and why the valid instrument gives **23** instead;
- why the top-8 concentration test keeps failing around 68-69% at an 80% bar: a decision carried by a 1% residue cannot be concentrated
  into a handful of names.

## Status of L4 after this

- **Instrument: now valid** (accounting closes at 1.776e-15). Every future claim about this layer must use `--fixed-runner`, and the
  three bugs above are fixed in the tool.
- **The 17-supply/19-oppose figure is withdrawn**; the corrected split is **23 supply / 13 oppose**.
- **The concentration bar is still NOT MET** (68.2% against 80%) - the honest conclusion from yesterday stands, but now for a measured
  reason rather than an instrument artefact.
- **The repair target is now precise:** the readable middle layer must be made to carry the answer **without** a -893% direct-path
  correction. That is a structural change to how the answer head reads the concepts, not a training-duration problem.
