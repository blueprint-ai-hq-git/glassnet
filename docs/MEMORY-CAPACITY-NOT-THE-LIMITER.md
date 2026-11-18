# The memory bank CAN hold 16 steps. The gate was never about capacity. (2026-11-18)

The memory gate - k=16 recall above 10% on real text - failed after every mechanism fix: **1.312% -> 1.605% (symmetric bank) -> 2.214% (symmetry broken)**, with the gate unmoved. This
line's pre-registered falsifier said the limiter was **demand**, not capacity: a 32-token window of ordinary text never asks the model to retain anything for 16 steps, so the slots have nothing
to learn for and a better bank mostly stores better noise. That was a claim about what the model is *asked* to do, and it is now measured.

## The test

A synthetic task with exactly the demand in question: a random token at position 0, filler after it, and the model must output that token at position D. Nothing else is supervised. The
memory module is the one already certified in `lens/memchan.py`, run in its **batched** form with one bank per sample.

The probe's own hand-rolled backward pass was **certified before any result was quoted**, against central finite differences on the full probe loss: worst relative error **6.789e-08** across
`Wo`, `Wq`, `Wk`, `decay`, the embedding table and the readout (`tools/delayed_recall_gradcheck.py`).

```
delay   bank arm   control arm
  1      100.00%       3.12%
  4      100.00%       3.32%
  8      100.00%       3.03%
 16       81.25%       3.22%      <- FULL budget (12000 steps)
chance     3.12%      (vocab 32)
```

The **control arm** is the same readout with the memory channel removed, so it sees only the final token. It sits **exactly at chance** at every delay it was run at, which is what makes the
result attributable to the bank rather than to a leak in the wiring.

## What this settles

**The bank can hold a fact for 16 steps when a task demands it.** The falsifier for the memory line ("delay 16 at or below 10% would mean capacity is the limiter and the falsifier's branch
is wrong") **did not fire**. So the k=16 gate on real text fails **not because the bank cannot hold, but because ordinary text never asks it to.**

Three defects were found and fixed in this mechanism and each paid (+0.293, +0.609), totalling **1.312% -> 2.214%** against a 10% gate. Capacity was never the binding constraint; the
binding constraint is the **training signal**.

## Honest limits of this result

- The synthetic task is **much easier than text**: a 32-token vocabulary, one deterministic fact per sequence, perfect recall of a single token. It bounds what the bank *can* represent, it does
  not show that it will reach the gate on natural text at this scale.
- Delay 16 was first read at **half** the budget (6000 steps, 81.93%) and is now confirmed at the **full** 12000-step budget: **81.25%** with its **control at 3.22%**. The two budgets agree
  within 0.68 points, so 16-step recall **plateaus near 81%** rather than climbing to 100% - capability at 16 steps is real but does not match the perfect scores at 1, 4 and 8.
- This says nothing yet about whether adding real-text demand *works* - that is a separate arm, in flight, with its own written predictions.

## The consequence for the line

The next experiment is not another bank fix. It is a training objective with genuine cross-step demand on the real corpus. That arm exists now (`TGT_DELAY` in the trainer: the target at
scored position t becomes `tok[t+1-16]`), and its own falsifier closes the line if the k=16 probe stays at or below 2.214%.
