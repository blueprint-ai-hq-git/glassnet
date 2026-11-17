
## Correction: the arm is stuck, not worse - I compared two different quantities (2026-11-17)

Two earlier write-ups (`WARMSTART-PILOT.md`, `WARMSTART-G3.md`) said the warm-start pilot "ends worse than the model it started from", quoting val **6.5568 / 6.5673
nats against 6.06 for the shipped model**. That comparison is invalid and is withdrawn here.

`tools/check_eval_protocol.py` scores the shipped model on **the pilot's exact 40 held-out windows and the exact same positions**:

```
SHIPPED MODEL, last-position loss on those 40 windows : 6.5694 nats
PILOT after 9 steps, identical windows                : 6.5700 nats
difference                                            : +0.0006 nats
```

**The 6.06 figure came from scoring every position of held-out text** (`tools/bpc_ours.py`: 8.7393 bits/token = 6.0566 nats). Scoring the last position of a window
and averaging is a different quantity - it conditions on 32 tokens of context instead of varying context length - and it is systematically higher. The two numbers
were never comparable.

## What is actually true

- The arm is **neutral**: +0.0006 nats after 9 steps at batch 16, learning rate 1e-4, clipped.
- Reported honestly: **9 steps of training moved nothing measurable**, on either side.
- The earlier framing - "worse than the model it started from" - overstated the failure and would have sent the next attempt hunting a regression that does not exist.

## Why this correction matters more than the number

The failure mode is the one this project keeps meeting: **two numbers that look comparable and are not.** Every time it has happened, the model was not the problem -
the instrument was. Here the pilot is stuck, which is a much easier problem than a regression: stuck means the update direction is not producing held-out
improvement, so the next steps are a proper rate schedule, sequence-length context, and far larger batches - not a hunt for what broke.

The full protocol for the arm is now pinned: evaluate on the same windows, the same positions, and compare against a re-measured baseline computed in the same run.
A number from a different protocol never enters the comparison again.
