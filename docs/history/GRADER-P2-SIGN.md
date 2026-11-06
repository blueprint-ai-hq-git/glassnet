# The P2 sign bug in `wall_ab_grade.py` — wrong conclusion stapled to right numbers

Written 2026-11-06 17:5x, after five of six arms existed. Read this before quoting any P2 delta.

## What was pre-registered

`docs/history/WALL-AB-PREDICTIONS.md:37-40`, written before any arm ran:

> P2 MAIN. `wall_off` is LOWER than `wall_on` on BOTH seeds. Predicted delta: -1.0 to -6.0 points final
> val acc. FALSIFIER: if `wall_on - wall_off` is smaller than 0.5 pp on both seeds, the bypass earns
> nothing at this rung ...

So the predicted quantity is `off - on` in the band -1.0 .. -6.0, and the falsifier is a **no-effect**
band (`wall_on - wall_off` under 0.5 pp, i.e. `|off - on| < 0.5`).

## What the grader computed (the bug)

```python
deltas = []
for s in SEEDS:
    off, on = d[("off", s)]["acc"], d[("on", s)]["acc"]
    deltas.append(on - off)                                  # <-- on - off, not off - on
    print(f"  s{s}: on {on:.2f}% - off {off:.2f}% = {on - off:+.2f}")
if all(x < 0.5 for x in deltas):
    print("  -> FALSIFIED: bypass earns under 0.5 pp on both seeds ...")
elif all(x > 0 for x in deltas):
    band = all(-6.0 <= x <= -1.0 for x in deltas)            # <-- band tested against on - off
    print(f"  -> PASS direction; magnitude {'inside' if band else 'OUTSIDE'} the -1.0..-6.0 band")
```

Two defects, both of the same kind (each produces a confident sentence that the numbers do not support):

1. **The falsifier swallowed a real loss.** `on - off < 0.5` is satisfied both by "no effect" (+0.2) and
   by "the bypass is clearly worse" (-3.1). Measured at seed 1, `on - off = -3.11`, so the old code would
   have printed *"FALSIFIED: bypass earns under 0.5 pp on both seeds"* — a no-effect claim about a 3.11 pp
   loss. The conclusion happens to point the same way as the truth, which is exactly why this kind of bug
   survives review.
2. **The predicted band was unreachable.** The band test only ran inside `elif all(x > 0 ...)`, where
   `-6.0 <= x <= -1.0` is impossible. A *perfectly* predicted result would still have printed
   "magnitude OUTSIDE the -1.0..-6.0 band". No run of the old code could ever have recorded a P2 hit.

## What it does now

`off - on` is the delta, the falsifier is `all(|off - on| < 0.5)` (a genuine no-effect band), a clear
loss with the opposite sign prints as `OPPOSITE SIGN ... do not spin it as noise`, and every seed prints
both differences side by side:

```
s0: off 12.18% - on 10.79% = +1.39   (on 10.79% - off 12.18% = -1.39)
s1: off 12.79% - on 9.68% = +3.11   (on 9.68% - off 12.79% = -3.11)
-> OPPOSITE SIGN: off is ABOVE on by more than 0.5 pp on both seeds ...
```

This is a fix of the grader to match the pre-registration, **not** a redefinition after seeing results:
the numbers printed by old and new code are identical (`on - off` -1.39 / -3.11), only the sentence
attached to them changed, and the change is towards the pre-registered reading. The five-arm grade in
full is `lens/runs/WALL-AB-GRADE.txt`.

## How it reached the repo, stated plainly

The edit was made by a **concurrent** agent turn (a `glassnet_wake.sh` worker started 17:45:10 while a
chain-triggered agent started 17:43:32 was still running). That turn's `git add -A` swept the
half-finished file into commit `68557e5`, whose message does not mention it. The edit was then checked
from the outside: `py_compile` clean, and the run quoted above is the file's real output on five arms.
This document exists because an unmentioned change to a grading script is a provenance hole, even when
the change is right.

The rule the project already uses covers this: print the compared values beside the verdict, because
numbers are safe and the sentence attached to them is the risky step.
