# The pointer works. It is being drowned out. (2026-11-11)

The 6,000-step arm scored **94.2% on taught words and 0.5% on held-out words**, which read as "the pointer only
memorises". That reading was wrong, and one diagnostic overturned it.

## The diagnostic
Hold the copy path fixed; compare the learned pointer against an ORACLE pointer that puts all its weight on the answer
position. Computed from the checkpoint's own weights, on the same items the run was scored on
(`tools/copy_pointer_oracle.py`).

```
            pointer-only accuracy   ORACLE pointer   pointer lands on the answer position
taught             84.0%                100.0%               83.5%
held-out           66.3%                 97.7%               85.7%
```

**The pointer copies words it was never trained on at 66.3%, and it looks at the right position 85.7% of the time.**
The copy operation generalises. The mechanism is not the problem.

## So why did the full model score 0.5% on held-out words?
Because the full model's logits are the SUM of three pathways - the concept answer layer, the old `dmix` copy path, and
the new pointer. The pointer alone is right 66.3% of the time; the other two, trained for days on corpus statistics,
happily outvote it and answer `the`. **A correct mechanism that cannot be heard is indistinguishable from a broken one,
and the harness could not tell the difference. The oracle test could.**

This is the same lesson as the concept audit earlier today, in a different costume: measuring the assembled system hides
which part is doing what.

## Two real bugs found while doing this
1. **`save`/`load` silently dropped the pointer.** The weights were written to the file but the `pointer` flag was not,
   so every reload rebuilt a model WITHOUT the pointer and then failed to find the parameters - or, for older files,
   quietly produced a different model than the one trained. This is exactly the class of bug already documented in
   `api.load` for the `direct` path, so it went in the same place, with the same warning.
2. **The Q/K matrices were created before `self.D` existed**, which crashed with
   `'LensNet' object has no attribute 'D'` on every load. Moved to where the width is known.

## What the fix has to be
The pointer needs a way to be HEARD: a learned gate on its output, so the network can turn the pointer up when the task
is a lookup, instead of being outvoted by pathways that were trained on a different objective. That is the next arm,
and it is one change.

Prediction, recorded before building it: with a gate, full-model held-out copying reaches at least 50%, because the
pathway underneath already does 66.3% on its own.
