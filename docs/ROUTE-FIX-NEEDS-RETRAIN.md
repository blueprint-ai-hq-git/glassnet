# The route ordering fix cannot be retrofitted: it changes behaviour and needs retraining (2026-11-13)

The bug: with `route` on, `forward()` captured `_route_base = logits` **before** the direct path contributed, so the `dmix` term was
computed and thrown away (`docs/PARAM-BUDGET-RETRACTION.md`). The fix moves that capture to after the direct path, so the switch
chooses between (main + direct) and the pointer readout.

## The patch was applied, checked, and automatically reverted
`lens/scripts/run_routefix.sh` waits for all trainers to stop, backs `model.py` up to `lens/model.py.before-routefix`, patches,
syntax-checks, then **tests its own behaviour**: on a route-on checkpoint, zeroing `dmix` must now cost more than 1.0 point of
language accuracy (it cost 0.00 before the patch). The measurement:

```
route: True
language: full 0.25%  |  dmix zeroed 14.75%  |  damage -14.50
```

So the patch changed behaviour enormously - and **in the wrong direction for this checkpoint**. Including the direct term in the route
base injects a contribution that the existing weights never learned to balance, and accuracy collapses from **14.75% to 0.25%**;
zeroing `dmix` restores exactly 14.75%. The script's verdict line ("PATCH CHANGED NOTHING") was **wrong** - it tested
`full - zeroed > 1.0`, a sign-dependent condition that cannot distinguish "no effect" from "catastrophic effect" - but its action was
right: it restored the backup, and `lens/model.py` is byte-identical to `lens/model.py.before-routefix` (`diff` confirms).

## What this actually establishes
1. **The ordering bug is real** and the patch changes the model's function, so the two orderings are genuinely different.
2. **A checkpoint trained under one ordering cannot be evaluated under the other.** The `dmix` weights in every route-on checkpoint
   were frozen at initialisation (drift exactly 0.0000), so there is nothing trained to contribute; including the term is not a fix, it
   is an untrained 8.4M-parameter injection.
3. **Fixing it requires retraining with the corrected ordering.** That is a new training arm, not a patch: the model must learn with
   the direct term present, and the first question for that arm is whether it beats the current reference (25.52 by position at epoch
   3 for `matched_d512_s0`) rather than whether the term now influences the output.

## The lesson recorded in the script
The self-check exists because a patch that changes no behaviour and stays in the tree is indistinguishable from a fix. That worked -
it prevented an untrained pathway from being left enabled. The check's *wording* was the part that was wrong, and it is fixed: the
verdict now compares the ablation damage before and after the patch (`|damage_after|` above a threshold), not a signed difference.

## Reproduction
`lens/scripts/run_routefix.sh`, output `logs/routefix/verify.log`, backup `lens/model.py.before-routefix`. Nothing is left patched.
