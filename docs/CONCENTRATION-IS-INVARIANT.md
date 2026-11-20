

---

# ONE COUNT-INVARIANT SCALE FOR EVERY CONCEPT ARM, AND IT CORRECTS MY LAST CLAIM (2026-11-20 13:15)

I flagged in the previous entry that the concentration test ("top-8 suppliers carry >= 80% of positive supply") is **vacuous at 9
concepts**, because top-8 of 5 positive units *is* the whole layer. So the instrument was changed to two measurements that do not
depend on layer size:

- **top_half_share** - share of positive supply held by the strongest **half** of the positive suppliers
- **supply_entropy** - Shannon entropy of the positive supply distribution / log(n_pos), where **1.00 = perfectly spread**

Every arm measured today, re-scored from its saved JSON with no retraining:

```
result file                   n   n+    supply  top-half entropy   recon
CONCEPT-EXACT-100K-ON        36   20     8.48x    79.6%   0.903   4.63e+00   <- direct path, large residual (bias+direct)
CONCEPT-EXACT-OFF            36   17     9.45x    74.2%   0.950   4.13e-02
CONCEPT-EXACT-ON             36   17    16.43x    74.6%   0.949   3.45e-01
CONCEPT-COUNT  (9 concepts)   9    5    19.84x    79.7%   0.927   7.10e-01
DECORR-SWEEP-d05             36   17    38.05x    76.6%   0.943   2.53e-01
DECORR-SWEEP-d10             36   19    38.36x    81.2%   0.906   2.50e-01
DECORR-SUPPLY-d02            36   17    62.60x    76.7%   0.943   2.54e-01
CONCEPT-EXACT-100K-OFF       36   19   236.03x    81.6%   0.906   2.56e-01
DECORR-SUPPLY-ctrl           36   19   236.04x    81.6%   0.906   2.56e-01
```

Two arms in `lens/runs/` (`CONCEPT-RANK-ON.json`, `CONCEPT-RANK-OFF.json`) come from the **non-additive** path and are labelled here
as **not quotable**; they are kept, not deleted, so nobody re-derives them.

## What the invariant scale says, and the correction it forces

**Concentration is flat across every configuration measured:** top-half **74.2% - 81.6%**, entropy **0.906 - 0.950**. Nothing in this
line has ever pushed attribution into a few concepts - the positive supply is close to perfectly spread in all of them, at 9 units and
at 36.

So the previous entry's reading needs correcting. "Fewer concepts cuts cancellation 48%" is still true **as a statement about
volume** - positive supply fell 38.05x -> 19.84x - but on the count-invariant scale the 9-concept arm's distribution shape
(top-half **79.7%**, entropy **0.927**) sits **inside the same band** as the 36-concept arms. Dropping from 36 concepts to 9 reduced
how much the units push, **not** how concentrated the push is.

That matters because the goal was never "less total push" - it was "a human can read a few rules and know the answer". On every
measurement available, **no configuration yet produces that**. The honest summary of the whole concept line today:

1. share is not specificity (a bottleneck pushes share to 100% while the named-vs-random ratio falls);
2. the layer answers by cancellation (supply up to 236x the net margin);
3. decorrelation removes most of the volume (236x -> 38x) and saturates at 0.5;
4. fewer concepts removes more volume (38x -> 20x) without an accuracy cost, and every one of the 9 passes its audit;
5. **but concentration never moves** - top-half stays ~74-82%, entropy ~0.91-0.95, at every count and every strength.

Point 5 is the one that decides the architecture bet, and it is why the next arm (`narrowdir`, 9 concepts WITH the direct path) must be
read on the invariant scale rather than on supply alone.
