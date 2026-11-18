# Breaking the memory bank's symmetry pays, but late and only a little (2026-11-18)

The 8-slot memory bank had a proven defect: every slot ended with a **bit-identical** decay, so the bank was one slot with eight times the write rate. Per-slot decay initialised from
`uniform(0.85, 0.95)` (`MEM_SYM=1`) breaks that symmetry. The question this arm asks is whether a differentiated bank actually stores more.

## Two checkpoints, two different answers

```
                                  symmetric      symmetry-broken
slot decay spread (max-min)       0.000000          0.049954   at step 3250
                                                    0.064555   at step 3750
k=16 memory, step ~3250             1.550%             1.512%   <- no benefit yet
k=16 memory, step ~3750             1.605%             2.254%   <- +0.649 points, +40% relative
held-out teacher-forced            21.03%             21.16%
loss                              6.3199             6.3322
```

**The benefit is delayed.** At step 3250 the differentiated bank was marginally *worse* (1.512% vs 1.550%). By step 3750 it is clearly better: **2.254% vs 1.605%**, a **+0.649-point**
gain at **+40% relative**, while held-out improved slightly (21.16% vs 21.03%). An early single-checkpoint reading in either direction would have been the wrong conclusion.

## What this does and does not establish

- **Does:** the symmetry defect was **suppressing** memory, and fixing it recovers real capability. The slots also keep diverging (spread 0.0500 -> 0.0646), so they are not collapsing back
  into one.
- **Does not:** the memory gate is **10%** at k=16. Best measured is **2.254%**, so the gate still **FAILS** by a factor of 4.4. Symmetry was a real but small part of the gap.
- **Caveat:** the arm's ep3750 is compared against the symmetric arm's **ep3500**, because that arm stopped there. 250 steps of training length sit inside the 0.649-point gain, so the clean
  number is the endpoint at 4500, not this one.

## The reading that matters for the line

Two defects have now been found in this mechanism and both were real: the symmetry collapse, and (earlier) the missing `dM` gradient path. Fixing them moved k=16 memory from
**1.312% -> 1.605% -> 2.254%**. Each fix paid, and the gate is still 4.4x away. That is the signature of a mechanism that is **right but starved** - which is what the pre-registered
falsifier for this arm says: a 32-token window never asks the model to remember anything for 16 steps, so there is little training signal for the slots to differentiate *for*.
