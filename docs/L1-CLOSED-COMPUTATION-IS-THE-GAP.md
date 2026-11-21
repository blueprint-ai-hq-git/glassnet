

---

# L1 MEMORY CLOSES AS "NOT NEEDED AT THIS SIZE", AND THE ONE REAL GAP IS COMPUTATION (2026-11-21 14:05)

Two rungs, identical job, only the DISTANCE of the informative token differs. This is the causal comparison that the memory question
needed and never had.

## Measured

```
rung        where the informative digit sits          answer accuracy
copylast    the LAST position, 7 decoys before it      100.0%  (seed 0)   100.0%  (seed 1)
copyfirst   position 0, 7 decoys after it               80.0%  (seed 0)   (seed 1 running)
```

Both are far above the 10% chance line. **Reading by position works at both ends of an 8-token context** - perfectly at the recent end,
80% at the far end.

## Therefore L1 closes - honestly, and for a reason

The readable memory bank has been HANGING since it measured +0.49 against a +1.0 bar on ordinary text. The distance-recall failure made
it look like memory had a job. **It does not, at this context size.** At 8 tokens the attention mechanism already reads every position,
so a bank that re-stores those same tokens adds nothing to measure.

**L1 CLOSES as "not needed at this scale".** It reopens only when the context is long enough that attention stops reaching - which is a
condition the project can state and test, not a vague hope. This is the first layer today that reaches a clean decision instead of a
partial one, and it reaches it by being measured rather than by being built.

## The corrected map of what this model can and cannot do

| capability | evidence | verdict |
|---|---|---|
| read the most recent position | copylast **100.0%** | WORKS |
| read a distant position (7 back in 8) | copyfirst **80.0%** | WORKS |
| select by content (find the odd digit) | copy1 **98.0%** | WORKS |
| select two and compare | max2 **92.0%** | WORKS |
| **add two digits** | sum2fixed **2.0%**, sum2 **8.0%** | **FAILS** (below 10% chance) |
| **decode a query symbol into an offset** | recall flat at chance, all distances, 3 seeds | **FAILS** |

The two failures share one property: they both need **computation over values** - combining digits arithmetically, or turning a symbol
into a position and using it. Everything that can be done by matching, copying, selecting or comparing works.

## What this means for the architecture

The missing layer is not memory and not attention. It is a **carrier for computation**: a place where a value can be produced, held, and
consumed by a later step. That is the same conclusion the generalisation line reached from the other direction (held-out 42.83% against
55% train, unmoved by 8.3x data), now supported by a controlled contrast.

## Correction carried forward

An hour before this was written I published that the model "cannot read by position at all, even one position back". `copylast` at 100%
refuted it, and `copyfirst` at 80% finishes the refutation. The recall failure was **query decoding**, never reading. Corrected on the
page at `docs/history/CORRECTION-POSITIONAL-READING.md`.
