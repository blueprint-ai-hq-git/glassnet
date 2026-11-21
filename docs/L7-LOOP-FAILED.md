

---

# L7 LOOP FAILS: REPEATED PASSES CHANGE NOTHING (7.0% at R=1, R=4, R=8) (2026-11-21 14:45)

The loop was built to supply the computation the architecture is missing, and it was certified before use (finite differences worst
relative **2.141e-07** across A, B, P, lnf_scale, blk_Wq, blk_W1; isolation **exactly 0.000e+00**; **zero extra parameters**, so a loop
could not be confused with more capacity).

## Measured

```
task       R=1     R=4     R=8
sum2fixed  7.0%    7.0%    7.0%      <- identical to the last decimal
sum2         -     15.0%    -
copy1        -    100.0%  100.0%     <- the loop runs correctly and costs nothing that works
```

- **P1 control MET:** R=1 gives 7.0%, which is the baseline in this configuration (the ladder's 2.0% came from a wider model, so the honest
  comparison is the in-run control, as stated before the run).
- **P2 NOT MET:** the loop was supposed to lift sum2fixed to >= 50%. It moved it by **0.0 points**.
- **P3 MET:** copy1 stays at 100.0% at R=4 and R=8, so the loop is not breaking anything.
- **FALSIFIER FIRED:** every R stays <= 10% on the arithmetic task.

## What this rules out, precisely

**Repeated application of the same block does not supply computation.** Two passes, four passes and eight passes over the same weights
produce the same answer. Combined with the ladder, the eliminations now read:

| candidate cause | test | result |
|---|---|---|
| needs more context / memory | L1 distance recall | not needed at this size (both ends readable) |
| needs selection or routing | copy1 100.0%, max2 100.0% | already perfect |
| needs more capacity | capacity allocation 3 seeds | no reliable gain (+0.27 +- 0.31) |
| needs more data | 600 -> 5000 pairs | +6.11 points, bar missed |
| needs more compute passes | **L7 loop R=1/4/8** | **0.0 points** |

## The honest caveat, stated before the next step

At this budget the model does **not learn the task at all** at R=1 either (7.0% is chance for a 10-way answer), so the loop was asked to
improve something that had never been learned. That is a real limitation of this test, and it is recorded as one. But the pre-registered
rule was explicit: *if every R stays <= 10%, L7 fails and the next attempt is serialising the computation into emitted tokens, not a bigger
loop* - so the next step is the scratchpad, and re-running the loop with longer training would be a rescue attempt that has to be
pre-registered separately rather than quietly folded in.

## Next, and its predictions are already written

`lens/scratchpad.py`: addition becomes counting. The model must **write** (d1+d2) ONE tokens and then STOP, so the computation happens in
the open, one token at a time, where a human can check every step. P1 exact-count >= 50%; P2 within +-1 >= 80%; falsifier <= 20% exact,
in which case output format does not fix it either.
