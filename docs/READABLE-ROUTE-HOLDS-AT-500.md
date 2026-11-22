

---

# THE READABLE ROUTE HOLDS AT 500 WINDOWS: 4.2-21.6% VERSUS 64.4-81.6% (2026-11-22 19:37)

The finding that decides the architecture was measured at 200 windows. Today showed twice that a result can move with sample size, so it was
re-measured at **500 windows** per checkpoint - 2.5x the sample.

```
checkpoint                     windows   concept route alone   direct route alone   mean rank of the model's own token
d1024_s1_best                    500          21.6%                 64.4%              1,021 of 12,002
d1024_e9_s1_best                 500          18.6%                 81.6%              1,995
direct1024_d512_s0_best          500           4.2%                 70.0%              2,092
P1 concept route below 50% on all three: MET
P2 mean rank worse than 5 on all three: MET
FALSIFIER not fired.  OUTCOME: DIRECT_PATH_CARRIES_THE_ANSWER
```

**At 200 windows it was 5.5-18.0% / 65.0-84.5%; at 500 windows it is 4.2-21.6% / 64.4-81.6%.** The same conclusion inside the same small band:
**the readable layer, on its own, does not reproduce the model's answer, and the unreadable path does.**

This is the number that any future claim about the readable middle layer has to be quoted against. It now rests on three checkpoints, two
sample sizes, three different readouts (linear fit, wider fit, nearest neighbour) and a parameter-free nearest-neighbour check - all
agreeing.

## What was launched next, and why

With the queue empty, the next experiment attacks the one gap the day's successes leave open. All seven written-reason tests used a **cue
token I provided** - a dedicated id that nothing else in the input resembles, which is a hint in itself. The new task, `trigword`, draws
the trigger word from the **same id pool as the decoys**: among the non-digit tokens one means something and the rest are identical-looking
noise. The model must find it, act on it, and **name it as its reason**.

Bars: reason >= 95%, all three tokens >= 90%, honesty >= 95%.
Falsifier: reason below 80% means the earlier perfect scores depended on the cue being special, and finding a meaningful word among
look-alike words needs something these models do not have.
