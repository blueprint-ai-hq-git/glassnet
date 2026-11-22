

---

# UNIVERSAL: THE READABLE LAYER IS NEVER THE MAJORITY OF THE DECISION - AND THE CONCENTRATION BAR IS GAMEABLE (2026-11-22 15:30)

Three chat checkpoints, the corrected aggregate statistic, 200 windows each. Every accounting check closes to ~2e-15, so the instrument
is valid on all three.

```
checkpoint                      concepts positive   MEDIAN ratio   AGGREGATE share   bias+direct   top-8 concentration
d1024_s1_best                        22 of 36           10.4%            18.2%          81.8%             63.9%
d1024_e9_s1_best                     20 of 36           -0.5%             0.2%          99.8%             81.3%
direct1024_d512_s0_best              14 of 36            4.5%            10.2%          89.8%             92.5%
```

## My predictions, and their fate

- **P1 "the aggregate concept share is below 35% on all three": MET** (18.2%, 0.2%, 10.2%).
- **Falsifier "any checkpoint above 50%": did NOT fire.**
- **P2 "the positive-supply count stays in 21..25 of 36": NOT MET** (22, 20, **14**).

## The load-bearing conclusion

**On every model measured, the readable concept layer supplies between 0.2% and 18.2% of the decision margin, and the bias plus direct
path supplies the other 81.8% to 99.8%.** The concept layer is **never** the majority of the answer. That is now measured on three
independent checkpoints with a valid instrument, and it is the single most important fact about the readable middle layer in this
project.

## The trap, and it is the best thing found in this run

Look at the last two columns together:

```
d1024_e9_s1_best        concept share 0.2%      top-8 concentration 81.3%   <- "MEETS" the 80% bar
direct1024_s512         concept share 10.2%     top-8 concentration 92.5%   <- "MEETS" the 80% bar
d1024_s1_best           concept share 18.2%     top-8 concentration 63.9%   <- fails
```

**The two models whose concepts explain almost nothing score HIGHEST on the concentration bar.** A layer contributing 0.2% of the
margin can have perfectly concentrated internal supply and mean absolutely nothing - and the bar would print **MET**.

So the 80% concentration bar is **gameable**: it measures concentration *within concept supply* and says nothing about how much of the
decision that supply is. From now on **concentration may never be quoted without the aggregate share beside it**, exactly as a
constant baseline must never be quoted without the accuracy it is being compared to. Any earlier "concentration MET" in this project's
record is now suspect for the same reason.

## Also corrected: the supply split is sample-sensitive

At 60 windows the split read 23 / 22 / 23; at 200 windows it reads 22 / 20 / **14**. So "22-23 of 36 concepts supply positive margin"
is **not** a stable constant - it depends on both the model and the number of windows. Any future statement of it must carry its n.

The honest version of that fact: **a minority of concepts supply positive margin, and which minority it is moves with the sample.**
