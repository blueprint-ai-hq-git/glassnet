

---

# CORRECTION: THE CONCEPT LAYER SUPPLIES ~15-18% OF THE ANSWER, NOT 993% - AND MY STATISTIC WAS THE BUG (2026-11-22 15:26)

Two claims published in the last hour are **withdrawn**: "the concept layer supplies 993.4% of the margin and the bias+direct path
removes 893.4%" and the "66-68% concentration ceiling across three models". Both rested on a statistic that is invalid.

## What was wrong

The tool divided **each window's** concept contribution by **that same window's** margin and **averaged the ratios**. Any window with a
tiny margin makes that ratio explode, so the mean is dominated by the least informative windows. Measured on the same model:

```
                        MEAN of per-window ratios      MEDIAN ratio      AGGREGATE (total contribution / total margin)
 60 windows                    993.4%                     33.5%                        14.8%
200 windows                    126.8%                     10.4%                        18.2%
```

**The mean of ratios moved 7.8x between two sample sizes.** That number was a sampling artefact, not a property of any model. The
**aggregate** moves 3.4 points and the universe of plausible values is stable. **Only the aggregate and the median may be quoted**,
and the tool now prints all three, labelling the mean as "UNSTABLE, do not quote".

## The real picture, and it is the opposite of what I said

```
at 60 windows:   concept layer 14.8% of the margin   |  bias + direct path 85.2%
at 200 windows:  concept layer 18.2% of the margin   |  bias + direct path 81.8%
```

**The readable middle layer carries roughly ONE SIXTH of the decision, and more than 80% of the answer comes from the bias plus the
direct path.** It is not "ten times bigger than the answer and cancelled" - it is **small, and the part that decides is the part that
is not readable through the concepts at all.**

## Which makes the concentration ceiling obvious instead of mysterious

The top-8 concepts carry **68.2%** (60 windows) / **63.9%** (200 windows) of *positive concept supply* against an 80% bar. That was
always going to fail: **a layer that accounts for ~18% of the margin cannot have a subset that accounts for 80% of it.** The bar was
measuring concentration *within the concept supply*, which is a different quantity from concentration *of the decision*, and the two
had been quietly conflated.

## What survives, and what the repair actually is

**Survives:** the supply split at **22-23 of 36 concepts positive** (stable across sample sizes and checkpoints); the accounting
identity closing at **1.8e-15 to 2.7e-15**; and **concentration NOT MET** - which is now explained rather than merely reported.

**Withdrawn:** 993.4% / -893.4%; "concept layer ten times larger than the answer"; the "66-68% ceiling across three models" as a
measured ceiling (all three of those runs used the invalid mean-of-ratios AND only 60 windows).

**The repair target, restated and now measured:** the answer head must take **most of its margin from the concepts** instead of ~1/6,
with the bias and direct path shrinking from ~82% to a minority. Until that changes, no amount of concept naming can make this layer
the explanation of the answer, because it is not where the answer comes from.
