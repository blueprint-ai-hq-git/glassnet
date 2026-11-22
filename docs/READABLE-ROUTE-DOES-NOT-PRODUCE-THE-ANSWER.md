

---

# THE READABLE ROUTE DOES NOT PRODUCE THE ANSWER: 5.5% TO 18.0% (2026-11-22 15:36)

The share numbers said the concepts supply 0.2-18.2% of the *margin*. The question that actually matters is different, and it now has a
direct measurement on three checkpoints with no training and no labels involved:

> **If the unreadable direct path were removed, would the model still give the same answers? No.**

```
checkpoint                      windows   concept route alone   direct route alone   mean rank of the model's own top token
                                          reproduces the answer                      under the concept-only logits
d1024_s1_best                     200            18.0%                  65.0%                    1,133
d1024_e9_s1_best                  200            17.0%                  84.5%                    2,159
direct1024_d512_s0_best           200             5.5%                  73.0%                    2,090
```

**P1 (concept route below 50% on all three): MET. P2 (mean rank worse than 5): MET. Falsifier not fired.
`OUTCOME: DIRECT_PATH_CARRIES_THE_ANSWER`.**

## What the numbers say in plain terms

- **The readable middle layer, on its own, picks the same token as the model only 5.5-18% of the time.** If you delete the unreadable
  path, the model's answer changes in 82% to 94.5% of cases.
- **The unreadable path alone** reproduces the model's answer **65% to 84.5%** of the time.
- The **mean rank** of the model's own chosen token, under the concept-only logits, is **1,133 to 2,159 out of 12,002**. So the concept
  route does not merely fail to rank the right answer first - **it usually does not have the right answer near the top at all.**

## This is the honest core of the readable-layer problem

Every readable-layer claim this project has made has to be read against this number. The concept layer:

- is **namable, auditable, and honest about what its units track** (12 of 14 image concepts matched their definitions, once the name
  file was corrected);
- but **is not the route the answer travels**, on any of the three language checkpoints measured.

So the failure was never "the concepts are badly named" or "the concepts are too few". **The architecture computes the answer along a
path that bypasses the concepts, and the readable path is a side channel.** That is why:
- the margin share is small,
- concentration within concept supply is gameable and meaningless on its own,
- and every attempt to explain the model by naming concepts has felt like a story told afterwards.

## The repair, stated as a single measurable target

**The concept route must reproduce the model's own answer at 100%, and the direct route must be worth nothing on its own.** Anything
less means the readable layer is still a side channel. That is now a gate with a number on it, measured the same way the negative result
was measured, so it cannot be talked around: run `tools/direct_ablation.py`, and read the first column.
