

---

# THE CONCEPT LAYER ANSWERS BY CANCELLATION (2026-11-20 13:35)

This is the structural result behind every earlier oddity - the concept share that overstated itself by up to 60x, the `explain()`
that was exact on 0 of 200 windows, and the fact that removing the direct shortcut raised the share to 100% without making the
concepts more specific.

## Two instruments, one of them mine and WRONG

`tools/concept_rank.py` first attributed each concept by ablating it alone and re-computing the top-1/top-2 margin. That produced
**negative "margin removed" for 19 of 36 concepts** and a top-8 total of **-302% (ON) / -204% (OFF)** against an all-36 figure of
66.41% / 85.63%. Those are not a failed prediction: they are an **invalid instrument**. The margin is not linear in the concepts,
because ablating a unit can change *which class is runner-up*.

The fix is to hold the runner-up class **fixed at its baseline value**, which makes the split exactly additive:

```
contribution_i = h_i * (W[i, pred] - W[i, runner])        sum_i contribution_i = margin   (checkable)
```

The residual of that check is **4.134e-02 (OFF)** and **3.445e-01 (ON)** - not zero, and honestly so: it equals the difference in the
class bias `(b + direct)` between the two classes, which the concept terms cannot supply. That check can fail, and its failure has
a name.

## What the exact attribution shows, 96 usable windows per arm

```
                                    --direct ON        --direct OFF
concepts supplying POSITIVE margin  17 of 36           17 of 36
positive supply vs the net margin   16.43x             9.45x
top-8 of the positive suppliers     68.7% of positive   67.4% of positive
strongest supplier                  is_a_request +223.8% is_a_request +123.9%
strongest opposer                  mentions_technology -262.0%  mentions_technology -153.7%
```

- **P1 NOT MET in both arms:** the top-8 suppliers carry **68.7% / 67.4%** of the positive supply against an 80% bar. Attribution is
  spread over 17 concepts, not concentrated in a handful.
- **P2 MET:** the **same** concepts lead in both arms (`is_a_request`, `is_a_list`, `mentions_place`, `mentions_feeling`, then
  `mentions_a_quantity`, `mentions_health`, `mentions_education`, `mentions_money`). The load-bearing set is a property of the
  architecture and the data, not of which arm trained it.
- **The finding that matters:** the concepts supply **9.4x to 16.4x** the net margin. The answer is not carried by the concepts; it
  is the **small residue of large opposing pushes**. 19 of 36 concepts actively oppose the prediction. `mentions_technology` alone
  pushes the margin **-262%** on the ON arm.

## Why this explains everything above it

A layer that decides by cancellation cannot be explained by naming its top-k contributors, because the named ones are a minority of
a far larger opposing sum. That is exactly the measured `explain()` exactness of 0 of 200, and exactly why the share metric could be
made to say 46% when the net was 1.3% of the total flow.

## The next experiment this dictates

Not "which concepts matter" - that question is answered: 17 positive, 19 opposing, spread. The question is whether the cancellation
can be **removed structurally**, and the codebase already has the knob for it: `lam_decorr` (a measured -0.307 decorrelation at
0.2 on the image model). The test is a decorrelation arm against its paired control at the same budget, with the same exact
attribution instrument applied to both checkpoints afterwards. If positive supply does not fall toward 1.0x the margin, the
cancellation is intrinsic to a 36-concept layer trained by cross-entropy and the remedy is FEWER, WIDER concepts instead.
