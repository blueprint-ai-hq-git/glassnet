# Retraction: the data judge already passed, and my |z| clip is not the fix (2026-11-13)

I picked up the data judge as "still fails its own gate (false alarm 0.0730 against a 5% bar, one-token loops 69.5% against 80%)"
and produced a mechanism, a fix and a claim. The control test killed the claim, and the numbers show the status line itself was out
of date. Both corrections are below.

## What I claimed, and what the control did to it
**Hypothesis:** on a one-token-loop window five of the judge's deviation features reach **|z| = 1e9**, because their MAD over clean
text is zero and the floor is 1e-9 (measured: `rep_adjacent`, `max_run`, `trigram_rep`, `pair_seen`, `triple_seen` all at 1e9 for a
loop window). With signed weights one such column decides the whole linear score, so whether a window that is ENTIRELY one repeated
token gets caught depends on that column's sign. That would explain 69.5%.

**Prediction, written before the run:** clipping |z| at 50 passes all three gates on three seeds, and the UNCLIPPED control
reproduces the old failure (~0.073 false alarm, ~69% loops). **"If the control does NOT reproduce the old numbers, the clip is not
what changed the outcome and the result is void."**

```
arm                          AUROC     false alarm   shuffled   one-token loop   verdict
clipped |z|<=50, seed 0      0.9822      0.0422       82.9%        90.1%         PASS
clipped |z|<=50, seed 1      0.9820      0.0423        -           89.7%         PASS
clipped |z|<=50, seed 2      0.9820      0.0422        -           90.0%         PASS
UNCLIPPED control, seed 0    0.9923      0.0257       80.7%        90.1%         PASS
```

**The control passed, and passed better.** Unclipped has a false alarm of **0.0257** against the clipped **0.0422**, and a higher
AUROC (0.9923 vs 0.9822). So clipping does not fix anything; it **costs margin**. My hypothesis is **retracted as a cause**, by my
own pre-registered rule.

The |z| = 1e9 values are real and still worth knowing - they are just not what was breaking the gate. The clip is now **off by
default** (`--zclip 0`), with a comment saying why, so the tool behaves as verified.

## The bigger correction: the judge was never failing
`docs/history/JUDGE-V2.md` (same day, earlier) already records a rebuild passing all three gates: AUROC **0.9923 +/- 0.0001**, false
alarm **2.59% +/- 0.02%**, every corruption kind >= 80%. My uncapped run reproduces it to the digit - **0.9923** and **0.0257**
against the recorded **0.9923** and **0.0259**.

So the "judge still fails its gate" line in my own status summary was **stale**: it described `docs/DATA-JUDGE.md` (the OLD judge),
while the rebuilt tool in the tree passes. That is a reporting error on my part, not a result, and it is the second time this session
that a stale status line nearly cost a wake of work.

## What is actually true about the judge, verified this run
```
                                  value            bar
AUROC                             0.9923           >= 0.90    PASS
false alarm on clean held-out     2.57%            <= 5%      PASS
soup / shuffled / loop / chunk / unknown   all >= 80%           PASS
```
One honest caveat that has not changed: `shuffled order` is the weakest kind at **80.7%**, sitting on its bar, so the judge is
fragile exactly on the corruption that preserves every token and only destroys order.

## Reproduction
`tools/loop_judge.py --windows 20000 --seed 0` (defaults are the verified configuration), `--zclip 50` for the clipped arm,
`--zclip-off` redundant now that 0 is the default. Logs: `logs/judge/{clip_s1,clip_s2,noclip_s0}.log`.
