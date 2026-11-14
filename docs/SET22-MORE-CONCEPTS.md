# More concepts is not the fix: +8 concepts bought 0.57 accuracy, cost 3 mislabeled labels, and WIDENED the readable penalty

Date: 2026-11-14. Tool `gates/phase2_train.py` with a 22-feature set (correlation limit relaxed to 0.85, max pairwise 0.810), plus
a matched plain control at the same parameter count from `tools/matched_plain_p2.py`. Predictions in
`lens/runs/SET22-PREDICTIONS.txt`, outputs `SET22-RESULTS.txt`, `SET22-AUDIT.txt`.

## The question

`docs/history/RULE-COUNT-SATURATED.md` decomposed the readable model's gap: **1.24 points concept bottleneck + 1.55 rule layer**.
The open question was whether the concept bottleneck is the **number** of hand-computed statistics or their **kind**.

## Result: count is not the answer

```
                                     params    accuracy (3 seeds)   vs its own matched plain
14 concepts (recommended)            11,800    81.58% +/- 0.50      84.37 +/- 1.04   -> -2.79
22 concepts (this run)               18,336    82.15% +/- 0.38      85.28 +/- 0.46   -> -3.13
  plain control, matched to 22       18,295    --                    (the 85.28 figure itself)
```

- **P1 HOLDS**: 22 concepts do beat 14 (82.15 vs 81.58, **+0.57**). The count is not irrelevant.
- **P2 HOLDS**: the gap to the matched plain control stays at 2+ points - and in fact **widens from 2.79 to 3.13**.
- **P3 FAILED**: **3 of 22 concepts are mislabeled** (mean own-label correlation 0.779, 1 of 22 below 0.50), against **0 of 14** for
  the clean set. The relaxed correlation limit put overlapping labels back into the set, exactly as the label-redundancy work
  predicted.

## The finding that matters: extra capacity helps the plain model more

```
                         11.8k params      18.3k params    gain from the extra parameters
matched plain            84.37 +/- 1.04    85.28 +/- 0.46   +0.91
readable                 81.58 +/- 0.50    82.15 +/- 0.38   +0.57
readable penalty         -2.79             -3.13            gets WORSE with size
```

**Adding parameters helps the plain network half again as much as the readable one.** So the readable architecture does not merely
start behind; it falls further behind as it grows. That is a scaling statement, not a tuning one, and it is the strongest evidence
yet that the binding limit is the **kind** of statistic: whole-image summary numbers (mean ink, edge counts, symmetry scores) throw
away the spatial detail that extra capacity would otherwise exploit.

## Decision: keep 14 concepts

The 22-concept set is **not adopted**. It buys +0.57 accuracy, costs **3 mislabeled concepts** (the exact defect this session spent
several runs eliminating), and widens the readable penalty by 0.34 points. On the project's own priority - honesty first, accuracy
second - it is a worse model.

## What the next run must be

Higher-resolution concepts, i.e. the same measurement computed **per image region** rather than once per image, so the concept layer
retains spatial information while every concept stays nameable ("ink in the upper-left quarter of the lower-right region"). The
prediction to write before that run: if per-region concepts close more than 1 point of the 3.13-point gap at matched parameters,
then spatial resolution was the constraint all along; if they do not, the concept bottleneck is fundamental to a 14-24 concept middle
layer and the honest conclusion is that the readable image model's price is about 3 points and cannot be paid down.

## Prediction scorecard

- P1 (22 beats 14) **HOLDS** at +0.57.
- P2 (gap stays >= 2 points) **HOLDS**, at 3.13 - worse than before.
- P3 (at most 2 of 22 mislabeled) **FAILED** at 3.
- Falsifier (a flat ~81.6% would have meant "kind, not count") **not triggered literally**, but the wideneing penalty says the same
  thing in a stronger form: count buys a little, and the readable penalty grows with size regardless.
