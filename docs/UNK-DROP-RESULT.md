# Stopping the model from giving up: drop-unk raises real token accuracy by +1.62 points

Date: 2026-11-14. Arm: `dropunk_d1024_s0` (d1024, 9 epochs, batch 512, the exact reference recipe from
`run_exposure_test.sh`, with **one** flag added: `--drop-unk`). Source: `logs/exposure/dropunk_d1024_s0.log`,
audit `lens/runs/UNK-DROP-RESULT.txt`.

## Why this arm existed

`docs/history/UNK-AUDIT.md` measured that the reference model PREDICTS `<unk>` on **21.73%** of held-out assistant positions
when only **5.40%** of targets are `<unk>`, and that 86% of those predictions are wrong. A fifth of its answers went to a token
that cost it accuracy - it was trained to give up, because `<unk>` targets are answerable-by-construction and the plain
objective pays for emitting them. `--drop-unk` sets the training weight to zero wherever the target token is `<unk>`
(83,211 of 2,000,000 training targets in this run, 4.161%), so those positions stop being paid for.

## Result, measured with the same tool and the same 20,000 held-out windows as the baseline

```
                          reference      drop-unk     change
overall accuracy            29.34%        27.83%      -1.51
accuracy, <unk> excluded    27.79%        29.41%      +1.62     <- the real test
model predicts <unk> on     21.73%         0.00%     -21.73
accuracy on <unk> targets   56.41%         0.00%     -56.41
```

## Predictions, scored (written before the run, in the script header)

- **P1 the `<unk>` prediction rate falls below 10% - HOLDS** at **0.00%**. The model never emits `<unk>` again.
- **P2 accuracy with `<unk>` targets EXCLUDED rises from 27.79% to at least 28.8% (+1.0 point) - HOLDS** at **29.41%**, a gain
  of **+1.62 points**. This was the real test, because it measures prediction of tokens that actually exist.
- **P3 the raw headline falls, but by less than 2 points - HOLDS** at **-1.51**. It had to fall: `<unk>` targets were being
  scored at 56.41% and are now untrained, so a model that scores 0% on them necessarily loses that much raw accuracy. The two
  numbers are not in conflict - they measure different things, and the one that matters went up.

## What this establishes

**For the same weights, the same data and the same schedule, refusing to train on unanswerable targets makes the model better
at the positions it can actually answer: 27.79% -> 29.41% real token accuracy, +1.62 points, with `<unk>` emissions at exactly
zero.** The raw headline falls 1.51 points, and that is the arithmetic of dropping 5.4% of positions that were scoring 56%.

It also settles the question that motivated it: the 17.32% generative accuracy in `docs/history/FAILURE-ATTRIBUTION.md` was not
caused by a `<unk>` dumping ground. Removing the dumping ground improved real accuracy but did not touch the retrieval/generation
split, so the generative ceiling has a different cause.

## A note on reading the audit output

`tools/unk_audit.py` prints its own P1/P2/P3 lines, which ask *"is the headline propped up by `<unk>`?"* for the checkpoint
being scored. On this checkpoint they report FAILED, and that is correct and expected rather than a problem: a model that never
predicts `<unk>` cannot lose accuracy by excluding `<unk>` targets, and has no `<unk>` predictions to be right about. Those
lines are diagnostics about the baseline hypothesis, not predictions of this arm.
