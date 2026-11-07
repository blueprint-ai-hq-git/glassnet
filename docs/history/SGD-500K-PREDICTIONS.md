# The missing corner: plain SGD, gates off, 500k - the configuration the headline actually used

2026-11-07 14:1x IST, written before `lens/scripts/run_sgd_500k_s0.sh` is launched.

## Why this arm exists

Yesterday's control (`docs/history/NOGATE-500K-RESULT.md`) left three configurations, and only two were measured:

    (a) standard / plain SGD / gates off    -> the 37.37% headline (d=512, full corpus, 6 epochs)   NEVER RUN AT d256/500k
    (b) ours / relgrad / gates ON           -> the whole data ladder: 20.81% at 500k, d256
    (c) ours / relgrad / gates OFF          -> 22.05% at 500k, d256, 7.7x faster than (b)

The best number this project has came from (a). The ladder that says "data scale is the lever" was measured in
(b). So "is our step rule better than plain SGD?" - the question behind the whole method - has never been asked at
a matched rung. This arm asks it: same rung (500k), same seed (0), same data, same `--lr 0.4 --lr-decay 0.5
--batch 256 --epochs 3`, with `--method standard` (plain SGD, gates off, verified by its own STEP RULE line).

## Predictions

S1  PLAIN SGD LANDS WITHIN 1.5 pp OF RELGRAD. VAL in **20.55 - 23.55%** (22.05 +- 1.5), i.e. the step rule is not
    the thing that moves this rung - the gates are.
    FALSIFIER: above 23.55% means plain SGD beats our step rule at a matched rung, which would say relgrad is
    costing accuracy; below 20.55% means relgrad is clearly better than the old loop on held-out accuracy even
    ungated.

S2  PERPLEXITY in **120 - 175**. Wider than the accuracy band on purpose: the ungated relgrad arm surprised me by
    31 ppl (171.47 -> 140.27), and perplexity has been the more sensitive of the two numbers all day.

S3  SPEED. ~72-75 s/epoch, like (c) and unlike (b): no gate means no per-step probe re-scoring. I will report the
    measured figure beside all three arms rather than assert it.

S4  MECHANICAL. 0 `agreement step` lines, 0 `gate step` lines, and the log prints `STEP RULE: plain SGD`. If not,
    the arm did not test what it claims and the result is void.

S5  CONCEPTS. Audit >= 36 of the kept set.

## What will not be claimed

* One seed, one rung, one width. A 1.5 pp band is wider than the 3.07 pp three-seed spread can resolve, so any
  difference inside it is a direction, not an effect.
* This arm does not "explain" the 37.37% headline: that run is d=512 on the full 16.9M-window corpus with 6
  epochs, and is 15 pp above anything at d256/500k. It tests the step rule, not the corpus.
* No headline change. Cost ~4-5 min at 72 s/epoch; checkpoints every epoch; CPU pinned to two cores.
