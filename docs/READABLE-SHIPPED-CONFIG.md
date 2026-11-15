

## The best operating point is stable: lam 1.0 across three seeds (shippable)

Predictions in `lens/runs/LAM10-3SEED-PREDICTIONS.txt`. The lam 1.0 arm (364 tokens, 21.08% VAL, 52/53 concepts, +3.35 over its
control) was a single seed, so seeds 1 and 2 were run.

```
seed   final VAL   final ppl   concepts passing   distinct tokens   effective dims
  0      21.08%      269.30         52/53               364           10.09 of 53
  1      21.15%      267.04         51/53               352           10.64 of 53
  2      21.17%      262.88         52/53               326           10.14 of 53

final VAL : mean 21.13%, sd 0.05
breadth   : mean 347 tokens, sd 19 (326-364)
concepts  : 51-52 of 53 pass on every seed
```

The paired lam 0.3 arms at the same seeds for comparison: 22.04%, 21.98%, 21.99% with 135, 131, 131 tokens.

- **P1 PASSED**: every seed landed inside the predicted 20.73-21.43% band, with a spread of only 0.09 points. This configuration does
  not depend on one seed's luck.
- **P2 PASSED**: breadth stayed above 300 on every seed (364, 352, 326).
- **P3 FAILED, and it failed because I wrote it wrong.** It said "final VAL above the paired **lam 0.3** value by at least 0.50 points
  on 2 of 3 seeds" while claiming to test the **direct-path control**. lam 0.3 IS the readable baseline, so the prediction as written
  asks lam 1.0 to beat the higher-accuracy readable arm, which it cannot do by construction - it loses 0.83 to 0.96 points on 3 of 3
  seeds. **This is my second prediction-writing error today** (the first was writing P3 of the lam 0.6 check against a fixed value
  instead of a paired delta). The test I intended is stated separately below rather than retro-fitted into the scorecard.
- **FALSIFIER NOT TRIGGERED**: validation never fell below 20.73%, so lam 1.0 is stable enough to be the shipped readable operating
  point.

## The comparison that matters, stated cleanly

```
readable lam 1.0, three seeds : 21.13% +/- 0.05, 347 +/- 19 tokens, 51-52 of 53 concepts
direct-path control, 18 epochs: 17.73%           (seed 0 only - the control has not been run at 18 epochs on any other seed)
```

**The readable model is +3.40 points more accurate than its matched control while being fully explainable.** That caveat is real and
deliberate: the control side of that comparison is single-seed, so the honest form is "the readable model's three seeds all sit 3.2 to
3.4 points above the one matched control run".

## Where the breadth question stands

`lam 1.0` at 347 tokens is the widest *stable* readable point that keeps accuracy above 21% and 51+ concepts honest. lam 1.5 and 2.0
reach 414 and 422 but lose 0.8 to 1.2 points and drop concepts below the name audit. So the shipped readable configuration is **lam
1.0**, and the breadth ceiling of ~422 stands as documented (`docs/history/BREADTH-IS-SUPERVISION.md`).
