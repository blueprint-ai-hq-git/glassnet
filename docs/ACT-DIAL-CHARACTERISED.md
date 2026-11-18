# The activation dial is characterised: pass count peaks at lam-act 1.0-1.5, then falls (2026-11-18)

Five arms, one change each, readable d512, seed 0:

```
lam-act   pass     mean AUROC   mean fire   failing concepts
0.0      30/36      0.6886        47.4%     is_a_list, mentions_a_name, mentions_a_quantity, mentions_body, mentions_family, mentions_sport
0.3      32/36      0.7038        36.2%     mentions_a_name, mentions_a_quantity, mentions_body, mentions_sport
1.0      34/36      0.7370        20.1%     mentions_body, mentions_sport
1.5      34/36      0.7573        14.2%     mentions_body, mentions_sport
2.0      32/36      0.7767        10.5%     mentions_a_quantity, mentions_body, mentions_sport, mentions_time
```

## Predictions for the 1.5/2.0 arms, scored

- **P1** "mean AUROC keeps climbing, at least 0.76 at lam-act 2.0" - **HELD**: 0.7767.
- **P2** "no NEW failures appear; the failing set stays inside {mentions_body, mentions_sport}" - **FAILED**: at 2.0, `mentions_a_quantity` and `mentions_time` start failing too.
- **P3** "pass count reaches at least 35/36" - **FAILED**: 34 at 1.5, 32 at 2.0.
- **FALSIFIER** "mean AUROC at 2.0 is not higher than 0.7370" - **did not fire** (0.7767).

## What the five points actually say

**AUROC rises monotonically** across the whole range: 0.6886 -> 0.7038 -> 0.7370 -> 0.7573 -> 0.7767. **But the pass count peaks and then falls**: 30 -> 32 -> 34 -> 34 -> 32.

That is the mechanism the arm's own rationale predicted before it ran: **uniform pressure that keeps climbing stops helping the rare concepts and starts suppressing the common ones.** At lam-act 2.0 the fire rate is 10.5% overall, and `mentions_a_quantity` - a concept with **13.9% support**, not a rare one - drops out, along with `mentions_time`. The pressure is now so strong that a concept which is genuinely present cannot make its neuron fire.

So the honest operating point is **lam-act 1.0-1.5**: **34/36 concepts (94.4%)**, and 1.5 dominates 1.0 because it reaches the same pass count with a higher mean AUROC (0.7573 vs 0.7370) and a lower fire rate (14.2% vs 20.1%).

**Uniform activation pressure is finished as a mechanism.** It bought 30/36 -> 34/36 (a real gain) and then started costing what it had bought.

## The next arm, and the fallback the doctrine already dictates

Two concepts fail at **every** pressure: `mentions_body` (support 1.32%) and `mentions_sport` (1.43%). They are the rarest in the set but rarity alone is not the explanation - `mentions_a_name` (0.82%, the rarest of all) passes.

`lens/scripts/run_boost_rare.sh` therefore applies **per-concept positive weighting**, which the trainer already supports (`TALK_BOOST=name:mult`), aimed only at those two:

- **P1** pass count reaches **36/36**
- **P2** no concept that passed at 1.5 fails now - the failing set shrinks, it does not move
- **P3** the two boosted concepts' firing rates rise toward their true support instead of sitting far below it
- **FALSIFIER** they still fail at 3x boost. Then per-concept weighting is not the mechanism either, and the project's own doctrine applies: **a concept that cannot be made label-honest is DROPPED, never forced** - and the next report becomes **34/34 = 100% honest** on a set of the 34 concepts that can actually be verified.
