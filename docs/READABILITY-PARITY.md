# Readability at parity: -0.08 points (single seed, replication running) (2026-11-18)

The matched readability gap has been ~-1.5 points in every configuration this project has tried. One arm now measures **-0.08 points**, which is parity within noise. It is
**one seed**, the replication is running, and this page will carry the retraction if the three-seed mean lands at or below -1.0.

```
READABLE: 15,977 params   4 patches x 56  ->  21 concepts  ->  10
PLAIN   : 15,910 params   784 -> 20 hidden ->  10   (0.42% difference in size)
delta -0.08 points        concepts passing 20/21  (95.2% honesty)
```

## What changed, and why it is a mechanism rather than a lucky seed

The previous configurations all gave the readable model a **large** front-end and a **wide** concept input:

```
patch 7, d=64  -> 16 patches x 64 = 1024 features per concept unit   gap -1.47 to -1.53
patch 7, d=49  -> 16 patches x 49 =  784 features per concept unit   gap -1.46
patch 14, d=56 ->  4 patches x 56 =  224 features per concept unit   gap -0.08
```

In the winning shape the readable model spends most of its budget on the patch embedding and keeps the concept layer **small and cheap**: 21 units sitting on 224 features each,
heavily regularised by its own narrowness. The plain control spends almost all of its budget on one wide hidden layer (784 -> 20 units). Two further reasons the coarse patches
should suit this architecture: the concepts are themselves **region-shaped formulas** (`top_half_heavier`, `centre_heavier`, `has_flat_region`), and a 14x14 patch covers a region
rather than a fragment; and unit cost falls from 1034 parameters to 234, so the readable model is no longer starved of units.

## Predictions, scored honestly

- **P1** "readable fields >= 30 concept units against the control's ~24" - **FAILED**. The front-end consumed the budget and the readable model fielded 21 concepts against the
  control's 20. The unit advantage I predicted did not happen; near-parity of units happened instead.
- **P2** "gap moves within +-0.5 points" - **HOLDS** at -0.08.
- **P3** "honesty >= 85%" - **HOLDS** at 95.2% (20 of 21).
- **Falsifier** "with a 25%+ unit advantage the gap is still at or below -1.0" - **did not fire**, and it also never got its 25% advantage, so it is a weak pass, not a strong one.

## Why this matters beyond the number

Every earlier attempt to close the gap by adding concepts, equalising unit cost, or giving the answer layer rules **failed**, and each failure was recorded. This one succeeded by
changing the *shape* of the model rather than its capacity: a small concept layer on coarse regional features.

If the three seeds hold, the architecture's honest headline changes from "readability costs 1.5 points" to "readability costs nothing measurable, in the configuration where the concept
layer is small and its inputs are regional" - with 95% of concepts label-honest and the layer still carrying all of the accuracy (zeroing it drops 84.15% -> 10.17%, shuffling it
-> 9.55%).

## The replication, pre-registered

`lens/scripts/run_capadv_seeds.sh`, seeds 0, 1, 2, same config:

- **P1** 3-seed mean gap between -0.6 and +0.6
- **P2** sd across seeds at most 0.6
- **P3** honesty >= 85% on every seed
- **FALSIFIER** 3-seed mean at or below -1.0: then -0.08 was a favourable seed, ~-1.5 stands as the reported number, and **this page gets the retraction in place**.
