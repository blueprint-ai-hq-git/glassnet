

---

# UNTIED READOUT, CLEAN TEST: THE FALSIFIER FIRED, AND THIS CORRECTS THIS MORNING'S CLAIM (2026-11-20 12:10)

This morning I published "the untied readout is the first thing to ever move the tail", with its confound stated: the checkpoint that produced it also carried ~2.8x
the parameters and a rarity-weighted training setting. The clean one-change test has now finished, and **the confound was the cause**.

## The clean test

`win6kon` (tied) vs `untied6k` (identical config, `TIE=0`, nothing else changed): same seed, same batch order, same 6000-step budget, same draw seed 4242,
scored on the same 104 windows.

```
                          tied control    untied (only change)
ranks 20-50                25.19%          21.25%     -3.94
ranks 50-100                0.22%           0.00%     -0.22   <- the band the whole idea rested on
ranks 100-250               0.00%           0.00%      0.00
ranks 250-500               0.00%           0.00%      0.00
r21-500 overall             8.52%           7.15%     -1.37
top-20 accuracy            35.81%          34.79%     -1.02
trainer line vs constant   22.88% BEATS    22.40% BEATS (+7.93)
```

- **P1 FAILED**: ranks 50-100 landed at **0.00%**, at or below the 0.22% tied baseline.
- **P2 FAILED**: top-20 fell 1.02 points, and it did not have to (the bar was "no lower than 35.81%").
- **P3 PASSED**: 22.40% is inside the 21.88% floor and still beats the constant by 7.93 points, so the arm trained properly.
- **FALSIFIER FIRED**, exactly as pre-registered: "ranks 50-100 at or below 0.22% -> the scan12 result was its config and parameter count, not the readout;
  capacity is not the lever either, and this line closes rather than being refined again."

## The correction, stated plainly

The tail movement I attributed to the untied readout in `docs/history/RANK-BOOST-VERDICT-AND-UNTIED-TAIL.md` came from **the rarity-weighted training setting
and the extra parameters**, not from untying. Untying by itself makes every band slightly worse. That page's "prediction MET" line is superseded here.

## Where the lever is not

Every simple lever has now been measured and closed:

| lever | result | status |
|---|---|---|
| loss weighting toward the tail | **14.23% vs a 14.47% trivial constant** - below a frequency table | closed |
| post-hoc calibration (global and gated) | +0.84 / +0.49 / +0.00, no replication | closed |
| readout shape (tied vs untied) | ranks 50-100 **0.00% vs 0.22%** | closed today |
| depth (2 recurrent layers) | 5.68% / 5.08% vs a 6.07% single-layer baseline | closed |

## What is left, and it is the queue's own next item

All four levers changed the *strength* or the *post-processing* of learning. None changed **what is supervised at each position**. The queue note says exactly
this: "the next mechanism should change WHAT is supervised per position while keeping window labels".

The measured reason to expect something here is the delayed-recall probe: with a target that genuinely requires memory, the readable window bank plus its age
key lifted recall from **20.61% to 85.61%** across three seeds. On next-token prediction the bank is worth nothing - which may simply mean next-token
prediction never needs the window it is being given.

So the next arm tests that directly: train with the target at **offset 2** instead of offset 1, which no longer follows from the immediately preceding token, and
run it **with and without the readable bank**. If the bank sells anything, it sells it there.
