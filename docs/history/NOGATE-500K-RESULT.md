# Gates off at 500k: +1.24 pp, 7.7x faster - and the headline was never gated at all

2026-11-07 13:59:30 IST, `DATASCALE-500K-NOGATE.txt`, `exit=0`, marker `NOGATE-500K DONE 2026-11-07 13:59`.
Predictions written before the launch in `docs/history/NOGATE-500K-PREDICTIONS.md`; graded by
`lens/scripts/datascale_ladder_grade.py nogate`, thresholds fixed in that file.

## Result, against the gated twin (same rung, same seed, same step rule, same data)

| | gated 500k | gates OFF | difference |
|---|---:|---:|---:|
| VAL acc | 20.81% | **22.05%** | **+1.24 pp** |
| ASST (by position) | 21.82% | **23.22%** | **+1.40 pp** |
| held-out ppl | 171.47 | **140.27** | **-31.20** |
| train acc, ep 3 | 22.25% | 35.20% | +12.95 pp |
| train - VAL gap | 1.44 pp | 13.15 pp | +11.71 pp |
| s/epoch | 554-558 | **72-73** | **7.7x faster** |
| concepts | 36/36 | 36/36 | same |

Sources: `DATASCALE-500K.txt:392` (gated), `DATASCALE-500K-NOGATE.txt` (ungated). Provenance:
`verify_wall_ab_ckpt.py` -> 1 arm, 0 problems; epoch-3 checkpoint 15,262,004 bytes.

## The verdicts

    G1  gates-off VAL >= 21.81%      22.05%  -> PASS (+1.24 pp): the gates were part of the ceiling
    G2  overfitting signature        gap 1.44 -> 13.15 pp -> PRESENT, and VAL still improved
    G3  mechanical                   0 agreement lines, 0 gate lines, METHOD reads GATES OFF -> PASS
    G4  concepts >= 36               36/36 -> PASS

G2 is the interesting one: removing the gates let the model fit the training set far harder (22.25% -> 35.20%)
and overfit far more (gap 1.44 -> 13.15 pp), **yet held-out accuracy still rose 1.24 pp**. So at this rung the
gates were not protecting held-out quality - they were suppressing learning that generalised anyway.

## What this does to the ladder

Every rung of the data ladder (20k, 200k, 500k, 2M) was run with `--method ours`, i.e. **gated**. The flattening
reported in `docs/history/DATASCALE-2M-RESULT.md` (1.58 -> 2.56 -> 0.24 pp per doubling) is therefore a property
of the **gated** method, not a property of the data alone. The ungated curve has exactly one point.

## The part that matters most

`lens/runs/TIED-D512-EXT6.txt` - the run behind the 37.37% headline - prints:

    METHOD: standard (the old loop: plain SGD, our gates OFF)

**The best number this project has was produced by the old loop with the gates off.** The ladder, by contrast,
has been scaling *our* method with the gates on. So the project has been comparing "how far our method scales"
against a headline produced by a different method, and the one control that isolates the gates at a matched rung
now says they cost 1.24 pp and 7.7x wall-clock.

## What is still missing, and the next arm

Three configurations exist, not two:

    (a) standard / plain SGD / gates off   -> the 37.37% headline (d=512, full corpus, 6 epochs)
    (b) ours / relgrad / gates ON          -> the whole data ladder (20k -> 2M, d=256)
    (c) ours / relgrad / gates OFF         -> this arm: 22.05% at 500k, d=256

The comparison (a) vs (c) has never been made at any matched rung, so "relgrad vs SGD" is still unmeasured at
this scale. That is the next arm: `standard` (SGD, gates off) at 500k, d=256, seed 0 - ~4 minutes at the measured
72 s/epoch. Only then does the ladder's configuration get chosen by evidence rather than by inheritance.

## Not claimed

* One seed, one rung. +1.24 pp is inside the 3.07 pp three-seed spread measured at 200k, so on its own this is a
  direction, not a settled effect; the 7.7x compute difference is a measured ratio, not a noisy one.
* The ungated arm had not converged - VAL rose every epoch (19.71 -> 21.25 -> 22.05), so "gates off is better"
  may understate what ungated training does with more epochs.
* No change to the headline: 37.37% stands, and this arm is 15.32 pp below it.
