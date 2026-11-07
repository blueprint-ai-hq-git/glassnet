# 16M windows: the last doubling the corpus contains (predictions, before the run)

2026-11-07 21:2x IST, written before `lens/scripts/run_datascale_16m_nogate.sh` is launched.

## Why this, and why it is the end of the axis

Three doublings have now been measured at d=256, gates off, seed 0, three epochs:

    1M -> 2M    +0.06 pp   (2M reached 23.12)
    2M -> 4M    +0.62 pp   (4M reached 23.74; three seeds 23.74 / 24.13 / 24.03, spread 0.10)
    4M -> 8M    +0.77 pp   (8M reached 24.51)

The rate is rising, not falling, and the 4M rung reproduces to 0.10 pp, so these deltas are not seed noise at that
rung. The prepared mix holds ~18.4M windows at stride 16 and 8M was 43% of it: **16M is 87%, i.e. the last doubling
that can be run without new data.** After this arm the axis is exhausted and the honest next question is a different
axis (width, or the gates contrast), not another doubling.

Cost: 16,000,000 windows is 62,500 updates/epoch; at 8M's measured 1,021-1,508 s/epoch this is ~2,000-3,000 s/epoch,
so ~2-2.5 h for three epochs. Each epoch checkpoints, well inside the 2 h rule.

## Predictions

Y1  STILL PAYING. VAL > 24.51% (strictly above 8M).
    FALSIFIER: <= 24.51% means the axis finally turned down between 8M and 16M - which would be the first time.

Y2  HOW MUCH. Land in **24.7 - 25.4%** (+0.2 to +0.9 pp). The last two doublings paid 0.62 and 0.77; a third equal
    step is unlikely to repeat exactly, so the band is centred near +0.5 and deliberately asymmetric-free.
    FALSIFIER: above 25.4% means the rate is still accelerating; below 24.7% means it decayed sharply.

Y3  PERPLEXITY below 87.55 (the 8M value). Every rung so far has fallen further.

Y4  MECHANICAL. 0 `agreement step` / 0 `gate step` lines, METHOD reads GATES OFF, 3,811,534 params.

Y5  CONCEPTS >= 36 of the kept set.

## What this arm cannot settle, stated before it runs

* One seed. At 4M that was measured to be a 0.10 pp risk; at 16M it is unmeasured, so a +0.2 pp result could be noise
  and will be reported as such rather than as a win.
* It does not test the gates (this is the gates-off ladder).
* It does not test width. d=256 is the whole ladder; the headline is d=512 at full corpus (37.37%).
* It is not comparable to any full-corpus d256 run (35.72% by position, `docs/history/STATE.md:146`): capped 16M
  windows for 3 epochs is a different regime from the full corpus for 6.
