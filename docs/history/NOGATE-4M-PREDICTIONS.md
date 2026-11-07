# 4M windows, gates off: does the ungated curve keep paying? (predictions, before the run)

2026-11-07 16:3x IST, written before `lens/scripts/run_datascale_4m_nogate.sh` is launched.

## Where the ladder stands, ungated

    20k    12.18%   (gated; the only 20k value that exists)
    200k   17.43%   (gated)
    500k   22.05%   (gates off)  0.54 pp/doubling from 500k to 2M
    2M     23.12%   (gates off)  <- best d256 on record

The 2M ungated arm was the first rung where the ungated curve was measured against another ungated rung, and it was
still rising (+1.07 pp for 4x, which overshot my own band). So the next rung is the honest one to spend GPU on:
**4M windows, gates off**, same seed, batch, step rule and corpus.

Cost, scaled from the 2M ungated arm (249-250 s/epoch at 7,813 steps): 4M is 15,625 steps, so ~500 s/epoch,
~25 min for three epochs plus corpus assembly. Affordable.

## Predictions

V1  STILL PAYING. VAL > 23.12% (strictly above the 2M ungated arm).
    FALSIFIER: <= 23.12% means the ungated curve saturates at 2M and the ladder is finished at this width.

V2  HOW MUCH. Land in **23.4 - 24.6%**, i.e. +0.3 to +1.5 pp for one doubling. The measured per-doubling rate from
    500k to 2M was 0.54, so the point estimate is ~23.66; the band allows for the decay every rung so far has shown
    eventually.
    FALSIFIER: outside the band. Above it means the lever is accelerating, which nothing so far supports; below it
    means the decay is steeper than the 500k->2M step suggested.

V3  PERPLEXITY below 104.71 (the 2M ungated value), same direction.

V4  MECHANICAL. 0 `agreement step` and 0 `gate step` lines, METHOD reads GATES OFF, 3,811,534 params.

V5  CONCEPTS. Audit >= 36 of the kept set.

## What will not be claimed

* One seed per rung, and the 3.07 pp seed spread measured at 200k is larger than every delta in this table. The
  curve's *shape* is the finding; no single point-to-point difference is an effect size.
* 4M windows is 4x the 2M arm but still a capped sample of the corpus (16.9M windows exist; 4M is 24% of them).
* No headline change (37.37%, d=512, full corpus, 6 epochs), and nothing here touches the d512 step-size confound.
* Checkpoints every epoch (~8 min each), CPU pinned to two cores, VRAM ~490 MiB.
