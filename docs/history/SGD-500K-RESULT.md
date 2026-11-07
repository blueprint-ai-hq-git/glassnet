# Three-way at 500k: our step rule beats plain SGD by 2.24 pp - and the headline used plain SGD

2026-11-07 14:20:18 IST, `SGD-500K.txt`, `exit=0`, marker `SGD-500K DONE 2026-11-07 14:20`.
Predictions written before the launch in `docs/history/SGD-500K-PREDICTIONS.md`.

## The three configurations, matched

Same rung (500k windows), same seed (0), same data, same `--d 256 --window 32 --batch 256 --epochs 3 --lr 0.4
--lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0`. Only the step rule and the gates change.

| configuration | VAL | ASST | ppl | train acc | concepts | s/epoch | source |
|---|---:|---:|---:|---:|---:|---:|---|
| relgrad + gates ON | 20.81% | 21.82% | 171.47 | 22.25% | 36/36 | 554-558 | `DATASCALE-500K.txt:392` |
| **relgrad, gates OFF** | **22.05%** | **23.22%** | **140.27** | 35.20% | 36/36 | 72-73 | `DATASCALE-500K-NOGATE.txt:69` |
| plain SGD, gates OFF | 19.81% | 20.38% | 221.78 | 19.04% | **31/36** | 31-32 | `lens/runs/SGD-500K.txt:66` |

Provenance: `verify_wall_ab_ckpt.py` -> 1 arm, 0 problems, on all three.

## The verdicts for this arm

    S1  VAL in 20.55-23.55%        got 19.81%  -> FAIL, 0.74 pp below the band
    S2  ppl in 120-175             got 221.78  -> FAIL
    S3  ~72-75 s/epoch             got 31-32   -> FAIL, and faster than predicted (my band assumed relgrad's cost)
    S4  mechanical                 0 gate lines, `STEP RULE: plain SGD` -> PASS
    S5  concepts >= 36             got 31/36   -> FAIL

Three of five missed, all in one direction: **plain SGD is the weakest of the three** at this rung - lower
accuracy, much worse perplexity, and it fails to learn the concept module (31/36 against 36/36 for both relgrad
arms). It is also 2.3x cheaper per epoch than relgrad, which is the one thing it wins.

## What the three-way says

    relgrad, gates off   22.05
    relgrad, gates on    20.81      gates cost 1.24 pp
    plain SGD, gates off 19.81      step rule costs 2.24 pp

So our step rule is worth **+2.24 pp** over the loop this project used before, and the gates cost **1.24 pp**.
The best measured configuration at d=256/500k is **relgrad with the gates off**, and it is 7.7x faster than the
gated arm.

## Why that is awkward for the headline

**37.37% was produced by plain SGD** (`TIED-D512-EXT6.txt`: `METHOD: standard (the old loop: plain SGD, our gates
OFF)`, 16.9M windows, 6 epochs, train acc 48.91%). At the only rung where the comparison has been made, that step
rule is the weakest of the three by 2.24 pp. So the project's best number sits on its worst-measured step rule -
which means the headline configuration has never been run in the configuration that is now best measured.

That is a hypothesis with a specific test, not a claim: **d=512, full corpus, 6 epochs, relgrad, gates off**. Cost
is unmeasured - plain SGD took 1,720 s/epoch at that scale (`docs/history/STATE.md:2771`), and relgrad costs ~2.3x
plain SGD per epoch at 500k, which would extrapolate to hours per epoch. That extrapolation is not a measurement,
so the honest first step is a **one-epoch cost probe** before committing 6 epochs.

## Not claimed

* One seed per configuration. The +2.24 and -1.24 pp are inside the 3.07 pp three-seed spread measured at 200k,
  so the *ordering* is a direction, not a settled effect. The epoch-time ratios (7.7x, 2.3x) are measured, not
  noisy.
* No headline change. 37.37% stands until something beats it at its own scale.
* Nothing here says the gates are useless: they were measured at one rung, ungated training has not been run to
  convergence, and the gates' stated purpose is protecting held-out quality, which at other rungs they may do.
