# 2M without the gates: 23.12% - the flattening was partly the gates, and this is the best d256 number on record

2026-11-07 16:16:25 IST, `lens/runs/DATASCALE-2M-NOGATE.txt:69`, `exit=0`, marker
`NOGATE-2M DONE 2026-11-07 16:16`. Predictions written before the launch in
`docs/history/NOGATE-2M-PREDICTIONS.md`.

## Result against the gated 2M arm (same windows, seed, batch, step rule)

| | gated 2M | gates OFF 2M | difference |
|---|---:|---:|---:|
| VAL | 21.29% | **23.12%** | **+1.83 pp** |
| ASST | 22.60% | **24.96%** | **+2.36 pp** |
| held-out ppl | 149.14 | **104.71** | **-44.43** |
| train acc | 21.14% | 35.10% | +13.96 pp |
| concepts | 36/36 | 36/36 | same |
| s/epoch | 2,081-2,279 | **249-250** | **8.75x faster** |

Sources: `DATASCALE-2M.txt:1142` (gated), `DATASCALE-2M-NOGATE.txt:69` (ungated). Provenance:
`verify_wall_ab_ckpt.py` -> 1 arm, 0 problems, 3,811,534 params, best = epoch 3 (ASST 24.96%). Mechanical: 0
agreement-step lines, 0 gate-step lines, `METHOD: OURS STEP RULE, GATES OFF`.

This is the **highest held-out VAL at d=256** this project has measured: 23.12% against 21.29% (gated 2M), 20.81%
(gated 500k), 20.50% (best seed at 200k) and the 19.21% three-seed mean.

## The verdicts

    U1  VAL >= 21.79%                       23.12%  -> PASS (+1.83 pp)
    U2  vs ungated 500k, lands 21.5-23.0%   23.12%  -> OUTSIDE, high by 0.12 pp
    U3  ppl < 149.14                        104.71  -> PASS
    U4  mechanical                          0 / 0 lines, GATES OFF -> PASS
    U5  concepts >= 36                      36/36   -> PASS

U2 was the interesting one and it missed on the side that matters: I predicted the ungated curve would buy at most
+1.0 pp for 4x data (band capped at 23.0), and it bought **+1.07 pp**. So the falsifier fired in the direction its
own text named - *"above 23.0 reopens the ladder"*.

## What that does to the flattening claim

    gated:    500k 20.81 -> 2M 21.29     +0.48 pp for 4x = 0.24 pp/doubling
    ungated:  500k 22.05 -> 2M 23.12     +1.07 pp for 4x = 0.54 pp/doubling

Both curves flatten between 500k and 2M, but they do not flatten to the same place, and the ungated one flattens
less. Earlier today's public statement - "the data lever bends flat past 500k" - was measured on a gated ladder and
is too strong: it is **the gated method that bends hardest**. At 0.54 pp/doubling the ungated curve is still
climbing, and a 4M ungated arm is now the obvious next rung rather than a questionable one.

## The gates, measured twice

    500k: gated 20.81 -> ungated 22.05   (+1.24 pp, 7.7x faster)
    2M:   gated 21.29 -> ungated 23.12   (+1.83 pp, 8.75x faster)

Two rungs now agree in direction and roughly in size, which is the replication the 500k arm could not provide on
its own. The mechanism is visible in both: ungated train accuracy runs far above gated (35.10% vs 21.14% here),
i.e. the gates were rejecting updates that generalised.

## Not claimed

* One seed per configuration, as always: the 200k rung showed a 3.07 pp spread across three seeds, which is larger
  than either delta here. Direction and speed ratios are solid; effect sizes are not.
* No headline change: **37.37%** (d=512, full corpus, 6 epochs) is still 14.25 pp above this number, and the d512
  step-size confound from `D512-PROBE-RESULT.md` is untouched by this arm.
* The gates' stated purpose is protecting held-out quality. At two rungs, on one seed each, they did the opposite.
  That is not the same as proving they never help.
