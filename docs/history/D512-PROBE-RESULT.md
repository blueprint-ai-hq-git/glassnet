# d512 probe: 2,854 s per epoch (P1 PASS) - and the relgrad arm trails by 6.34 pp at epoch 1 for a reason that is not the step rule

2026-11-07 15:31:45 IST, `lens/runs/PROBE-D512.txt` (second segment; the first is the 900k calibration),
`exit=0`, `PROBE-D512 DONE 2026-11-07 15:31`. Predictions: `docs/history/D512-COST-PROBE-PREDICTIONS.md`.

## Result, against the headline's epoch 1

| | headline ep 1 (plain SGD) | this probe ep 1 (relgrad, gates off) | difference |
|---|---:|---:|---:|
| VAL | 31.52% | **25.18%** | **-6.34 pp** |
| ASST | 33.46% | 26.67% | -6.79 pp |
| held-out ppl | 46.99 | 110.24 | +63.25 |
| train acc | 42.01% | **21.12%** | -20.89 pp |
| concepts | 36/36 | 36/36 | same |
| epoch time | 1,715 s | **2,854 s** (wall 3,238 s incl. load) | **1.66x slower** |

Headline figures from `lens/runs/TIED-D512-EXT6.txt:83`; probe from its own `ep 1/1` line. Provenance:
`verify_wall_ab_ckpt.py` -> 2 arms, 0 problems; both probe checkpoints are 15,567,566 params, best = epoch 1,
sha and arrays identical, `direct=tied/True`. Mechanical: 0 agreement-step lines, 0 gate-step lines,
`METHOD: OURS STEP RULE, GATES OFF`.

## The verdicts

    P1  one epoch 2,500-6,000 s        got 2,854 s  -> PASS. Six epochs scale to ~4.8 h (17,124 s).
    P2  VAL 28-36%                     got 25.18%   -> FAIL, 2.82 pp below the floor
    P3  mechanical (gates off, direct tied, 15.57M params) -> PASS
    P4  the ext6 run untouched          its best checkpoint is still dated 25 Oct, 62,288,028 bytes; the probe
                                        wrote its own tag (probe_d512_relgrad_full) -> PASS

## Why the deficit is NOT evidence against the step rule

The probe's **train** accuracy is 21.12% against the headline's **42.01%** - the model learned half as much from
the same epoch of the same data, at the same width, with the same head. A rule that produces better generalisation
from less fitting would still fit *something*; this looks like a step that is too small, and the log says why:

    STEP RULE: relgrad (OUR rule, lens/opt.py)   opt_lr 0.0441256  momentum 0

`--lr` is not used by this rule at all (`train_chat.py:983`). The step size is the value calibrated at **d=256** to
reproduce SGD at lr 0.4, carried over unchanged to a **d=512** run whose SGD twin used **lr 0.3**. Whether that
constant transfers across width and lr is exactly the thing this arm did not control.

So the honest statement is narrow: **at d=512 with relgrad's d256-calibrated step size, one epoch lands 6.34 pp
behind plain SGD and takes 1.66x longer.** It is not "our step rule is worse at d512", and it is not "the gates
would help here" either - nothing in this arm separates a mis-scaled step from a weaker rule.

## The next experiment this points at

Calibrate relgrad's `opt_lr` at d=512 to match plain SGD at lr 0.3 the way `lens/runs/OPT-GRAD-SCALE.txt` did at
d=256, then re-run one epoch. That is a gradient-scale measurement, not a training run, and it is the only way to
make the d512 comparison mean anything. Until then, the 37.37% headline stands untouched and this arm is a cost
answer plus a flagged confound.

## Not claimed

* One epoch, one seed. Nothing here says anything about a completed six-epoch d512 run.
* No comparison to 37.37%: that is six epochs on the same corpus; this is one epoch of a different step rule at a
  possibly-wrong step size, and it is 12.19 pp below the headline.
* Cost is measured for this configuration only (relgrad at batch 1024, d 512, 16.9M windows).
