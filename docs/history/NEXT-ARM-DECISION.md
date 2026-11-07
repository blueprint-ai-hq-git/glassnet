# What runs after the 2M arm - decided before the 2M arm finishes

2026-11-07 13:2x IST, written while arm `datascale_2m_s0` was in its last epoch. The point is that the next launch
is a pre-decided branch, not a judgement made while looking at the 2M number - which is how a result gets read
backwards.

## Measured costs (from the logs, not estimates)

| arm | windows | batch | steps/epoch | s/epoch measured | source |
|---|---:|---:|---:|---:|---|
| d256 500k | 500,000 | 256 | 1,953 | 554, 558, 558 | `DATASCALE-500K.txt` ep lines |
| d256 2M | 2,000,000 | 256 | 7,813 | 2,279, 2,082 | `DATASCALE-2M.txt` ep lines |
| d512 4M | 4,000,000 | 1024 | 3,906 | 419, 416, 417 | `TIED-D512.txt` ep lines |

Epoch cost scales with **steps**, not with windows: 4x windows at the same batch cost 3.7x the time, while the
d512 arm with 4x the windows but 4x the batch is cheap per epoch despite 4x the parameters. Any "d512 at 2M"
estimate therefore depends on the batch chosen, and **that combination has never been measured** - it needs a
one-epoch probe, not a guess.

## The branch, decided now

**B1 - 2M lands ABOVE 24.0% (outside T2's band, high side).** The steepening is real and stronger than the band.
Next arm: **4M windows, d256, batch 256, seed 0** (~2x the 2M epoch cost, ~70 min/epoch, ~3.5 h). It is the same
ladder with one flag changed, which is the only kind of arm that can be compared cleanly to the others.

**B2 - 2M lands inside 21.5-24.0% (T2 PASS).** The lever pays but decays, exactly as predicted. Next arm:
**width, with the data it needs - d512 at 2M windows**. Measured basis: d512 is 15,567,566 params
(`TIED-D512.txt`), and 2M x 32 tokens / 15,567,566 = **4.11 tokens/param**, the first d512 rung the project
would run that is not starved (the d512 full-corpus rung sits at 8.22). Before committing 3 epochs, run **one
epoch as a cost probe** and record s/epoch, then decide. Do not quote a d512@2M result against the d256 ladder -
different width is a different ladder, and against the 37.37% headline it will be undertrained.

**B3 - 2M lands at or below 20.81% (T1 FAIL).** The lever saturated between 500k and 2M. Two candidate arms, and
the choice between them is the interesting question, not a formality:
  * **gate-off control at 500k** - tests the hypothesis in `docs/history/GATE-REFUSALS-VS-LENGTH.md`: the arm
    refuses ~2/3 of its late steps, so extra data may be buying less than it could. This arm is cheap (measured
    554-558 s/epoch) and it is the only way to tell "data stopped helping" from "the gate stopped accepting".
  * **width with data (d512 at 2M)** - moves the frontier instead of explaining it.
  Under B3 I would run the gate-off control first, because it can explain the ceiling, and a ceiling that is
  explained is worth more than a frontier that is moved by accident.

**B4 - the run dies (INCOMPLETE marker, no ep3 checkpoint).** No result, no branch: relaunch the same arm once
(the runner already refuses to call a partial block done), then treat the outcome as B1/B2/B3.

## Rules this branch keeps

* One flag changes per arm; anything else is a new ladder and must be labelled as one.
* Predictions are written to `docs/history/` **before** the launch, with a falsifier that can fire.
* No number is quoted from an arm that has not finished, and no delta is quoted across ladders.
* The 37.37% headline (d512, full corpus, 6 epochs) is untouched by any of these arms.
