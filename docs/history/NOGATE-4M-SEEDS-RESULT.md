# The 4M rung is reproducible to 0.10 pp - and that makes the 8M gain real

`DATASCALE-4M-NOGATE-SEEDS.txt` (seed 1 at `:151`, seed 2 at `:301`). Block finished 21:16:32, exit 0, 2 of 2 arms.

    4M windows, gates off, 3 epochs, d=256, batch 256, identical config
    seed 0   23.74%  VAL   ppl 95.99   (DATASCALE-4M-NOGATE.txt:69)
    seed 1   24.13%  VAL   ppl 93.88   ASST 25.84%   concepts 36/36
    seed 2   24.03%  VAL   ppl 93.52   ASST 25.84%   concepts 36/36
    three-seed mean 23.97%   range 23.74 - 24.13   seed1-seed2 spread 0.10 pp

## The pre-registered verdicts

| # | prediction | result | verdict |
|---|---|---|---|
| R1 | each seed in 22.2 - 25.3% | 24.13, 24.03 | **PASS** |
| R2 | the two seeds differ by <= 3.1 pp | 0.10 pp | **PASS** (31x inside the bound) |
| R3 | three-seed mean in 22.6 - 24.9%, quoted with its range | 23.97%, range 23.74-24.13 | **PASS** |
| R4 | zero gate lines | 0 | **PASS** |
| R5 | concepts >= 36 both | 36/36, 36/36 | **PASS** |

Provenance: 2 arms verified, 0 problems; both carry ep3 checkpoints with matching sha, arrays and meta.

## What it actually establishes

**Reproducibility is not a constant - it improves with data.** The 200k rung, where three seeds also exist, spans
**3.07 pp** (17.43 / 19.71 / 20.50). The 4M rung spans **0.10 pp**. Eight times the windows bought twenty times less
seed-to-seed wobble, which is the expected shape (more distinct windows, less sampling variance) and the first time
this project has measured it across rungs rather than assumed it.

**Consequence for the ladder.** The 4M block mean is 23.97% with a range of 23.74-24.13. The 8M arm reached
**24.51%** (`DATASCALE-8M-NOGATE.txt:235`), which is above *every one of the three 4M seeds* by 0.38 to 0.77 pp. So
the "one more doubling still pays" finding published an hour ago no longer rests on a single seed: it beats the whole
measured 4M range.

**Consequence for the 200k claim.** It goes the other way and must be stated: at 200k the three-seed spread (3.07 pp)
is *wider* than the +1.24 pp gates-off delta measured at 500k seed 0. That number stays unpublishable as a delta until
gated twins exist at those rungs. Nothing about this block fixes that.

## What is not claimed

* This replicates the ungated **rung**, not the gates-off **contrast** - a contrast needs gated twins at the same
  seeds, i.e. a four-arm block, which this is not.
* No p-value. Three seeds give a range; the range is what is quoted.
* Not the d256 record: full-corpus d256 reaches 35.72% by position (`docs/history/STATE.md:146`).
* Epoch times in this block ran 487-1,523 s depending on machine load (another loop was training during seed 1); they
  are upper bounds and are not used for any comparison here.
