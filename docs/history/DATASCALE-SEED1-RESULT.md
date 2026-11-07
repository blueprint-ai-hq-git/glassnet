# DATASCALE-HONEST seed 1: the 10x-data gain replicated, and so did the concept jump

2026-11-07 08:23:59 IST. Seed 1 finished with `exit=0` and the runner's own marker
`DATASCALE-HONEST DONE 2026-11-07 08:23` (`lens/runs/DATASCALE-HONEST-s1.txt`). Predictions were written
before the launch (`docs/history/DATASCALE-SEED1-PREDICTIONS.md`); this file grades them and nothing else.
Grader: `lens/scripts/datascale_grade.py`, which re-reads every pinned source line on each run.

## The two arms, and their 20k baselines

All four rows: same corpus (`prepared_mix_ctrl_honest`), same command, same 36 kept concepts, same
**3,811,534** parameters, no bypass, same 87,035-window held-out set (n=44,418 masked / 58,191 assistant
positions). The only difference within each seed pair is `--max-windows 20000` vs `200000`.

| run | seed | windows | VAL acc | ASST acc | final ppl | concepts | source |
|---|---:|---:|---:|---:|---:|---:|---|
| `wall_off` | 0 | 20,000 | 12.18% | 13.00% | 399.50 | 20/36 | `WALL-AB.txt:116` |
| `datascale_honest_s0` | 0 | 200,000 | 17.43% | 18.61% | 207.29 | 36/36 | `DATASCALE-HONEST.txt:230` |
| `wall_off` | 1 | 20,000 | 12.79% | 13.21% | 395.80 | — | `WALL-AB.txt:850` |
| `datascale_honest_s1` | 1 | 200,000 | **19.71%** | **20.82%** | **184.32** | 36/36 | `DATASCALE-HONEST-s1.txt:262` |

    within-seed gain  seed 0: +5.25 pp      seed 1: +6.92 pp      mean +6.09 pp
    mean VAL over the two seeds: 18.57%     mean ASST: 19.71%     mean ppl: 195.81
    seed-to-seed spread: 19.71 - 17.43 = +2.28 pp

Seed 1's epoch path: VAL 16.24 → 19.22 → 19.71, ASST 16.52 → 20.35 → 20.82, ppl 239.50 → 194.83 → 184.32, i.e.
it was still improving at epoch 3, where seed 0 had flattened.

## The verdicts, as the grader printed them

    R1 ACC REPLICATES   seed 1 VAL 19.71% inside 15.18-20.18%        PASS
    R2 LOSS REPLICATES  seed 1 ppl 184.32 < 300                      PASS
    C1 CONCEPT JUMP     seed 1 audit 36/36 passing (>= 30 required)  PASS
    R3 SEED SPREAD      recorded, not judged: +2.28 pp

So both of the predictions' falsifiers stayed quiet, and the concept jump is not a seed-0 accident: 20/36 at
20k windows, 36/36 at 200k, twice, at identical parameter count.

R3 is the one number that argues for caution: the two seeds differ by 2.28 pp, which is **larger than the
1.03 pp run-to-run spread measured at the 20k rung** (`REF-REPRO.md`). So the honest statement of the result
is the two values and their mean, not a tight interval: **+5.25 and +6.92 pp within seed, mean +6.09**, with
about +/-1.1 pp of seed-to-seed variation implied by two draws.

## Checkpoint provenance, checked in this run

`verify_wall_ab_ckpt.py lens/runs/DATASCALE-HONEST-s1.txt` -> **1 arm verified, 0 provenance problems**:
`best.npz` is the epoch the log named (ep 3), sha and arrays match, `__meta__` says seed 1 / no bypass, and
the parameter count counted off the stored arrays is 3,811,534 - the same as seed 0 and as the 20k baseline,
so the architecture really is unchanged between the two window counts.

## What this does and does not licence

* It turns "one seed, a lead" into a **replicated effect**: two seeds, same direction, both inside the
  pre-registered band, both with the concept audit at 36/36.
* It does **not** change the headline: best held-out next-token accuracy remains **37.37%** (d=512, full
  corpus, 6 epochs). 18.57% mean at d=256 / 200k windows is a different, smaller rung.
* It is still **two seeds**, so no p-value is quoted and no interval is invented; the mean and the two raw
  values are what is published.
* It says nothing about talk quality, "smart", or shipping.
