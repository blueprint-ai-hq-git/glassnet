# Seed replication of the best configuration (4M ungated, seeds 1 and 2): predictions, before the run

2026-11-07 18:3x IST, written before `lens/scripts/run_datascale_4m_nogate_seeds.sh` is launched (after the 8M arm
finishes).

## Why

Everything this project has published about the gates-off result rests on **one seed each**:

    500k ungated  22.05%  (seed 0)
    2M   ungated  23.12%  (seed 0)
    4M   ungated  23.74%  (seed 0)   <- 3 epochs; 23.92% at 6 epochs
    vs gated      20.81 / 21.29       (+1.24 and +1.83 pp at the same seeds)

The 200k rung, where three seeds exist, showed a **3.07 pp spread** - larger than either gates-off delta. So the
headline claim ("the gates cost 1-2 pp") currently has no error bar of its own. This arm puts one on the rung the
ladder ends at: **4M windows, gates off, 3 epochs, seeds 1 and 2**, run back to back in one unit.

Cost: 2 x ~25 min = ~50 min plus assembly, because each seed re-assembles its window sample (the cap is drawn with
the seed).

## Predictions

R1  EACH SEED LANDS IN **22.2 - 25.3%** VAL (23.74 +- 1.55, i.e. the 200k rung's 3.07 pp spread halved).
    FALSIFIER: a seed outside that band means this rung's seed spread is wider than the rung below it, which would
    make every single-seed ladder number weaker than the text currently implies.

R2  THE TWO SEEDS DIFFER BY **<= 3.1 pp** (the largest spread measured at any rung so far).
    FALSIFIER: more than 3.1 pp means the 4M rung is noisier than 200k, and the +0.62 pp "data still pays" step from
    2M to 4M cannot be separated from noise at all.

R3  THE THREE-SEED MEAN (23.74, seed 1, seed 2) LANDS IN **22.6 - 24.9%** and is reported with its range, not a
    +- interval.

R4  MECHANICAL. Both arms: 0 `agreement step` / 0 `gate step` lines, METHOD reads GATES OFF, 3,811,534 params.

R5  CONCEPTS. Both arms >= 36 of the kept set.

## What will not be claimed

* Two extra seeds do not make a p-value. They make a range, and the range is what will be published.
* This does not replicate the gates-off *contrast* (that needs gated arms at the same seeds); it replicates the
  ungated rung itself. Stated because the difference matters: the contrast's error bar would need gated twins too,
  which is a four-arm block and not this one.
* No headline change; 37.37% (d512, full corpus, 6 epochs) stands.
