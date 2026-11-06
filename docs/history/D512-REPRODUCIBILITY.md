# D512-REPRODUCIBILITY — two independent runs of the same d=512 run, same seed

Written 2026-11-06 17:1x IST by the autonomous wake turn (label `wall_ab_resume`).
Every number below was copied from `lens/runs/GPU-QUEUE-RUNS.txt`; the line numbers are given so
anyone can check them. Nothing here is estimated, averaged or inferred.

**Why this file exists.** `docs/history/WALL-AB-PREDICTIONS.md` justifies running its own control arm
inside the block by citing "the measured run-to-run spread on this GPU path is 0.01-0.02 pp
(`docs/history/D512-REPRODUCIBILITY.md`)". That citation pointed at a file that did not exist, i.e.
the A/B was leaning on an unsourced number. This file is the measurement behind that sentence. If the
pair below had disagreed, the honest outcome would have been to correct the A/B document instead.

## The pair

Same wrapper `lens/scripts/run_gpu_d512.sh` (comment: d pinned once as of 2026-11-06),
same data dir `data/chat/prepared_mix_ctrl_honest`, seed 0, 3 epochs.

| | run A (2026-11-05) | run B (2026-11-06) |
|---|---|---:|
| wrapper start line | `GPU-QUEUE-RUNS.txt:148` (16:10) | `GPU-QUEUE-RUNS.txt:317` (15:35) |
| parameter count printed | **15,567,566** (d=512, K=36) — `:205` | **15,567,566** (d=512, K=36) — `:384` |
| epoch 3 line | `:212` | `:391` |
| train acc @ ep3 | 26.39% | 26.39% |
| **VAL ppl** | **109.90** | **109.90** |
| **VAL acc** | **24.39%** | **24.40%** |
| MASKED (legacy by-id) ppl / acc | 304.13 / 16.98% (n=44418) | 304.15 / 16.98% (n=44418) |
| ASST (by-position) ppl / acc | 93.60 / 25.75% (n=58191) | 93.59 / 25.77% (n=58191) |
| epoch-3 wall time | 189 s | 185 s |
| final VAL line | `:289` → ppl 109.90 acc 24.39% concepts 35/36 | `:468` → ppl 109.90 acc 24.40% concepts 35/36 |
| concept audit: failing concept | `mentions_sport` AUROC 0.5153 (`:276`) 35/36 pass | `mentions_sport` AUROC 0.5148 (`:455`) 35/36 pass |

Measured spread between the two runs: **0.01 pp on VAL acc, 0.02 pp on ASST acc, 0.00 on VAL ppl**
(109.90 printed identically at 2 decimals in both). Same single failing concept in both audits.

## What it supports

The A/B's falsifier thresholds are 0.5-1.0 pp (P2 "smaller than 0.5 pp" / P3 "more than 1.0 pp
below"), i.e. 50-100x the measured drift of this path. That is why the A/B can use a same-block
control and still quote a delta.

## What it does NOT show — stated so the number is not over-read

* **One seed, one config.** This is repeatability of a single d=512 / 3-epoch / 2M-window run, not a
  determinism guarantee, and not evidence about the 6-epoch 37.37% run or about d=256.
* **The two wrappers are not byte-identical.** The 2026-11-05 wrapper passed `--d 256` early and
  `--d 512` later (argparse keeps the last value) and passed `--batch` twice; the 2026-11-06 wrapper
  passes each once (the fix is commented in the script). The *effective* config matched, which the
  identical parameter count 15,567,566 in both runs confirms, but the command lines differ.
* **Run B's `=== DONE ... 15:35 exit=0` stamp is not evidence.** The queue runner wrote it at the
  start time, before the training lines (`GPU-QUEUE-RUNS.txt:326`). Run B's completion is evidenced by
  its epoch-3 line (`:391`), its final VAL + concept audit (`:468`), and the checkpoint files in
  `lens/runs/chat/gpu_d512_s0_*` stamped 15:43-15:49.
* These runs are the plain-SGD path (`METHOD: standard`), not the OURS/relgrad path the A/B arms use.
  The drift measured here is therefore a *lower bound* argument for the A/B, not a measurement of the
  A/B arm's own variance.
