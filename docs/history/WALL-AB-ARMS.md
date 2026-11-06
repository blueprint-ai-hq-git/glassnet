# WALL-AB arms — the numbers as they land

Source of every figure: `lens/runs/WALL-AB.txt` (line numbers given). Nothing is computed or estimated.
Design and predictions, written before any arm ran: `docs/history/WALL-AB-PREDICTIONS.md`.
Status: **5 of 6 arms done** at the time of writing (2026-11-06 17:5x IST). Last arm (`wall_r64` s1)
launched 17:42:23. No verdict is possible until it finishes.

Checkpoint provenance for every finished arm is machine-checked by
`lens/scripts/verify_wall_ab_ckpt.py` (read-only), output kept in `lens/runs/WALL-AB-CKPT-VERIFY.txt`:
5 arms verified, 0 with a provenance problem.

## Arm table

| arm | seed | final VAL ppl | final VAL acc | ASST (by pos) ppl / acc | MASKED ppl / acc | concepts | epoch-3 line | final line |
|---|---|---|---|---|---|---|---|---|
| `wall_off` (no bypass) | 0 | 399.50 | **12.18%** | 355.22 / 13.00% | 1402.15 / 5.94% | 20/36 | `:116` | `:201` |
| `wall_on` (tied, +2,097,152 params) | 0 | 575.00 | 10.79% | 505.00 / 11.42% | 1502.15 / 7.31% | 20/36 | `:392` | `:477` |
| `wall_r64` (rank 64, +1,292,416 params) | 0 | 550.85 | 11.28% | 486.97 / 11.59% | 1447.35 / 9.71% | 24/36 | `:580` | `:664` |
| `wall_off` | 1 | 395.80 | 12.79% | 350.02 / 13.21% | 1321.54 / 7.09% | 26/36 | `:765` | `:850` |
| `wall_on` (tied, +2,097,152 params) | 1 | 628.41 | 9.68% | 549.10 / 10.30% | 1603.56 / 6.79% | 22/36 | `:952` | `:1036` |
| `wall_r64` | 1 | — | — | — | — | — | running (17:42:23) | — |

Parameter counts are the ones the trainer printed in each arm's announcement and were re-counted off the
stored arrays by the verifier: `off` 3,811,534 (`:733`), `on` 5,908,686 (`:360`), `r64` 5,103,950 (`:547`).
The deltas against `off` are therefore +2,097,152 and +1,292,416 exactly as the design states.

## Per-arm detail

`wall_off` seed 0 (`:97`, `:107`, `:116`): ep1 7.54% (ppl 588.26) → ep2 7.45% (668.92) → ep3 12.18%
(399.50); epoch times 49/42/42 s. Gate: 188/188 steps accepted, 0 refused. Agreement: 49 of 237 refused
(20.7%). Result checkpoint `lens/runs/chat/wall_off_s0_ep3.npz`, sha256 `40d4120a…4c16` (15,261,816 bytes).

`wall_on` seed 0 (`:373`, `:383`, `:392`): ep1 9.39% (544.13) → ep2 9.01% (804.04) → ep3 10.79% (575.00);
94/89/87 s. Gate: 0 of 235 refused (0.0%), probe 8.9699 → 6.2616. Agreement: 2 of 237 refused (0.8%),
A_train 0.2192 → 0.2048. Result `wall_on_s0_ep3.npz`, sha256 `99a4ec80…5907` (23,650,664 bytes).

`wall_r64` seed 0 (`:560`, `:570`, `:580`): ep1 8.76% (563.74) → ep2 **13.21%** (453.27) → ep3 11.28%
(550.85) — the arm peaked at epoch 2, so its `best.npz` is the epoch-2 file, not epoch 3 (see provenance
below); 81/81/87 s. Gate: 1 of 232 refused (0.4%), probe 9.0590 → 6.1469. Agreement: 5 of 237 refused
(2.1%), A_train 0.2039 → 0.1980. Result `wall_r64_s0_ep3.npz`, sha256 `cf979050…deaa` (20,431,974 bytes).

`wall_off` seed 1 (`:745`, `:756`, `:765`): ep1 7.91% (581.39) → ep2 10.47% (450.93) → ep3 12.79%
(395.80); 69/69/72 s. Gate: 0 of 221 refused (0.0%), probe 9.3102 → 5.7582. Agreement: 16 of 237 refused
(6.8%), A_train 0.2513 → 0.2085. Result `wall_off_s1_ep3.npz`, sha256 `70162257…2106` (15,261,828 bytes).
Its announcement (`:733`) prints the same 3,811,534 params as seed 0 and the same seeded-1 20,000-window
cap (`:678` region), i.e. the seed-1 arm is the same configuration, not a drifted one.

`wall_on` seed 1 (`:854`, `:932`, `:942`, `:952`): ep1 8.22% (ppl 564.26) → ep2 **10.94%** (529.38) → ep3
9.68% (628.41); 93/88/89 s. It peaked at epoch 2, so its `best.npz` is the epoch-2 file (provenance
below). Gate: 0 of 236 refused (0.0%), probe 9.2767 → 6.1598. Agreement: 1 of 237 refused (0.4%),
A_train 0.2062 → 0.2174. Result `wall_on_s1_ep3.npz`, sha256 `d28c4146…8684` (23,650,676 bytes).
Announcement `:919` prints 5,908,686 params, the same as `wall_on` seed 0, so the seed-1 arm is the same
configuration.

## What these five lines do and do not say

* **Not a verdict.** P1-P5 are graded by `lens/scripts/wall_ab_grade.py` after all six lines exist. One
  arm is still missing (`wall_r64` s1, running since 17:42:23), and every prediction in
  `WALL-AB-PREDICTIONS.md` is a both-seeds claim. The grader's output, not this file, is the verdict.
* **Both seeds now point against P2's predicted direction.** P2 says `wall_off` is LOWER than `wall_on`;
  the measured pair is off 12.18% / 12.79% versus on 10.79% / 9.68%, i.e. the no-bypass arm is **1.39 pp**
  higher at seed 0 and **3.11 pp** higher at seed 1, with 2,097,152 fewer parameters. Both seeds are in,
  so P2's falsifier is now met on its own terms - but it is graded by the script, not asserted here.
* **The rank-64 arm sits between them at seed 0:** 11.28%, which is 0.49 pp above `wall_on` and 0.90 pp
  below `wall_off`. Seed 1 for it is running.
* **Train accuracy runs far ahead of held-out in every arm** (e.g. 39.23% train vs 11.28% val at `:580`;
  39.03% vs 10.79% at `:392`; 39.39% vs 9.68% at `:952`), which is the overfit direction P4 asks about.
* **No arm talks.** The first logged greedy reply in each of the five finished arms is word salad
  (`:132`, `:408`, `:595`, `:781`, `:967`). The bypass question is about the concept wall, not about
  fluency, and nothing here is claimed as talk progress.

## Provenance: checked for every finished arm, not just recorded

`train_chat.py` starts its "new best" tracking empty and overwrites `<out>_best.npz` on every run, so a
restarted arm can leave `best.npz` holding a *different* run's weights. That happened to `wall_off` s0
earlier today (below), which is why every arm is now checked mechanically instead of by eye:

| arm | epoch the log named NEW BEST | `best.npz` sha256 == that epoch's | arrays equal | `__meta__` matches arm |
|---|---|---|---|---|
| `wall_off` s0 | ep3 (13.00%, `:119`) | yes `40d4120a…` | yes | yes (`direct_mode off`, seed 0) |
| `wall_on` s0 | ep3 (11.42%, `:395`) | yes `99a4ec80…` | yes | yes (`tied`, seed 0) |
| `wall_r64` s0 | ep2 (13.61%, `:573`) | yes `dca4c38c…` | yes | yes (`lowrank`, seed 0) |
| `wall_off` s1 | ep3 (13.21%, `:768`) | yes `70162257…` | yes | yes (`off`, seed 1) |
| `wall_on` s1 | ep2 (10.94%, `:945`) | yes `1298bbc5…` | yes (`ep2.npz`, 23,650,676 B) | yes (`tied`, seed 1) |

At seed 0 on this rung the arm numbers sit in the band P1 calls comparable, but P1 is graded only from
the `wall_on` pair, so it is not called here either.

## The arm-1 provenance defect (found 17:1x, repaired 17:2x, kept as the reason for the check above)

`wall_off` seed 0 was trained twice: it completed at 17:07:13 (`:116`/`:201`, ep3 12.18%), then a
supervisor with a broken skip test restarted it at 17:07:27 and that second process was stopped at
17:10:39 after its own epoch 1. Markers on disk:

| file | mtime | sha256 |
|---|---|---|
| `wall_off_s0_ep3.npz` (the result) | 17:07:13 | `40d4120a…4c16` |
| `wall_off_s0_ep2.npz` (same run) | 17:06 | `b4745f6d…f30a` |
| `wall_off_s0_ep1.npz` (second attempt) | 17:11:54 | `714d15c6…60b0` |
| `wall_off_s0_best.npz` (second attempt) | 17:11 | `714d15c6…60b0` — identical to ep1 |

The wrong file was archived (not deleted) to
`lens/runs/archive/wall_ab_bestfix_2026-11-06/wall_off_s0_best.npz.OVERWRITTEN-714d15c6`
(sha256 re-verified after copy) and `best.npz` was restored from `ep3.npz`, so it now reads
`40d4120a…` as the table above records. Full write-up: `docs/history/WALL-AB-BESTFIX.md`.
Whether the two attempts' epoch-1 weights were byte-identical is **not established** — the printed
epoch-1 losses agree to four decimals (6.9867 / 588.26 / 7.54%) in both attempts, which is suggestive but
is not a hash comparison, and the overwritten file means it can no longer be checked.

The code fix (make `best.npz` tracking resume from an existing checkpoint, or refuse to overwrite a
`best.npz` whose run did not start at epoch 1) is still **not applied**: `train_chat.py` is executing the
A/B right now, and editing it between arms would make the arms non-identical, which is the one thing the
design does require. Until it is fixed, an arm's number must be quoted from its log line plus `ep3.npz`
(or the epoch the log named), never from `best.npz` alone.
