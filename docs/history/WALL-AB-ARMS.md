# WALL-AB arms — the numbers as they land

Source of every figure: `lens/runs/WALL-AB.txt` (line numbers given). Nothing is computed or estimated.
Design and predictions, written before any arm ran: `docs/history/WALL-AB-PREDICTIONS.md`.
Status: **1 of 6 arms done** at the time of writing (2026-11-06 17:2x IST). No verdict is possible yet.

## Arm table

| arm | seed | final VAL ppl | final VAL acc | ASST (by pos) ppl / acc | MASKED ppl / acc | concepts | epoch-3 line | final line |
|---|---|---|---|---|---|---|---|---|
| `wall_off` (no bypass) | 0 | 399.50 | **12.18%** | 355.22 / 13.00% | 1402.15 / 5.94% | 20/36 | `:116` | `:201` |
| `wall_on` (tied, +2,097,152 params) | 0 | — | — | — | — | — | running | — |
| `wall_r64` (rank 64, +1,292,416 params) | 0 | — | — | — | — | — | queued | — |
| `wall_off` | 1 | — | — | — | — | — | queued | — |
| `wall_on` | 1 | — | — | — | — | — | queued | — |
| `wall_r64` | 1 | — | — | — | — | — | queued | — |

`wall_off` seed 0 detail (`:97`, `:107`, `:116`): ep1 7.54% (ppl 588.26) → ep2 7.45% (ppl 668.92) →
ep3 12.18% (ppl 399.50). 42 s per epoch, 20,000 train windows, 87,035 val windows, `opt relgrad`.
Gates: 188/188 steps accepted, 0 refused; agreement gate 237 checked, 49 refused (20.7%).
Checkpoint of the result: `lens/runs/chat/wall_off_s0_ep3.npz`,
sha256 `40d4120a08b5e86a82498386165411754031fb29d80ae413b6058f859e854c16` (15,261,816 bytes, 17:07:13).

## What this number is NOT

* Not a comparison. The design states the control (`wall_on`) must be measured inside this block
  (P1), so the stored 11.02% reference is **not** a legal comparator. 12.18% alone says nothing
  about whether the concept wall leaks.
* Not a contradiction of P2 yet. P2 predicts `wall_off` < `wall_on`; with `wall_on` unmeasured there
  is nothing to subtract from. Quoting "off is already above the old reference" would be exactly the
  cross-job comparison the predictions forbid.

## Provenance defect found while recording it (real, not bookkeeping)

`wall_off` seed 0 was trained twice: it completed at 17:07:13 (`:116`/`:201`, ep3 12.18%), then a
supervisor with a broken skip test restarted it at 17:07:27 and that second process was stopped at
17:10:39 after its own epoch 1. Markers on disk:

| file | mtime | sha256 |
|---|---|---|
| `wall_off_s0_ep3.npz` (the result) | 17:07:13 | `40d4120a…4c16` |
| `wall_off_s0_ep2.npz` (same run) | 17:06 | `b4745f6d…f30a` |
| `wall_off_s0_ep1.npz` (second attempt) | 17:11:54 | `714d15c6…60b0` |
| `wall_off_s0_best.npz` (second attempt) | 17:11 | `714d15c6…60b0` — identical to ep1 |

So `best.npz` for this arm currently holds the **restarted process's epoch-1 weights**, not the
13.00% epoch-3 checkpoint, because `train_chat.py` starts its "new best" tracking empty and overwrites
`<out>_best.npz` unconditionally. Nothing is lost (`ep3.npz` is intact and its sha256 is recorded
above), and no file was deleted or rewritten by this turn. Whether the two attempts' epoch-1 weights
are byte-identical is **not established** — the printed epoch-1 losses agree to four decimals
(6.9867 / 588.26 / 7.54%) in both attempts, which is suggestive but is not a hash comparison, and the
overwritten file means it can no longer be checked.

Consequence for the remaining five arms: `best.npz` must not be used as the source of an arm's number
(use `ep3.npz` + the log line, as above), and a restarted arm must not be allowed to overwrite
`best.npz`. The code fix is **not** applied in this turn on purpose: `train_chat.py` is executing the
A/B right now, and editing it between arms would make the arms non-identical, which is the one thing
the design does require.
