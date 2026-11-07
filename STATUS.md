# Where GlassNet is right now

Updated: 2026-11-06. Every figure below comes from a run recorded in `docs/history/` or `lens/runs/`.

## Honest completion

| Area | Done | Note |
|---|---:|---|
| Autograd engine | 100% | gradients verified to 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) |
| Our optimiser | 100% | built and measured; the matched 3-seed sweep puts it +0.78 points over SGD's best setting (28.59% vs 27.81%) and level at matched epoch, and it fails the stability bar - not the 4.3 points once published |
| KV cache | 100% | 2.29x faster (0.483 -> 0.211 ms/token), identical output |
| Exports (npz/safetensors/gguf) | 95% | safetensors export of the best checkpoint pending |
| Local server | 95% | serves, not yet load-tested |
| Live inspector | 85% | real weights on :8770, read-only |
| Measurement rig | 95% | every run logged to JSONL |
| Named concepts | 60% | 256 neurons, audit runs, coverage printed |
| Our own architecture | 40% | built and measured; **lost** to the plain shape |
| Safety layer | 10% | designed, not built |
| Coherent conversation | 15% | grammar real, content weak |
| Complete readability | 10% | partial coverage only |
| **Machine (engineering)** | **~90%** | |
| **Intelligence (model quality)** | **~22%** | |
| **Project ready to ship** | **~32%** | |

## What this means in one paragraph

The scaffolding around the model is genuine engineering: hand-derived gradients that pass numerical checks,
an optimiser that measurably beats SGD, a cache that is 2.29x faster with identical output, and a system that
stops and resumes without losing work. The model itself is not good yet - 37.37% held-out next-token
accuracy at best. Anyone claiming this is a working assistant would be lying, and the numbers in this
repository are the proof of that.

## Best numbers

| Metric | Value | Where |
|---|---:|---|
| Best held-out accuracy | 37.37% | d=512, full corpus, 6 epochs |
| d=256 full corpus | 35.26% | by position, `lens/runs/ATTN-GRADE-RUN.txt:28` (legacy by-id: 26.31%) |
| d=256, 2M windows | 24.20% | seed 1, by position, `lens/runs/GPU-BEST.txt:64` |
| d=512, 3 epochs, 2M windows (run twice) | 24.39% / 24.40% (ppl 109.90 both) | same script, 0.01 pp spread between two runs a day apart, `docs/history/D512-REPRODUCIBILITY.md` |
| Parameters, shipped d=256 model | 5,908,686 | counted off the model object; `docs/history/PARAM-AUDIT.md`, also `lens/runs/GPU-BEST.txt:57` |
| Optimiser vs SGD (matched sweep, 3 seeds) | 28.59% vs 27.81% best-epoch mean; 29.12% vs 27.15% at matched epoch | `docs/history/PUBLIC-NUMBER-AUDIT.md` |
| Gradient check | 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) max error | `tests/test_gradcheck.py` |
| KV cache speedup | 2.29x (0.483 -> 0.211 ms/token, 2,069 -> 4,747 tokens/s) | `lens/runs/KV-CACHE-MICRO.txt` |
| Arithmetic, unseen operands | 0.0% exact | digit 0 alone: 100%; two different runs, `lens/runs/ARITHMETIC.txt` and `lens/runs/ARITH_DIAG.txt` |

## Levers measured and closed

These were tested one change at a time against a matched reference, and lost on every seed.
They are listed so nobody has to re-run them.

| Lever | Result | Where |
|---|---|---|
| Attention (bare single head, no position) at 20k windows | 8.21% vs 11.02% reference, 2 seeds, -2.81 | `docs/history/CTX-ATTN-ARMS.md` |
| Window 64 (double tokens per example) at 20k windows | 9.45% vs 11.02% reference, 2 seeds, -1.57 | `docs/history/CTX-ATTN-ARMS.md` |

## Measurement bugs found and fixed

A wrong number is worse than a missing one, so these are listed with the correction.

| Bug | What it did to the numbers | Where |
|---|---|---|
| Talk decoder left-padded short prompts with token id 0 (`<unk>`) | 20-23 of 32 context slots were `<unk>` for the project prompts, so 95.1% (d256) / 96.5% (d512) of the reply tokens were `<unk>`; pad-free the same weights measure 26.4% / 21.5% | `docs/history/PROMPT-ENCODING.md` |
| Architecture table added the proposed recurrent core and counted the tied embedding twice | published "7,081,122 parameters"; the shipped model prints **5,908,686**, so the count was overstated by 1,172,436 (19.8%) | `docs/history/PARAM-AUDIT.md` |
| Cost table added a stage labelled "runs 32x" exactly once | published 0.1446 ms/token; with the core counted the 32 times it runs, the same rows give 0.2283 ms | `docs/history/PARAM-AUDIT.md` |
| KV cache speedup "9.7x" had no run file behind it | only the prose figure existed (0.217 vs 2.115 ms, `docs/history/STATE.md:2730`); the recorded micro-benchmark is **2.29x** (0.483 -> 0.211 ms/token, identical to 5.3e-06) | `lens/runs/KV-CACHE-MICRO.txt` |
| `wall_ab_grade.py` P2 computed `on - off` but graded the pre-registered `off - on` band | the falsifier (`on - off < 0.5`) fired on a real 3.11 pp LOSS and called it "earns under 0.5 pp"; the -1.0..-6.0 band was unreachable, so a perfect result would still have printed "OUTSIDE". Numbers unchanged (-1.39 / -3.11), only the sentence was wrong | `docs/history/GRADER-P2-SIGN.md` |
| A 200k-window arm ran at 19:05 on `data/chat/prepared` with plain SGD | its VAL 13.12% looks like progress next to the 20k numbers but is **not comparable**: 13.3M vs 294.2M cached tokens, gates OFF vs `ours`, 13,651 vs 87,035 val windows, 40 vs 36 concepts. Recorded as a run, never as evidence - it also had no queue entry and no pre-registered prediction | `docs/history/UNSCHEDULED-DATASCALE-RUN.md` |
| Same command, same seed, three runs at the 20k-window rung | VAL acc 11.32 / 10.79 / 10.29 and ASST 11.96 / 11.42 / 10.92 - a **1.03 pp spread**, not the 0.01-0.02 pp measured on the d512 / 2M-window path. The small figure must not be carried to this rung, and a +/-1.25 band around one draw cannot serve as a control at n=1 seed | `docs/history/REF-REPRO.md` |
| A restarted run rewrote `<out>_best.npz` from its own epoch 1 | `wall_off_s0`'s epoch-3 checkpoint was replaced by an aborted second attempt's epoch 1. Repaired earlier; the **guard is now wired into the trainer and proven at the call site** (2 hunks, patch sha256 f10dc89e -> e5acd0aa, 4/4 unit tests, `--check` import=True call=True, and the end-to-end pair: run 2 printed 1 `CLOBBER GUARD` line, archived run 1's best.npz as sha256 b349a10b... with a receipt, and wrote its own 422ebc7e...) | `docs/history/BEST-CKPT-GUARD-APPLIED.md`, `lens/runs/BEST-CKPT-GUARD-E2E.txt` |

## Wall A/B (concept-wall bottleneck) - FINISHED 2026-11-06 17:51, six arms, P1 FAILs as coded (band mixes two columns)

Question: docs/history/PROBLEMS.md OPEN #1. Design + predictions written before any run:
docs/history/WALL-AB-PREDICTIONS.md (P1-P5, plus P6 "supervision must not move a number").
Full write-up with the grader's own words: `docs/history/WALL-AB-VERDICT.md`.

| arm | seed | final val ppl | val acc | concepts | params | source |
|---|---|---:|---:|---:|---:|---|
| wall_off (no bypass) | 0 | 399.50 | **12.18%** | 20/36 | 3,811,534 | lens/runs/WALL-AB.txt:201-202 |
| wall_on (tied full-rank bypass, +2,097,152 params) | 0 | 575.00 | 10.79% | 20/36 | 5,908,686 | lens/runs/WALL-AB.txt:477-478 |
| wall_r64 (rank 64, +1,292,416 params) | 0 | 550.85 | 11.28% | 24/36 | 5,103,950 | lens/runs/WALL-AB.txt:664-665 |
| wall_off (no bypass) | 1 | 395.80 | **12.79%** | 26/36 | 3,811,534 | lens/runs/WALL-AB.txt:850-851 |
| wall_on (tied full-rank bypass) | 1 | 628.41 | 9.68% | 22/36 | 5,908,686 | lens/runs/WALL-AB.txt:1036-1037 |
| wall_r64 (rank 64) | 1 | 521.68 | 10.87% | 26/36 | 5,103,950 | lens/runs/WALL-AB.txt:1222-1223 |

Runner marker: `lens/runs/WALL-AB.txt:1226 ===== WALL-AB DONE 2026-11-06 17:51 arms_missing=0`.
Reference for the same 20k-window rung: 11.96 / 10.08 (mean 11.02%), lens/runs/CTX-ATTN.txt.

Verdicts as printed by `lens/scripts/wall_ab_grade.py` (output `lens/runs/WALL-AB-GRADE.txt`,
sha256 ac55e35b..., 6 rows in `lens/runs/WALL-AB-RESULTS.tsv`):

* **P1 FAIL as coded, but the band mixes two columns** - the 10.0-12.5 band was derived from `ASST` values
  (11.96 / 10.08 at `LEARNCURVE.txt:4773` and `:4960` - not in the file the predictions cite, which has no
  such numbers) and then applied to the `VAL` column. On the graded column `wall_on` s0 10.79% is inside
  the band and s1 9.68% is 0.32 pp below the floor. Compared on the SAME column the control reproduces
  within 0.55 pp on both seeds (VAL -0.53 / +0.15, ASST -0.54 / +0.22), and the trainer commands and the
  five config-header lines match field for field. So this FAIL is a metric mismatch, **not** evidence that
  the rung drifted. The gate keeps its coded verdict - re-deriving it after seeing the result is exactly
  what a gate must not do - so until the maintainer rules, no delta against the stored rung is quoted. Both readings
  with all numbers: `docs/history/WALL-AB-P1-METRIC.md`.
* **P2 OPPOSITE SIGN** - off minus on = +1.39 (s0), +3.11 (s1). Removing the bypass did not cost the
  predicted 1-6 points; on both seeds it was ahead.
* **P3 MIXED** - r64 minus on = +0.49 (s0), +1.19 (s1): not below `on` on either seed, so "rank 64 is
  under-provisioned" is unsupported, but the dose is not clean at seed 1 either.
* **P4 PASS** - mean train-minus-val gap off +8.35 vs on +28.98, i.e. 20.63 pp smaller with 2.1M fewer
  parameters: at this rung the big gap follows capacity.
* **P5 CONTRADICTION CONFIRMED** - the no-bypass arm is above the bypass arm by more than 1.0 pp on both
  seeds, reported as a finding against the plan, not as noise.
* **P6 not triggered** - the resumed arms' recorded command lines match `run_wall_ab.sh` field-for-field
  (only shell quoting of the seed differs), so the resume is valid and its arms are poolable.

What that means: the A/B does **not** establish that the concept wall is lossy, and it does not license
the wider-residual build. It is a coherent lead - same direction on both seeds, 2.1M fewer parameters,
far less overfit - and its control reproduces on a matched comparison, but the pre-registered gate as
coded still says FAIL, so nothing is quoted against the stored rung until that is ruled on. If the maintainer
adopts the metric-consistent reading, the deltas become quotable and the plan's next lever is data scale;
if he keeps the gate as coded, the control has to be re-run before the block can be read at all.

A third run of the same command, same seed, launched at 18:07 and finished at 18:15, landed at VAL 10.29% /
ASST 10.92% (`lens/runs/REF-REPRO.txt`) - so three draws of one configuration span **1.03 pp**, and P1's
+/-1.25 band around a single draw cannot work as a control at n=1 seed. Full write-up:
`docs/history/REF-REPRO.md`.

### Replication on fresh seeds (WALL-REP) - finished 2026-11-06 18:56

A separate pre-registered block (`docs/history/WALL-REP-PREDICTIONS.md`), own log
`lens/runs/WALL-REP.txt`, own tags - its arms are **not** in the six-arm table above.

| arm | seed | VAL acc | ASST acc | source |
|---|---|---:|---:|---|
| wall_rep_off (no bypass) | 2 | **13.40%** | 13.83% | lens/runs/WALL-REP.txt:329 |
| wall_rep_on (tied bypass) | 2 | 9.98% | 10.65% | lens/runs/WALL-REP.txt:110 |
| wall_rep_off (no bypass) | 3 | **13.26%** | 13.72% | lens/runs/WALL-REP.txt:517 |
| wall_rep_on (tied bypass) | 3 | 11.21% | 11.99% | lens/runs/WALL-REP.txt:705 |

**R1 direction PASS** (+3.42 at seed 2, +2.05 at seed 3), **R2 magnitude PASS** (both inside the predicted
1.0-4.0 pp), **R3 noise bound FAIL** (the `on` arm spans 1.53 pp over four draws against a 1.5 pp bound -
missed by 0.03), **R4 not pooled PASS**. All four seeds now put the no-bypass arm ahead:
+1.39 / +3.11 / +3.42 / +2.05. Because R3's pre-registered falsifier says a span above 1.5 pp means "no
delta may be quoted at all", **no delta is quoted here and none is claimed as significant** - the numbers
are shown, the rule is not re-read. Checkpoints: 4 arms, 0 provenance problems
(`verify_wall_ab_ckpt.py lens/runs/WALL-REP.txt`). Full write-up: `docs/history/WALL-REP-RESULT.md`.

Every arm's checkpoint has been machine-checked against the epoch its log line named
(`lens/scripts/verify_wall_ab_ckpt.py`, output `lens/runs/WALL-AB-CKPT-VERIFY.txt`,
sha256 4e11965f...: **6 arms, 0 provenance problems**; the 5-arm version is kept, not deleted, at
`lens/runs/archive/ckpt_verify_5arms_2026-11-06/`, sha256 6b55befd...).
Supervisor: lens/scripts/run_wall_ab_resume2.sh via the chain (label wall_ab_resume2), log
`lens/runs/WALL-AB.txt` (1,226 lines).

## Data scale at d=256 - the first lever that moved this rung (2026-11-06 19:29)

One flag changed (`--max-windows 20000` -> `200000`), everything else identical: same corpus, same command,
same 36-concept set, **same 3,811,534 parameters**, no bypass, seed 0 - and the **same 87,035-window
held-out set** in both runs (`n=44418` masked / `n=58191` assistant positions in each epoch-3 line, verified
from both logs), so the extra data cannot have come from resizing the scored set.

| run | seed | windows | VAL acc | ASST acc | final ppl | source |
|---|---:|---:|---:|---:|---:|---|
| `wall_off` | 0 | 20,000 | 12.18% | 13.00% | 399.50 | `lens/runs/WALL-AB.txt:116` |
| `datascale_honest_s0` | 0 | 200,000 | **17.43%** | 18.61% | 207.29 | `lens/runs/DATASCALE-HONEST.txt:230` |
| `wall_off` | 1 | 20,000 | 12.79% | 13.21% | 395.80 | `lens/runs/WALL-AB.txt:850` |
| `datascale_honest_s1` | 1 | 200,000 | **19.71%** | **20.82%** | **184.32** | `lens/runs/DATASCALE-HONEST-s1.txt:262` |

**Replicated on the second seed.** Seed 1 was pre-registered (`docs/history/DATASCALE-SEED1-PREDICTIONS.md`)
and passed every check: VAL 19.71% inside the 15.18-20.18% band (R1 PASS), ppl 184.32 < 300 (R2 PASS), and the
concept audit again **36/36** (C1 PASS). Within-seed gains: **+5.25 pp (seed 0) and +6.92 pp (seed 1), mean
+6.09**; mean VAL 18.57%, mean ASST 19.71%. The two seeds differ by 2.28 pp, which is larger than the 1.03 pp
run-to-run spread measured at the 20k rung, so the honest statement is the two values and their mean, not a
tight interval. Grader: `lens/scripts/datascale_grade.py`; write-up `docs/history/DATASCALE-SEED1-RESULT.md`.

The concept jump was not predicted by D1/D2, and it repeated: the audit went **20/36 at 20k windows ->
36/36 at 200k** on both seeds, counted from the AUROC rows rather than from the summary lines.

**Is the 3-epoch schedule truncating it? No.** A 6-epoch arm at 200k, seed 0, one knob changed
(`docs/history/DATASCALE-EP6-PREDICTIONS.md` written first), reached final VAL **18.43% / ppl 201.69** - a gain
of **+1.00 pp** over the 3-epoch 17.43, landing exactly on the predicted threshold, so E1 is graded **PARTIAL**
by its own wording (E2 ppl PASS). The path there was a swing, not a climb: 17.48, 17.90, 17.43, **15.13**,
**19.68**, 18.43 - and the best saved checkpoint is epoch 5 (ASST 20.83% vs 18.87% for the 3-epoch arm). So
doubling the schedule doubles the cost for +1.00 pp of endpoint and a better best-checkpoint, and this rung is
unstable epoch-to-epoch. Write-up: `docs/history/DATASCALE-EP6-RESULT.md`; the 3-epoch result is not averaged
with it.

Caveats, stated because they matter: **two seeds**, so no p-value and no invented interval - the two values
and their mean are what is published. The 20k rung's measured run-to-run spread is 1.03 pp
(`docs/history/REF-REPRO.md`), and the two seeds themselves differ by 2.28 pp, i.e. seed-to-seed variation at
this rung is the larger of the two. The headline is unchanged - best held-out accuracy is still **37.37%**
(d=512, full corpus, 6 epochs). Checkpoints verified with `verify_wall_ab_ckpt.py`: the seed-1 arm is 1/1
clean, the parameter count off the stored arrays is 3,811,534 in both seeds, and `best.npz` equals the epoch
the log named. Write-ups: `docs/history/DATASCALE-RESULT.md` (seed 0) and
`docs/history/DATASCALE-SEED1-RESULT.md` (seed 1).
