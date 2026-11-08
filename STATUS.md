## 2026-11-07

Today's changes, each with its log line: `docs/history/DAILY-2026-11-07.md`.

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
| `train_chat.py` printed `len(Xtr) / params` under the label "tokens per parameter" - but `len(Xtr)` counts **windows** | every such figure in every log is low by exactly 32x, and the "UNDER-TRAINED" verdict was computed on it: the d=512 / 4M-window rung printed **0.26 (UNDER-TRAINED)** when it truly had **8.22 (`adequate`)**, and the d=256 arms are 0.17 / 1.68 / 4.20 at 20k / 200k / 500k windows, not 0.01 / 0.05 / 0.13. Recorded once as a *prediction* error (`AUTOPILOT-LOG.md:4552`) and never fixed in code; now computes tokens and prints the old ratio beside it so old logs stay comparable. Second unit question under it: `chat_data.py:166` uses `stride = window // 2`, so the figure counts token INSTANCES - the headline's 34.74 is 18.90 distinct tokens/param per epoch, i.e. about half. Thresholds left alone | `docs/history/TOKENS-PER-PARAM-MISLABEL.md`, `lens/scripts/train_chat.py:830` |
| A restarted run rewrote `<out>_best.npz` from its own epoch 1 | `wall_off_s0`'s epoch-3 checkpoint was replaced by an aborted second attempt's epoch 1. Repaired earlier; the **guard is now wired into the trainer and proven at the call site** (2 hunks, patch sha256 f10dc89e -> e5acd0aa, 4/4 unit tests, `--check` import=True call=True, and the end-to-end pair: run 2 printed 1 `CLOBBER GUARD` line, archived run 1's best.npz as sha256 b349a10b... with a receipt, and wrote its own 422ebc7e...) | `docs/history/BEST-CKPT-GUARD-APPLIED.md`, `lens/runs/BEST-CKPT-GUARD-E2E.txt` |

Every percentage on this page and in `README.md` that carries a `file:line` citation is now checked mechanically: `lens/scripts/check_public_cites.py` opens each cited line and fails if the quoted number is not on any cited line of that row (missing file, line past the end, blank line, or a number that appears nowhere it cites). Current state: **33 cited rows on these two pages and 56 across all of `docs/history`, 0 problems**. It has already earned its keep three times - it found a citation to a **blank** line (`STATE.md:2768`, the 37.37% claim, now `:2770`) and a table row that cited the `final VAL` line while also quoting that arm's ASST, a number only on the epoch line (`WALL-AB.txt:850` -> `:765`) - and a third time on this very page, catching a new 2M row that quoted a
best-checkpoint 22.82% cited only to the final-epoch line. It does not judge which column a number came from - that mistake (P1, VAL vs ASST) still needs a human eye.

**The 17.43% arm's checkpoint now verifies mechanically, and one arm has none.** `verify_wall_ab_ckpt.py` used to split a log at `----- off seed N` markers, so a run launched before that convention existed - the seed-0 200k arm behind 17.43% - was invisible to it and the tool answered about a stale block appended after it. With leading unmarked segments parsed, **20 arms across 11 logs** now verify: `datascale_honest_s0` has 3,811,534 params as announced, best checkpoint = epoch 2, and `sha256(best) == sha256(ep2)` with identical arrays. The single remaining problem is the 11.04% / 41-concept run on `data/chat/prepared`: it announces **3,912,509** params but the tag its save line names resolves to the honest seed-1 checkpoint at **3,811,534**, so that run has **no checkpoint of its own** - which matches `docs/history/UNSCHEDULED-DATASCALE-RUN.md` (recorded as a run, never as evidence). Note: `docs/history/VERIFIER-SEGMENTATION.md`.

**Held-out sets are disjoint by construction, at every scale.** Above 200,000 train windows the trainer skips its exact content walk and prints the skip (`train_chat.py:783-787`); every data-scale arm here - 500k, 2M, 16.9M - exceeded that, so none printed a walk. It is redundant: the train/val split is assigned per conversation (`chat_data.py:133`), windows are selected per conversation by that flag (`chat_data.py:165`), and windows never cross a conversation boundary (`chat_data.py:166`), so a train and a val window cannot share tokens; the vocabulary is built from train text only (`chat_data.py:119`). What is still unchecked is corpus-level duplication (a val conversation that is a near-copy of a train one) - that would inflate every number here and has never been tested. Note: `docs/history/DISJOINTNESS-STATED-LIMIT.md`.

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
same concept module (the auto-kept set is data-dependent: 37 kept at 20k windows, 36 at 200k), **same 3,811,534 parameters**, no bypass, seed 0 - and the **same 87,035-window
held-out set** in both runs (`n=44418` masked / `n=58191` assistant positions in each epoch-3 line, verified
from both logs), so the extra data cannot have come from resizing the scored set.

| run | seed | windows | VAL acc | ASST acc | final ppl | source |
|---|---:|---:|---:|---:|---:|---|
| `wall_off` | 0 | 20,000 | 12.18% | 13.00% | 399.50 | `lens/runs/WALL-AB.txt:116` |
| `datascale_honest_s0` | 0 | 200,000 | **17.43%** | 18.61% | 207.29 | `lens/runs/DATASCALE-HONEST.txt:230` |
| `wall_off` | 1 | 20,000 | 12.79% | 13.21% | 395.80 | `lens/runs/WALL-AB.txt:765` |
| `wall_off` | 2 | 20,000 | 13.40% | 13.83% | 384.39 | `lens/runs/WALL-REP.txt:329`, reproduced exactly by `DATASCALE-20K-S2.txt` |
| `datascale_honest_s1` | 1 | 200,000 | **19.71%** | **20.82%** | **184.32** | `lens/runs/DATASCALE-HONEST-s1.txt:262` |
| `datascale_honest_s2` | 2 | 200,000 | **20.50%** | **21.47%** | **180.30** | `lens/runs/DATASCALE-HONEST-s2.txt:256` |
| `datascale_500k_s0` | 0 | **500,000** | **20.81%** | **21.82%** | **171.47** | `lens/runs/DATASCALE-500K.txt:392` |
| `datascale_2m_s0` | 0 | **2,000,000** | **21.29%** | **22.60%** | **149.14** | `lens/runs/DATASCALE-2M.txt:1142`; best checkpoint ep2 ASST **22.82%** at `lens/runs/DATASCALE-2M.txt:817` |
| `datascale_500k_nogate_s0` (**gates OFF** control) | 0 | 500,000 | **22.05%** | **23.22%** | **140.27** | `lens/runs/DATASCALE-500K-NOGATE.txt:69` |
| `sgd_500k_s0` (**plain SGD**, gates OFF) | 0 | 500,000 | 19.81% | 20.38% | 221.78 | `lens/runs/SGD-500K.txt:66` (concepts 31/36) |
| `datascale_2m_nogate_s0` (**gates OFF**, 2M) | 0 | 2,000,000 | **23.12%** | **24.96%** | **104.71** | `lens/runs/DATASCALE-2M-NOGATE.txt:69` |
| `datascale_4m_nogate_s0` (**gates OFF**, 4M) | 0 | 4,000,000 | **23.74%** | **25.71%** | **95.99** | `lens/runs/DATASCALE-4M-NOGATE.txt:69` |
| `datascale_4m_nogate_ep6_s0` (**gates OFF**, 4M, 6 epochs) | 0 | 4,000,000 | **23.92%** | **25.92%** | **93.72** | `lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73` |
| `datascale_4m_nogate_s1` / `_s2` (**gates OFF**, 4M, seeds 1-2) | 1, 2 | 4,000,000 | **24.13% / 24.03%** | **25.84% / 25.84%** | **93.88 / 93.52** | `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` (seed 1) and `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` (seed 2) |
| `datascale_8m_nogate_s0` (**gates OFF**, 8M) | 0 | 8,000,000 | **24.51%** | **26.46%** | **87.55** | `lens/runs/DATASCALE-8M-NOGATE.txt:235` |
| `datascale_16m_ours_s0` (**our step rule**, 16M) | 0 | 16,000,000 | **24.58%** | **26.40%** | **84.47** | `lens/runs/DATASCALE-16M-OURS.txt:225` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS.txt:226` (final, concepts) |
| `datascale_4m_ours05_s0` (**our rule, opt-lr 0.5**, 4M) | 0 | 4,000,000 | **26.06%** | **27.98%** | **85.60** | `lens/runs/DATASCALE-4M-OPT05.txt:154` (VAL/ASST/ppl) and `lens/runs/DATASCALE-4M-OPT05.txt:155` (concepts) |
| `datascale_16m_ours05_s0` (**our rule, opt-lr 0.5**, 16M) | 0 | 16,000,000 | **26.76%** | **28.85%** | **71.69** | `lens/runs/DATASCALE-16M-OURS05.txt:225` (VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS05.txt:226` (concepts) |
| `datascale_16m_nogate_s0` (**plain SGD**, 16M) | 0 | 16,000,000 | **28.30%** | **30.42%** | **80.15** | `lens/runs/DATASCALE-16M-NOGATE.txt:136` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-NOGATE.txt:213` (final, concepts) |

**Three seeds, three matched pairs: mean gain +6.42 pp.** Val acc 17.43% / 19.71% / 20.50% against 20k
baselines of 12.18% / 12.79% / 13.40%, so the paired gains are **+5.25, +6.92 and +7.10 pp** (mean **+6.42**),
mean VAL **19.21%**. The baseline and the arm rise together - a seed that gives a stronger 20k run also gives a
stronger 200k run - which is why the paired gain spans only 1.85 pp while the 200k values themselves span
3.07 pp; the paired gain is the number to quote. Seed 1 passed every pre-registered check (band R1, ppl R2,
concepts C1); seed 2's band check **failed by 0.32 pp on the high side**, the direction its prediction doc had
pre-named as "the band was too conservative", while its mean and concept checks passed. Graders:
`lens/scripts/datascale_grade.py`, `lens/scripts/datascale_seed2_grade.py`; write-ups
`docs/history/DATASCALE-SEED1-RESULT.md`, `DATASCALE-SEED2-RESULT.md`, `DATASCALE-20K-S2-RESULT.md`.

**The lever bends flat between 500k and 2M in the GATED method - and the ungated rung says that was partly the gates.** 2,000,000 windows at seed 0 reached
**21.29% VAL / 22.60% ASST / ppl 149.14**, concepts 36/36, and the pre-registered checks split: T1 (> 20.81%)
PASS at **+0.48 pp**, T3 and T4 PASS, **T2 FAIL** (landed 0.21 pp below its 21.5% floor) and **C4 FAIL** (epoch 3
at 21.29% is 0.40 pp *below* epoch 2, so the best checkpoint - ASST 22.82% - is epoch 2, not the final one). The
whole seed-0 ladder is now 12.18 -> 17.43 -> 20.81 -> 21.29%, i.e. **1.58, then 2.56, then 0.24 pp per
doubling**: 100x data bought +9.11 pp in total and the last 4x bought 5.3% of it, while perplexity kept falling
(399.50 -> 149.14) as accuracy stopped rising. So the earlier "not flattened at 200k" note is now completed by
the rung above it: the curve rose one more step and then flattened, and 0.24 pp/doubling is inside the 3.07 pp
seed spread, so the last rung is not distinguishable from noise on one seed. Source: `lens/runs/DATASCALE-2M.txt:1142`;
grader `lens/scripts/datascale_ladder_grade.py 2m`; write-up `docs/history/DATASCALE-2M-RESULT.md`. Not
comparable to the older 2M result at batch 1024 (24.20%) - different batch and sampling path.

**Re-run at 2M with the gates off, the flat curve moves: 23.12% VAL / 24.96% ASST / ppl 104.71**, against the gated
twin's 21.29 / 22.60 / 149.14 - **+1.83 pp for the same data, and 8.75x faster** (249 s/epoch against 2,081-2,279).
All five pre-registered checks pass except U2, which missed *high*: I predicted at most +1.0 pp over the ungated
500k arm and it bought **+1.07 pp** (21.5-23.0 band, got 23.12) - the direction its own falsifier named as "above
23.0 reopens the ladder". So the two curves are 500k 20.81 -> 2M 21.29 gated (**0.24 pp/doubling**) and 22.05 ->
23.12 ungated (**0.54 pp/doubling**): both flatten, the gated one flattens hardest, and the ungated curve is still
climbing. **23.12% is the highest d=256 held-out number on record** (previous best 21.29% gated), and still 14.25 pp
below the d=512 headline. The gates-off effect has now been measured at two rungs, same direction, similar size
(+1.24 pp at 500k, +1.83 pp at 2M), with the mechanism visible in train accuracy (35.10% ungated vs 21.14% gated).
Source: `lens/runs/DATASCALE-2M-NOGATE.txt:69`; write-up `docs/history/NOGATE-2M-RESULT.md`. One seed per
configuration - the 3.07 pp seed spread at 200k is larger than either delta, so direction and speed are solid and
effect size is not.

**And it is still paying at 4M.** 4,000,000 windows, gates off: **23.74% VAL / 25.71% ASST / ppl 95.99**, concepts
36/36, 484-490 s per epoch (`lens/runs/DATASCALE-4M-NOGATE.txt:69`). Paired to the 2M ungated arm that is **+0.62 pp
for 2x data = 0.62 pp/doubling**, against 0.54 pp/doubling for the 500k->2M step: the ungated rate has not decayed,
which is the opposite of what the gated ladder reported. All five pre-registered checks passed (V1 > 23.12, V2 inside
23.4-24.6, V3 ppl < 104.71, V4 mechanical, V5 concepts), and the one generic check that failed is C4 - **epoch 3
equals epoch 2 to two decimals (23.75 -> 23.74)** while perplexity kept falling, so the arm plateaued on accuracy
inside the run. That plateau is the next question, not more data: the arm is limited by its 15,625 steps x 3 epochs,
so "more epochs at 4M" and "bigger batches" are different experiments and neither has been run. 23.74% is the best d=256 number in this **capped-window ladder** (<=4M windows, 3 epochs, honest mix) - 2.45 pp
above the gated 2M arm and 13.63 pp below the d512 headline - but it is **not** the best d=256 number in the
project: full-corpus d=256 runs reach **35.72% by position** (`docs/history/STATE.md:146`, 16.9M windows,
6 epochs), 10.01 pp higher on the comparable metric. Source for the 23.74% row: `lens/runs/DATASCALE-4M-NOGATE.txt:69`. One seed;
the 3.07 pp spread at 200k dwarfs the +0.62 pp. Write-up: `docs/history/NOGATE-4M-RESULT.md`.

**At 16M windows plain SGD beats our step rule by 3.72 pp, reversing the 500k result.** Matched pair, identical corpus, windows, seed, epochs and batch, only the optimiser differs: **SGD 28.30% / ASST 30.42% / ppl 80.15** against **our rule 24.58% / 26.40% / 84.47** (`lens/runs/DATASCALE-16M-NOGATE.txt`, `lens/runs/DATASCALE-16M-OURS.txt`). I predicted our rule would still win, by less than the +2.24 pp it won at 500k; the pre-registered comparator returned PREDICTION FALSIFIED. **The +2.24 pp is therefore a small-data result** and must not be quoted as a property of the optimiser. Mechanism, written down before epochs 2-3 existed and then observed: our rule's accuracy stayed flat (24.65 -> 24.49 -> 24.58) while perplexity fell 3.80, the signature of a fixed-size unit step overshooting; SGD's decaying steps climbed 27.27 -> 27.90 -> 28.30 over the same epochs. This arm also lost one concept: 35/36, `mentions_a_name` fires on 1.8% of rows (AUROC 0.9982). One seed, and this rule's own seed spread is 1.89 pp - the 3.72 pp gap exceeds it, but two seeds would settle it.

**The 4M rung reproduces to 0.10 pp - and the 8M gain beats the whole range.** Three seeds of the same 4M ungated config: **23.74 / 24.13 / 24.03%**, mean **23.97%** (computed; recorded in `docs/history/NOGATE-4M-SEEDS-RESULT.md:9`), range 23.74-24.13, seed-to-seed spread **0.10 pp** (`lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` for seed 1, `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` for seed 2, and `lens/runs/DATASCALE-4M-NOGATE.txt:69` for seed 0); provenance 2 arms, 0 problems. Reproducibility improves with data: the 200k rung's three-seed spread is **3.07 pp** (`docs/history/DATASCALE-SEED2-RESULT.md`), twenty times wider. So the 8M result (`lens/runs/DATASCALE-8M-NOGATE.txt:235`) is above every one of the three 4M seeds by 0.38-0.77 pp - the "one more doubling pays" claim no longer rests on one seed. The same fact cuts the other way at 200k, where the 3.07 pp spread is wider than the +1.24 pp gates-off delta at 500k, so that delta stays unpublishable until gated twins exist at those seeds.

**Halving the step recovers 2.18 pp - and leaves the rule 1.54 pp behind SGD.** 16M windows, gates off, seed 0, our rule at `--opt-lr 0.5`: **26.76% VAL / ASST 28.85% / ppl 71.69**, concepts 36/36 (`lens/runs/DATASCALE-16M-OURS05.txt:225`). Against the same arm at `1.0` (24.58%, `lens/runs/DATASCALE-16M-OURS.txt:226`) that is **+2.18 pp**, Z1's PASS, and it confirms the overshoot mechanism written down before the arm ran: the 1.0 unit step was too large for 187,500 updates. Against plain SGD (28.30%, `lens/runs/DATASCALE-16M-NOGATE.txt:213`) the shortfall shrinks from 3.72 to **1.54 pp** - roughly 2.2 of the 3.7 was a fixable configuration error and the rest is the rule itself. The same signature reappears one level up: accuracy 26.74 -> 26.85 -> 26.76 while perplexity fell to 71.69, which is why C4 (ep3 >= ep2) failed. Because every ladder rung ran at 1.0, its absolute levels are probably depressed by about this much - deltas stand, ceilings do not.

**One caveat that applies to every number in this table, now measured at two rungs.** Every gates-off runner passes `--opt-lr 1.0`, and that step costs about **2.2-2.3 pp**: halving it took 4M from 23.74% to **26.06%** (+2.32) and 16M from 24.58% to **26.76%** (+2.18) - four times apart in update count, so it is not a scale effect (`docs/history/NOGATE-4M-OPT05-RESULT.md`, `docs/history/NOGATE-16M-OPT05-RESULT.md`). So every level in this table sits roughly 2.2 pp below what its own configuration can reach, and none should be read as a ceiling. Rung-to-rung deltas survive (a constant handicap cancels), and the curve keeps its shape: at 0.5, 4M -> 16M is +0.70 pp, the same flattening the 1.0 curve showed. The 3-seed optimiser sweep used 0.5 (`docs/history/PUBLIC-NUMBER-AUDIT.md:73`), so it and this ladder were never the same configuration.

**The data axis has not flattened: doubling to 8M paid more than the doubling below it.** 8,000,000 windows, gates off, seed 0, three epochs: **24.51% VAL / 26.46% ASST / ppl 87.55**, concepts 36/36 (`lens/runs/DATASCALE-8M-NOGATE.txt:235`; the concept audit 36/36 is on `:236`). The 4M rung was 23.74% (`lens/runs/DATASCALE-4M-NOGATE.txt:69`), so this doubling bought **+0.77 pp** against **+0.62 pp** for 2M -> 4M - the gain grew rather than decaying, which falsifies the expectation written into the prediction (X2 was centred below a repeat of 0.62). Path 23.46 / 24.03 / 24.51 with ppl 95.71 / 91.04 / 87.55. Two limits travel with it: the prepared mix holds ~18.4M windows, so only one more doubling is available before the axis runs out of data, and with one seed this says the axis has not turned down, not that 0.77 pp is precise (the 200k rung's three-seed spread was 3.07 pp). The machine rebooted mid-arm at 18:52; the arm resumed from its epoch-1 checkpoint (about 11 minutes lost, not 51).

**Doubling the steps was not the lever.** Six epochs at the same 4M windows reached **23.92% VAL / 25.92% ASST /
ppl 93.72** (`lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73`) - **+0.18 pp** for doubling the updates from 46,875 to
93,750. The hypothesis behind that arm ("the plateau is step-limited") was falsified by its own pre-registered
threshold (W1 needed >= 24.04, got 23.92). At this rung, **doubling the data bought +0.62 pp and doubling the steps
bought +0.18 pp**, so data is worth ~3.4x what steps are worth per doubling; ppl fell the whole way (95.89 ->
93.72) while accuracy wobbled in a +-0.3 pp band. Neither closes the **11.80 pp** gap to the project's d256 record
(35.72% by position, full corpus, `docs/history/STATE.md:146`). 8M windows were then run: 24.51% (`lens/runs/DATASCALE-8M-NOGATE.txt:235`). One seed;
only the ratio of the two deltas is readable, not their sizes. Note: `docs/history/NOGATE-4M-EP6-RESULT.md`.

**The gates were part of that ceiling, and the headline was never gated.** Same 500k rung, same step rule, same
data, gates OFF: **22.05% VAL / 23.22% ASST / ppl 140.27** against the gated arm's 20.81 / 21.82 / 171.47 - and
**72 s per epoch instead of 558, 7.7x faster** (`lens/runs/DATASCALE-500K-NOGATE.txt:69`, twin at
`DATASCALE-500K.txt:392`). All four pre-registered checks passed (G1 +1.24 pp, G2, G3 mechanical 0 gate lines,
G4 36/36 concepts). The gates let the model fit the training set far harder (22.25% -> 35.20% train) and overfit
far more (train-minus-VAL gap 1.44 -> 13.15 pp) **while held-out accuracy rose** - so at this rung they were
suppressing learning that generalised. That matters beyond the rung: every ladder arm ran `--method ours` (gated),
while the run behind the **37.37%** headline prints `METHOD: standard (the old loop: plain SGD, our gates OFF)`
(`lens/runs/TIED-D512-EXT6.txt`). The project's best number therefore comes from the ungated old loop, and "our
method vs the old loop" has never been measured at a matched rung - that is the next arm. One seed, one rung, and
+1.24 pp is inside the 3.07 pp seed spread, so this is a direction, not a settled effect; the 7.7x is a measured
ratio. Write-up: `docs/history/NOGATE-500K-RESULT.md`.

**And the step rule is worth more than the gates.** Adding the third configuration - plain SGD, gates off, same
rung and seed - gives **19.81% / 20.38% ASST / ppl 221.78**, concepts only **31/36** (`lens/runs/SGD-500K.txt:66`).
The matched three-way at 500k is therefore **relgrad ungated 22.05 > relgrad gated 20.81 > plain SGD 19.81**: our
step rule is worth **+2.24 pp** over the loop this project used before **at this rung** (at 16M windows the ordering reverses - see the paragraph above), the gates cost **1.24 pp**, and plain SGD
is 2.3x cheaper per epoch while learning the least. That is awkward for the headline: **37.37% was produced by
plain SGD** on 16.9M windows in 6 epochs (`TIED-D512-EXT6.txt`), so the best number sits on the weakest-measured
step rule. The next test is that configuration at that scale - d=512, full corpus, 6 epochs, relgrad, gates off -
and it needs a **one-epoch cost probe** first, because relgrad is ~2.3x plain SGD per epoch while plain SGD already
took 1,720 s/epoch at that scale (`docs/history/STATE.md:2771`). One seed per configuration and all three deltas
sit inside the 3.07 pp seed spread: the ordering is a direction, not a settled effect. Write-up:
`docs/history/SGD-500K-RESULT.md`.

**What one d512 epoch actually costs, and a confound that blocks the obvious conclusion.** A one-epoch probe in
the headline's own configuration (d=512, `--direct --direct-mode tied`, batch 1024, lr 0.3, 16.9M windows, our
step rule, gates off) took **2,854 s** - so six epochs scale to **~4.8 h** - and reached **25.18% VAL / 26.67%
ASST / ppl 110.24**, against the headline's epoch 1 at **31.52% / 33.46% / 46.99** (`lens/runs/PROBE-D512.txt`,
headline at `TIED-D512-EXT6.txt:83`). That is **-6.34 pp** and 1.66x slower per epoch. It is **not** evidence
against our step rule: the probe's train accuracy is **21.12%** against the headline's **42.01%**, and its log
shows the step size is still the constant calibrated at d=256 (`opt_lr 0.0441256`, `--lr` unused by this rule)
while the headline ran SGD at lr 0.3. A step that is too small explains both the deficit and the slower
convergence, and this arm cannot separate that from a weaker rule - the next step is to calibrate the step size at
d=512, not to run six epochs. One epoch, one seed, and 12.19 pp below the headline: nothing here changes 37.37%.
Note: `docs/history/D512-PROBE-RESULT.md`.

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

Caveats, stated because they matter: **three seeds**, so no p-value and no invented interval - the three
values and their mean are what is published. The 20k rung's measured run-to-run spread is 1.03 pp
(`docs/history/REF-REPRO.md`), and the three seeds span 3.07 pp, i.e. seed-to-seed variation at this rung is
three times the measurement noise and is what limits precision here. The headline is unchanged - best held-out accuracy is still **37.37%**
(d=512, full corpus, 6 epochs). Checkpoints verified with `verify_wall_ab_ckpt.py`: the seed-1 arm is 1/1
clean, the parameter count off the stored arrays is 3,811,534 in both seeds, and `best.npz` equals the epoch
the log named. Write-ups: `docs/history/DATASCALE-RESULT.md` (seed 0) and
`docs/history/DATASCALE-SEED1-RESULT.md` (seed 1).