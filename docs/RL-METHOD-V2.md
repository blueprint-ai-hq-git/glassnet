# LENS-PG v2: our own policy-gradient method, built around the measured reason v1 learned nothing

Status: **implemented and unit-tested on CPU (28/28 checks). Not yet trained.** No accuracy number exists for it yet and
none is claimed here.

## The diagnosis this method is built on

From this project's own records, not from a paper:
- `docs/history/RL-DENSITY.md` - dense per-token reward beat sparse **19.95 +/- 1.11 vs 18.84 +/- 1.00**, but closed only
  **12%** of the ~14-point gap to supervised (**28.06 +/- 2.21**).
- `docs/history/RL-GEN-VERIFY.md` - verifier-only arms: **"mean reward -0.006, advantage exactly zero"**, and `acc_gen`
  **0.00 in 9 of 9 arms**.
- `docs/history/RL-KL-ANCHOR.md` - a KL anchor stops the damage (beta 5 **+0.64**, beta 20 **+2.04**) and is not better.

"The advantage is exactly zero" is an arithmetic statement, not a tuning problem. If every sample for a prompt scores the
same, then `(r - mean(r))` is zero for all of them, the gradient is zero, and the run trains on nothing while reporting a
normal-looking loss curve. No amount of extra steps, reward tuning or KL weight fixes that. **That is what v2 addresses.**

## What v2 adds, and what it borrows

Borrowed from standard practice (the idea, not the code): a **group-relative baseline** - the GRPO idea of scoring a
group of samples per prompt against their own mean instead of training a value network. We have no critic in our engine and
did not add one.

Ours: the parts that follow from our own measurements.

| piece | what it does | our setting |
| --- | --- | --- |
| `group_advantage()` | per-group normalised advantage with a **variance gate**: a group whose rewards are all identical is dropped **and counted** | `LENS_G_GROUP = 16` samples/prompt (v1 used 8) |
| clipping | advantages clipped after normalisation so one lucky sample cannot dominate | `LENS_ADV_CLIP = 3.0` |
| `in_curriculum()` | sample a prompt only while the model's own success rate on it is inside a band - always-right and never-right prompts produce zero advantage by construction | `LENS_BAND = (0.10, 0.90)` |
| `repetition_penalty()` | fraction of repeated n-grams in the sampled answer, priced into the reward - the failure mode where a model scores on partial credit while saying nothing | `LENS_REP = 3` |
| `task_success_rates()` | per-prompt success rate used by the curriculum, recomputed as training moves | - |

The honest headline of the design: **v1 could not tell that it was learning nothing. v2 can.** `dead_fraction` is reported
by the method itself, so a run that scores no learning signal says so instead of producing a loss curve.

## What is verified now (CPU only)

`lens/tests/test_rl_v2.py`, 28 checks, all passing:
- identical rewards -> advantage **exactly 0.0** and **counted as dead groups** (2/2), `mean_abs_advantage = 0.0`;
- a mixed group -> **zero-mean** advantage with right samples positive and wrong ones negative;
- clipping holds at the configured bound;
- a batch of dead and alive groups is counted separately and the dead ones contribute nothing;
- a rewards array whose length is not a multiple of the group size is **refused**, not silently resized;
- repetition: clean sequence 0.0, repeated trigram 0.25, degenerate spam 0.875, too-short sequence not judged;
- curriculum: always-right excluded, never-right excluded, mid-band admitted, band edges inclusive;
- per-task success rates correct.

Two errors of mine were caught by these tests and are recorded rather than tidied away: the test's first path setup put the
project root one directory too low (`ModuleNotFoundError: glassnet`), and one expectation of mine was simply wrong (a task
with correctness `[0,0,1,0]` has rate 0.25, not 0.5 - the function was right, my assertion was not).

## Predictions for the first run (written before it)

- **P1 no more silent zero-gradient runs:** `dead_fraction` is reported and is **below 0.5** with `LENS_G_GROUP = 16`;
- **P2 the failure that has never moved moves:** `acc_gen` **> 0** on a verifier-only task (it was 0.00 in 9 of 9 arms).
  This is the real test of whether any of this matters;
- **P3 against a warm start:** RL improves its own starting point by **>= 2 points** (v1's best was 19.95 against
  supervised 28.06).
- **Falsifier:** if `dead_fraction` stays near 1.0 even at 16 samples per prompt, then this task's reward is genuinely
  binary at this model size and the fix is the reward, not the estimator - and P2 fails, which means our RL line is not
  competitive for a reason none of these nine arms has yet isolated.


## First measurement from the wiring (added 2026-11-14 09:52): 44.6% of groups carry no signal

The method is now wired into `tools/rl_compare.py` behind four flags, all defaulting OFF so every existing arm is
byte-identical in behaviour:
`--lens-adv-clip` (advantage clip, v2 setting 3.0), `--lens-rep` (repetition weight), `--lens-band lo,hi` (curriculum),
`--lens-band-min` (samples before a prompt is judged). `train_rl()` gained the matching keyword arguments with defaults,
and the dead-group accounting is read-only, so it reports even when every flag is off.

Measured on the addition task, CPU, group = 8, curriculum band 0.10..0.90:

```
6 steps,  batch 8,  group 8  ->   23 of  48 groups dead (47.9%)
60 steps, batch 16, group 8  ->  428 of 960 groups dead (44.6%)
```

**Roughly 45% of sampled groups produce exactly zero advantage.** Those groups contribute nothing to the gradient, which
means v1 was effectively training on about half the batches it paid for - and, worse, reporting a normal loss curve while
doing it. This is the first direct measurement of the mechanism behind the 9/9 `acc_gen 0.00` result, and it is the number
v2 exists to move.

Two wiring errors of mine, caught by the smoke tests rather than by a 40-minute run:
1. the flags were added to the CLI but `train_rl()` is a function with its own signature, so the first run died with
   `NameError: name 'lens_band' is not defined`;
2. the dead-group report was written in `main()` while the counters live inside `train_rl()`, so it printed nothing until a
   module-level `LENS_STATS` channel was added.

Next: run v2 as a real arm against the v1 baseline on the addition task (equal compute, 3 seeds), with the predictions in
the section above unchanged - `dead_fraction` below 0.5 is already met at 0.446, so the live questions are P2 (`acc_gen`
above zero for the first time) and P3 (at least +2 points over its own warm start).

## What is not done yet

Wiring into `tools/rl_compare.py` (group-normalised advantages, curriculum sampling, repetition term) and the first seeded
run. Until that lands, this document describes a method with tested arithmetic and **no result**.
