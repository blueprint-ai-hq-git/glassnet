# The copy win is real: 5 seeds, paired, +1.98 points (t = 6.2) (2026-11-13)

`docs/COPY-KEEP-LEARNING.md` reported the per-batch freeze at **+2.1 points** on three seeds and I recorded it as **suggestive, not
decisive**, with a pre-registered falsifier: "if the gap falls below 1.0 point once five seeds are in, the +2.1 from three seeds was
noise". Seeds 3 and 4 were then run. The falsifier did not trigger - and the analysis I used the first time was the wrong one.

## Five seeds, both arms, identical otherwise
Mixed 50/50 stream, 4000 steps, from `early_s0_ep1000`, one nice-19 core, `keep_scoped` (copy path frozen only on language batches)
against `keep_none` (nothing frozen). Late held-out copy = mean of the last three evaluations.

```
seed   keep_scoped   keep_none   difference
0         76.2         73.9         +2.3
1         78.9         75.9         +3.0
2         76.8         75.7         +1.1
3         71.5         69.7         +1.8
4         76.5         74.8         +1.7
mean   75.97 +/- 2.72   74.02 +/- 2.52   +1.98
```

## The statistical correction matters more than the extra seeds
Read as two independent groups the result looks weak: gap **+1.98** with an unpaired SE of **1.66**, i.e. about **1.2 SE**. That is
what I quoted first, and it is the wrong test here. These arms are **seed-matched**: seed 3 is a low-scoring seed in *both* arms
(71.5 and 69.7), seed 1 a high one in both. That shared variation is noise common to the pair, and an unpaired test counts it
against the effect.

Paired on seed, the differences are **+2.3, +3.0, +1.1, +1.8, +1.7** - mean **+1.98**, sd **0.71**, paired SE **0.32**, **t = 6.2**,
and **5 of 5 seeds favour the freeze** (sign test **p = 0.031** one-sided).

**So the per-batch freeze result stands, and is stronger than the version I published: +1.98 +/- 0.71 points across five seeds,
positive on every seed.** What was wrong was not the effect, it was my test.

## Language cost, on the same five seeds (CORRECTED - the first version of this doc quoted three-seed numbers)
```
arm            language accuracy            taught-word copy
keep_scoped    15.56 +/- 0.20               84.26 +/- 0.67
keep_none      16.30 +/- 0.28               84.14 +/- 0.70
paired language difference: -0.73 points
```
So the trade is not free: **+1.98 points of held-out copy for -0.73 points of language**, positive on 5 of 5 seeds for copy.
Taught-word copy is unchanged (84.26 against 84.14), so the cost is specifically in the language objective, not in copy. That remains
the honest price of shielding the copy path from language gradients without giving it its own parameters.

## What is still not right about this line
- The freeze protects a capability by not letting it learn from language; the copy path only learns from copy batches, so it cannot
  improve from language either.
- The recorded successor - a copy path with its **own** parameters, trained on copy batches and invisible to language gradients -
  is still unbuilt. This result is the target that change has to beat, not a substitute for it.

## Reproduction
Arms: `lens/scripts/run_keep_arms.sh` (seed 0), `run_keep_seeds.sh` (seeds 1-2), `run_keep_seeds2.sh` (seeds 3-4, predictions in the
header). Logs: `logs/keep/` and `logs/keep2/`. Checkpoints `keep_scoped_s*` / `keep_none_s*` in `lens/runs/chat/`.
