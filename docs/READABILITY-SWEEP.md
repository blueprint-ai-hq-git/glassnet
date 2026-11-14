# Readability sweep: the readable image model measures 1.4-2.2 points behind plain here - and why that is NOT yet a smaller gap

Date: 2026-11-14. Runner: `lens/scripts/run_readability_gap.sh`, 8 arms planned (4 configs x 2 seeds), 24 epochs,
`--extra none`. Parsed from each arm's own json (`results.readable.history[-1].test_acc`,
`results.plain.history[-1].test_acc`, `audit.n_pass / audit.n_concepts`).

## What ran

```
arm                     n   readable    plain     gap     concept honesty
A shipped  d64 patch7   2   84.36/84.27  86.42/85.81  -2.06 / -1.54   100.0% / 92.3%
B wide     d96 patch7   2   84.20/84.17  86.43/86.08  -2.23 / -1.91   100.0% / 100.0%
D softlam  lam 0.15     2   84.79/84.45  86.42/85.81  -1.63 / -1.36    84.6% / 92.3%
C fine     patch 5      -   CRASHED, no result (see below)
```

Every arm prints READABLE and PLAIN at **matched parameter counts inside the same run** (A: readable 24,458 vs plain
24,467 - 0.04% apart), so the gap is not an artefact of comparing separate runs.

## The honest reading of those numbers

**Do not report this as "the gap shrank from 4.96 to 1.8".** The published figure the project carries is readable
**81.73 +/- 0.54** against plain **86.69 +/- 0.59** (gap 4.96) from `docs/history/PHASE2-IMAGE-GATE.md`, and it describes a
**different pipeline**: that number is post **rule-pruning** and 3 seeds, while this sweep is unpruned, `--extra none`, and 2
seeds. Two different measurements of two different configurations are not a before-and-after.

What this sweep does establish, on its own terms:
- at matched parameters and 24 epochs with no extra data, the readable architecture is **1.4 to 2.2 points** behind the plain
  MLP, with concept honesty between 84.6% and 100%;
- **softer honesty pressure buys accuracy and costs honesty, as predicted**: lam-concept 0.15 (D) gives the smallest gap
  (-1.36 / -1.63) and the lowest honesty (84.6% on seed 0, below the 85% bar), while the shipped 0.3 (A) holds 92.3-100%
  honesty for a slightly larger gap;
- **wider concepts (B, d96) do not close the gap**: -2.23 / -1.91 against A's -2.06 / -1.54. P1 predicted the opposite
  ordering (finer patches should beat width); the finer-patch arm is the one that crashed, so that prediction is **untested**,
  not refuted.

## Two flaws in my own harness, both recorded

1. **The `patch 5` arm (C) crashed silently.** The log shows its banner (`READABLE: 22,617 params patches 25 x 64 -> 13
   concepts -> 10`, and a matched PLAIN at 22,270 params) and then no result at all. My runner pipes each arm through
   `grep -E "READABLE|PLAIN|TEST|honest"`, so the traceback was **filtered out of the log** and the failure left no evidence
   except a missing json. A harness that hides its own error is worse than one that fails loudly - the arm must capture the
   full output and the exit code.
2. **The summary parser crashed too** (`ValueError: too many values to unpack`), because I widened each row to three values
   and left `gaps = [a - b for a, b in rows]` expecting two. The numbers above were recovered by parsing the json directly.

## What the next run must do

- capture full per-arm output plus exit codes, so a crashed config is visible as an error instead of an absence;
- run the **pruned** configuration at 3 seeds, so the published 4.96 and this sweep are finally the same measurement;
- re-test `patch 5` once it runs, because P1 (finer patches beat width) is still open.
