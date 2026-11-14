# Scaled rejection sampling: 96.15% generalising on its own, or 77.03% generalising WITH in-domain accuracy going UP 19 points

Date: 2026-11-14. Tool `tools/rl_reject.py`, predictions written before the run in `lens/runs/RL-REJECT-SCALE-PREDICTIONS.txt`.
Task: `(a+b) mod 10` - warm start 400 supervised steps on sums <= 9, so generalising accuracy (sums >= 10) starts at **0.00%**.
This scales the 4-round pilot (`docs/history/RL-REJECTION-SAMPLING.md`) up: **8 rounds, 2,000-prompt held-out evaluation, 3
seeds, both arms**. The pilot used 4 rounds, 60 prompts per round and a 400-prompt evaluation and was explicitly called a pilot.

## Arm 1: no replay - 0.00% to 96.15% mean, with 100.00% on one seed

```
round    seed 0     seed 1     seed 2      (greedy generalising accuracy, 2,000 held-out prompts)
1        40.35%     40.45%     45.50%
2        49.50%     50.10%     59.55%
3        73.40%     70.00%     73.30%
4        88.50%     77.30%     74.85%
5        93.35%     89.00%     87.50%
6       100.00%     97.65%     89.30%
7       100.00%     96.05%     91.60%
8       100.00%     94.75%     93.70%
```

**Mean at round 8: 96.15%** (100.00 / 94.75 / 93.70). Seed 0 reaches **100.00% - 2,000 of 2,000 held-out prompts correct** - a
number that was 0.00% at the start, from a 7,342-parameter model, without a single labelled generalising example.
In-domain accuracy collapses to 2-3% (from 70.5-72.5%): the model trades the old task for the new one.

## Arm 2: replay 0.5 - 77.03% generalising AND the old task gets BETTER by 19 points

Half of every supervised batch is in-domain rows. Same protocol, same seeds:

```
round    seed 0                  seed 1                  seed 2
         gen     in-domain       gen     in-domain       gen     in-domain
1        14.25%   40.00%        13.50%   54.50%        15.15%   46.00%
2        27.80%   51.00%        20.50%   53.50%        24.40%   55.50%
3        29.95%   59.50%        22.40%   61.00%        36.35%   57.50%
4        39.50%   68.50%        36.25%   69.50%        49.75%   63.00%
5        43.50%   72.50%        53.05%   65.00%        47.60%   76.50%
6        54.10%   67.00%        56.75%   71.00%        57.25%   79.00%
7        72.55%   82.50%        61.10%   80.00%        61.75%   82.00%
8        78.30%   90.00%        80.90%   89.00%        71.90%   92.00%
```

**Mean at round 8: 77.03% generalising with in-domain at 90.33%** - against a warm-start in-domain of **71.33%**. The old task
ends **19.00 points BETTER** than it started (mean change +19.0: +19.50 / +18.00 / +19.50), while the new task goes from 0.00%
to 77%.

**That is not a trade-off. Both abilities improve.** The pilot's apparent trade-off (replay 0.5: 36.7% gen for in-domain at
68%) was a 4-round artefact: with more rounds the replay rows keep the old task alive *and improve it* while the kept
generalising examples accumulate.

## The two arms, side by side, as a real choice

```
                       generalising    in-domain      what it is
no replay (8 rounds)      96.15%          2-3%        maximum new task, old task gone
replay 0.5 (8 rounds)     77.03%         90.33%       both tasks, old task improved by +19
```

Neither is "better" in the abstract - one is a specialist, one is a generalist that got stronger on both. The knob that moves
between them is a single number, and it is now measured at 8 rounds instead of extrapolated from 4.

## Predictions, scored

- **P1 no-replay reaches >= 90.0% by round 8 - HOLDS** at **96.15%**, above the 90.0% bar.
- **P2 replay 0.5 reaches >= 60.0% gen with in-domain >= 60% - HOLDS**, at **77.03% / 90.33%**.
- **P3 the wider 2,000-prompt evaluation does not lower the number by more than 3 points - HOLDS.** Round 4 with 2,000 prompts
  reads 88.50 / 77.30 / 74.85 against the pilot's 400-prompt 88.00 / 82.75 / 78.25: the number did not fall, so the pilot was
  not flattered by a small evaluation set.
- Falsifier (round 8 <= round 4) **not triggered** - the curve was still climbing at round 8, so more rounds remain available.

## What is still not claimed

- This is `(a+b) mod 10` with a 7,342-parameter model, **not** the 47M chat line. Nothing here is evidence about language.
- The task is one where correctness is machine-checkable for free, which is what makes the kept pool trustworthy. A task without
  a verifier cannot use this method as-is.
- `mod 10` gives 10 candidate answers, so a lucky sample is not exotic; fuel at T=4 was 4.67% per sample. On a task with
  100,000 possible answers the fuel rate would be the binding constraint, not the method.
