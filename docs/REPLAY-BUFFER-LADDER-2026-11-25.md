# 2026-11-25 - the replay buffer ladder: 1,024 examples clear the bar, and the whole curve

Sources: lens/runs/REPLAY-BUFFER.out (part 1: none/128/512/4096, crashed at the save step after printing - numbers valid,
JSON missing), lens/runs/REPLAY-BUFFER2.out and REPLAY-BUFFER2.json (part 2: 1024/2048, clean). Two seeds each, seeds 7 and 8,
24,000 steps per task, held-out operand pairs disjoint from training, exact-match scoring.

    buffer   retention mean   worst seed   forgetting   task B
    none            0.8%          0.5%      92.9 pts     98.4%
    128            23.5%         22.4%      71.9 pts     97.1%
    512            75.9%         74.7%      22.6 pts     96.2%
    1024           88.4%         87.6%      10.9 pts     96.4%
    2048           96.4%         96.3%       3.4 pts     97.0%
    4096          100.0%         97.9%       0.0 pts     96.2%

## What this establishes

  * Forgetting is a buffer-size problem with a measurable curve, and the curve is steep then saturating.
  * The pre-registered 80% bar is cleared at **1,024 replayed examples** (worst seed 87.6%), not at 512 (worst seed 74.7%).
    So the honest answer to "how much memory does continuous learning cost" is roughly A QUARTER OF THE TASK, not all of it
    and not a handful.
  * Replay is nearly free for the new skill: task B stays between 95.8% and 98.4% across every arm, so nothing is being
    bought by sacrificing the new task.

## Two corrections made along the way, both mine

 1. Part 1 CRASHED at the save step: `NameError: name 'os' is not defined` because the script used os.makedirs and never
    imported os. The numbers had already printed and are valid; the JSON was never written. Import added.
 2. The part-1 summary line printed "CONSTANT MEMORY HOLDS" while the pre-registered 80% bar had FAILED at 74.7%. It keyed
    that sentence off the falsifier (<50%) instead of the bar it had set. Fixed to print RETENTION_BELOW_BAR when the bar is
    missed, so a partially-met result can no longer be summarised as solved.

## Scope

One task pair (2-digit addition then subtraction), one architecture, two seeds, one learning rate. The curve is a
within-experiment comparison so it is internally valid; it does not establish the same buffer fractions for other task
pairs, larger models, or longer task sequences.
