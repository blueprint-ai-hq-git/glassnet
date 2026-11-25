# 2026-11-25 - the forgetting test was VOID; an instrument probe measured what mastery actually costs

## What happened

exp_forgetting.py compared four learning rules on "learn addition, then learn subtraction, measure what survived". It ran
clean and printed this (lens/runs/FORGETTING.json, first version):

    plain backprop           A before 23.0%  A after 0.5%   FORGET +22.5 pts   B 94.0%
    + replay                 A before 23.0%  A after 53.2%  FORGET -30.2 pts   B 79.2%
    + replay + consolidation  A before 22.8%  A after 28.7% FORGET  -6.0 pts   B 43.5%
    frozen core, readout only A before 22.8%  A after 0.5%  FORGET +22.2 pts   B 13.2%

None of those "FORGET" numbers means anything, because task A was only **23.0% learned** when task B began. A skill that was
never acquired cannot be forgotten. Worse, the retention RATIO then produced nonsense: the replay arm "retained 231.5% of A"
because replay kept teaching A during B - an improvement dressed up as retention.

**Verdict: VOID.** No arm credited, numbers kept on disk, nothing deleted.

## The probe that fixed the question before asking it again

exp_mastery_probe.py measured how long task A actually takes, reporting train and held-out together so under-fitting is
never confused with poor transfer:

    steps   4000  train  23.3%  held  19.5%    8s
    steps   8000  train  87.7%  held  83.2%   16s
    steps  16000  train  95.9%  held  92.2%   33s
    steps  24000  train  99.3%  held  94.8%   50s
    steps  32000  train  99.5%  held  95.0%   66s

MASTERY sits at 24,000 steps (held 94.8%). That is six times the budget the void run used, which is exactly why the void run
had nothing to measure.

## What changed in the experiment

  * STEPS_A = STEPS_B = 24,000, taken from the probe rather than guessed
  * a MASTERY GATE: if task A has not reached 90% held-out before task B starts, the script prints
    VOID_INSUFFICIENT_MASTERY and refuses to print any forgetting number at all
  * consolidation penalty 0.1 -> 0.05 (the 0.1 run crushed task B to 43.5%, which would make "retention" look good for the
    wrong reason - a model that cannot learn the new task retains the old one trivially)

The lesson is the instrument-first rule again, and it is the thirteenth time: the void here was not in the numbers, it was in
the PREREQUISITE. The test asked "did it forget?" before asking "did it ever know?".
