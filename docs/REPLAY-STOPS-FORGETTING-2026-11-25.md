# 2026-11-25 - REPLAY STOPS FORGETTING: 93.5 points lost becomes 2.5, for 2.2 points of new-task cost

Source: lens/runs/FORGETTING.out (exit 0), lens/runs/FORGETTING.json. Mastery gate PASSED (task A at 94.8% held-out before
task B began), so these are real forgetting numbers, unlike the void run this morning.

    arm                        A before   A after    FORGET     task B
    plain backprop                94.8%      1.2%   +93.5 pts    98.0%
    + replay                      95.0%     92.5%    +2.5 pts    95.8%
    + replay + consolidation      95.0%     91.8%    +3.2 pts    92.0%
    frozen core, readout only     95.0%      0.2%   +94.8 pts    91.2%

Held-out operand pairs, disjoint from training, exact-match scoring, 24,000 steps per task.

## The result that matters

Catastrophic forgetting is real and near-total: plain backprop kept **1.2%** of a skill it had at **94.8%**. One task later
the old knowledge was gone. Framed honestly: 93.5 points of a mastered skill destroyed by teaching a new one.

Replaying old examples while learning the new task reduced that to **2.5 points**, and the new task still reached **95.8%**
against 98.0% with no replay. So continuous learning cost 2.2 points of new-task performance, not a retrain.

## Predictions vs measured

    P1 plain forgets >=40 pts                          MET   (93.5)
    P2 replay retains >=80% of A                       MET   (97.4%)
    P3 consolidation retains >=90% AND B within 5 pts   NOT MET
    P4 frozen core keeps A but loses >=10 pts of B      NOT MET

P3 failed on the second clause: consolidation retained 96.6% but its task B came in 6.0 points BELOW plain, worse than the
5-point allowance, and slightly worse than plain replay on retention too. So the extra mechanism bought nothing here and
cost new-task accuracy. Recorded as a failure rather than massaged.

P4 failed in an interesting way and the failure is informative: the frozen-core arm did NOT keep task A (0.2%). Freezing the
trunk while retraining the readout destroys the old task anyway, because the readout is where the answer digits are produced.
"Freeze the body" is not the brain-copy people assume it is.

## Scope, stated plainly

  * two-digit addition then two-digit subtraction, one architecture, single seed per arm. The effect is 91 points, far above
    the 0.012-0.013 run-to-run spread measured on this GPU, so it is not noise - but it is one task pair, one seed.
  * the replay arm replayed ALL 4,000 old examples. That is keeping the entire past, which does not scale. The next
    experiment asks the practical question: does a small buffer do the same job?
