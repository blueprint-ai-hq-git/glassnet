# GlassNet - morning picture (2026-10-20 08:55)

Nothing ran overnight and nothing broke. Machine was left up, but all jobs were stopped and every cron paused,
so the night was free: newest checkpoint is still cl_d384_clean_12_ep12 from 19:19, no file in the project was
touched after 22:30, GPU idle at 123 MiB / 36 C, 76 GB free, 4,349 checkpoints kept.

## SCOREBOARD (every number measured, nothing estimated)

ENGINE (from scratch: own tensors, forward, backward, loss, optimiser - no torch)
  gradient checks PASS, worst absolute error ~1.3e-09 across sizes; XOR learns to 100%.

DATA
  294,225,090-token dialogue corpus with control tokens; val certified DISJOINT from train by exact bytes
  (VERDICT CLEAN, 0 overlapping windows) after a fixed-point repair (pass 2 left 4, pass 3 left 0).

MODELS TRAINED (honest masked accuracy on the clean val, n=44,418)
  d=256   6 epochs  27.72% final
  d=256  12 epochs  28.30% final, PEAK 28.52% at epoch 9   <-- BEST SETTING FOUND
  d=256  24 epochs  27.35% final, peak 28.48% at epoch 9
  d=384  12 epochs  28.32% final, peak 28.42% at epoch 11  (tied, not better)
  d=512  48 epochs  23.07% final (train 36.03%, val ppl 447: overfitting, not learning)
  standalone eval of d=256/24ep: 27.24% masked (n=45,403), eval exit 0

TWO SETTLED CONCLUSIONS
  1. Training longer COSTS accuracy: both long runs peak at epoch 9 and decay after. ~9-12 epochs is right.
  2. Capacity is NOT the limit: doubling to d=512 made the honest val WORSE by 4 points.

RULE-LEARNING: still none. 450 trials (methods x sizes 32..512 x lr 0.01..0.4 x epochs 100..800 x RL samples
2/8/32) ALL scored 0% extrapolation. Best trials: train 83.75% with held-out 5-15%. RL memorises harder than
SFT (train 51.2%, held-out 10%, extrapolation 0%).

THE ONE LEVER FOUND (structural, measured by lens/scripts/slot_binding_probe.py on addition)
  presence 65% / 0% / 0%   |  slot binding 65% / 0% / 0%  |  slot+interactions 100% / 0% / 0%
  digit-decomposed 100% / 0% / 0%  |  numeric+MLP 95% / 65% / 0%
  numeric + REGRESSION (answer as a quantity) -> MAE 0.008 / 0.010 / 0.041, EXACT 100% on unseen 10..19
  Binding alone does not fix it. Interactions alone do not fix it. Quantities in AND quantities out does.

TALK: word salad, verbatim: "what is 2 + 2 ?" -> "* * sample input / * * * 5 = 5 % 4 !";
"tell me about the sun ." -> "calculate this process can also use this simple step by understanding time
complexity ...". It has absorbed technical vocabulary and none of the rule.

READABILITY GATE: last measured FAIL (2,334 false fires, 17.2% honest-fire rate, 14.1 units/window).

## NEXT, in order
1. CODE: the numeric/magnitude path - magnitude concepts beside the token-presence concepts, plus a numeric
   output head beside the token classifier. Highest-value change in the project; the probe says it is the lever.
2. RUN: readability gate on the newest ship (queued).
3. Then, and only then, re-run the settings search looking for a non-zero extrapolation.

## BLOCKER
The autopilot cron's last run died with HTTP 401 (API key expired). The script-only watchdog and digest need no
key and ran fine. One autopilot run will confirm whether the key works again.
