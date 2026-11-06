# A NETWORK THAT KNOWS IT MUST LEARN (design, 2026-10-20)

Yahya: "every setting that you made is a normal setting, it does not force the model to learn ... set the
neural network so it already knows it has to learn not memorise, like it is alive while it is training so it
knows what he has to do, and that thing is still readable to 100% while training and after training, and we
have to get that number consistent."

He is right that a setting cannot do this. A learning rate is a number a human picks; it has no opinion about
memorising. What he is describing is a RULE THE MODEL ENFORCES ON ITSELF, and it has three parts.

## 1. THE MODEL CARRIES ITS OWN MEMORISATION METER (live, readable, part of the model)
The model holds its own probe slice of data (drawn once, never trained on, seed recorded). Every step it
measures itself on both the batch it just saw and the probe, and the difference is one number:

    gap = loss(train batch) - loss(own held-out probe)

That number is the memorisation meter. It is a named, printable value in the model's own state, so it is
readable while training and after. It is not a script watching from outside; the model owns it.

## 2. THE STEP IS REFUSED IF IT MEMORISES (the rule that forces learning)
This is the part that cannot be a setting: an update is applied ONLY if it does not worsen the gap. Concretely,
compute the candidate step, measure the gap after it, and REVERT the step if the gap grew. So the model
literally cannot walk in the memorising direction, however it is tuned. Two independent detectors:

  a. GAP GATE (what happened): did held-out loss get worse while training loss improved? Then that step was
     memorisation and it is refused.
  b. AGREEMENT GATE (why it happened): per-example gradients are compared. Learning produces a direction the
     examples AGREE on; memorising produces a different direction per example. Low agreement -> the step is
     scaled down or refused. Both numbers are printed every step.

## 3. IT IS ALIVE IN THE SENSE THAT IT DECIDES (and it shows its reasoning)
The model, not the trainer, answers the question "did I learn or did I memorise?" at every step, using its own
probe and its own gradients, and its answer changes what happens next. Its concept layer prints WHICH named
concepts fired for the batch it just refused, so a human can see the reason, not just the verdict.

## WHAT MUST BE MEASURED BEFORE THIS COUNTS AS BUILT
  1. the gap number, printed every step, and shown to be CONSISTENT: same seed -> same number within tolerance,
     and different seeds -> the same verdict (this is the "get that number consistent" requirement)
  2. steps refused by each gate, counted, on a run where memorisation is expected (a small corpus that CAN be
     memorised) - the gate must fire there, otherwise it is decoration
  3. the honest val on the disjoint val, ours versus plain SGD with the same seed: learning must show up as
     held-out accuracy, not as train accuracy
  4. 100% readable during AND after training: every gate value and every concept firing printable per step

## MEASURED 2026-10-20 (item 2 done, items 1/3/4 still open)
  Item 2 is measured. Files: lens/scripts/gap_gate_calibration.py, numbers in
  lens/runs/gap_gate_calibration.json, 3 corpus seeds x 3 training seeds.
    - a corpus that CAN be memorised (y independent of X): the shipped rule refused 0.000 of 300 steps, i.e.
      it was DECORATION. FILE: lens/runs/gap_gate_calibration.json, key part1_grid.
    - at ema_band beta=0.5 tol=0.0 it refuses 0.837 of steps there and 0.660 of a corpus that CAN be learned.
      The cost bar declared in part 4 (<= 0.20 on the learnable corpus) is therefore FAILED at the shipped
      step size. The outcome test still passes: held-out 1.0152+-0.1554 (gate on) against 1.2637+-0.2562
      (gate off) on the memorisable corpus, and 0.0024+-0.0005 against 0.0028+-0.0005 on the learnable one.
    - a full joint search over step rule x detector x beta x tol (parts 4 and 5) found the only admissible
      configuration to be the one with the step throttled, which the same turn measured as costing the model
      its ability to generalise (0.0% vs 100.0% exact on unseen pairs). No threshold tuning fixed this.
      Reason recorded there: a rule that compares the held-out loss with a smoothed version of ITSELF fires
      often on any noisy series. The measurement is the problem, not the threshold.
  Item 1 (the number consistent across seeds) is now checked: same seed reproduces the numbers exactly and 3
  seeds give the same verdict (part 3 of the calibration, PASS).
  Item 3 is NOT done on the real corpus - every verdict above is from synthetic corpora where the answer is
  known by construction, never from the chat corpus. Item 4 is done for the gates (both EMAs, the gap, the
  agreement and the verdict print every step, lens/unified.py) but not for the concept layer.

## DETECTORS 3 AND 4 (2026-10-20 14:5x) - THE GATE NOW FIRES, AND STILL NO CONFIGURATION PASSES
  Four more variants were added to GapGate.decide and tested online (the offline screen was measured unfaithful
  and is now a pre-filter only: replaying the shipped detector over a gate-off trace gives NOISE 0.943 / RULE
  0.739 against the known online 0.837 / 0.660, total deviation 0.185):
    'gap_level'  train loss persistently below the held-out loss   -> FAIL, and at tol 0.1/0.2 it does not merely
                 cost learning, it DESTROYS it (RULE held-out 1203.9x the gate-off value, 3.31 against 0.00275)
    'rate_ratio' the two losses' relative improvement rates        -> FAIL, fires on both corpora (NOISE 0.85-0.96,
                 RULE 0.45-0.85)
    'trend_band' the smoothed held-out loss against its own value 'lag' steps ago (lag=1 IS detector 1)
    'gap_trend'  the same lagged comparison on the gap
  FILES: lens/runs/GAP-GATE-DETECTOR3.txt + .json, lens/runs/GAP-GATE-DETECTOR4.txt + .json, predictions written
  first in lens/runs/GAP-GATE-DETECTOR{3,4}-PREDICTIONS.txt.
  WHAT IS NOW TRUE (item 2's literal requirement): the gate FIRES on the corpus that can be memorised. Best
  configuration, 'trend_band' beta=0.9 lag=50 tol=0.02: NOISE refused per seed [0.787, 0.833, 0.277] (mean 0.632),
  RULE refused per seed [0.037, 0.083, 0.187] (mean 0.102), RULE held-out 0.987x gate-off, NOISE held-out 0.990x
  gate-off. Against the shipped rule's 0.660 refusals on the learnable corpus, this costs one sixth as much.
  WHY IT IS STILL A FAIL: the NOISE bar is >= 0.50 on EVERY seed and seed 2 gives 0.277. The near-miss is
  recommended to the user but NOT applied; the shipped default is unchanged.
  THE MEASURED REASON NOTHING PASSES: once the learnable corpus reaches its noise floor (0.0025, the y-noise) its
  held-out loss is a noisy flat series, and the memorisable corpus's held-out loss in that same late phase only
  rises slowly (1.000 -> 1.418 over 300 steps). Any rule strong enough to fire on the second fires on the first in
  that tail, and the lag a trend rule needs delays it exactly where the rise is slowest.
  IDENTITY CHECK: 'trend_band' at lag=1 reproduces detector 1's part-1 row exactly (NOISE 0.601, RULE 0.311), so
  the lag is the only difference between the two.


## MEASURED 2026-10-21 20:3x - ITEM 2 IS NOW MET IN THE TRAINER THAT SHIPS (PLAN item d2b)
  The rule is wired into lens/scripts/train_chat.py (--gates gap, default OFF) and measured there on real chat
  text. Files: predictions lens/runs/TRAIN-GATES-PREDICTIONS.txt (P58a-j), verdict table
  lens/runs/GATE-REPORT.txt, raw output lens/runs/GATE-TEST.txt, per-arm
  lens/runs/chat/gate_*_{history.jsonl,gates.jsonl,gates_summary.json}.
    - ON A CORPUS THAT CAN BE MEMORISED (data/chat/prepared_gate_small, 1,204 train / 360 val windows of real
      chat text, built by lens/scripts/build_gate_corpus.py with prepared_clean's own split labels): the gate
      REFUSED 74 of 400 steps = 0.185. Same seed twice: 74 and 74. So it is not decoration - and this is the
      first time the rule has fired inside the trainer that ships rather than in the arithmetic harness.
    - ON A CORPUS THAT CAN BE LEARNED (prepared_clean, 20,000 windows): 2 of 237 = 0.008 refused. The cost bar
      (<= 0.20) PASSES here, which CONTRADICTS the synthetic calibration's 0.660 - the honest reading is that
      the synthetic learnable corpus was noisier than real text.
    - THE OUTCOME BAR FAILED (P58d): held-out VAL ppl 1059.04 with the gate vs 1035.72 without it, ratio 1.023
      against <= 1.000. The gate fired, the train loss ended 10% higher (5.2024 vs 4.7194), and the held-out
      loss ended 2.3% higher: refusing memorising steps did not turn into a held-out gain at 80 epochs on
      1,200 windows. Recorded as a FAIL, not smoothed.
    - THE PROBE IS SCORED WITH THE SAME LOSS THE TRAINING USES (assistant-masked when --assistant-only is on),
      which is the rule the 2026-10-20 teacher-mode bug taught; the probe is drawn once from the held-out
      split with a recorded seed and is checked disjoint from the training windows on window CONTENT (0 of 256
      on the small corpus).
  WHAT IS STILL NOT DONE, NAMED: (i) item 3 (ours vs plain SGD on the disjoint val, same seed) is not done for
  the gates - and the val-overlap check this turn built shows it must be done on a corpus whose val is
  actually held out (prepared_mix_ctrl_honest: 0 of 87,035 val windows overlap train;
  prepared_clean: 2,197 of 7,943 = 27.660% do - lens/runs/VAL-TRAIN-LEAK*.txt); (ii) the AGREEMENT GATE
  (section 2b) is not wired at all and nothing is claimed about it; (iii) the concept layer's per-step
  printout (item 4) is still only done for the gates.
  QUEUED, NOT IDLE: bash lens/scripts/run_gate_arm.sh runs the rule on the REAL recipe (the stored opt_sgd04_s2
  arm with --gates gap as its only change) on the clean corpus; predictions P59a-e are already on the record.
  (ii) IS NOW DONE - see UPDATE 7 below (the agreement gate is built, wired and measured on both corpora).


## UPDATE 7 - THE AGREEMENT GATE (SECTION 2b) IS BUILT, WIRED AND MEASURED: IT FIRES WHERE MEMORISATION IS
## POSSIBLE AND IS NEARLY SILENT WHERE LEARNING IS, ON EVERY SEED - BUT THE SEPARATION IS NOT STABLE
## (2026-10-21 21:5x, PLAN item d2d)

WHAT SECTION 2b ASKED FOR: "per-example gradients are compared. Learning produces a direction the examples
AGREE on; memorising produces a different direction per example. Low agreement -> the step is scaled down or
refused. Both numbers are printed every step."

WHAT WAS BUILT (lens/unified.py::AgreementGate + lens/scripts/train_chat.py --gates agreement / gap+agreement):
the batch is cut into G = --agree-groups equal sub-batches (default 8), each gets its own forward+backward
through the SAME loss the training uses (mask and mtp target included), and the number is
    A = || sum_g grad_g || / sum_g || grad_g ||
over every parameter concatenated: 1 when the sub-batches share one direction, 1/sqrt(G) = 0.3536 when they are
mutually independent, 0 when they cancel. It is printed every step and written to <out>_agreement.jsonl.
A refused step is SKIPPED (the weights do not move), which is the same outcome as the gap gate's revert and
needs no snapshot of the weights. The measurement runs BEFORE the update and does NOT change it (the
full-batch gradient is snapshotted and restored around the pass), so the control ('off') is digit-identical to
no gate at all. FOUR DETECTORS, all judged on the same series: 'off' (control), 'level' (A_train below a FIXED
threshold), 'ema_band' (A_train below its own EMA - the gap gate's detector-1 analogue), and 'vs_probe'
(A_train below A_probe, the model's coherence on the data it has NEVER seen). DEFAULT = 'vs_probe', the only
one with no human-chosen scale in it. STATED, NOT HIDDEN: scaling the step is NOT implemented - refusal is the
half of section 2b that ships, and the doc allows either.

MEASURED, THE GAP GATE'S OWN TWO ARMS SO THE TWO RULES ARE COMPARABLE (M = data/chat/prepared_gate_small,
80 epochs = 400 steps; L = prepared_clean, --max-windows 20000, 3 epochs = 237 steps; d=128, batch 256,
lr 0.4, CPU, 3 seeds each). Files: lens/runs/AGREEMENT-REPORT.txt (verdict table), AGREEMENT-TEST.txt (raw),
lens/runs/chat/agree_*_agreement.jsonl (per step):
    'vs_probe' (the DEFAULT)  M refused 0.383 / 0.190 / 0.158     L refused 0.013 / 0.110 / 0.076
    'ema_band' tol 0.0        M refused 0.983                     L refused 0.932   <- DECORATION, both
    'level' tol 0.5           M refused 1.000                     L refused 1.000   <- DECORATION, both
  THE ITEM'S OWN QUESTION, ANSWERED: on every seed the default detector clears the two-sided bar (>= 0.05 on
  the corpus that CAN be memorised, <= 0.20 on the one that CAN be learned) -> PASS. It is NOT decoration.
  THE PART THAT MATTERS: the SEPARATION is not stable. The corpus gap is +0.370 at seed 0 and +0.080 / +0.082
  at seeds 1 and 2, and the L arm's own rate moves 8.5x across seeds. A rule that clears a per-corpus
  threshold on three seeds is not the same claim as a rule that tells the two corpora apart, and only the
  first is what was measured.
  WHY THE MARGIN IS THIN, MEASURED: A_train starts at 0.3134 (M) and 0.3047 (L) - BELOW the independence floor
  0.3536 - so the sub-batch gradients are ANTI-correlated, not merely independent. A is therefore a small dip
  rather than a shared direction, and 'vs_probe' fires on a 0.005-0.035 difference between two numbers whose
  means are 0.2040 and 0.2006. THE NAME "AGREEMENT" IS MISLEADING FOR THIS QUANTITY AT THIS SCALE and that is
  recorded rather than renamed away.
  THE TWO DEGENERATE DETECTORS ARE INSTRUCTIVE: 'ema_band' refuses almost everything (0.983 / 0.932) because
  A falls almost monotonically, so "below its own EMA" is true nearly every step - the mirror image of the gap
  gate's detector 1, which barely fired. 'level' tol 0.5 refuses EVERYTHING because A never reaches 0.5; the
  arm's own record is the hazard: train_loss 9.1595, train_acc 0.00%, VAL ppl 10730.54 against 1035.72 with no
  gate. A fixed threshold below the observed range silently stops ALL learning.
  THE OUTCOME, MEASURED AND NOT PREDICTED (recorded as an observation, not a verdict, and the missing bar is my
  own error): the M arm with 38.3% of steps refused ended at train_acc 17.79% and VAL ppl 1172.17 against the
  ungated 30.83% and 1035.72 - the gate fired and did NOT protect the held-out loss. The queued real-recipe arm
  carries a two-sided outcome bar this time (P61c).
  MY OWN PREDICTIONS: of the 14 written before the run, 8 PASS, 5 FAIL as worded and 1 (P60h) FAILED AS A
  PREDICTION while the bar it described PASSED - I predicted the learnable-corpus bar would fail at 0.25-0.60
  and it measured 0.013. Every FAIL is on the record beside its corrected comparison.
  THE COST, DECLARED: 4.53x the ungated per-epoch time (1.83 s/epoch against 0.40 s/epoch), against a <= 3.0x
  bar -> FAIL. THE NUMBER THAT BUYS IT BACK IS ALREADY MEASURED: the sum of the sub-batch gradients divided by
  G differs from the single-pass full-batch gradient by 2.4e-07 RELATIVE (all 11 arms), so a version that uses
  that sum AS the batch gradient needs no extra pass at all - to be built after the same equivalence is
  re-measured under --assistant-only, where the loss is a weighted mean and the split is not exactly equal.
  NOT DONE, NAMED: (i) the agreement gate on the real recipe - QUEUED, not idle (lens/scripts/run_agree_arm.sh,
  predictions P61a-e in lens/runs/AGREEMENT-GATE-PREDICTIONS-APPENDIX.txt); (ii) scaling instead of refusing;
  (iii) the free version above. NO ACCURACY CLAIM OF ANY KIND: these are rule-firing numbers on small arms, not
  a statement about how well the chat model talks.

## HOW IT RELATES TO THE REST
  - the rule-versus-lookup test (train 0..9, score 10..19) is the arbiter: a lookup table scores 0%, a rule
    scores 100%. Already measured: one-hot 0%, numeric in + numeric out 100% exact.
  - the numeric path is what makes the arithmetic case learnable at all; the gates are what stop the model
    from taking the memorising shortcut while it trains.
  - EASY MODE: the measured best settings are baked in and the gates are ON by default, so a normal person
    gets a model that only learns. HARD MODE: every knob and every gate threshold exposed, for people who want
    to set it themselves.
