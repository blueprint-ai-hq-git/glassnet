# ONE TRAINING METHOD - status (2026-10-20)

the maintainer: "make one single training method for distillation and RL, there is no SFT ... one training method and
that is it ... in easy mode just a few lines of change, in hard mode all the settings."

## WHAT IS BUILT
lens/unified.py - ONE loop. The only thing that differs between modes is WHERE THE TARGET COMES FROM:
  mode "verifier"  -> reinforcement: the model answers, the verifier says right or wrong, it is reinforced on
                      what it got right. Give it a dataset of problems and it works them out itself.
  mode "teacher"   -> distillation: a teacher's answer replaces the label.
  mode "data"      -> ordinary training on a distilled dataset.
Same loop, same gates, same readable per-step output. No stages, no pretrain-then-SFT-then-RL pipeline.

## MEASURED, FIRST REAL RUN (smoke test: learn 2+2, then answer pairs it never saw)
  data mode     : 98 accepted, 202 refused by the gap gate | held-out 48.85 -> 10.80 | unseen 0%
  teacher mode  : identical numbers, because the test teacher returned exactly a+b, the same as the labels --
                  an invalid distillation test. A real test needs a teacher that differs from the labels.
  verifier mode : 123 accepted, 177 refused | cold-start partial credit used 72 times | unseen 0%

## MEASURED 2026-10-20 (later the same day) - THE REAL DISTILLATION TEST, AND THE METHOD'S CURRENT STATE
  VALID distillation test: lens/scripts/distillation_test.py, output kept in lens/runs/DISTILLATION-TEST.txt.
  Labels y = a+b, teacher t = a+b+3, 3 seeds, 3000 steps, scored on pairs the model never saw:

      data mode (target = labels)      : matches the labels 90.7% +- 13.2
      teacher mode, t = a+b+3          : matches the TEACHER 90.7% +- 13.2, the labels 0.0%
      teacher mode, t = a*b (control)  : matches the teacher 9.3% (must stay low; it is not representable)

  So the mode really does select the target source, and the non-representable teacher is really not learned.
  PREDICTIONS P12/P13 (both "PASS") were written before the run and both hold.
  A BUG WAS FOUND DOING THIS, in lens/unified.py: before the fix, teacher mode matched neither the teacher
  (6.7%) nor the labels (14.7%). The model was trained on the teacher while its probe, and therefore the gap
  gate, was scored against the LABELS. Fixed: the probe is scored against the same target the model trains on.

  THE LOOP NOW, AFTER THE FIXES (lens/runs/UNIFIED-RUN.txt, 6000 steps, 1 seed, honest held-out):
      data mode    : 100.0% exact on trained pairs, 100.0% exact on pairs NEVER seen (10..19)
      teacher mode : same numbers, but the smoke teacher equals the labels - use distillation_test.py instead
      verifier mode: 9.0% / 0.0%. STILL NOT CONVERGING, the one live flaw left in the method: a right/wrong
                     verifier over rounded answers gives too little exact signal to climb from cold
                     (cold-start partial credit used 1015 times). Next target once the gates are settled.
                     SOLVED 2026-10-21 - see UPDATE 3 at the bottom: the cause was not the cold start, it was
                     that the loop converged to a SLOPE 6.5% low, which rounding hid on the trained range.


## THREE REAL FLAWS FOUND BY RUNNING IT (and their status)
1. COLD-START DEADLOCK in verifier mode: at initialisation nothing is right, a right/wrong verifier gives no
   signal, and ZERO steps were accepted. FIXED: partial credit - keep the closest attempts and reinforce those.
2. THE GATES BLOCKED LEGITIMATE LEARNING: the first agreement rule refused at 0.25 and blocked 161 of 300
   honest learning steps. CHANGED: agreement now SCALES the step and only near-total disagreement is refused.
3. THE GAP GATE IS STILL TOO AGGRESSIVE AND NOT YET TRUSTWORTHY: 202 of 300 steps refused, and the model still
   scores 0% on unseen numbers. It compares held-out loss on a SINGLE noisy mini-batch, so ordinary
   fluctuation is read as memorisation. NOT FIXED. Needs smoothing (an exponential moving average over steps)
   before any run is believed. Until then this gate is decoration, and it must be tested on a corpus that CAN
   be memorised, where it has to fire, before it is trusted anywhere else.

## WHAT "DONE" REQUIRES
  - the gap number consistent: same seed -> same number, different seeds -> same verdict
  - the gates fire on a deliberately memorisable corpus (otherwise they are decoration)
  - unseen-pair accuracy above 0%, decided on held-out, never on train accuracy
  - easy mode: the measured best settings, gates ON by default, a few lines to use
  - hard mode: every knob and every gate threshold exposed

## UPDATE - THE SMOOTHED GATE (same day, after the fix)

The gap gate now compares an exponential moving average of held-out loss instead of one noisy mini-batch.
Result: it went from refusing 202 of 300 steps to refusing 0 of 300. So it is now TOO PERMISSIVE, the opposite
failure. Honest reading: a 5%-relative-worsening tolerance on an EMA is not calibrated, and the gate cannot be
trusted until it is tested on a corpus that CAN be memorised, where it MUST fire.

Also found: the smoke-test model itself is undertrained (9% exact even on the pairs it trained on, held-out
loss ~11 where it should approach 0), so this harness is too weak to judge ANY gate. Fix the harness before
drawing conclusions. Any claim about the gates made with this harness would be worthless.

STATUS OF THE THREE FLAWS: cold-start deadlock FIXED and verified. Agreement gate FIXED (scales, no longer
blocks). Gap gate: rewritten twice, still NOT trustworthy, needs calibration on a memorisable corpus.

## UPDATE 2 - THE GATE TESTED ON A MEMORISABLE CORPUS, AND WHY THE HARNESS WAS WEAK (2026-10-20 11:3x-12:0x)

FILES: lens/scripts/gap_gate_calibration.py (the test), lens/scripts/unified_harness_probe.py (the harness
diagnosis), lens/runs/gap_gate_calibration.json (raw), lens/runs/UNIFIED-RUN.txt (this turn's output).

THE TEST, which is the one this document demanded. Two corpora with the answer known in advance: NOISE
(y independent of X, so memorisation is the only possible outcome) and RULE (y = X @ w_true + 0.05 noise).
Verdict rule fixed before the run: PASS iff every seed refuses >= 50% on NOISE and <= 20% on RULE.

  1. GATE VERDICT: the shipped rule (ema_band, beta=0.9, tol=5%) refused 0.000 of 300 NOISE steps -> FAIL,
     as predicted before the run. The best of 24 configurations under the step rule that existed at that
     moment was beta=0.5, tol=0.0, NOISE 0.614 / RULE 0.163 -> PASS. CORRECTION LATER THE SAME TURN: when
     the step throttle was removed (the fix below) the same gate refused 0.660 of the learnable corpus and
     the verdict became FAIL, and a joint search over step rule and detector (parts 4-5 of the calibration)
     found no admissible configuration at the shipped step size. The PASS above belongs to a step size that
     no longer exists. Current state: FAIL against the pre-declared cost bar, outcome test passing.
  2. IT IS NOT JUST COUNTING REFUSALS. Same seeds, gate ON versus gate OFF, 3 corpus seeds x 3 training
     seeds (final numbers at the shipped step rule, agreement_mode='one'): on NOISE it ends at held-out
     1.0152+-0.1554 against 1.2637+-0.2562 gate-off, so refusing
     actually PREVENTS the damage; on RULE 0.0025+-0.0005 against 0.0028+-0.0005, so it does not block
     learning. A gate that changed no outcome would have been decoration however many steps it refused.
  3. PREDICTION THAT WAS WRONG: I expected the 'divergence' rule (held-out worse WHILE train improves) to
     win. It scores NOISE 0.433 against the winner's 0.614, because on a fully memorisable corpus the
     training loss eventually rises too. It is kept as variant='divergence' with that number recorded.

THE HARNESS WAS WEAK, AND NOT BECAUSE OF THE DATA. The smoke-test model reached only 9% exact on pairs it
trained on. Plain gradient descent on the very same corpus, same seeds, reaches 100.0% exact on trained pairs
and 100.0% on unseen 10..19 (closed-form least squares residual 2.7e-29). The corpus was always solvable; the
fault was inside the loop: every step was multiplied by max(agreement, 0.05) and the agreement number is about
0.3 on this data, so every honest step was throttled roughly 3x.

      shipped  (throttle on,  gap gate on) : 30.0% exact on trained pairs,  0.0% on unseen pairs
      FIXED    (throttle off, gap gate on) : 88.7% exact on trained pairs, 66.7% on unseen pairs
      control  (throttle off, gap gate off): 100.0% / 100.0%

So if a human reads this document and asks "why was it undertrained", the answer is: it was not the harness,
it was a throttle on every step of the method. The agreement number still exists, is still printed, and still
refuses near-total disagreement, but it no longer scales the step (agreement_mode='one').

REMAINING REQUIREMENT FOR "DONE" HERE: the gate verdicts above are on synthetic corpora where the answer is
known. The gate has still never been run on the real chat corpus, and that is the next honest test.


## UPDATE 3 - VERIFIER MODE FIXED, AND THE REASON IT WAS NEVER THE COLD START (2026-10-21 17:5x)

FILES: lens/scripts/unified_verifier_test.py, lens/scripts/run_unified_verifier.sh,
lens/runs/UNIFIED-VERIFIER-PREDICTIONS.txt (P51a-i + ADDENDUM 1/2/3), lens/runs/UNIFIED-VERIFIER.txt,
lens/runs/_tmp/PROBE-ANNEAL-DEBUG.txt. Predictions were written BEFORE any code changed.

THE SHIPPED RULE, MEASURED (P51a/b/c PASS). It reproduces the stored state exactly (9.0% trained, 0.0%
unseen, 1015 cold-start steps), and its selection is biased LOW: the quarter it keeps has a mean target below
the batch mean on 0.901 of the 1015 cold-start steps (5.167 against 9.250 at step 0). The endpoint is a
NEAR-CONSTANT predictor - w = [1.71, 1.75, 4.16], predictions spanning 8.31..9.87 - which is exactly what a
constant scores on this corpus by rounding.

THE REAL FAULT, AND IT IS NOT THE COLD START. A verifier asked for a DIRECTION (which of pred, pred+s,
pred-s is closest to right) gets 95%/91% on the trained pairs on two seeds and still 0.0% on pairs it never
saw. The weights say why: w = [18.66, 18.80, 0.28] against the TRUE [20, 20, 0] - a SLOPE 6.5% LOW. On the
trained range (y = 0..18) 6.5% hides inside the +/-0.5 rounding tolerance; on the unseen range (y = 20..38)
the same 6.5% is 1.3-2.5 units and every answer is wrong. ROUNDING ON THE TRAINING RANGE HID A WRONG SLOPE.

MY OWN FIRST TWO DIAGNOSES WERE WRONG AND STAND WRONG. I predicted an absorbing state ("once |err| <= s the
picked candidate is pred itself and the example stops contributing"); the hand-instrumented run refutes it -
at s = 0.0625 the picks are still spread over all three candidates and the gradient norm is still 0.02-0.09 at
step 2800. Annealing s made the run WORSE, not better, because the update MAGNITUDE is proportional to s.

THE RULE THAT WORKS: probe step fixed SMALL (1/16, so the zero-gradient band |err| < s is negligible) and the
update NORMALISED by 1/s so the step does not shrink with the probe. Numbers, all 3 seeds:
    3000 steps : slope error 23.2% / 0.45% / 8.0%  ->  unseen  0% / 100% / 0%     (P53a/b/c FAIL as worded)
    12000 steps: slope error 0.47% / 0.45% / 0.44% ->  unseen 100% / 100% / 100%   (P54a/b/c PASS)
    the SHIPPED rule at the same 12000 steps: slope error 74-85%, 9-11% trained, 0% unseen - WORSE, so the
    gain is the rule and not the extra steps (P54c PASS).
  The unseen score tracks the slope error seed by seed with no exception. Controls fire: the reversed
  verifier stays at 1.0% trained / 0.0% unseen in every variant; the data-mode upper bound is 88.7%/66.7%.

STATUS OF THE FIVE FLAWS IN THIS DOCUMENT: cold-start deadlock FIXED. Agreement gate FIXED. Gap gate NOT
TRUSTWORTHY (detectors 3 and 4 both failed their own calibration; the near-miss trend_band is recommended,
not applied). Verifier mode: the CORRECTED RULE WORKS, but it is NOT the default and nothing calls it -
verifier_mode='probe3' + probe_step + probe_normalise are named parameters, and the shipped default
('hard_quarter') is deliberately unchanged so every stored number still reproduces (P51i/P52e/P53e PASS,
`python3 -m lens.unified` byte-identical to lens/runs/UNIFIED-RUN.txt). Next: make it the default FOR VERIFIER
MODE and re-record the smoke test.
NO ACCURACY CLAIM OF ANY KIND: this is the arithmetic harness the method is developed on, not the chat model.


## UPDATE 4 - THE CORRECTED RULE IS THE DEFAULT NOW, AND IT NEEDS A VERIFIER THAT CAN RANK (2026-10-21 18:3x)

FILES: lens/runs/UNIFIED-VERIFIER-PREDICTIONS-ADDENDUM4.txt (P55a-P55k) + ADDENDUM5 (P56a-P56d), both written
BEFORE the code changed; lens/runs/UNIFIED-VERIFIER.txt (18:38 block); lens/runs/UNIFIED-RUN.txt (the smoke
test, re-recorded); the pre-change output KEPT at lens/runs/UNIFIED-RUN-pre-probe3.txt.

WHAT CHANGED: `train_unified` now defaults to verifier_mode='probe3', probe_step=1/16, probe_normalise=True.
The old cold-start rule is kept as a named variant ('hard_quarter') and the test names it explicitly, so every
stored number describing it still reproduces (P55f PASS: 9.0% trained / 0.0% unseen / 1015 cold-start steps).
The smoke test's verifier arm runs at 12000 steps, the budget the item itself said was too short.

MEASURED, 3 seeds, 12000 steps, same harness:
    default WITH a rank oracle        slope error 0.472 / 0.448 / 0.436%   unseen 100.0%   (P55h PASS, and
                                      it reproduces arm G's stored numbers to the last digit)
    default with ONLY right/wrong     trained 97 / 98 / 99%               unseen   0.0%   (P55g FAIL)
    the OLD default (hard_quarter)    trained  9 / 10 / 11%               unseen   0.0%
So the default is measurably better on what it trained on and STILL cannot extrapolate without a rank-capable
verifier. That is written down as a FAIL of the item's own claim, not as a partial win.

THE CAUSE, AND IT IS THE USEFUL RESULT (PART 6, probe step swept, one seed):
    with a right/wrong verifier the unseen score is 0.0% at s = 1, 1/4, 1/16, 1/64, and the SLOPE ERROR is
    6.797% at every one of them - identical to five decimal places - against the rank oracle's 0.472% at the
    same steps. A right/wrong verifier declares "right" inside its own 0.5 tolerance, so the band where an
    example stops giving a direction is 0.5 wide no matter what the probe step is. SHRINKING THE PROBE CANNOT
    FIX A BINARY VERIFIER; the fix is an oracle that can rank two candidates. (My ADDENDUM 4 guess that the
    fallback would behave like the oracle "outside the rounding band" was WRONG - the band is the whole
    trained range - and it stands wrong in the predictions file.)
A bug was found by running the new default before any number was taken: the fallback branch had never been
executed and crashed on a shape mismatch (BUGS.md). Making a rule the default is what runs the code behind it.

STATUS OF THE FIVE FLAWS: cold-start deadlock FIXED. Agreement gate FIXED. Gap gate NOT TRUSTWORTHY
(detectors 3 and 4 failed their own calibration; the near-miss trend_band is recommended, not applied).
Verifier mode: the corrected rule IS the default and CONVERGES with a rank-capable verifier; with only a
right/wrong verifier it still cannot extrapolate, and the next item is a rank-capable verifier ('verifier3').
NO ACCURACY CLAIM OF ANY KIND: arithmetic harness, not the chat model, not the real corpus.


## UPDATE 5 - A DIRECTION OUT OF A RIGHT/WRONG VERDICT: THE LADDER (2026-10-21 19:3x, PLAN item verifier3)

FILES: predictions lens/runs/UNIFIED-LADDER-PREDICTIONS.txt (P57a-P57l, written BEFORE any code changed);
measurements lens/runs/UNIFIED-LADDER.txt (PART 7a/7b/7c + PART 8); the smoke test re-recorded as
lens/runs/UNIFIED-RUN.txt (md5 037addb803c0484a26e38ced3f68f898, 46 lines) with the PRE-MOVE output KEPT at
lens/runs/UNIFIED-RUN-pre-ladder.txt (md5 035ac81b7ad1ce3b4d004de1489eaa38).

THE PROBLEM (UPDATE 4): a right/wrong verifier stops giving a direction inside its own 0.5 tolerance, so the
fallback parks every example on the edge of that band: slope error 6.797% at every probe step, 97/98/99%
exact on trained pairs, 0.0% unseen.

THE FIX (new, `binary_sign_ladder` in lens/unified.py + the `binary_probe='ladder'` path): the verdict is a
MEMBERSHIP test. A candidate pred + d is right iff |pred - y| is in the window (d - tol, d + tol) on that
side, so for one rung d >= tol the two sides cannot both be right, and EXACTLY ONE of them being right names
the SIDE of the target. Scan rungs d = tol, tol + 1.5*tol, ... (windows overlap, so no residual inside the
range falls between two rungs) and take the first rung that resolves. Every example keeps its own direction,
so an example the model already answers correctly is still given one - it is not parked, and parking is what
made the slope error. The rule's dead zone is a single point (the residual exactly 0) instead of the whole
0.5 band, and the cost is 2 verifier questions per rung.

MEASURED, 3 seeds, 12000 steps, same harness, same gate defaults as UPDATE 4 (lens/runs/UNIFIED-LADDER.txt):
    ladder, right/wrong verifier    trained 100.0 / 100.0 / 100.0%   unseen 100.0 / 100.0 / 100.0%
                                    slope error 0.189 / 0.230 / 0.192%          <- the fix, and it beats the
                                    rank oracle's 0.472 / 0.448 / 0.436% on the same harness
    old fallback ('nearest')        trained  97.0 /  98.0 /  99.0%   unseen   0.0 /   0.0 /   0.0%
                                    slope error 6.797 / 6.489 / 5.948%
    rank oracle (unchanged)         slope error 0.472 / 0.448 / 0.436%, unseen 100.0% (identical at 5 dp)
    CONTROL flipped direction       slope error 100.12%, unseen 0.0%     (the ladder can fail)
    CONTROL always-right verifier   slope error 100.00%, trained 1.0%    (no information in, no direction out)
    COST                            mean verifier calls per step 4.8 / 3.86 / 4.39 over the run, 2.0 in the
                                    last 1000 steps, against the old fallback's 3; unresolved examples 0.0%
DIRECT CHECK of the helper on 200,000 synthetic residuals: WRONG SIGN on 0 of the 199,206 it resolved; the
794 it did not resolve all have |r| >= 40.75, i.e. beyond the declared top rung (ladder_max 40.0 + tol) - the
ladder is simply not asked about them. Inside its range: 199,049 residuals, 0 unresolved, 0 wrong signs.

DEFAULT MOVED (P57l, because P57b/P57c PASSED): `train_unified` now defaults to binary_probe='ladder';
'nearest' is kept as a named variant so every stored number that describes it still reproduces. The smoke
test moved with it, ON PURPOSE, and only in the verifier arm: trained 100.0% / unseen 100.0% (was 97.0% /
0.0%), held-out 72.61969 -> 0.00013 (was -> 0.05199), accepted 2340 / refused-for-gap 9660 / refused-for-
disagreement 0 (was 2585 / 6280 / 3135). The data arm, the teacher arm and STAGE 2 are digit-for-digit
identical before and after the move (measured line by line, PART 8).

THREE OF MY OWN BARS WERE WORDED WRONG AND FAILED AS WORDED; the FAILs stand in UNIFIED-LADDER.txt and the
corrected comparisons are recorded beside them (PART 8), not in place of them: P57e applied PART 6's
ONE-SEED number to three seeds; P57i compared full-precision floats against five-decimal printed numbers
with a 1e-9 tolerance; P57a swept residuals past the ladder's top rung and counted, by design, the ones the
ladder never asks about. Corrected, all three PASS. A fourth error of mine was caught before it could reach
the record: the first version of the cost log divided the verifier calls by the batch size, understating the
cost 16x (a units error, found by the smoke test).

STATUS OF THE FIVE FLAWS: cold-start deadlock FIXED. Agreement gate FIXED. Gap gate NOT TRUSTWORTHY
(detectors 3 and 4 failed their own calibration; the near-miss trend_band is recommended, not applied).
Verifier mode: FIXED AND THE DEFAULT - it converges with a rank oracle AND with a plain right/wrong verifier,
on this harness. The shipped TRAINER still does not use the method (PLAN item (d2b)).
NO ACCURACY CLAIM OF ANY KIND: arithmetic harness, not the chat model, not the real corpus.

## UPDATE 6 - THE METHOD IS IN THE SHIPPED TRAINER: THE GAP GATE FIRES THERE, AND THE OUTCOME BAR FAILED
## (2026-10-21 20:3x, PLAN item d2b)
The last line of UPDATE 5 said the shipped trainer still does not use the method. It does now, in part: the
GAP GATE (section 2a of LEARN-NOT-MEMORISE.md) runs per STEP inside lens/scripts/train_chat.py behind
--gates gap (default OFF), with the model's own held-out probe drawn once with a recorded seed and scored
with the SAME loss the training uses, a refused step REVERTED, and one JSONL row per step. The rule wired in
is the METHOD's own default (train_unified builds GapGate(beta=0.5, tol=0.0, variant='ema_band')), not a new
threshold. Predictions first: lens/runs/TRAIN-GATES-PREDICTIONS.txt (P58a-j).
MEASURED on real chat text (verdict table lens/runs/GATE-REPORT.txt, raw lens/runs/GATE-TEST.txt):
  * the gate FIRES where memorisation is possible: 74 of 400 steps refused = 0.185 on a corpus small enough to
    fit (data/chat/prepared_gate_small), 74 and 74 on two runs of the same seed. It is not decoration.
  * it is nearly SILENT where learning is possible: 2 of 237 = 0.008 on prepared_clean with 20,000 windows -
    inside the <= 0.20 cost bar, and my prediction said that bar would FAIL (0.660 from the synthetic
    calibration). The synthetic learnable corpus was noisier than real text; the prediction stands wrong.
  * **THE OUTCOME BAR FAILED**: held-out VAL ppl 1059.04 gated vs 1035.72 ungated (ratio 1.023, bar <= 1.000).
    Refusing 18.5% of the steps cost 10% of the train loss and did NOT buy held-out loss.
  * the default path is unchanged (P58a: digit-identical epoch lines against the pre-change copy), and
    relgrad + the gate runs (P58j).
NOT DONE, NAMED: the AGREEMENT GATE (2b) is not wired; the concept layer's per-step printout is still only
for the gates; and the rule has not been measured at the real recipe yet - that arm is QUEUED
(lens/scripts/run_gate_arm.sh, predictions P59a-e) rather than run inline.
A CORPUS DEFECT THE CHECK TURNED UP, and it belongs here because every val number in this file depends on it:
prepared_clean's val overlaps its train at the window level - 2,197 of 7,943 val windows = 27.660% have their
exact content (32 tokens + target) in the full train window set - while data/chat/prepared_mix_ctrl_honest,
the corpus the comparison arms use, reads 0 of 87,035. FILES: lens/runs/VAL-TRAIN-LEAK.txt,
VAL-TRAIN-LEAK-mix.txt. See BUGS.md.
NO ACCURACY CLAIM: the numbers above are about the RULE firing and the held-out loss it does or does not
protect, on small arms; nothing about how well the chat model talks.

## UPDATE 7 - THE SECOND DETECTOR OF THE METHOD IS BUILT: AgreementGate, and it is in the SHIPPED trainer
## (2026-10-21 21:5x, PLAN item d2d)
lens/unified.py now carries the second rule the design names beside the gap gate: AgreementGate, with the
coherence number A = ||sum_g grad_g|| / sum_g ||grad_g|| over G equal sub-batches of the batch, four detectors
(off / level / ema_band / vs_probe, DEFAULT vs_probe - the model compared against its OWN held-out probe), and
a refused step SKIPPED rather than reverted. lens/scripts/train_chat.py wires it behind
--gates {agreement,gap+agreement}. Predictions first: lens/runs/AGREEMENT-GATE-PREDICTIONS.txt (P60a-n),
scored in lens/runs/AGREEMENT-REPORT.txt.
MEASURED, 3 seeds, the gap gate's own two arms: the DEFAULT detector refused 0.383/0.190/0.158 of 400 steps on
the corpus that CAN be memorised against 0.013/0.110/0.076 of 237 on the one that CAN be learned - every seed
clears the pre-registered two-sided bar -> PASS - but the corpus GAP moves +0.370 / +0.080 / +0.082 across
seeds, so the rule fires where it should and does not yet SEPARATE the corpora reliably. The other two
detectors are decoration on both corpora (ema_band 0.983/0.932, level 1.000/1.000). Cost 4.53x, and the
sub-batch sum / G equals the single-pass batch gradient to 2.4e-07, which is what would make it free.
THE FULL RECORD IS IN LEARN-NOT-MEMORISE.md UPDATE 7 (it owns section 2b) and BUGS.md (two hazards).
NOT DONE, NAMED: scaling instead of refusing; the free version; the real recipe (queued:
lens/scripts/run_agree_arm.sh, predictions P61a-e). NO ACCURACY CLAIM OF ANY KIND.
