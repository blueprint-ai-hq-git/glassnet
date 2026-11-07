# SETTINGS REGISTRY - every knob, ours to define (2026-10-20)

the maintainer: "why are you just thinking about learning rate, there I like 50 billion different settings we have to go
through and make our own so it only learns and still does not hide anything ... list me all of the settings
there are already in the training, that we have to set, of RL, SFT, distillation, everything single, and then
make everything for that our own setting that forces the model to learn."

This is that list. Section A is what EXISTS in our code today. Section B is what standard training has and we
are MISSING. Section C is our own invention for each area, aimed at the one rule that matters: a lookup table
stores per-example facts (memorising), a rule is shared across examples (learning). 2+2=4 must be COMPUTED.

=========================================================================================================
## A. SETTINGS THAT ALREADY EXIST IN OUR CODE (real, not aspirational)
=========================================================================================================
TRAINER (lens/scripts/train_chat.py):
  --d                model width
  --window           context window in tokens
  --batch            batch size
  --epochs           training length
  --lr               learning rate (hand-set; the thing that must stop being hand-set)
  --lr-decay         linear decay factor (0 = constant)
  --lam-concept      weight of the concept loss
  --lam-act          weight of the activation penalty
  --mtp              multi-token prediction head count (1 or 2)
  --assistant-only   mask the loss to assistant tokens only
  --train-turn-end   train the turn-end (stop) target
  --attention        attention on/off
  --max-gap          memorisation gate: stop when train-heldout gap exceeds this (EPOCH level, a post-mortem)
  --gates            THE MEMORISATION RULE FROM lens/unified.py, PER STEP, inside the shipped trainer
                     (added 2026-10-21, PLAN item d2b). 'off' (default) = the behaviour before the flag
                     existed; 'gap' = the candidate update is applied, the model's own held-out probe is
                     re-scored with the SAME loss the training uses, and a step that worsens the smoothed
                     probe loss is REVERTED. MEASURED on a corpus that CAN be memorised: 74 of 400 steps
                     refused (0.185); on a corpus that can be learned: 2 of 237 (0.008). The held-out ppl
                     outcome bar FAILED there (1059.04 gated vs 1035.72 ungated, ratio 1.023) - see
                     lens/runs/GATE-REPORT.txt. The agreement gate (2b) is NOT wired.
  --gate-variant     which GapGate detector: ema_band (default, the method's own), divergence, best_band,
                     gap_level, rate_ratio, trend_band, gap_trend, or 'off' (the CONTROL - refuses nothing
                     and is digit-identical to --gates off). 'trend_band' is the near-miss RECOMMENDED to
                     the owner and not applied anywhere.
  --gate-beta / --gate-tol / --gate-lag   the gate's smoothing and thresholds (0.5 / 0.0 / 25 = the method's)
  --probe-windows    size of the model's OWN probe slice, drawn ONCE from the held-out split, never trained
                     on (default 256)
  --probe-seed       the seed for that draw (default: --seed); both are printed with the draw
  --gate-log-every   print cadence for the gate numbers; the per-step JSONL is written EVERY step to
                     lens/runs/chat/<out>_gates.jsonl (one row per training step)
  --max-windows      how many training windows to draw (memory control)
  --seed             seed
  --device           cpu/gpu
  --resume / --resume-ep / --out

METHODS (lens/methods.py, run_distill.py):
  --samples          RL: samples per prompt
  --rounds           RL: rounds
  --reward           RL: reward mode (exact verifier)
  --pretrain         SFT warm start before RL
  --seq-in / --seq-out  sequence lengths in and out

METHOD, ONE LOOP (lens/unified.py - the single training method, all three target sources):
  mode               where the target comes from: 'data' | 'teacher' | 'verifier'
  verifier_mode      HOW the verifier is asked. DEFAULT 'probe3' since 2026-10-21: the verifier names which
                     of three candidates (pred, pred + probe_step, pred - probe_step) is closest to right.
                     'hard_quarter' is the OLD default, kept as a named variant (measured collapse: 9.0%
                     trained / 0.0% unseen). MEASURED 2026-10-21: probe3 reaches slope error 0.472/0.448/
                     0.436% and 100% unseen with a RANK-capable verifier, but 0.0% unseen with only a
                     right/wrong verifier, and the probe step does not change that (slope error 6.797% at
                     s = 1, 1/4, 1/16, 1/64 - identical). The dead zone is the VERIFIER'S OWN tolerance
                     (0.5 for a right/wrong test), not the probe step. lens/runs/UNIFIED-VERIFIER.txt.
                     FIXED 2026-10-21 19:3x by binary_probe='ladder' (PLAN verifier3): with a plain
                     right/wrong verifier the slope error is 0.189/0.230/0.192% and 100.0% exact on
                     unseen pairs, 3 of 3 seeds. lens/runs/UNIFIED-LADDER.txt.
  probe_step         the probe offset (default 1/16)
  probe_normalise    divide the update by probe_step so the step does not shrink with the probe (default True)
  binary_probe       where the DIRECTION comes from when there is no rank oracle. DEFAULT 'ladder' since
                     2026-10-21 19:3x (PLAN verifier3): ask the right/wrong verifier about pred +- d on a
                     ladder of rungs spaced ladder_spacing*ladder_tol and take the first rung where exactly
                     one side is right - that side is the direction. MEASURED 3 seeds, 12000 steps: slope
                     error 0.189/0.230/0.192% and 100.0% exact on trained AND unseen pairs (against the old
                     'nearest' fallback's 6.80/6.49/5.95% and 0.0% unseen). 'nearest' is the OLD fallback,
                     kept as a named variant so its stored numbers reproduce. lens/runs/UNIFIED-LADDER.txt.
  ladder_tol         the verifier's DECLARED tolerance (default 0.5 = the 0.5 in |round(pred) - y| < 0.5).
                     The ladder is only as good as this declaration: rungs are placed relative to it.
  ladder_max         top rung (default 40.0). Residuals beyond ladder_max + ladder_tol are NOT resolved and
                     the example parks, exactly like the old fallback - so this must cover the target range.
  ladder_spacing     rung spacing in units of ladder_tol (default 1.5; must be < 2 or the rung windows stop
                     overlapping and residuals can fall between two rungs and be lost)
  agreement_mode     what the agreement number does to a step that is not refused: 'one' (default) | 'floor' | 'raw'
  gate_beta/gate_tol the memorisation gate's EMA smoothing and tolerance (see GATES below)

OPTIMISER (lens/opt.py, ours): lr, eps, mode (relgrad/rmsprop), momentum, scale (rmsprop only, now 1/sqrt(n);
  for relgrad the per-tensor update norm is exactly lr so no factor is needed), decay (step size schedule;
  1.0 = constant, bit-identical to the behaviour before the knob existed). decay MEASURED 2026-10-20: with
  decay 0.999 the numeric path answers unseen operand pairs 100% exactly at every step size from 0.005 to
  0.5, where a constant step was non-monotonic (100.0/54.7/66.7/100.0%) and did not fit a 5-parameter
  regression (19-28%). lens/runs/NUMERIC-PATH-OPT.txt, lens/runs/NUMERIC-PATH-OPT-DECAY.txt.
AUTOPILOT (lens/autopilot.py): d_candidates, lr_candidates, epochs_max, patience, probe_epochs,
  probe_frac, val_frac, batch  -- it chooses these itself, on a held-out slice it never trains on
EASY/HARD MODE (lens/modes.py, recipes.json): method, d, window_in, window_out, epochs, batch, opt, lr,
  lam_concept, lam_act, max_gap, sft_seq_in, sft_seq_out, budget_params
MODEL (lens/model.py): d, window, number of concepts, attention, mtp, seed
STREAMING (lens/stream.py): decay, stride, window

=========================================================================================================
## B. STANDARD SETTINGS WE ARE MISSING (must be added to be complete)
=========================================================================================================
SFT:  label smoothing, dropout, weight decay, gradient clipping, gradient accumulation, warmup steps,
      LR schedule shape (constant/linear/cosine), sequence packing, curriculum ordering, data mixture
      weights, early-stopping patience, checkpoint/eval interval, prompt-completion template,
      padding/truncation policy, class balancing
RL:   reward source (verifier vs learned reward model), sampling temperature, top-p, top-k, samples per
      prompt, rounds, KL penalty to the reference policy, advantage normalisation / group baseline,
      reward clipping, group size, max new tokens, prompt set, on/off-policy ratio, reward shaping,
      length penalty, entropy bonus, rejection threshold, verifier strictness
DISTILL: teacher choice, temperature, alpha (hard vs soft target mix), top-k soft targets, layer matching,
      sequence length, student size
OPTIM:  optimiser family, momentum, adaptive betas, weight decay, per-parameter scaling, gradient
      normalisation mode, update-rejection rule
MEMORY: hot/warm/cold tier sizes, quantisation bits for concepts, context-length policy, cache capacity,
      eviction policy
THINKING BLOCK: tag format, think-block loss weight, max think length, whether the trace is concept-based
GATES:  max-gap, eval interval, number of seeds before a win is claimed

=========================================================================================================
## C. OUR OWN INVENTIONS - every one aimed at "learn the rule, do not store the fact"
=========================================================================================================
1. AGREEMENT-GATED UPDATE (the central idea, ours). For each example in a batch, compute its own gradient.
   Learning produces a direction all examples agree on; memorising produces a different direction per
   example. So: measure the agreement between per-example gradients and scale (or reject) the step by it.
   An update that only helps individual examples is refused by construction. Nothing is hidden: the
   agreement number is printed every step.
2. RULE-VERSUS-LOOKUP TEST (already measured): train on operand pairs 0..9, then score on unseen 10..19.
   A lookup table scores 0%. A rule scores 100%. Measured so far: one-hot structures 0%, numeric in +
   numeric answer 100% exact. This test is the arbiter for every setting in this registry.
3. NUMERIC PATH: values enter as numbers, the answer leaves as a number. Measured: 100% exact
   extrapolation where every categorical design gave 0%.
4. CONCEPT HONESTY LOSS: a concept unit must fire only when its own evidence is present (the existing
   false-fire gate). A unit that memorised examples fires wrongly and is DROPPED, never shipped with a
   disclaimer.
5. CAPACITY AS A LEARNING FORCE: our d=256 model is 3.9M parameters against a 294M-token corpus. It is
   physically too small to store the corpus, which is exactly why it must generalise. Keep it that way.
6. THINKING BLOCK (standard tags, per the maintainer: use the ready-made ones so any runtime understands them).
   Our corpus already uses standard ChatML <|im_start|>/<|im_end|>. Adopt the standard reasoning block
   inside the assistant turn, and fill it with the NAMED CONCEPTS so the private thinking stays readable
   and checkable by our own gate.
7. HARDWARE-DRIVEN CONTEXT (per the maintainer: "it can be extended to unlimited as much as their hardware can
   support ... we do not follow math, math follows us"). No fixed cap; the cap is the machine:
     hot tier  (exact K/V/O, d=512): 6,144 bytes/token -> 1 GB ~ 175,000 tokens
     warm tier (concepts, int8):    1.28 bytes/token   -> 1 GB ~ 835,000,000 tokens
     cold tier (tokens, int16):     2 bytes/token      -> 1 GB ~ 536,000,000 tokens
   So: model in the 6 GB GPU (ours is ~7 MB), context in system RAM, overflow to disk.
     64 GB RAM  -> ~50 billion tokens of concept memory
     1 TB RAM   -> ~780 billion tokens (~0.78 trillion), and more with disk tiers
8. SPEED IS A SETTING TOO (per the maintainer: "our model should be fast at everything even the token per second").
   Current measured: cached step 7.66 ms/token vs 0.51 ms for a full window recompute = 14x SLOWER. That
   is our Python per-token overhead, not the model: the model is 7 MB and fits in cache. The rewrite
   (preallocated buffers, batched step, no rebuild-per-token) must produce a real tokens/second number.
