# GlassNet problem board

Rule (the maintainer): one problem at a time until 100% measured. No jumping ahead.
Legend: [x] closed · [~] running · [ ] open.

## AUTONOMOUS LOOP (mission 2026-10-01) — RUNG 1 PASSED ✅ (STOPPED.md)
Goal: small readable model within 2 pts of a same-size normal net on
Fashion-MNIST (Rung 1), then beat it; then Rung 2 (CIFAR-10, license check
first), Rung 3 (tiny text task). Video: OUT OF SCOPE. Decisions on the
VALIDATION split; TEST quarantined (runs/loop/TEST-USE.md: 2 touches, logged).
- [x] L0. Setup: split train50k/val10k/test10k, concept registry, harness.
- [x] L1. Broken concepts FIXED (quad_top_diff/edge_balance replaced them).
- [x] L2. Added concepts — audits caught 5 lying names, all rejected honestly.
- [x] L3. Widened concept layer — no accuracy gain (concept list ≠ bottleneck).
- [x] L4. Learned-but-audited concepts: 32→128-unit gate, names still audited.
- [x] L5. Better rule layer: linear rules FAILED (49.7, reverted); removing
       the rule layer entirely WON (85.8; +3.2 pts).
- [x] L6. Auto-pilot untouched during loop (its rebuild respects new flags).
- [x] R1. RUNG 1 GATE PASSED: test 86.26% ± 0.12% vs same-size normal
       86.26% ± 0.33% (Δ +0.00, readable ≥ normal; 102.4k vs 102.6k params).
       STOPPED.md written; waiting for the maintainer before Rung 2.
- [x] R2. PATHMNIST (MedMNIST v2, CC BY 4.0 — CIFAR-10 BLOCKED, no license):
       iterations 011–014 (concept set rebuilt for textures, audit metric
       fixed for imbalanced targets, lie control re-engineered to pinned
       probe, rescue reflex tested+FAILED+reverted). RUNG 2 GATE: TEST
       readable **46.20% ± 1.18%** vs same-size normal **37.99% ± 4.25%**
       (Δ +8.21 pts, audit PASS on the submitted model). STOPPED.md rewritten;
       waiting for the maintainer before Rung 3 (tiny text task — license first).

## Phase 1 — engine + baseline
- [x] G1. Every hand-written gradient passes the numerical gate — worst rel err 1.8e-07, worst abs 4.8e-10, 7/7 checks. (The gate caught a real bug on first run: nets were being checked in float32, where finite differences are blind; fixed by running the gate in float64. Training stays float32.)
- [x] G2. The engine learns XOR — 100.00% accuracy, loss 0.0008, 33 params, 0.6s.
- [x] G3. Fashion-MNIST BASELINE recorded — 86.69% ± 0.59% final-epoch test acc over seeds 0/1/2 (85.86/87.01/87.21); best-epoch mean 86.99% ± 0.19%. runs/phase1/BASELINE.md.

## Phase 2 — readable architecture ✅ (gate passed 2026-10-01)
- [x] P1. Concept layer: 14 named neurons, exact formulas in CONCEPTS.md; per-class stats docs/concept-class-stats.md.
- [x] P2. Sparse rule layer: L1 (sum-|w|) + prune to top-3; avg 3.00 conns/rule, 0 dead of 30. (Mean-L1 was too weak — fixed with sum-L1, cost found by sweep.)
- [x] P3. Answer layer: class = weighted sum of rule activations.
- [x] P4. explain(): receipts + pictures (runs/phase2/explain/). Trouser 94.2%, bag 71.0%; sandal→sneaker near-miss explained honestly.
- [x] P5. Gate report: readable 79.04±1.97 vs baseline 86.69±0.59 (−7.65 pts); 12/14 neurons faithful (0.798 mean AUROC); BROKEN named: vertical_edges 0.529, quad_LR_ink 0.501; lie-neuron control CAUGHT (fidelity 0.969→0.442, secretly ink_amount 0.905).
- [x] P6. Ablations (3 seeds each): no-supervisor → names collapse to chance (0.486) · no-sparsity → 14 conns/rule +2.7pts · no-rules → 83.13 best acc but no explanation · prune −0.45pts. runs/phase2/ablations.md.

## Phase 3 — auto-pilot trainer ✅ (gate passed 2026-10-01)
- [x] A1. Watcher acting on signs: loss flat → lower step size · loss explodes → restart smaller · train≫test → stop early · silent neurons → flag/prune. ALL FOUR reactions proven by forced-sign controls (runs/phase3/CONTROLS.md); controls caught 2 real bugs (answer-axis prune, weak explosion detector) both fixed.
- [x] A2. Every action logged in plain English — runs/phase3/actions_seed{0,1,2}.log (real run: 0 interventions, healthy task; logged honestly).
- [x] A3. HIDDEN_DEFAULTS.md — all 30 defaults + reasons; runner verifies 100% coverage and refuses to run otherwise.
- [x] A4. GATE: zero settings → 80.16% ± 0.98% vs manual 79.04% ± 1.97% — beats manual on all 3 seeds (+0.03/+0.82/+2.50), within seed noise. PASS.

## Phase 4 — data judge ✅ (gate passed 2026-10-01; v1 FAILED and was fixed)
- [x] J1. Hand-labeled pile (2500 rows: 1200 good + 8 corruption types ×150 + 100 injected junk) + from-scratch judge (our engine, 784→32→1, BCE). v1 gate FAILED unseen-type (54.67%±4.45%) — experiment (experiments/judge_features.py) showed epoch count was the lever (concepts scored 75.2% unseen vs pixels 72.2% at 8 epochs, but pixels 79.8% held-out); epochs 6→10 + 3-seed ensemble.
- [x] J2. Cheap automatic cleaning: removed 100/100 injected (20 NaN + 20 blank + 20 gray + 40 exact dupes; 0 false near-dupes).
- [x] J3. GATE: held-out 79.69% ± 0.61% (ensemble 79.79%) · unseen-type 71.48% ± 0.69% (ensemble 71.50%) · both ≫ 50% chance · example mistakes shown (runs/phase4/mistakes.png + report). Known weakness: quiet corruptions (box_overlay 11%, partial noise 10%, upside-down 39% single-seed recall on the all-types judge) — an honest risk, not claimed away. PASS.

## Phase 5 — library ✅ (gate passed 2026-10-01)
- [x] L1. model = GlassNet(task); model.train(data); model.explain(example) — glassnet/api.py; train() runs the auto-pilot (zero settings); explain() = text receipt + optional figure; accuracy(), train_judge()/judge(), save/load.
- [x] L2. Visual report (matplotlib): glassnet/report.py — image, named concept bars, rule contributions, class probabilities. Examples: runs/phase5/docs/assets/reports/report_18.png / docs/assets/reports/report_42.png.
- [x] L3. Beginner README (docs/BEGINNER.md) + basic tests (tests/test_library.py — 7/7 PASS, incl. the Phase 6 evo-switch test).
- [x] L4. GATE: examples/example_20_lines.py — 11 lines, trains to 80.78%, explains 2 examples (one correct bag 69.9%, one genuine miss dress→shirt 37.7% fully receipted), judge scores 5 images. PASS.

## Phase 6 — brand-new learning method ✅ (passed 2026-10-01)
- [x] N1. (1+1)-Evolution Strategy (Gaussian mutation, same-batch selection, 1/5 sigma rule, ZERO gradients) behind optimizer="evo"; SGD untouched and default. FAIR: same arch/seeds/objective/budget (5,628 batch-evals)/post-processing, 3 seeds. RESULT: SGD 80.16% ± 0.98% vs EVO 34.73% ± 2.04% (best-in-run 46.7-52.3%). Amplitude 45.4 pts > noise 1.6 pts → real, SGD wins. Mechanistic finding: pruning costs EVO ~30% (diffuse evolution weights) vs ~0.5% for SGD (sparse structure) — the readability pipeline assumes GD-like sparsity.
## T12 talk ladder (2026-10-13, in flight)
- Goal: does BIGGER help the talk line from scratch? (the queued main-power push, gate now open:
  all six lines at 100% strict)
- Design: ONE variable per run vs the t11 baseline (d256 w32, 80.59%, 40/40). L1 d384, L2 d512,
  L3 w48. 96 ep, same data/seed/lr, checkpoints each epoch. Audit 40/40 per member = ship gate.
- Predictions (pre-registered in run_talk_ladder.sh): d384/d512 <= +1.5 pts (data-bound), w48
  +1..+3 pts (context). ALL must stay 40/40 honest.
- Next after the ladder: extend the winner to 192 ep, scenario battery (run_talk_scenarios.sh),
  re-pin selfcheck, update REVIEW_PACKET.

## MORNING RESUME (2026-10-14) -- everything below is queued, nothing is lost
1. `bash lens/scripts/run_morning.sh` = selfcheck + attention run + chained splitfix run.
2. Bars: attn > 56.07% clean-val + 40/40 + no word salad; splitfix > 56.07% + 40/40.
3. Then: distill re-measure on clean data (teacher clean number 54.89% is the new reference,
   NOT 80.59).

## OPEN (next after the six-option matrix lands)
1. **Concept-wall bottleneck test (the biggest open lever for "smart").**
   The answer layer can only see the concept layer, so any signal the concepts cannot express is
   lost. Planned A/B on the WINNING corpus/dims from the matrix (identical budget, seed 0):
     A: current head (concepts -> answers)
     B: concepts + LOW-RANK RESIDUAL (x ->64 -> vocab), so the answer may bypass a lossy concept wall
   Honesty rules for this experiment, decided BEFORE running:
     * explain() must report how much of each answer came from concepts vs the residual path, or the
       "readable" claim would be quietly false;
     * a residual that earns real accuracy but is not explainable gets reported as an ACCURACY-vs-
       READABILITY trade, not as a free win;
     * a low-rank path (not a full x->vocab projection) because 16384x12000 would be ~196M params
       and would not fit the 6 GB VRAM rule.
2. **Distill-to-convergence** claim ("+2.8 pts") is leak-contaminated and unquotable until re-run on
   the clean corpora -- do not cite it anywhere.
3. **Audio/video accuracy metrics undefined** (both lines pass behaviour gates but have no accuracy
   number; speech now has one, 37.43% on 6,243 official-val clips).
4. **ODC-By datasets** (tulu-v2-sft-mixture, OLMo-mix, finemath) are permissive but not on the
   four-license allowlist -- ask before taking; skipped, not silently used.
