# GLASSNET — STATUS

Updated 2026-10-08 18:50. **The three-variant vision-language product exists, is transparent, and
its ceiling was measured from four independent directions today.**

## TODAY (2026-10-08) — what runs, with artifacts

1. **SEE + TALK (fused model)** — `lens/scripts/train_seetalk.py`: image -> NAMED concepts ->
   caption word. The ONLY pixels->words path is the concept layer; explain residual 0.00e+00 (CPU)
   / 6.7e-06 (GPU). Gates 4/4 at four rungs: tiny 72.5 / small 77.3 / medium 77.5 / big (60k,
   d=128) 75.5-78.7 vs text-only control pinned at 61.0 everywhere (+14.5 to +16.3 delta).
   Checkpoints round-trip bit-identical; vocab words + concept names + formulas + supports +
   image/patch size all travel inside the .npz. **K=22 shipping run: peak 78.69%.**
2. **VARIANTS (one interface, three models)** — `lens/scripts/variants.py`:
   `v.see(img)` class+confidence+fired concepts · `v.talk(ids)` next WORD via the checkpoint's own
   vocab file (hard error on out-of-range id) · `v.see_and_talk(img, prefix)` · `v.explain(...)`.
   REFUSAL TEST PASSES: see-only asked to talk raises "Refusing instead of guessing".
   see_and_talk on 500 TEST images: **386/500 = 77.2%** noun position (first & only test use,
   logged in TEST-USE.md).
3. **CAPTION CAMERA VIDEO** — `caption_camera.py` -> `caption_cam2.mp4` (+ ~/caption_cam.mp4):
   fused model narrates 48 TEST frames word by word, **34/48 = 70.8% full-name correct**, with
   driving concepts and MATCH/MISS on screen. First render scored 62.5% under a wrong first-word-only
   rule (multi-word classes like "ankle boot" undercounted) — caught by re-scoring the saved log,
   regenerated, per-frame lines kept.
4. **CLEANER (any dataset in, ON/OFF)** — `clean.py`: 75,197 items -> 99.95% kept; self-test drops
   match counts exactly (dup 5, short 3, empty 2, repetition 2, low-signal 2); OFF provably
   pass-through; rules+thresholds frozen to `clean_rules_frozen.json`.
5. **TEACHER LOOP — CLOSED ON EVIDENCE (4 experiments, 3 seeds):**
   * **t1 PASS 4/4: 10% data + teacher lands within 8.6-9.8 pts of the 100%-data ceiling** —
     "less data but smart" SHOWN.
   * t2 FAIL 4/4 (+0.06 / -0.64 / -0.45 / +0.74): teacher never beats 10% REAL labels.
   * Mechanism measured: teacher mass on truth 0.743 vs real label 1.000 — on deterministic-label
     tasks ground truth dominates; 30% uniform corruption didn't flip it either (only -1.2 pts;
     SGD shrugs off uniform junk). Distillation pays only where labels can't dominate.
   * Caught & fixed en route: no-control arm bug; teacher/student vocab ORDER mismatch (would have
     silently mislabelled every soft target — remap by word string); distill_kl verified END-TO-END
     against float64 reference at 2.5e-06 on all 7 param groups after two float32-FD PROBE false
     failures (signal vs rounding floor, documented; the op was never wrong).
6. **CONCEPT CEILING — three experiments + sweep, final shape:**
   * K=12 is information-STARVED: +10 dims buys **+4.4** (74.1 -> 78.5).
   * Flattens hard: K32 -0.1, K48 +0.6; **K=48 DUMMY (random projections) = 80.4, nominally top**.
   * real ~= dummy at every width: **transparency costs NOTHING**. "Readable concepts improve
     learning" NOT supported at this rung — not claimed. What concepts demonstrably own: exact
     explain (0.00e+00), audited firing, refusal behaviour.
   * Same wall from p1's miss (6x data no gain), eyes v2 (2x resolution no gain), image line:
     **78-80% is this TASK's ceiling at 784 px, not the architecture's.**
   * Eyes v1 was a design flaw (native-generation arms; thincross literally empty at 28); v2
     physical-pooling rerun: resolution effect = 0 inside seed noise. Honest from the first failure.
   * **Shipped decision: K=22 real concepts = the product config** (width where it pays, no cost,
     full explainability).

## BLOCKED ON OWNER DECISIONS (not on me)
* **FACES**: (a) skip / (b) owner's own photos / (c) name a dataset — all checked datasets have
  unclear or non-commercial licences (recorded DATA_LICENSES.md).
* **OPEN-VOCAB CAPTIONS**: COCO SKIPPED — annotations CC BY 4.0 but images stay under Flickr terms
  (official statement; a contradictory wiki claim was not relied on). Needs licence-clean source.

## HOUSEKEEPING
* Root disk 87% full (31 GB) — watch; the project disk fine (213 GB).
* Game guard live on every run; GPU idle after ktest completion.
* Failed/partial artifacts all KEPT and labelled: eyes v1, teacher v1 (no-control), gradcheck
  probe failures, caption v1 scoring rule, tiny2 no-artifact run, ktest_SMOKE.
* NEW today: `feed_customer.py` one-command pipeline (both clean arms PASS f1 on foreign data;
  exact f2 noted as follow-up), `selfcheck.py` 28-check ship gate, `API.md` quickstart,
  load_vision extended registry (dummy checkpoints REFUSE to load), see arm = ktest_real K23.

---

## PREVIOUS (2026-10-07) — the chat-model optimisation ledger (still true)

1. **THE SWEEP — all five variants beat the baseline** (d=128, K=37, vocab 12k, W=8, 12ep, 2,029,925 params, ppl 148.33, 1.576 bits/char):
   - **A vocab 3,500** — ppl **90.43**, **1.420 bits/char**, 618,925 params — best perplexity AND 3.3× smaller
   - **B window 16** — ppl 109.95, 1.482 bits/char, acc 24.69%, 24/41 concepts — best accuracy
   - C 24ep + lr decay — 143.19 / 1.565 · D d=256 — 147.26 / 1.574
2. A is the single biggest win: 3.3× fewer params, 9.9% better bits/char (embedding was 93% of the model).
3. B confirms drift is a context problem, not a perplexity problem.
4. D/big1 buy ~0.1% — extra width is nearly worthless once the concept layer is wide enough.
5. Predictions: 2 wrong, 3 held.
6. **BUG 31 — third verdict-check bug**: compared bits-per-token against bits-per-char and printed the wrong conclusion. Pattern named: **the risky step is converting numbers into a claim.**
7. Decoder fixed: top-k sampling + repetition penalty.
8. Combined run `best1` (vocab 3500 + window 16 + lr decay + 24ep, d=128).
9. CODE data ready: Magicoder-OSS (MIT), codeparrot/apps (MIT, with test cases).
10. Licence discipline held throughout; skips recorded with reasons.
11. **THE CEILING:** 10M = 3.5 h, 50M = 3.7 d, GPT-3 = 10,513 years on this hardware.
12. 31 bugs at that point, every one caught by a gate, control, or self-check.
13. Game guard live.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md` · `PLAN.md`.
