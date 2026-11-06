"""record_g2.py — record the Target G2 verdict in LOG.md and STATUS.md.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 16:05 — TARGET G2 VERDICT: FAILED on G2-NOWORSE (coverage and readability PASSED)

```
    D    params  readable acc  normal acc  readable ppl  normal ppl   cov%      concepts
   32     65355        29.77%      29.79%        79.569      67.108   100.0000    7.33/9
   64    115691        29.76%      29.73%        82.057      69.968   100.0000    7.00/9
  128    216363        29.70%      29.97%        84.845      70.367   100.0000    6.67/9
  D= 32: coverage PASS | readable PASS (81.5%) | acc -0.02 (bar 0.45) | ppl +12.46 (bar 5.81) FAIL
  D= 64: coverage PASS | readable PASS (77.8%) | acc +0.03 (bar 0.61) | ppl +12.09 (bar 6.90) FAIL
  D=128: coverage PASS | readable PASS (74.1%) | acc -0.27 (bar 0.32) | ppl +14.48 (bar 5.77) FAIL
```

**Scope: 3 of the 5 pre-registered rungs** (D=32/64/128, a 3.3x parameter range). D=256 and D=512
were not run. The verdict covers the completed rungs only.

**G2 = FAILED** because G2-NOWORSE fails at every rung on perplexity. **Parity is not a win, and
this is worse than parity.**

**A metric turned out to be DEGENERATE.** Validation accuracy is pinned at 29.7-30.0% for every
model at every size, and `<unk>` is 29.64% of validation targets -- so predicting `<unk>` scores
~30%. The accuracy clause of G2-NOWORSE is **satisfied trivially by both models** and discriminates
nothing. Only perplexity can separate them here. The project had already noted ppl is the honest
metric; this run is the clearest demonstration of why.

**THE REAL FINDING -- readability survived scale better than any earlier target suggested:**
* coverage **100.0000% at all three rungs**, measured
* **74-82% of concepts pass at every size** (7.33 / 7.00 / 6.67 of 9)
* **five concepts are perfect at all three sizes**: contains_a_wh_word, mentions_speech,
  mentions_giving, mentions_knowing (AUROC 1.0000 everywhere), is_archaic_address (0.9994-0.9997)
* **two are structural failures, not scale failures:** `is_a_dialogue_line` **never passes at any
  size** (0/3 everywhere), `pronoun_subject` passes at D=32 then fails at D=64 and D=128.
  Making the model bigger does not fix them.

**Predictions scored:**
1. Coverage 100% at every rung -- **RIGHT**
2. Pass-rate rises with size (predicted 3-5/9 at rung 1) -- **WRONG**: flat-to-declining
   (7.33 -> 7.00 -> 6.67), and rung 1 was 7.33/9, far above the 3-5 predicted
3. The two collapsed concepts recover at larger rungs -- **WRONG** (one never passes at all)
4. is_archaic_address passes at every rung -- **RIGHT**
5. NOWORSE fails at the large rungs -- **PARTIALLY RIGHT**: it fails at every rung
6. Accuracy rises monotonically -- **WRONG**, and the metric is degenerate

**Consequence for the project:** the readable architecture's readability **does carry across a
3.3x size range**, but it pays a stable **12-14 ppl** penalty that does not close with size. Under
the frozen rule that is a FAIL. The next question is whether the penalty is the bottleneck's price
or a fixable design issue -- that is a new pre-registered target, not a retune of this one.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-06 16:05. **Record: A = NOT MET, A2 = NOT MET, A3 = MET, C = PARTIAL, D = 100% readable / BEHIND, E = 100% readable / PARITY, F = LM collapse (concept claim withdrawn, corrected), G = NO WIN (teacher too weak), G2 = FAILED on perplexity (coverage + readability PASSED).**

1. **TARGET G2 finished and FAILED — on the frozen rule, not on a judgement call.** Coverage passed and readability passed at every rung; **perplexity failed at every rung** (+12.46 / +12.09 / +14.48 against bars of 5.81 / 6.90 / 5.77).
2. **The good news is real and bigger than expected:** coverage **100.0000% at all three sizes**, and **74–82% of concepts pass** (7.33 / 7.00 / 6.67 of 9). Five concepts are perfect at every size. Targets D and E made readability look far shakier than it is across this range.
3. **Two concepts are structural failures, not scale failures:** `is_a_dialogue_line` **never passes at any size**, `pronoun_subject` passes at D=32 then fails at D=64 and D=128. A bigger model does not fix them.
4. **A metric is degenerate:** validation accuracy is pinned at ~30% because `<unk>` is 29.64% of targets. The accuracy clause passes trivially for both models. **Only perplexity discriminates.**
5. **Scope:** 3 of the 5 pre-registered rungs (D=32/64/128 = a 3.3× range). **D=256 and D=512 were not run** — nothing is claimed about them.
6. **My predictions: 2 right, 3 wrong, 1 partial.** Wrong on the pass-rate rising with size (it is flat-to-declining, and rung 1 was 7.33/9 vs my predicted 3–5). Wrong on the two collapsed concepts recovering. Recorded as wrong.
7. **The first G2 sweep was VOID and is recorded as void:** my audit used the wrong statistic, and my runner never set `T2_CONCEPT_SET` so the models trained on the word set while the audit compared the 9 meaningful targets. Caught by a contradiction (D=64 gave 0.33/9 where Target E gave 7/9).
8. **Two hard gates now prevent both:** the script refuses to start on the wrong concept set (verified both ways), and the runner aborts unless the audit reproduces a known result.
9. **Tests: 21 passing** (7 library + 10 readable-model + 4 regressions). Every wrong number today was caught by a control, a contradiction, or a self-check.
10. **Owed from you:** the **Target G decision** (easier task / bigger model / both / stop) and the **human R4 readability score** — the AI reviewer's 60/60 was informed, not blind.

Detail: `TARGET-G2-RESULTS.md` · `TARGET-G-RESULTS.md` · `TARGET-H.md` (pre-registered, not run) · `TARGET-F-RESULTS.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
