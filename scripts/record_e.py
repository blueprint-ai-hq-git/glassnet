"""record_e.py — append the Target E entry to LOG.md and refresh STATUS.md (10 lines max).
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-05 19:45 — TARGET E: K=9, all nine taught. PARITY, coverage 100%, R1 7/9

**One change from D: 6 concepts -> 9.** Floor unchanged (2.5%), bars unchanged, no rule moved.

**Coverage: 100.0000%**, minimum over all 10,124 validation windows, zero unnamed columns.
Prediction 1 CONFIRMED.

**Margin: PARITY.** readable ppl 90.8859+-15.6409 / acc 29.98%+-0.1961 vs normal
81.5138+-5.3461 / 30.19%+-0.2223 (both 115,691 params). Readable is WORSE IN DIRECTION on both;
neither difference reaches 2x pooled std, so the frozen rule says parity. Parity is not a win.

**R1 = 7/9** (prediction 5 CONFIRMED). **R5 = 6/9.**
FAIL R1: `pronoun_subject` (AUROC 0.5125, contrib 0.0769) and `is_a_dialogue_line` (0.5460,
0.0770). **The same two that failed in D** -- so their failure is not simple starvation; it
persists at K=6 and K=9 and appears only when concepts must carry the whole model.
`is_archaic_address` passes R1 (AUROC 0.7485, contrib 0.4147) but FAILS R5: ablating its whole
word set removes only 11.6% of its firing activation (bar 0.25) -- real, but not load-bearing.

**Independent agreement with the owner's reviewer.** R4 caveat 2 said `pronoun_subject`'s name
overclaims ("fires on windows ending in 'I', not on all subject pronouns"). The machine audit
says the same from a different direction: AUROC 0.5125 against its own pre-registered target --
indistinguishable from chance -- and contribution 0.0769 vs a 0.25 bar. Two independent methods,
same concept, same diagnosis.

**Predictions scored:** 1 CONFIRMED - 2 WRONG in magnitude (better than D by only 1.19 ppl /
0.14 acc, not >10 / >0.3) - 3 half right (worse in direction, within the bar -> PARITY) -
4 WRONG (neither collapsed concept recovered) - 5 CONFIRMED (exactly 7/9) - 6 WRONG AND
BACKWARDS (the three NEW concepts are the strongest: AUROC 1.0000, contrib 10.2-10.6, all PASS).

**C / D / E together:** coverage 6.40% / 100.0000% / 100.0000%; R1 6/6 / 4/6 / 7/9; margin
MIXED / BEHIND / PARITY. **The pattern: making the model readable costs accuracy. C only looked
competitive because ten unnamed columns did 93.6% of the work.** And across BOTH 100%-coverage
runs, 2 concepts stop meaning what their names say.

**PROCESS NOTE (bug caught before reporting).** The first E evaluation audited columns against
Target C's target list. E's columns are in a different order and number nine, so column 2 would
have been audited against the wrong concept; the run crashed at column 6 with
`IndexError: list index out of range`. **The invalid numbers were never recorded and never
reported.** Added `load_targets("meaningful_ext")` with an explicit order warning, verified the
mapping name-by-name against the training loader (output: MATCH), then re-ran.

**Artifacts:** `TARGET-E-RESULTS.md`, `runs/t3/e_eval.json`, `runs/t3/reader_test_key_e.json`,
`runs/t3/models_e_*.json`, `runs/t3/e_run.log`, `runs/t3/ckpt/*_meaningful_ext_e_seed*.npz`.
**Needs the owner: a human R4 score** (2 cards minimum).
"""

STATUS = """# GLASSNET — STATUS

Updated 2026-10-05 19:45. **Record: A = NOT MET, A2 = NOT MET, A3 = MET, C = PARTIAL, D = 100% readable / BEHIND, E = 100% readable / PARITY. Parity is not a win.**

1. **Target C readability, as a count:** `R1 6/6 · R2 0/6 · R3 6/6 · R4 6/6 by an informed AI reader (human PENDING)`. **No final readability number is reported, and none may be inferred from R4 alone.**
2. **Target D:** coverage **100.0000%**, **BEHIND** the matched normal, R1 4/6. **Target E (K=9, all named):** coverage **100.0000%**, **PARITY**, R1 7/9, R5 6/9.
3. **The pattern across C/D/E — readable costs accuracy.** Coverage 6.40% / 100% / 100%; margin MIXED / BEHIND / PARITY. **C only looked competitive because ten unnamed columns did 93.6% of the work.**
4. **Stable across both 100%-coverage runs: 2 concepts stop meaning what their names say.** `pronoun_subject` (AUROC 0.5125) and `is_a_dialogue_line` (0.5460) fail at K=6 AND K=9. This is the unsolved part.
5. **Your reviewer and the machine audit agree independently** — caveat 2 said `pronoun_subject` overclaims; the audit says AUROC 0.5125 against its own target (chance) and contribution 0.0769 vs a 0.25 bar. Two methods, same concept, same diagnosis.
6. `is_archaic_address` passes R1 but fails R5: ablating its whole word set removes only 11.6% of its firing (bar 0.25) — real, but not load-bearing with nine columns.
7. **My predictions were wrong three times here:** the new concepts are the *strongest*, not the weakest; neither collapsed concept recovered; the gain over D was 1.19 ppl, not >10.
8. **A bug was caught before it was reported:** the first E audit used Target C's target list, which is in a different order. Invalid numbers were never recorded; mapping re-verified (MATCH) before re-running.
9. **Your R4 instructions are in LOG.md verbatim**; the three follow-ups are in `QUEUE-AFTER-B.md` as ideas only. Cards, key, rules and Target B untouched.
10. **Needs you: a human R4 score** (2 cards minimum, existing key). Next build: the literature check for the training-methods goal.

Detail: `TARGET-C-RESULTS.md` · `TARGET-D-RESULTS.md` · `TARGET-E-RESULTS.md` · `METER-04-NAME-PRECISION.md` · `METER-05-NAME-TYPES.md` · `LOG.md` · `QUEUE-AFTER-B.md`.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print(f"STATUS.md rewritten, {len(STATUS.splitlines())} lines, "
      f"{sum(1 for l in STATUS.splitlines() if l[:2].strip().rstrip('.').isdigit())} numbered")
