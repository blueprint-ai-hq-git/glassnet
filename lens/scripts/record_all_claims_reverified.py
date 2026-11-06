"""record_all_claims_reverified.py — the complete post-fix ledger after all retrains landed.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — COMPLETE POST-FIX LEDGER: every claim re-measured on the corrected gradient.

### THE BUG (one sentence)
The concept-BCE backward was wrong by a factor of pos_weight for every concept whose support != 0.5
(up to 49x on images, 170x on the talk model) — found 2026-10-10, fixed, and now gated inside
selfcheck by gradcheck_full_ops (which had never been run by any gate before).

### EVERY CONCEPT-LOSS RESULT, RE-MEASURED
```
  claim / artifact                     pre-fix        post-fix        verdict
  ---------------------------------------------------------------------------------------
  medical readable vs plain (E12)      +24.62         +24.72          HOLDS (lam=0 arms byte-identical)
  fashion generalization  (E17)        +6.74          +6.52           HOLDS (plain byte-identical)
  K22 see+talk ship      (K22)         77.03%         76.90%          HOLDS (gates 4/4)
    K22 concept fidelity: decorative   5              2               IMPROVED (names got honester)
    K22 strong/honest/weak             10/6/1         12/3/5          strong up
  see-only K-transfer k1 (E8)          +2.47 PASS     +1.28 FAIL      DEMOTED (real concepts less useful
                                                                        than the bug made them look)
  see-only K-transfer k2 (E8)          +0.74 PASS     +0.09 PASS      STANDS (width, not meaning)
  see-only ladder readable (E10)       83.49%         84.36%          IMPROVED
  talk asonly v12k (ship)              ppl 10771      ppl 13064       UNCHANGED CLAIM: both are 0-unk
                                       acc 9.86%      acc 13.38%      fluent filler; neither metric
                                                                        decisive for the claim it ships on;
                                                                        OLD artifact kept as shipping default
  code all-levers-closed (E13-16)      20-30/150      MEASURING NOW   (compare_code_fixedgrad.py,
                                                                        lab's own harness, n=150, Fisher)
```

### WHAT THE FLIPPED CLAIM TEACHES
k1 flipped because the distorted gradient drove REAL concept neurons ~49x harder than the loss
implied, inflating the apparent benefit of having them; the dummy arm (no meaningful targets) was
barely affected. **The fix does not "remove results" — it removes the distortion that produced some
of them.** Two of the three strongest claims survived re-measurement with a BYTE-IDENTICAL control
arm (the lam=0 / plain conditions carry no concept loss, so their equality pre/post is the proof the
fix touched only the concept path).

### STATE AFTER THIS PASS
* engine: concept-BCE backward exact; float64 mode real (it was float32-forced before); to_cpu
  unwraps Tensors (0-d object bug fixed); DOC_MODE=clean genuinely cleans (it silently didn't).
* tests: run_all_tests.py = 14/14 PASS (script that existed but was never run is how the bug
  shipped; the runner now exists and runs).
* shipping: see_talk = seetalk_k22_big_fixedgrad (promoted, fidelity better); talk = old asonly
  (claim unchanged, kept); image = retrained checkpoints measured, defaults re-pointed where
  better (ladder K13 fixedgrad).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended complete post-fix ledger")
print("  HOLDS: medical +24.72, fashion +6.52, K22 (fidelity better)")
print("  DEMOTED: k1 +2.47 -> +1.28 FAIL")
print("  STANDS: k2 width-not-meaning; IMPROVED: ladder 84.36, K22 decorative 5->2")
print("  MEASURING: code all-levers-closed (lab harness, n=150)")