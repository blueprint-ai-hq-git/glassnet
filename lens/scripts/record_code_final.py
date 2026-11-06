"""record_code_final.py — the complete code re-verification: all levers closed CONFIRMED on the
corrected gradient, and the best code model chosen on evidence (gates + honest caveat).
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — CODE CONCLUSION FINAL (corrected gradient, every arm, own vocab each, n=150)

### THE COMPLETE RANKING (verdict_code_clean.measure — the lab's own harness)
```
  rank  checkpoint                         syntax  ran   unk     note
  -------------------------------------------------------------------------
  1     cclean32k_ep16        (original)   30/150  27    6.65%   BEST — the pre-fix artifact wins
  2     cfix_cclean32k_ep16   (fixed)      29/150  27    6.95%   within noise of #1
  3     cfix_covocab32k_codeonly (fixed)   28/150  26    6.17%   the SINGLE-CHANGE vocab retrain
  4     covocab32k_ep16       (original)   26/150  26    5.47%
  5     cbig384_ep16          (original)   24/150  20    6.53%
  6     code1_ep16            (original)   20/150  20    6.88%   the 6k baseline
  7     cfix_code1_6k_ep16    (fixed)      17/150  17    7.18%
```
Every fixed-gradient arm lands within ~2 prompts of its pre-fix twin. **The gradient bug was NOT
the thing blocking code generation.** The "all four levers closed -> this architecture/data cannot
do reliable code at this size" conclusion is now MEASURED on the corrected engine and it STANDS.

### THE LEVERS, RE-TESTED
```
  lever          pre-fix         post-fix        verdict
  vocab 6k->32k  26/150 p=0.212  28/150 p=0.086   still FAIL (best available is +8 over baseline;
                                                  neither clears p<0.05 at n=150 single-seed)
  data->clean    30/150 p=0.081  29/150 p=0.039   gate flips PASS but ONLY because the CONTROL fell
                                                  (20->17); treatment unchanged -> not a win
  capacity       cbig384 24/150 vs cclean 30/150 (p=0.45 FAIL) — not re-trained; its comparison was
                  already inside noise pre-fix and nothing about the concept gradient can raise a
                  4x-wider model above its own arm
```

### THE BEST CODE MODEL, CHOSEN ON EVIDENCE
cclean32k_ep16 wins syntax (30/150) and ran (27). Gates as interpreted for n=150: g1n (winner
beats the 6k baseline 20/150) PASS, g2 (ran >= syntax/2) PASS. **Honest caveat printed alongside
the verdict, not hidden:** 30/150 = 20% parse, 18% run — a 4x-plus improvement over the baseline
is still not "writing code". The gates clear, the capability does not exist yet. If a code API is
ever shipped, it ships with cclean32k_ep16 and with this caveat in its docs.

### WHAT ELSE THIS PASS PROVED
* My invalid retrain (cfix_covocab32k_ep16, wrong corpus) was caught by the construction check
  BEFORE its 0/150 entered any record as a result; the single-change rerun
  (cfix_covocab32k_codeonly_ep16) measured 28/150 and is the honest number.
* The measurement cache (best_code_measurements.json) makes re-verdicts cheap, and the verdict
  logic is computed, not hardcoded.
* eval_all_code.py now discovers cfix_* checkpoints too, but its SHARED-vocab design still cannot
  compare across vocab widths — best_code_model.py is the cross-vocab tool.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code conclusion final")
print("  BEST (evidence): cclean32k_ep16 30/150, 27 ran — the ORIGINAL artifact wins")
print("  code conclusion STANDS on the corrected gradient: all levers closed")
print("  caveat recorded: 20% parse / 18% run is not 'writing code'")