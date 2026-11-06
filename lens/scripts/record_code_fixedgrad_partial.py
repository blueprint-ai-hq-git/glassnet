"""record_code_fixedgrad_partial.py — what the code re-verification has established so far, plus two
honest flags (a gate that flips for the wrong reason, and one of MY OWN arms that was invalid).
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — CODE CONCLUSION RE-TESTED (part 1). A gate flips — for a reason that does NOT
## support the claim. And one of my own retrains was INVALID.

### CONSTRUCTION CHECK ON THE RETRAINS THAT WERE RUN
```
  arm                original corpus   my retrain corpus   valid?
  baseline 6k        v6000_both        v6000_both          YES  (20/150 -> 17/150)
  clean corpus       v32000_clean      v32000_clean        YES  (30/150 -> 29/150)
  vocab 32k          v32000_code       v32000_both         ** NO ** (2 things changed at once)
```

### THE DATA (clean corpus) LEVER — the pre-registered gate now PASSES, but NOT because the lever works
```
  pre-fix : clean 30/150  vs  baseline 20/150   p=0.081  -> FAIL
  post-fix: clean 29/150  vs  baseline 17/150   p=0.039  -> PASS
  mechanism: the TREATMENT moved -1 (30->29, noise); the BASELINE moved -3 (20->17).
```
**The gate passes because the CONTROL got worse, not because cleaning the corpus helped.** Reporting
this as "clean data now unlocks code generation" would be exactly the failure mode this project has
banned. The honest statement: the treatment is unchanged within noise (30 vs 29) and the pre-registered
p<0.05 is now met against a degraded baseline. At n=150 with one seed per arm, a 3-prompt swing is not
evidence of anything. **Nothing about the code capability is proven either way by this.**

### MY OWN ERROR — THE VOCAB ARM RETRAIN WAS INVALID (caught before it was reported as a result)
The corrected-vocab retrain ran with `CD_VOCAB=32000` but left `CODE_DOC` at its default (`both`), so it
trained on **v32000_both** while the original `covocab32k` was trained on **v32000_code**. Two things
changed at once — the exact violation of the project's one-change rule that the original label
attached to it. Worse, the eval then loaded the `v32000_code` vocab (its own 32k token->id map) into a
model trained on the `v32000_both` map, so token ids were scrambled and the model produced
`0/150` — a number that looks like a catastrophic regression but was an artifact of feeding the model
a vocabulary it had never seen. Tag `cfix_covocab32k_ep16` is marked INVALID in
`compare_code_fixedgrad.py` and excluded from every comparison. The single-change retrain is
`cfix_covocab32k_codeonly_ep16` (`CD_VOCAB=32000 CODE_DOC=code`) and its measurement is the remaining
open item.

### ALSO FIXED: A VERDICT-CHECK BUG I INTRODUCED AN HOUR AGO
`compare_code_fixedgrad.py` was printing a hardcoded "the lever FAILS post-fix" line next to a
computed `p=0.039 PASS`. Right number, wrong conclusion — the same pattern as the three earlier
verdict bugs. The verdict is now COMPUTED, and a pass driven by a falling control is printed as
"PASS DRIVEN BY A FALLING CONTROL — NOT evidence the lever works". Measurements are also cached to
JSON so re-running the verdicts does not re-generate 150 samples per checkpoint.

### STANDING (unchanged by this)
* code1 20/150, covocab32k 26/150, cclean32k 30/150, cbig384 24/150 — the lab's harness reproduces
  every recorded number exactly (re-run today), so the harness itself is sound.
* the capacity lever (d192->384, p=0.4526 FAIL) is not yet re-measured on the fixed gradient.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended code re-verification part 1")
print("  DATA lever: gate flips PASS by a FALLING CONTROL (treatment -1, baseline -3) -> not a win")
print("  vocab arm: my retrain was INVALID (2 variables changed) -> marked, single-change rerun queued")
print("  fixed: hardcoded verdict line contradicting its own computed p-value")