"""record_ktransfer_correction.py — the SEE-ONLY K-transfer claim CHANGES under the corrected
gradient: k1 flips from PASS to FAIL. A recorded claim in PROBLEMS.md E8 is now wrong.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — CLAIM CORRECTION: the SEE-ONLY K-transfer result CHANGES under the corrected
## gradient. k1 FLIPS PASS -> FAIL.

### THE RECORDED CLAIM (PROBLEMS.md E8, unchanged until now)
"ktest 3 arms x 3 seeds on the PROVEN train_image loop: none 81.89±0.23 / real 84.36±0.16 / dummy
83.62±0.18. **k1 +2.47>2 PASS**, k2 |real-dummy| 0.74<=2 PASS" — i.e. adding 10 REAL concepts
transfers to the pure classifier, and dummy/random concepts do NOT (so the gain is meaning, not width).

### RE-MEASURED ON THE CORRECTED GRADIENT (same recipe, same seeds, 9 runs)
```
  arm      OLD (buggy grad)      NEW (corrected)      delta
  none     81.89 ± 0.29          83.35 ± 0.26          +1.46
  real     84.36 ± 0.20          84.62 ± 0.30          +0.27
  dummy    83.62 ± 0.22          84.54 ± 0.29          +0.92

  k1  real > none by > 2.0     OLD +2.47 PASS   ->  NEW +1.28 FAIL
  k2  |real - dummy| <= 2.0    OLD +0.74 PASS   ->  NEW +0.09 PASS
```
**k1 FAILS under a correct gradient.** The "extra REAL concepts transfer" effect shrinks from +2.47 to
+1.28 — below the pre-registered bar of 2.0. The gate is NOT being moved: it is recorded as FAILED on
the corrected engine.

### WHY THIS HAPPENED (the mechanism, not a mystery)
The old backward pushed concepts whose support != 0.5 up to 49x harder than the forward loss implied.
For the REAL arm that meant the real concept neurons were driven much harder than intended — which
INFLATED the measured benefit of having them. The dummy arm (random projections) has no meaningful
concept targets, so it was far less affected (+0.92). Removing the distortion removes most of the
apparent "real concepts help" gap. **Part of the recorded K-gain was an artifact of the bug.**

### WHAT SURVIVES
* **k2 PASSES on both**: real ~= dummy (0.74 -> 0.09). The "it is WIDTH, not MEANING" conclusion is
  UNCHANGED and if anything strengthened — real and dummy are now nearly identical.
* **The readability claim does not depend on this.** The +24.62/+6.74 readable-vs-plain results come
  from the VL pipeline and were re-measured separately (medical +24.72 HOLDS).
* The SEE-ONLY ladder improved: readable 83.49% -> **84.36%** on the corrected gradient (plain
  unchanged at 86.42%, as expected — it carries no concept loss).

### THE HONEST SUMMARY
The claim "adding real concepts transfers +2.47 pts to the pure classifier" is **DEMOTED**: on the
corrected gradient it is +1.28, not significant at the pre-registered bar. The complementary claim
"it is width, not meaning" (k2) STANDS. PROBLEMS.md E8 must be corrected — the entry currently
records k1 as PASS, which is no longer true on the fixed engine.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended K-transfer correction")
print("  k1: +2.47 PASS (buggy) -> +1.28 FAIL (corrected) — claim DEMOTED")
print("  k2: PASS both — 'width not meaning' STANDS (0.74 -> 0.09)")
print("  ladder readable improved 83.49 -> 84.36 on the fixed gradient")