"""record_meter06.py — METER-06, the arithmetic fix, and the honest bottom line.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 17:50 — METER-06 CORRECTED. SEVEN ARMS, SEVEN FAILURES. HERE IS WHAT IS CERTAIN.

### THE METER BUG, MINE, AND HOW IT WAS HANDLED

The first METER-06 run reported **0/9 for every arm** -- including the arms that pass 9/9 under the
old meter. A perfect concept scoring exactly 1.00 against a bar of 2.0 is the tell: the random
baseline for the JOINT rate is `q * s`, and I divided by `q` alone. Even a flawless concept was
capped below the bar by construction.

**The bar (2x random) was NOT changed. Only the miscomputed baseline was fixed.** The broken run's
output is kept at `lens/runs/meter06/rescore_BROKEN_firstrun.json`, not deleted.

### METER-06, CORRECTED

```
arm                    METER-05    METER-06
lc=0.1 binary               7/9        7/9     agrees
lc=0.3 binary               9/9        9/9     agrees
lc=0.5 binary               9/9        9/9     agrees
lc=1.0 binary               9/9        9/9     agrees
two-phase                   9/9        9/9     agrees
FIX-02 soft MSE             0/9        2/9     still bad, confirmed
FIX-03 soft BCE             0/9        7/9     ranks perfectly, does not saturate
```

**The corrected meter reproduces the old verdict on every old arm**, so it is not a rubber stamp.
It changes exactly one verdict -- FIX-03 -- and for a reason it was pre-registered to detect.

**FIX-03, both numbers side by side:**
```
mentions_speech      AUROC 1.0000   M5 fires 0.0010  M5 FAIL   M6 lift 27.52x  M6 PASS
mentions_giving      AUROC 1.0000   M5 fires 0.0013  M5 FAIL   M6 lift 27.84x  M6 PASS
mentions_knowing     AUROC 1.0000   M5 fires 0.0020  M5 FAIL   M6 lift 26.82x  M6 PASS
contains_a_wh_word   AUROC 1.0000   M5 fires 0.0057  M5 FAIL   M6 lift 18.12x  M6 PASS
is_archaic_address   AUROC 0.9992   M5 fires 0.0107  M5 FAIL   M6 lift 11.23x  M6 PASS
is_a_negation        AUROC 1.0000   M5 fires 0.0077  M5 FAIL   M6 lift  9.08x  M6 PASS
pronoun_object       AUROC 1.0000   M5 fires 0.0047  M5 FAIL   M6 lift  8.14x  M6 PASS
pronoun_subject      AUROC 0.5443   M5 fires 0.0743  M5 FAIL   M6 lift  1.03x  M6 FAIL
is_a_dialogue_line   AUROC 0.5047   M5 fires 0.0067  M5 FAIL   M6 lift  0.98x  M6 FAIL
```

**The seven concepts are real and strong.** Their most-active windows are 8-28x more likely to be
the true ones. They rank perfectly; they simply do not saturate, which is what graded targets do.
**The two hard concepts fail under BOTH meters** -- the new meter is not hiding them.

### THE HONEST BOTTOM LINE

```
arm                         concepts   ppl       gap      acc
lc=0.1 binary                    7/9    90.085   +12.281   29.73%
lc=0.3 binary                    9/9   100.850   +23.045   29.62%
lc=0.5 binary                    9/9   107.020   +29.216   29.65%
lc=1.0 binary                    9/9   115.829   +38.025   29.65%
two-phase                        9/9   116.430   +38.626   29.65%
FIX-02 soft MSE (concepts dead)  0/9    99.313   +21.508   30.03%
FIX-03 soft BCE (unsaturating)   1/9   103.386   +25.582   29.50%
matched normal                     -    77.804         -    28.43%
BAR                                          ~6
```

**SEVEN ARMS. SEVEN FAILURES.** Nothing has come below **+12.281**, against a bar of about **6**.
The blocker is not solved and no configuration tested has moved it.

### WHAT IS CERTAIN NOW, AND WHAT IS NOT

**CERTAIN:**
1. **The concept layer works.** Seven of nine concepts detect exactly what their names say: AUROC
   1.0000, lift 8-28x. This replicates across the old engine and the rebuilt one.
2. **The blocker is real and reproducible.** +12.281 on the rebuilt engine against +12.46 on the
   old one -- an independent replication.
3. **The cost is in the concept layer being a BOTTLENECK**, not in the supervision. Three attempts
   that attacked the supervision (Target I, the sweep, two-phase) all failed, and the two-phase run
   proved it directly: perfect concepts (BCE 0.00168) with a near-random language model
   (CE 5.0158). Nine concepts cannot carry 1,500 words.
4. **Graded targets change the failure mode, not the cost.** MSE collapses to the mean (0/9
   concepts). Soft-BCE keeps the ranking (7/9) but the gap is still +25.582 -- worse than the
   binary arm at the same strength.

**NOT CERTAIN, AND NOT TO BE CLAIMED:** whether the gap can be closed at all while the middle layer
stays readable. Every measurement so far says readability costs perplexity, and the best result in
the project's history is parity, never better.

### THE NEXT LEVER IS AN OWNER DECISION, NOT A SILENT ONE

The remaining lever is **more concepts**. Nine numbers cannot carry 1,500 words; sixty might. But
more concepts means more for a human to read, which is the exact thing this project exists to
protect. **That trade-off is the owner's call**, so it is being put to him rather than taken.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 17:50. **Seven arms against the blocker. Seven failures. Here is what is certain.**

1. **The model reads as WORDS and has its own number** (`ef1f6a3763d0`). `is_archaic_address = +89.44*thee +88.77*thou +86.41*thy -17.11*you` — the whole middle layer, 9 readable lines. Your idea, built.
2. **Decoding found the bug I could not find for days:** `pronoun_subject` was hunting character names; `is_a_dialogue_line` had its speaker-label weights NEGATIVE (wired backwards).
3. **Those two were never broken — they were UNDER-TRAINED.** At `lam_concept=1.0` both hit AUROC 1.0000. My "they compete for the same signal" diagnosis was WRONG and is withdrawn.
4. **The concept layer WORKS:** 7/9 concepts detect exactly what their names say — AUROC 1.0000 with their most-active windows **8–28× more likely** to be the true ones.
5. **THE BLOCKER: 7 arms, 7 failures.** Best gap **+12.281**, worst +38.626, **bar ~6**. Nothing below +12.
6. **The cause is now proven:** the middle layer is an **information bottleneck**. Nine concepts cannot carry 1,500 words. The two-phase run shows it directly — **perfect concepts (BCE 0.00168) with a near-random language model (CE 5.0158)**.
7. **Three attempts attacked the supervision; all three failed.** Target I (dead end), the sweep (clean monotonic curve, no free lunch), two-phase (worst result). The supervision was never the problem.
8. **Graded concepts change the failure mode, not the cost.** MSE collapses to the mean (0/9 — best language numbers, because a constant layer isn't a bottleneck: a false win, recorded as one). Soft-BCE keeps the ranking (7/9) but the gap is +25.582.
9. **A meter bug of mine, found and fixed honestly:** METER-06 first reported 0/9 for everything because I divided by the support once instead of twice. **The bar was not changed** — only the arithmetic. Broken output kept.
10. **NEXT IS YOUR CALL:** the remaining lever is **more concepts** — nine numbers can't carry 1,500 words, sixty might. But more concepts means more for a human to read, which is what this project exists to protect. **I'm not taking that trade-off silently.**

Gates: L0 18/18 · L1 12/12 · L2 8/9 · L3 8/8 · L4 12/12. 16+ bugs, every one caught by a gate.
Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `METER-06-GRADED.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
