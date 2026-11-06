"""record_blast_radius.py — every shipped artifact affected by the concept-gradient bug, and the
retrains now running for each.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — BLAST RADIUS: every SHIPPED artifact trained with the distorted concept gradient.
## Retrains launched for all of them on the corrected engine.

The concept-BCE backward was wrong by a factor of pos_weight for concepts whose support != 0.5
(found + fixed today). An audit of EVERY shipped checkpoint's concept supports:

```
  artifact                          concepts   w range      w==1   status
  SEE+TALK ship (k22, FIXED)        22         0.03-48.7    0     retrained, PROMOTED earlier today
  SEE-ONLY ship (K23 ktest_real)    23         0.03-48.9    0     AFFECTED -> retraining
  SEE-ONLY ladder (K13)             13         0.03-48.9    0     AFFECTED -> retraining
  TALK ship (asonly_v12000)         41         1.56-170.1   0     AFFECTED -> retraining (worst!)
```

All four shipped artifacts trained with `lam_concept = 0.3` and ZERO concepts at the safe w==1.
The talk model is the worst hit: 41 concepts with w up to 170, meaning the absent-side of every
concept neuron was pushed up to 170x harder than the forward loss implied.

WHAT IS ALREADY DONE
  * K22 see+talk: retrained, promoted to shipping default (val_acc 77.03 -> 76.90, decorative
    concepts 5 -> 2). Verified live through the interface; 14/14 tests pass.

WHAT IS RUNNING NOW (run_all_fixedgrad_retrains.sh, backgrounded)
  1. lad_image_cls_fixedgrad  (K13 ladder, same 24-epoch recipe)
  2. ktest_<none/real/dummy>_s0..2_fixedgrad  (K23, 3 arms x 3 seeds, same recipe)
  3. asonly_v12000_fixedgrad  (talk, same 24-epoch assistant-only recipe)

Once they land: re-verify accuracy and concept fidelity vs the pre-fix artifacts, then re-point the
defaults ONLY if the numbers hold or improve (the K22 pattern: same-or-better accuracy, stricter
concept fidelity). Pre-fix artifacts kept, not overwritten.
"""
with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended blast-radius record")
print("  affected: K13 ladder, K23 see-only, talk asonly (w up to 170)")
print("  retrains running: lad_image_cls_fixedgrad, ktest_*_fixedgrad, asonly_v12000_fixedgrad")