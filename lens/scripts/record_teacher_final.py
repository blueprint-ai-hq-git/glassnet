"""record_teacher_final.py — distillation: t2 failed 3 arms; the mechanism is now MEASURED absent.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d1 = json.load(open(os.path.join(P, "lens", "runs", "teacher", "teacher_on_f0.1.json")))
d2 = json.load(open(os.path.join(P, "lens", "runs", "teacher", "teacher_soft_f0.1.json")))

LOG = """
---

## 2026-10-08 — TEACHER LOOP, FINAL VERDICT AT THIS RUNG: t2 fails, and we now KNOW why

### THE THREE ARMS TESTED (all t1 PASS, all t2 FAIL)

```
  experiment                          A (10% real)   B (teacher)   t2 margin    t1
  argmax labels, same-pool teacher      66.54±0.22    66.60±0.22     +0.06      PASS (within 9.0)
  SOFT labels, same-pool teacher        66.52±0.25    65.88±0.29     -0.64      PASS (within 9.8)
  SOFT labels, DISJOINT d128 teacher    66.52±0.25    66.06±0.72     -0.45      PASS (within 9.6)
```

The disjoint teacher (d128, 20k images from 10000+, 78.00% of its own val, 4/4 gates, sha
b348ad70c8dad71c) has NEVER SEEN a student row, so memorised-regurgitation is ruled out. It still
fails t2. That closes the mechanism question.

### THE MEASUREMENT THAT ENDS IT (diagnostic on the 3,824 student-pool rows)

```
  teacher top1 mean:                          0.769
  teacher argmax agrees with the REAL label:  77.8%
  probability mass on the TRUE word:          mean 0.743   (median 0.998)
  on its WRONG rows (22.2%): mass on truth:   0.226
```

**A one-hot real label puts 1.000 on the truth on EVERY row. The teacher puts 0.743.**
The student's targets here are a DETERMINISTIC function of the image (the class noun after
"this is a"), so ground truth strictly dominates any estimate of it. Distillation earns its keep
when labels are NOISY, AMBIGUOUS, or EXPENSIVE — on this task none of the three is true, and the
measurement says so. The negative result is honest and final at this rung:

* **"less data but smart": SHOWN** — 10% of the data plus ANY of these teachers recovers to within
  9.0-9.8 points of the 100%-data ceiling (t1 three times, 3 seeds).
* **"teacher beats the same real data": NOT AVAILABLE on deterministic-label tasks**, and we can
  now name the boundary instead of retrying blindly: the teacher must know something the label
  cannot say. That is where the next experiment goes (below), not a retry loop.

### BUGS CAUGHT BY THE EXPERIMENT ITSELF (kept, as always)
1. Control bug: first teacher_loop run trained BOTH arms on teacher labels (identical rows) —
   the +0.00 result was the experiment having no control. Fixed and stated before any claim.
2. VOCAB-ORDER INTEGRITY: the disjoint teacher's vocab has the same WORDS as the student's but a
   DIFFERENT ORDER (pool frequencies differ). Without the remap-by-word-string, every soft label
   would have been a silently WRONG WORD — the camera-vocab-bug class, caught by the check that
   exists because of that bug. (Log line: `vocab order identical: False`, remap applied.)
3. distill_kl op verified END-TO-END against a float64 reference (worst rel 2.5e-06, all seven
   parameter groups, tau=2) after the float32 value-FD probe produced two FALSE FAILURES
   (rel 2.3e-2 at eps 1e-2; exactly 1.000 at eps 1e-4 where the FD signal dropped below the
   float32 rounding floor of the loss value). The op was never wrong; the probe was — and the
   replacement probe is strictly stronger (checks wiring through embedding/reshape/concat too).

### THE PRE-REGISTERED NEXT ARM (so "final at this rung" is not "final, full stop")
Where the label CANNOT dominate the teacher: **multi-word captions with template variety** — the
teacher sees image+prefix and knows which NOUN-FOLLOWING CONTINUATIONS are plausible ("looks like
a bag" vs "this is a bag"): distributional knowledge the one-hot next word does not carry. Also
open-vocab captions (COCO) where true ambiguity exists. Test stays the same: t2, 3 seeds, same bars.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars")
print("=" * 90)
print("  t1 PASS x3 (10%+teacher within 9.0-9.8 of full data): less-data-but-smart SHOWN")
print("  t2 FAIL x3 including disjoint d128 teacher: mechanism measured absent (0.743 < 1.000)")
print("  next pre-registered: arms where the label cannot dominate the teacher")
print("=" * 90)
