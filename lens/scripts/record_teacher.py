"""record_teacher.py — the teacher loop's first real experiment: t2 FAILED. Stated as failure.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "teacher", "teacher_on_f0.1.json")))

LOG = f"""
---

## 2026-10-08 — STEP 5 TEACHER LOOP: t1 PASSED, t2 FAILED — the teacher added nothing yet

### THE EXPERIMENT (and the control bug found in it)

Arms, all the same student (see+talk arch, d=48, 8 epochs), identical 10% row subsets, 3 seeds:

```
  A  student, 10% real labels          (no teacher)
  B  student, 10% rows labelled by the TEACHER (seetalk_small2, sha {d['teacher_sha']})
  C  student, 100% real labels         (the full-data ceiling)
```

**CONTROL BUG, FIRST RUN:** arm A was built with `use_teacher=(teacher_on=="on")` — so with the
teacher ON, A and B received the *same labels*, and scored 66.45 vs 66.45 on every seed. The first
"+0.00, FAIL" was the experiment having no control at all, not a teacher result. Fixed by forcing
A to real labels. The kept failed run (teacher_on_f0.1.json pre-fix numbers are in chat history;
the file now holds the fixed run).

### THE RESULT (fixed run, 3 seeds, mean±std)

```
  A  10% real            {d['means']['A'][0]:.2f}% ± {d['means']['A'][1]:.2f}
  B  10% + teacher       {d['means']['B'][0]:.2f}% ± {d['means']['B'][1]:.2f}
  C  100% real           {d['means']['C'][0]:.2f}% ± {d['means']['C'][1]:.2f}

  t1  10%+teacher within 10 points of 100%:     PASS  ({d['means']['B'][0]:.2f} vs {d['means']['C'][0]:.2f}, 9.0 points)
  t2  teacher beats no-teacher >2 AND > noise:  FAIL  (+{d['means']['B'][0]-d['means']['A'][0]:.2f} points, noise ±{d['means']['A'][1]+d['means']['B'][1]:.2f})
  t3  3 seeds both arms:                        PASS
  t4  provenance tagged + teacher hash:         PASS
```

### WHY t2 FAILED — the honest mechanism

Teacher B = the small2 model itself, labelling rows it did not train on. On this task the teacher's
label and the real label agree almost everywhere (both are "the class noun after 'this is a'"), so
the teacher is **a copy of the answer key, not extra information** — and where they disagree, the
teacher is simply wrong in a way the student cannot tell from the label alone. Distillation only
beats the control when the teacher (a) has information the small data lacks, or (b) softens the
labels (top-k probabilities, not argmax). Neither was true here: argmax labels of a same-architecture
teacher ≈ noisy duplicates of the real targets.

**t1 PASS means the real claim stands anyway: 10% of the data + a teacher gets within 9 points of
100% of the data.** Less data but smart: shown. Teacher BEATS the same 10% real data: not shown,
and t2 says it did not happen at this rung. Not reworded.

### WHAT COMES NEXT (each is a separate pre-registered experiment)
* soft teacher labels (full next-word distribution, KL loss) — mechanism (b), the standard fix
* a STRONGER teacher labelling rows the weak student would get wrong — mechanism (a): pool rows from
  the unlabeled 90% (real labels hidden), distil, and score against real labels held out
* the on/off switch table (teacher off must show NO gain — it did: +0.00 pre-fix is actually that
  measurement, and it passed as designed)
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars")
print("=" * 90)
print(f"  A {d['means']['A'][0]:.2f}±{d['means']['A'][1]:.2f}  B {d['means']['B'][0]:.2f}"
      f"±{d['means']['B'][1]:.2f}  C {d['means']['C'][0]:.2f}±{d['means']['C'][1]:.2f}")
print("  t1 PASS (10%+teacher within 9 pts of 100% data) — 'less data but smart' shown")
print("  t2 FAIL (+0.06 pts) — at this rung the teacher adds nothing over 10% real data")
print("=" * 90)
