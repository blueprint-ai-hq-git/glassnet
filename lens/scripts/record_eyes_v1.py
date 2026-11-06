"""record_eyes_v1.py — the FIRST eyes56 run failed its own premise, kept as evidence.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "eyes", "eyes56.json")))

LOG = f"""
---

## 2026-10-08 — EYES EXPERIMENT v1: e1/e2 FAILED, AND THE REASON KILLED THE EXPERIMENT'S PREMISE

```
  28px: overall 74.44±1.08   detail-pair 100.00±0.00   size-independent 49.70±1.52
  56px: overall 76.11±9.66   detail-pair  92.72±2.01   size-independent 61.00±17.38
  e1 56 overall > 28+5:    +1.67  FAIL
  e2 56 detail  > 28+10:   -7.28  FAIL
  e3 no cost to 28 classes: +11.30 PASS
```

**e2's numbers are the whole story: the 28-pixel model scored 100.00% on the detail pairs.** The
experiment was built on the assumption "these classes differ ONLY in detail that 28 pixels cannot
resolve" — and that assumption was never measured. At 28px the shapes were GENERATED NATIVELY at
28, so the thin cross at 28 is a 1-px stroke AT 28: perfectly visible to 28-pixel eyes. The two
arms saw DIFFERENT IMAGES, not one image at two resolutions. As designed, the test could not
detect a resolution benefit no matter what was true — the same class of design flaw as the
"28 destroys detail" claim I made about the video.

**Three more design bugs caught before the rerun:**
1. thincross at 28 was an EMPTY image (zero-width slice `c-0:c+0`) — the class did not exist. A
   generator bug would have "proven" the physics claim.
2. `SeeTalkNet` built PatchEmbed without `image=S`, so the 56 arm mis-batched and crashed concat —
   fixed with image/patch stored in the checkpoint meta (old checkpoints still load: defaults
   match how they were trained).
3. `np.diff` shortens its axis: adding the two diff arrays broadcast-failed. Measured supports
   printed for every concept (nothing at 0.5 assumed); one dead concept (corners never fire on
   these shapes) replaced with a measured one.

**v2 design (now running): ONE source dataset at 56px; the 28 arm receives the PHYSICAL 2x2 mean
pool — exactly what a real 28-pixel camera would capture. Same image, two resolutions, honest
gates (same bars e1/e2/e3).**
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
