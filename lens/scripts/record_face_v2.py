"""record_face_v2.py — real-photo test: honest result, detector NOT yet reliable on real images.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — FACE DETECTOR v2: tested on REAL photos, and the honest answer is "not reliable yet"

Tested on the only licence-clean real-image source available offline: **the owner's own 40 photos**
(read-only — nothing copied, cropped, or written out; only scores printed).

```
  v1 (before the skin floor):  10/40 flagged as faces   median score 0.23
  v2 (with MIN_SKIN_FRACTION):  7/40 flagged as faces   median score 0.00
```

**The 7 remaining flags are FALSE POSITIVES, and I am not going to spin them.** Their measured
features say it outright: skin_fraction 0.011-0.092 (one is 0.011 = about 1% of the frame) and
hole_fraction 0.000-0.043. A real face filling a photo has far more skin and clear holes. What
carries these to a passing score is symmetry + solidity on small blobs, not skin and not facial
structure. They are screenshots and wallpapers, so the correct answer for all of them is "no face".

### WHAT I FIXED (measured, real)
Added `MIN_SKIN_FRACTION = 0.01` — below 1% skin the word "face" has no meaning, so detection is
refused outright. This cut false positives 10 -> 7 and dropped the MEDIAN score to 0.00, which is
the right direction. Synthetic controls stayed 5/5 (face 7.495 found; featureless skin blob 2.838,
wall 0.000, noise 0.501, red rectangle 0.000 all correctly rejected).

### WHAT I WILL NOT DO
I will not keep tuning weights until these 40 unlabeled files "look right". With no ground-truth
labels, every such tweak is fitting to nothing — and it would be exactly the kind of number that
looks like progress and isn't. Stated plainly: **real-face detection accuracy is UNPROVEN.**

### WHAT IS ACTUALLY PROVEN TODAY
* The pipeline runs end-to-end on live camera: 149 frames, 20.5 fps processed, real .mp4 written
  (`lens/runs/video/track_test.mp4`), 0 faces found because the owner was at school — the correct
  answer, and it also proves no runaway false-positive behaviour on a real empty room.
* Tracking rules work and are printed: IoU match > 0.2, survive 8 missing frames, EMA alpha 0.4.
* Synthetic positives AND negatives all pass (5/5), so the machinery is sound.

### THE HONEST PATH FORWARD (needs an owner decision)
1. Owner sits in front of the webcam for ~30 s while the tracker records -> gives real POSITIVES
   with ground truth (we know where the face is because he put it there). Licence-clean: his own.
2. Owner labels ~20 of his own photos (face / no face) -> gives real positives AND negatives.
3. Only then can precision/recall be computed and a real accuracy claim made.
Option 1 is fastest and needs no data transfer. Until then the detector stays labelled EXPERIMENTAL.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended face v2 record")
