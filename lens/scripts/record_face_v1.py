"""record_face_v1.py — from-scratch face detection: built, failed twice, fixed, 5/5 on controls.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — FROM-SCRATCH FACE DETECTION (webcam, no dataset, no pretrained weights)

Built `face_concepts.py`: face detection as NAMED, MEASURABLE features, every formula readable and
every weight printed. No Haar cascade, no OpenCV (cv2 is absent on this machine and using its
cascades would be pretrained weights anyway), no dataset required.

Features (each with its exact formula in the file): skin_fraction, skin_cluster, skin_centred,
blob_aspect, blob_solidity, hole_fraction, vertical_symmetry, eye_dark_bands, mouth_dark_band.
Detector = transparent weighted sum + a structure gate. Weights are STATED CONSTANTS, not trained.

### IT FAILED TWICE, AND THAT IS THE USEFUL PART

**Failure 1 — it was a colour detector wearing a face detector's name.**
First weight set scored a FEATURELESS skin blob (8.20) ABOVE a face with eyes and mouth (7.70).
skin_cluster + blob_solidity + vertical_symmetry are all satisfiable by a plain ellipse, and the
eye/mouth terms were too small to matter. A detector that says "face" to any skin-coloured oval is
worthless. (Caught by including a NEGATIVE CONTROL in the test — a skin blob with no features.)

**Failure 2 — the eye feature could never fire, by construction.**
`eye_dark_bands` was measured over pixels that passed the SKIN rule. But eyes and mouth are DARK,
so they FAIL the skin rule — they are HOLES in the skin blob, not part of it. Measured proof: every
row of the face region came out 0.573, std 0.000. The feature was mathematically incapable of
detecting what it was named after.

### THE FIX (two changes, both measured)
1. `fill_holes()` — flood-fill the blob's interior so the face region INCLUDES eyes and mouth.
2. `hole_fraction` becomes its own weighted concept, because the holes ARE the facial features:
   measured face 0.111 vs featureless blob 0.000 — a clean separator that colour cannot fake.

### RESULT (5/5 known cases correct, real run)
```
  synthetic face WITH eyes+mouth   found=True   score 7.495   CORRECT
  synthetic skin blob, NO features found=False  score 2.838   CORRECT  <- the hard negative
  flat grey wall                   found=False  score 0.000   CORRECT
  random noise                     found=False  score 0.501   CORRECT
  red rectangle                    found=False  score 0.000   CORRECT
  REAL webcam frame (owner absent) found=False  score 0.000   (no face in frame — consistent)
```

### HONEST LIMITS, STATED
* The positive is drawn BY HAND, so 5/5 tests the MACHINERY, not real-world accuracy. Real accuracy
  needs real faces — which means either the owner's own captures or a licence-clean dataset.
* LICENCE REALITY (checked, not assumed): CelebA, LFW, UTKFace, VGGFace2 are research/non-commercial;
  FFHQ is NC; FairFace's annotations are CC BY 4.0 but its images come from YFCC100M with unclear
  per-image rights. ALL FAIL OR ARE AMBIGUOUS under the standing safe-licence rule. Own webcam
  captures pass. This is why TRACKING (no identities, no dataset) is the honest thing to ship first
  and RECOGNITION needs an owner decision.
* Webcam verified working headlessly: ffmpeg v4l2, 640x480 YUY2, real frame captured (154 KB).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended face v1 record")
