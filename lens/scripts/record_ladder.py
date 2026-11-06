"""record_ladder.py — the ladder results and the camera, written into LOG.md.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
L = os.path.join(PROJ, "lens", "runs", "ladder")

rows = {}
for rung in ("tiny", "small", "medium"):
    p = os.path.join(L, f"ladder_{rung}.json")
    if os.path.exists(p):
        rows[rung] = json.load(open(p))

lines = []
for rung, t in rows.items():
    n = sum(r["passed"] for r in t)
    lines.append(f"  {rung:<8} {n}/{len(t)}   " + "  ".join(
        f"{r['capability']}:{'PASS' if r['passed'] else 'FAIL'}" for r in t))

LOG = f"""
---

## 2026-10-08 — THE LADDER, THE VIRTUAL CAMERA, AND A CORRECTION

Owner's instruction: "don't do it first small to big testing everything ability everything".
I had jumped straight to full-size GPU runs. This is the fix.

### THE LADDER — every capability, small first, then stepped up

```
{chr(10).join(lines)}
```

Gates are BINARY and were fixed BEFORE each run:
  text LM    final val ppl < 2x chance AND loss fell
  image cls  test acc > 2x chance (10%)
  VL / VQA   test acc > 2x chance (8.3%)
  code LM    final val ppl < 2x chance
  explain    residual < 1e-5
  gradcheck  worst relative diff <= 1e-2
  audit      runs and produces a number for every concept

**At the tiny rung, text_lm and image_cls FAILED.** Text: loss fell 9.37 -> 9.09 but ppl 8339 against
a chance of 3500. Image: 9.94% against a chance of 10% -- random. **Neither failure was reported as
broken code and neither was reworded.** At the small rung both pass on the same code, so they were
under-trained, not broken. That distinction is only available because the rung exists.

**Why this matters:** a failure at 60,000 samples and 24 epochs tells you far less than a failure at
2,000 samples and 2 epochs, because you cannot separate a broken code path from an under-trained
model. The ladder makes the cheap failure happen first.

### THE VIRTUAL CAMERA — built and working

`lens/scripts/virtual_camera.py` + `lens/scripts/frameio.py`.

There are NO image libraries on this machine (no PIL, no cv2, no imageio) and none are on the
project's allowed list. So:
  * `frameio.py` contains a **5x7 bitmap font written out by hand** (A-Z, 0-9, punctuation) and
    draws text, boxes and bars directly into a NumPy RGB array
  * frames are piped as raw rgb24 into **ffmpeg**, which was already on the machine, producing a
    real h264 .mp4
  * sources: `synthetic` (moving shapes, generated in code), `fashion` (MIT), `pathmnist` (CC BY 4.0,
    medical), or any `video` file decoded with ffmpeg

**NO PHYSICAL CAMERA IS TOUCHED ANYWHERE.**

Result: `cam_fashion.mp4`, 60 frames, 940x460 @ 8fps, the model's own answer, confidence and firing
concepts drawn on every frame. **41/60 correct (68.3%).**

A frame was verified by reading it back rather than assumed:
```
VIRTUAL CAMERA / FRAME 14 LIVE
MODEL SAYS  BAG          CONFIDENCE 39.7%
TRUTH       BAG
Q WHAT DO YOU SEE
NAMED CONCEPTS FIRING
  HAS_HIGH_CONTRAST 0.99   BOTTOM_HALF_HEAVIER 0.99   ASKS_WHAT 0.98
  MOSTLY_INK 0.94          IS_BRIGHT 0.82            IS_THIN_STRUCTURE 0.56
```

### A CORRECTION I OWE

The first camera build fed `Qtr[0]` -- **the same question on every frame**. That was both a bad demo
and it HID a defect. Reading one frame, I saw `ASKS_NAME 0.89` on a "what is in the photo" question
and reported to the owner that **the text concepts are mislabeled**.

**That was one frame, and I generalised from it.** Fixed the camera to cycle questions per frame, then
measured the text concepts properly:

```
             AUROC     fires    support   verdict
asks_what    0.5352    0.608    0.500     FAIL      <- 0.5 is a coin flip
asks_name    0.6343    0.064    0.250     PASS
asks_see     0.5859    0.291    0.250     PASS
asks_identify 0.6300   0.182    0.250     PASS
```

**My specific claim was wrong, and the real finding is worse: EVERY text concept sits at AUROC
0.53-0.63, barely above chance, while image concepts reach 0.95-1.00.** The text half of the fused
layer is carrying almost nothing. "Passing" the 0.55 bar is not the same as being meaningful, and the
bar should not be read as a quality certificate.

**This is the seventh time a claim of mine has outrun its evidence** -- and the first time it ran the
OTHER way, inventing a defect that was not the actual defect. The lesson is the same as the other six:
a claim attached to a measurement that cannot support it. One frame is not a measurement.

### STILL BLOCKED

**Faces: every candidate rejected** (UTKFace, LFW, FFHQ -- no resolvable dataset, no stated licence,
or non-commercial). Nothing downloaded. The project's own licence rule blocks face recognition, and
"beats a 500M model" is ~1 year of continuous training by the measured ceiling.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
for rung, t in rows.items():
    n = sum(r["passed"] for r in t)
    print(f"  {rung:<8} {n}/{len(t)}")
