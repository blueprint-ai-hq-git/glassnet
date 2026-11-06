"""record_day_summary.py — what was built and measured on 2026-10-08.
"""
import json
import os

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
V = os.path.join(PROJ, "lens", "runs", "vl")
IMG = os.path.join(PROJ, "lens", "runs", "image")
LAD = os.path.join(PROJ, "lens", "runs", "ladder")


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


LOG = """
---

## 2026-10-08 — THE VISION DAY: WHAT WAS BUILT, WHAT WORKED, WHAT FAILED

### BUILT TODAY

**`lens/vision.py`** — the image front-end on the SAME concept layer as text.
An image becomes 16 PATCHES, exactly like a 16-word text window, so the middle of the network does
not know whether the 16 things came from a sentence or a picture. That is the from-scratch VL design.

**`lens/scripts/frameio.py`** — video I/O with NO image libraries. This machine has no PIL, no cv2,
no imageio, and none are on the project's allowed list. So: a **5x7 bitmap font written by hand**,
drawn straight into a NumPy RGB array, piped as raw rgb24 into ffmpeg (already installed).

**`lens/scripts/virtual_camera.py`** — the live-video pipeline. Frames in, the model's own answer,
confidence and firing concepts drawn on, real .mp4 out. **No physical camera is touched.** Verified
by reading a frame back rather than assuming: `MODEL SAYS BAG / CONFIDENCE 39.7% / TRUTH BAG /
Q WHAT DO YOU SEE / ASKS_WHAT 0.98`.

**`lens/scripts/ladder.py`** — every capability, tiny rung first, with BINARY gates fixed before
each run.

**New gradient-checked ops:** `reshape` and `concat` in `lens/tensor.py`, and class-weighted
cross-entropy. All checked numerically before use.

### THE LADDER — small to big, as instructed

```
RUNG TINY    5/7   text_lm FAIL (ppl 8339 vs 3500), image_cls FAIL (9.94% vs 10%)
RUNG SMALL   7/7   text_lm ppl 1162 | image 65.81% | vl 72.95% | code ppl 439
RUNG MEDIUM  7/7   text_lm ppl 484.95 | image 82.52% | vl 80.84% | code ppl 185.51
RUNG BIG     7/7   BUT SEE THE BUG BELOW
```

The tiny failures were under-training, and only the ladder makes that visible: at 2,000 samples and
2 epochs a broken code path and an untrained model look identical.

### BUG 32 — THE "BIG" RUNG WAS NOT BIG

```
RUNG SMALL   text_lm ppl 1162.0    code_lm ppl 439.27
RUNG MEDIUM  text_lm ppl  484.95   code_lm ppl 185.51
RUNG BIG     text_lm ppl 1722.39   code_lm ppl 533.63    <- WORSE than small
```

Worse at big than at small is impossible unless big is not big. It computed
`--max-windows max(2000, n*2)` and for the big rung `n=0` (meaning "full dataset"), so it evaluated
to **2000** -- LESS than the small rung's 20,000. **Two of seven capabilities trained on less data at
"big" than at "small", and the weak ppl gate (bar 7000) passed them anyway.**

Fixed: a full rung gets a genuinely large budget. Re-run launched.

**This is the failure mode the ladder was built to catch, and it caught it -- in the ladder.**
A gate that passes a run which cannot be big is a gate that is not measuring size.

### THE VL FIX — THE ONE CLEAN WIN OF THE DAY

The first VL design had all four questions asking for the same answer, so the question carried no
information and the text concepts sat at **AUROC 0.53-0.63** (a coin flip is 0.50).

Made the questions matter (`is this footwear` -> yes/no, `name the object` -> the class name):

```
                          text-concept   image-concept   TEST acc
vqa1 (questions same)       0.5963         0.5684        69.47%
vqa2 (questions matter)     0.9640         0.8560        88.13%
```

**Prediction: "above 0.70". Measured: 0.9640. Held.**

### MEDICAL — TWO FAILURES, A DIAGNOSIS, AND A FIX

PathMNIST (CC BY 4.0, colon histology, 9 tissue classes).

```
attempt                 text AUROC  image AUROC     TEST  concepts   vs majority
1 UNBALANCED                0.5035       0.4692   41.41%     0/16     at   (41.41%)
2 BALANCED                  0.0000       0.3545    8.68%     0/16     below
3 ADAPTIVE concepts         0.9962       0.6842   52.87%     8/20     ABOVE
```

**Attempt 1**: TEST equalled the majority baseline to the digit -- it answered "no" to everything.

**Attempt 2**: I hypothesised class imbalance and added weighted cross-entropy. **It was WORSE** --
it collapsed onto a different single class. Both predictions wrong. Reported as a failure.

**Then a DIAGNOSTIC instead of a third guess**, which found three causes:
  * `is_bright` was true for **93.5%** of histology images, `has_low_contrast` 82.8%, `mostly_ink`
    92.4% -- the concept thresholds were chosen for Fashion-MNIST;
  * a near-constant concept gets `pos_weight` up to **19.96**, and its balanced BCE at init
    (`1.31 x 0.3 = 0.39`) competed with the answer loss (`2.56`);
  * **and separately**: PathMNIST RGB channel means are **188.8 / 136.0 / 180.0** and this loader
    AVERAGES them -- destroying the stain colour that separates tissue types. Not yet fixed.

**Attempt 3** fitted every threshold on TRAIN as a quantile (`is_bright = mean > q50 of TRAIN
means`). Concept set alone: **12 kept / 6 dropped / max pos_weight 19.96  ->  16 kept / 0 dropped /
max pos_weight 5.67.** Result 52.87%, above the majority baseline, **readable +11.46 over plain**
(the plain control sat exactly ON the majority baseline), text concepts 0.9962.

**Three predictions, three held.**

**Guessing twice cost two runs. Measuring once fixed it.**

### WHAT IS STILL WEAK

* **Medical image concepts: 5/16, mean AUROC 0.6842.** Still Fashion-MNIST-shaped (brightness, edges,
  symmetry) on histology texture where the signal is COLOUR and the loader cannot see it.
* **The 500M-parameter comparison is not reachable** -- the measured ceiling puts a 500M model at
  about **one year** of continuous training on this GPU. Nothing today changes that.
* **Faces remain blocked** by the project's own licence rule; no dataset downloaded.
* Code execution correctness: **still 0 verified correct**. Unchanged, and not claimed otherwise.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

print("=" * 100)
print("TODAY, IN ONE TABLE")
print("=" * 100)
rows = []
for tag, p in (("vl vqa1 (fashion)", os.path.join(V, "vlsmoke_log.json")),
               ("vl vqa2 (fashion)", os.path.join(V, "vqa2_small_log.json")),
               ("vl vqa3 (fashion)", None),
               ("medical att.1", os.path.join(V, "med_small_log.json")),
               ("medical att.2", os.path.join(V, "med_bal_log.json")),
               ("medical att.3", os.path.join(V, "med_adapt_log.json"))):
    if p is None:
        continue
    r = j(p)
    if not r:
        continue
    a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    pl = r["results"]["plain"]["history"][-1]["test_acc"] * 100
    print(f"  {tag:<20} txt {np.mean(txt):.4f}  img {np.mean(img):.4f}  "
          f"TEST {acc:6.2f}%  plain {pl:6.2f}%  concepts {a['n_pass']}/{a['n_concepts']}")
print("=" * 100)
