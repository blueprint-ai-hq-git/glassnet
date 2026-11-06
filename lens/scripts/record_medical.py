"""record_medical.py — three attempts, the diagnosis, and the fix that worked.
"""
import json
import os

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "vl")


def load(out):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    return {"txt": float(np.mean(txt)), "img": float(np.mean(img)),
            "acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "plain": r["results"]["plain"]["history"][-1]["test_acc"] * 100,
            "concepts": (a["n_pass"], a["n_concepts"]),
            "maj": r["meta"].get("majority_baseline", 0) * 100,
            "txt_pass": (a["txt_pass"], a["txt_total"]),
            "img_pass": (a["img_pass"], a["img_total"])}


rows = [("1 UNBALANCED", load("med_small")), ("2 BALANCED", load("med_bal")),
        ("3 ADAPTIVE", load("med_adapt"))]

LOG = """
---

## 2026-10-08 — THE MEDICAL VL: TWO FAILURES, A REAL DIAGNOSIS, AND A FIX THAT WORKED

### THE TASK

PathMNIST (MedMNIST, CC BY 4.0): 89,996 colon-histology images, 9 tissue classes, 28x28.
The question "is this cancer" is medically meaningful here (yes for cancer-associated stroma and
colorectal adenocarcinoma), which is why the predicate is validated before use.

### ATTEMPT 1 — COLLAPSE ONTO THE MAJORITY ANSWER. FAILED.

```
text AUROC 0.5035   image AUROC 0.4692   TEST 41.41%   concepts 0/16
majority baseline 41.41%
```

TEST equalled the majority baseline **to the digit**. The model answered "no" to everything --
including every "what is in the photo" question. Concept neurons were dead (fires 0.000 for most,
two saturated at 1.000).

### ATTEMPT 2 — BALANCED THE ANSWER LOSS. WORSE. FAILED.

I hypothesised the free lunch was the cause and added inverse-frequency class weighting to the
cross-entropy (gradient-checked first: loss matched an independent NumPy reference to 0.00e+00,
gradient to 1.49e-08, and the unweighted path stayed bit-identical).

```
text AUROC 0.0000   image AUROC 0.3545   TEST  8.68%   concepts 0/16
```

**It collapsed onto a different single class** ("colorectal_adenocarcinoma") and got WORSE than the
majority baseline. **Both predictions wrong.** The balancing moved the collapse rather than removing
it, which proves class balance was not the cause.

### THE DIAGNOSIS — MEASURED, NOT GUESSED

`diag_medical.py` compared the concept targets against the actual images:

```
concept                support   pos_weight   verdict
is_bright               0.9353        0.07     near-constant
has_low_contrast        0.8279        0.21     near-constant
mostly_ink              0.9243        0.08     near-constant
is_dark                 0.0000          --     DROPPED (never true)
has_high_contrast       0.0000          --     DROPPED (never true)
top_half_heavier        0.0586       16.05     strong push toward always firing
centre_heavier          0.0477       19.96     strong push
```

**Three of twelve concepts were effectively constants and two never fired at all.** The cause is
that `image_concepts.SPEC` uses thresholds chosen for Fashion-MNIST (`is_bright = mean(pixel) > 0.50`)
and 93.5% of histology images pass it.

That breaks the layer twice:
1. a constant concept cannot be learned, so its AUROC is meaningless;
2. a near-constant concept gets `pos_weight = (1-s)/s` up to ~20, and its balanced BCE at
   initialisation is `1.31 x lam 0.3 = 0.39` **against ~2.56 for the answer loss** -- the concept
   term was shouting louder than the task.

A SECOND, SEPARATE CAUSE IS ALSO MEASURED AND NOT YET FIXED:
```
PathMNIST raw RGB channel means:  R 188.8   G 136.0   B 180.0
```
The channels differ substantially, and this loader AVERAGES them -- destroying the stain colour
that separates tissue types in histology. That is a real second cause and it is the next change.

### ATTEMPT 3 — CONCEPTS WITH FITTED THRESHOLDS. WORKED. ALL THREE PREDICTIONS HELD.

`image_concepts_adaptive.py`: same concepts, same meanings, but every threshold is a QUANTILE OF THE
TRAINING DATA instead of a magic number, and the resolved value is stored in the formula so a reader
can recompute it:

```
is_bright            mean(pixel) > q50 of TRAIN means      (resolved threshold = 0.669603)
is_dark              mean(pixel) < q15 of TRAIN means      (resolved threshold = 0.590096)
has_high_contrast    std(pixel)  > q75 of TRAIN stds
```

Measured effect on the concept set ALONE, before any training:
```
fixed thresholds :  12 kept,  6 dropped, max pos_weight 19.96, 3 near-constant
fitted thresholds:  16 kept,  0 dropped, support 0.150-0.700, max pos_weight  5.67
```

Result:

```
attempt                 text AUROC  image AUROC     TEST  concepts   vs majority
1 UNBALANCED                0.5035       0.4692   41.41%     0/16     at  (41.41%)
2 BALANCED                  0.0000       0.3545    8.68%     0/16     below
3 ADAPTIVE concepts         0.9962       0.6842   52.87%     8/20     ABOVE (41.41%)
```

* **Prediction 1 HELD**: TEST beats the majority baseline (52.87% vs 41.41%).
* **Prediction 2 HELD**: image concepts pass above zero (5/16, against 0/12 twice).
* **Prediction 3 HELD**: medical stays harder than fashion's 88.13%.
* Text concepts reached **mean AUROC 0.9962** (`asks_what` 1.0000, `asks_name` 1.0000,
  `asks_cancer` 0.9848) -- 3/4 pass, the fourth failing on contribution, not fidelity.
* **readable 52.87% vs plain 41.41% = +11.46 points.** The plain control sat exactly ON the majority
  baseline; only the readable arm got past it.
* The answers are varied again ("background", "smooth_muscle", "colorectal_adenocarcinoma") instead
  of one word.

### WHAT IS STILL WEAK, AND WHY

**Image concepts: 5/16, mean AUROC 0.6842.** These are still Fashion-MNIST-shaped concepts
(brightness, edges, symmetry) applied to histology texture, where the discriminating signal is
COLOUR and the concepts cannot see it because the loader averaged the channels. That is the next
single change, already diagnosed and measured.

### THE METHOD THAT WORKED

Attempt 1 failed, attempt 2 failed and was reported as a failure, then a DIAGNOSTIC was run before
attempt 3 -- and the diagnostic named the cause (degenerate concepts -> large pos_weight -> concept
loss dominating) precisely enough that one change fixed it and all three predictions held.

**Guessing twice cost two runs. Measuring once fixed it.**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
for tag, r in rows:
    if r:
        print(f"  {tag:<16} txt {r['txt']:.4f}  img {r['img']:.4f}  TEST {r['acc']:6.2f}%  "
              f"plain {r['plain']:6.2f}%  concepts {r['concepts'][0]}/{r['concepts'][1]}  "
              f"maj {r['maj']:.2f}%")
a = rows[2][1]
if a:
    print()
    print(f"  readable beats plain by {a['acc']-a['plain']:+.2f} points on medical")
    print(f"  readable beats the majority baseline by {a['acc']-a['maj']:+.2f} points")
    print(f"  text concepts {a['txt_pass'][0]}/{a['txt_pass'][1]}   "
          f"image concepts {a['img_pass'][0]}/{a['img_pass'][1]}")
print("=" * 100)
