"""record_image_run.py — the image model, the honest loss, and two wrong predictions.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "image")
r = json.load(open(os.path.join(P, "img1_log.json")))
rr, pp = r["results"]["readable"], r["results"]["plain"]
ra = rr["history"][-1]["test_acc"] * 100
pa = pp["history"][-1]["test_acc"] * 100
a = r["audit"]

LOG = f"""
---

## 2026-10-07 — THE IMAGE MODEL: BUILT, GRADIENT-CHECKED, AND IT LOSES.

### THE FUSION POINT

Text is `16 tokens x D -> concept layer`. Images are now `16 PATCHES x D -> concept layer`.
28x28 -> a 4x4 grid of 7x7 patches = 16 patches, deliberately the same count as the text window so
the two front-ends are interchangeable. **One concept layer, two front-ends.** That is the whole
basis for a from-scratch VL model, and the middle of the network does not know or care whether the
16 things came from a sentence or a picture.

New code: `lens/vision.py` (PatchEmbed, VisionNet, ImageConcept), a `reshape` op added to
`lens/tensor.py` with its backward, and `lens/scripts/image_concepts.py` (18 named pixel concepts,
each with its formula).

### IT LOSES, AND THAT IS THE RESULT

```
                 params      TEST acc
readable         16,665       83.49%
plain            16,705       86.42%
delta                          -2.93 points
concepts passing  9/13
```

**Readable is 2.93 points BEHIND a plain network of the same size.** This reproduces the project's
central finding on a second modality: on text the readable model paid ~12-14 ppl against its matched
normal; on images it pays 2.93 accuracy points. **Readability costs accuracy, in both places.**

The smoke (2 epochs, 6,000 images) had shown +0.16 for readable. That was 2 epochs of an
under-trained model, and reporting it as a trend would have been wrong. The trained result reverses
it and is larger. **A 2-epoch smoke does not predict a 24-epoch result.**

### THE AUDIT — 9/13, and the failures are informative

```
is_bright               AUROC 0.9995  fires 0.063  contrib 0.021  FAIL
left_half_heavier       AUROC 0.9993  fires 0.018  contrib 0.008  FAIL
has_high_contrast       AUROC 0.5356  fires 0.620  contrib 1.341  FAIL
centre_heavier          AUROC 0.4313  fires 0.499  contrib 1.043  FAIL
```

Four failures, three distinct causes, all worth keeping:

1. **`is_bright` learned it PERFECTLY (AUROC 0.9995) and the answer layer ignores it** (contrib
   0.021, bar 0.05). A faithful neuron that the model does not use. Fidelity and usefulness are
   different things, and this is the cleanest example the project has produced.
2. **`left_half_heavier` is perfect (0.9993) and fires on 1.8% of images** -- below the 2% floor.
   The floor is right and does not move, exactly as decided on the text side.
3. **`centre_heavier` INVERTED: AUROC 0.4313, worse than chance.** It fires more when its named
   concept is FALSE. Its support is 0.9688 -- it is true for 97% of images, so it is nearly a
   constant and there is barely any signal to learn. A concept that fires on almost everything
   cannot discriminate. Reading its name would mislead a human, which is precisely what the audit
   exists to catch.
4. `has_high_contrast` at 0.5356 is the same story more mildly (support 0.62).

### PREDICTIONS — TWO WRONG

1. **"readable 87-90%"** -> readable got 83.49%. **WRONG** (plain got 86.42%, inside the band).
2. **"readable within 1 point of plain"** -> **-2.93 points. WRONG, and I was too optimistic.**
   On the text side readability was expensive, so I should have predicted a loss, not a near-tie.
3. **"concepts above 7/13 but not 13/13"** -> 9/13. **HELD.**
4. **"left_half_heavier fails or is marginal"** -> fired 0.018, below floor. **HELD.**

### THE GRADIENT CHECK FAILED FIRST, AND THE BUG WAS MINE

The new `reshape` op and the image network are checked against central differences before use.
The first version reported **FAIL** and the fault was the test:

* it used **eps=1e-6**, but the `lens` engine is float32-only and the precision floor near 1.0 is
  **1.2e-07** -- so the perturbation was pure rounding noise, not a gradient measurement.
* an absolute 1e-6 tolerance is not achievable by any float32 computation.
* the project's own L1 gate already documented the right convention: **eps=1e-2, tol=1e-2**.

Corrected, with an eps sweep:
```
reshape alone                 2.534e-05   PASS
VisionNet at eps=3e-2         3.457e-03   PASS
VisionNet at eps=1e-6         FAIL        <- below the float32 floor
```
**And then this file's own summary text was wrong too**: it claimed the best eps sat "in the middle
of the sweep" and blamed truncation, while the table showed the best eps at the EDGE and a monotone
trend (smaller eps = worse, always). That was my claim contradicting my own measurement --
**the sixth time.**

### LICENCES

Faces: **every candidate rejected.** UTKFace, LFW, FFHQ and the rest either do not resolve, state no
licence, or are non-commercial. **Face recognition is blocked by the project's own licence rule**,
and no dataset was downloaded.
Captions: `HuggingFaceM4/COCO` CC BY 4.0, `textvqa` CC BY 4.0. Images: `medmnist-v2` CC BY 4.0,
`fashion_mnist` MIT.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("IMAGE MODEL — the numbers")
print("=" * 100)
print(f"  readable {rr['params']:,} params   TEST {ra:.2f}%")
print(f"  plain    {pp['params']:,} params   TEST {pa:.2f}%")
print(f"  delta    {ra-pa:+.2f} points      <- readable LOSES")
print(f"  concepts {a['n_pass']}/{a['n_concepts']}")
print()
print("  failures and their causes:")
for c in a["concepts"]:
    if not c["pass"]:
        au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
        why = []
        if c["auroc"] is not None and c["auroc"] < 0.55:
            why.append("AUROC below 0.55")
        if c["fires"] < 0.02:
            why.append("fires below the 2% floor")
        if c["contrib"] < 0.05:
            why.append("answer layer ignores it")
        print(f"    {c['name']:<24} AUROC {au}  fires {c['fires']:.3f}  -> {', '.join(why)}")
print("=" * 100)
