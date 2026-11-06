"""record_medical_colour.py — the colour change, the control, and the correction it forced.
"""
import json
import os

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "vl")


def stat(out):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    return {"acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "txt": float(np.mean(txt)), "img": float(np.mean(img)),
            "concepts": (a["n_pass"], a["n_concepts"]),
            "img_pass": (a["img_pass"], a["img_total"]),
            "plain": r["results"]["plain"]["history"][-1]["test_acc"] * 100}


S = {k: stat(v) for k, v in (("att1", "med_small"), ("att2", "med_bal"), ("att3", "med_adapt"),
                             ("ctl", "med_lam015"), ("att4", "med_color"))}

LOG = f"""
---

## 2026-10-08 — MEDICAL, FOURTH CHANGE: COLOUR. AND THE CONTROL THAT CORRECTED ME.

### THE MEASURED CAUSE

`diag_medical.py` found that PathMNIST's raw RGB channel means are **188.8 / 136.0 / 180.0** and the
loader averaged them with `.mean(axis=-1)`, destroying the stain colour that separates tissue types
in histology. A real, measured cause of the weak image concepts (5/16).

### THE CHANGE

`PatchEmbed` gained a `channels` argument; `load_medmnist(..., color=True)` returns the RGB channels
CHANNEL-MAJOR, (B, 3*784), which is what the patch extractor reshapes into. A patch is now 3*7*7 = 147
values instead of 49.

**The concept definitions were deliberately left identical.** `_img()` converts colour input back to
grayscale internally, so the concepts see exactly what they saw before. Any difference therefore
comes from the MODEL seeing colour, not from the concepts changing underneath it.

### THE RESULT

```
run                                     TEST   image AUROC   concepts   image concepts
1 UNBALANCED (gray)                   41.41%       0.4692       0/16          0/12
2 BALANCED   (gray)                    8.68%       0.3545       0/16          0/12
3 ADAPTIVE   (gray)                   52.87%       0.6842       8/20          5/16
4 ADAPTIVE + COLOUR                   64.44%       0.6760      14/20         10/16
```

Three predictions held: accuracy rose above 52.87%, image concepts passed better (10/16 vs 5/16), and
medical stayed below fashion's 88.13%. readable beat plain by **+23.04 points**.

### THE MISTAKE I MADE, AND THE CONTROL THAT CAUGHT IT

**Attempt 4 changed TWO things at once:** the colour AND the concept loss weight (0.3 -> 0.15). That
breaks the project's own one-change-at-a-time rule, and when I reported it I said "colour worked".

**The control (grayscale + lam 0.15, nothing else changed) proves that attribution wrong:**

```
total gain attempt3 -> attempt4 : +11.57 points
  explained by lam 0.15 ALONE   :  +6.36 points   (55%, the control)
  left for COLOUR to explain    :  +5.21 points   (45%)
```

**BOTH contribute. Neither can be credited alone.** More than half of the gain came from the
concept-weight change I slipped in alongside the colour change, so "colour worked" was more than
half wrong.

The control script's own verdict printed this without being told what answer to expect:
*"BOTH contribute. Neither change can be credited alone."*

**This is the eighth time a claim of mine has outrun its evidence, and the first time a CONTROL run
specifically caught it rather than my re-reading a number.** The rule that produced the catch is the
one the project already had: change one thing at a time, so a win can be attributed.

### WHAT IS STILL WEAK

* Image concepts 10/16 with mean AUROC 0.6760. Better, still not good. These remain brightness, edge
  and symmetry concepts on tissue texture; colour is going into the MODEL but there are still no
  colour CONCEPTS, so the readable middle cannot describe what the model is using.
* The next honest step is colour concepts (`is_pink_dominant`, `green_channel_low`, ...) measured on
  the RGB channels directly, so the concept layer can name the signal instead of only consuming it.
* Medical remains below fashion. Nothing here claims otherwise.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
for tag, s in (("1 UNBALANCED (gray)", S["att1"]), ("2 BALANCED (gray)", S["att2"]),
               ("3 ADAPTIVE (gray)", S["att3"]), ("CONTROL lam0.15 (gray)", S["ctl"]),
               ("4 ADAPTIVE+COLOUR", S["att4"])):
    if s:
        print(f"  {tag:<24} TEST {s['acc']:6.2f}%  img AUROC {s['img']:.4f}  "
              f"concepts {s['concepts'][0]}/{s['concepts'][1]}  img {s['img_pass'][0]}/{s['img_pass'][1]}")
if S["att3"] and S["ctl"] and S["att4"]:
    print()
    print(f"  total  : {S['att4']['acc']-S['att3']['acc']:+.2f}")
    print(f"  control: {S['ctl']['acc']-S['att3']['acc']:+.2f}  (lam 0.15 alone)")
    print(f"  colour : {S['att4']['acc']-S['ctl']['acc']:+.2f}")
print("=" * 100)
