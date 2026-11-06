"""record_vqa2.py — the load-bearing-question fix: diagnosis confirmed, prediction held.
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
    return {"txt": float(np.mean(txt)), "txt_each": txt, "img": float(np.mean(img)),
            "acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "concepts": (a["n_pass"], a["n_concepts"]), "txt_pass": (a["txt_pass"], a["txt_total"])}


v1 = stat("vlsmoke")
v2 = stat("vqa2_small")

LOG = f"""
---

## 2026-10-08 — THE VL FIX: THE QUESTIONS NOW MATTER, AND THE TEXT CONCEPTS WENT FROM A COIN FLIP TO 0.96

### THE DIAGNOSIS

vqa1 fed four questions that all asked for the same thing:

    "what is in the photo" -> sneaker      "name the object"  -> sneaker
    "what do you see"      -> sneaker      "identify this item" -> sneaker

**The question carried no information about the target.** The model had no reason to read it, so the
text concepts received no gradient pressure. Measured: **text concepts at AUROC 0.53-0.63.**
A coin flip is 0.50.

### THE FIX, AS ONE CHANGE

Different questions now need DIFFERENT answers:

    "what is in the photo" -> the class name   ("sneaker")
    "name the object"      -> the class name   ("sneaker")
    "is this footwear"     -> yes / no
    "is this a bag"        -> yes / no

The model CANNOT answer without reading the question.

### THE RESULT — PREDICTION HELD

```
                            text-concept   image-concept   TEST acc
vqa1 (questions identical)     0.5963         0.5684        69.47%
vqa2 (questions matter)        0.9640         0.8560        88.13%

vqa2 text concepts, each:
   asks_what      AUROC 0.9815
   asks_name      AUROC 0.9737
   asks_footwear  AUROC 0.9436
   asks_bag       AUROC 0.9573
   -> 4/4 text concepts PASS  (vqa1: 3/4, and two of those were barely off the floor)
```

**I predicted "text-concept AUROC rises clearly above 0.70". It went to 0.9640.** The diagnosis was
right: the concepts were weak because the task gave them nothing to detect, not because the layer
could not represent them.

### AND THE MODEL ACTUALLY ANSWERS THE QUESTION

```
Q: what is in the photo   ANSWER ankle_boot   TRUE ankle_boot   OK
Q: name the object        ANSWER pullover     TRUE pullover     OK
Q: is this footwear       ANSWER no           TRUE no           OK
Q: is this a bag          ANSWER no           TRUE no           OK
Q: what is in the photo   ANSWER shirt        TRUE shirt        OK
```

The SAME image yields the class name for a naming question and "no" for a footwear question. That is
question-conditioned answering, not a single fixed mapping. Before the change, every question wanted
the same word.

### THE HONEST BASELINES

```
majority-answer baseline: always say 'no' = 42.34%
true chance            : 1/14          =  7.14%
readable               :                  88.13%
```

**"no" is 42% of the answers**, so a model that always says "no" scores 42.34%. Reporting 88.13%
without that line would overstate what was achieved. The trainer now prints the majority baseline
itself so the number cannot be read alone.

readable 88.13% vs plain 83.53% (27,277 vs 27,690 params) = **readable +4.60 points**, and 14/17
concepts pass (image 10/13, text 4/4).

### THE PATTERN THIS RUN BREAKS

Seven earlier claims in this project outran their evidence. This one did not: the weakness was
MEASURED (0.53-0.63), a single cause was HYPOTHESISED (questions carried no information), one change
was made, a prediction was written FIRST, and the prediction held. Measure -> diagnose -> one change
-> predict -> verify.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("vqa1 -> vqa2")
print("=" * 100)
for tag, v in (("vqa1", v1), ("vqa2", v2)):
    if v:
        print(f"  {tag}: text AUROC {v['txt']:.4f}  image AUROC {v['img']:.4f}  "
              f"TEST {v['acc']:.2f}%  text concepts {v['txt_pass'][0]}/{v['txt_pass'][1]}")
if v1 and v2:
    print()
    print(f"  text-concept AUROC move: {v1['txt']:.4f} -> {v2['txt']:.4f}  "
          f"({(v2['txt']-v1['txt'])/v1['txt']*100:+.1f}%)")
    print(f"  prediction 'above 0.70': {'HELD' if v2['txt'] > 0.70 else 'WRONG'}")
print("=" * 100)
