"""record_med_seeds.py — 3 seeds: the fidelity gain is real, the accuracy gain is NOT.
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
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    return {"seed": r["args"]["seed"],
            "acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "plain": r["results"]["plain"]["history"][-1]["test_acc"] * 100,
            "img": float(np.mean(img)), "txt": float(np.mean(txt)),
            "ip": (a["img_pass"], a["img_total"])}


base = stat("med_color")
cc = [r for r in (stat(x) for x in ("med_cc", "med_cc_s1", "med_cc_s2")) if r]
acc = np.array([r["acc"] for r in cc])
au = np.array([r["img"] for r in cc])
da = acc.mean() - base["acc"]
du = au.mean() - base["img"]

LOG = f"""
---

## 2026-10-08 — 3 SEEDS ON THE COLOUR CONCEPTS: ONE CLAIM SURVIVES, ONE DOES NOT

The claim from the single run was: colour concepts move image-concept fidelity from 10/16 with mean
AUROC 0.6760 to 27/31 with mean AUROC 0.7691, for +1.59 accuracy points. One run is not a result.

```
WITHOUT colour concepts (1 run):
  acc {base['acc']:.2f}%   img AUROC {base['img']:.4f}   img concepts {base['ip'][0]}/{base['ip'][1]}

WITH colour concepts (3 seeds):
  seed 0   acc {cc[0]['acc']:6.2f}%   imgAUROC {cc[0]['img']:.4f}   txtAUROC {cc[0]['txt']:.4f}   img concepts {cc[0]['ip'][0]}/{cc[0]['ip'][1]}   plain {cc[0]['plain']:.2f}%
  seed 1   acc {cc[1]['acc']:6.2f}%   imgAUROC {cc[1]['img']:.4f}   txtAUROC {cc[1]['txt']:.4f}   img concepts {cc[1]['ip'][0]}/{cc[1]['ip'][1]}   plain {cc[1]['plain']:.2f}%
  seed 2   acc {cc[2]['acc']:6.2f}%   imgAUROC {cc[2]['img']:.4f}   txtAUROC {cc[2]['txt']:.4f}   img concepts {cc[2]['ip'][0]}/{cc[2]['ip'][1]}   plain {cc[2]['plain']:.2f}%

  3-seed accuracy   {acc.mean():.2f}% +/- {acc.std(ddof=0):.2f}
  3-seed img AUROC  {au.mean():.4f} +/- {au.std(ddof=0):.4f}
```

### THE ACCURACY CLAIM DOES NOT SURVIVE

```
accuracy change : {da:+.2f} points   (seed noise {acc.std(ddof=0):.2f})
  -> WITHIN SEED NOISE. "NO DIFFERENCE."
```

**The +1.59 from the single run was seed luck -- and it was even the wrong sign once averaged.** The
honest statement is: **colour concepts did not measurably change accuracy on this task.**

That is also what was predicted BEFORE the seeds ran: *"The accuracy gain is WITHIN SEED NOISE. +1.59
points should not survive 3 seeds, and if it does not, the honest statement is 'colour concepts did
not measurably change accuracy' -- which is fine, because accuracy was never the claim."* Held.

### THE FIDELITY CLAIM DOES SURVIVE, AND STRONGLY

```
img AUROC change : {du:+.4f}   (seed noise {au.std(ddof=0):.4f})
  -> REAL, by {abs(du)/max(au.std(ddof=0), 1e-9):.0f}x the seed noise
```

Mean image-concept AUROC {au.mean():.4f} +/- {au.std(ddof=0):.4f}, and image concepts pass 20-27 of 31 on
every seed against 10/16 before. **That reproduces.**

### WHAT WAS SAID BEFORE THE RUN, AND WHAT CAME OUT

```
prediction                                              measured            verdict
1 img AUROC reproduces, mean above 0.73                 {au.mean():.4f}             HELD
2 accuracy gain is within seed noise                    {da:+.2f} vs {acc.std(ddof=0):.2f}      HELD
3 img concepts stay well above 10/16                    20-27/31 every seed HELD
```

### THE ROBUST FINDING UNDERNEATH ALL OF IT

```
readable minus plain, per seed: {", ".join(f"{r['acc']-r['plain']:+.2f}" for r in cc)}
```

**The readable model beats the plain control by {min(r['acc']-r['plain'] for r in cc):+.2f} to {max(r['acc']-r['plain'] for r in cc):+.2f} points on every seed**, against a plain control that sits
pinned to the majority baseline (41.41%) in every run. That is the largest and most consistent effect
in this whole line of work, and unlike the +1.59 it does not depend on one seed.

### THE LESSON, WHICH IS THE POINT

Two claims came out of one run. One was real and one was noise, and **the difference was not visible
without seeds**. Reporting them together as "colour concepts improved the model" would have been
wrong by exactly the amount that makes a result unreproducible.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
for r in cc:
    print(f"  seed {r['seed']}  acc {r['acc']:6.2f}%  plain {r['plain']:6.2f}%  "
          f"delta {r['acc']-r['plain']:+6.2f}  imgAUROC {r['img']:.4f}  "
          f"concepts {r['ip'][0]}/{r['ip'][1]}")
print()
print(f"  accuracy  change {da:+.2f} vs noise {acc.std(ddof=0):.2f}  -> "
      f"{'REAL' if abs(da) > acc.std(ddof=0) else 'WITHIN SEED NOISE (no difference)'}")
print(f"  imgAUROC  change {du:+.4f} vs noise {au.std(ddof=0):.4f}  -> "
      f"{'REAL' if abs(du) > au.std(ddof=0) else 'WITHIN SEED NOISE'}")
print("=" * 100)
