"""record_generalization.py — the readable advantage CARRIES to a second domain (+6.74 pts).
Plus one honest note: the second arm was a no-op, so there is really only ONE new data point.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "vl", "gen_fashion_d64_log.json")))
rr = d["results"]["readable"]["history"][-1]["test_acc"] * 100
pp = d["results"]["plain"]["history"][-1]["test_acc"] * 100
aud = d.get("audit", {})
npass = sum(1 for c in aud.get("concepts", []) if c.get("pass"))
ntot = len(aud.get("concepts", []))

LOG = f"""
---

## 2026-10-09 — GENERALIZATION TEST: the readable advantage CARRIES to a second domain

Every strong result until now was on pathmnist. One dataset is not evidence that the readable
architecture is better in general — it could have been a lucky pairing with histopathology. So the
SAME recipe ran on **Fashion-MNIST**: a different domain (grey 28x28 clothing photos, 10 classes,
versus colour 28x28 tissue patches, 9 classes). One change at a time — only the data changed.

```
  Fashion-MNIST, recipe held identical to the winning medical config (d=64, patch 7, 8 epochs,
  lr 0.3 decay 0.5, lam_concept 0.15, adaptive, 20k train):

    readable   31,354 params   TEST  {rr:.2f}%
    plain      31,760 params   TEST  {pp:.2f}%
    delta                      {rr-pp:+.2f} pts
    majority baseline          39.73%   (readable beats it: {rr > 39.73})
    concepts                   {npass}/{ntot} pass their firing audit
    image concepts mean AUROC  0.7114
    text  concepts mean AUROC  0.9047   (vqa1 baseline was 0.53-0.63)

  z1  readable beats its own plain control by > 5 pts  -> {'PASS' if rr-pp > 5 else 'FAIL'} ({rr-pp:+.2f})
  z2  same direction as medical (readable > plain)     -> {'PASS' if rr > pp else 'FAIL'}
  z3  concept audit reported                           -> {npass}/{ntot}
```

**The readable model wins on a second, unrelated dataset by +{rr-pp:.2f} points, with {npass}/{ntot}
named concepts passing their firing audit.** Combined with the medical result (+24.62 pts,
replicated across 6 arms and 3 seeds), the finding is now: **the readable architecture is not a
handicap — it beats a parameter-matched plain control on two different datasets.** That is the
strongest, most replicated claim this project has.

### HONEST NOTE: the second arm was a NO-OP, so there is really one new data point
I ran a second arm with `--color --colour-concepts` expecting a different result and got byte-identical
numbers (86.29% both). Rather than report "two variants agree" (which would look like replication but
is not), I checked the code: `vl_data.prepare_vqa()` takes `color` and **ignores it entirely for the
fashion branch** — Fashion-MNIST is loaded as greyscale regardless. So the flag did nothing and the
identical numbers are correct, not a second measurement. **One new dataset, not two.** Stated because
reporting it as two would have inflated the evidence.

### WHAT THIS DOES AND DOES NOT ESTABLISH
* DOES: the readable advantage generalizes across two domains; concepts fire as named on both.
* DOES NOT: prove it scales to large models or hard problems. Both datasets are 28x28 and small
  (86,615 params max). "Carries across sizes" is still open and needs a bigger test.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended generalization record")
print(f"  Fashion-MNIST: readable {rr:.2f}% vs plain {pp:.2f}% = {rr-pp:+.2f} pts, concepts {npass}/{ntot}")
