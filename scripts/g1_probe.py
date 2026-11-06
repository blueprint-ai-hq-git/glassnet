"""g1_probe.py — WHY do is_a_dialogue_line and pronoun_subject fail at every size?

G2 showed these two fail at D=64 and D=128 while seven others pass (five with AUROC 1.0000). Before
changing the architecture or the targets, measure whether the concept is even readable from the
window by the simplest possible reader.

THE TEST: a LINEAR PROBE. For each concept, fit the best linear readout from the window's
word-presence vector to the concept's own binary label, on TRAIN windows, and score AUROC on
VALIDATION windows. This is a ceiling for a one-layer readout -- our own code, no sklearn.

  * probe AUROC HIGH  -> the concept IS linearly present in the window, so the model's failure is
                         an optimisation/architecture problem
  * probe AUROC CHANCE-> the concept is not linearly readable from word presence; a nonlinear model
                         might still find it, but the target may also be badly designed

Also measured: each concept's support, its firing rate in the trained model, and how much it
overlaps with the other eight concepts (redundancy).
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

from glassnet import tiny_data                 # noqa: E402
from glassnet.tensor import set_backend        # noqa: E402

set_backend("numpy")
R = os.path.join(PROJ, "runs", "t3")

DATA = tiny_data.build(win=8, vocab_cap=1500)
Xtr, Xva = np.asarray(DATA["Xtr"]), np.asarray(DATA["Xva"])
V = DATA["V"]
TARGETS = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]


def auroc(score, t01):
    o = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = int(t01.sum()), int((~t01).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[t01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# feature space: presence of every word that appears in ANY concept target, plus a bias
words = sorted({i for t in TARGETS for i in t["ids"]})
wid = {w: k for k, w in enumerate(words)}
print("=" * 100)
print("WHY DO TWO CONCEPTS FAIL? — linear probe from window word-presence to each concept's label")
print("=" * 100)
print(f"probe features: {len(words)} words (the union of all 9 targets' word sets) + bias")
print(f"train windows {len(Xtr)}, validation windows {len(Xva)}")


def feats(X):
    F = np.zeros((len(X), len(words) + 1), dtype=np.float64)
    for i, w in enumerate(X):
        for tok in w:
            k = wid.get(int(tok))
            if k is not None:
                F[i, k] = 1.0
    F[:, -1] = 1.0
    return F


Ftr, Fva = feats(Xtr), feats(Xva)
labs_tr = np.stack([np.isin(Xtr, t["ids"]).any(1) for t in TARGETS], axis=1)
labs_va = np.stack([np.isin(Xva, t["ids"]).any(1) for t in TARGETS], axis=1)

print(f"\n{'concept':<24} {'support':>8} {'probe AUROC':>12} {'model AUROC':>12}  verdict")
print("-" * 100)

# model AUROC from the G2 D=64 run, for comparison
g2 = json.load(open(os.path.join(R, "g2_scaling.json")))
d64 = [x for x in g2 if x["D"] == 64 and x["kind"] == "readable"]
model_auroc = {}
if d64:
    names = [r["name"] for r in d64[0]["rows"]]
    for i, n in enumerate(names):
        model_auroc[n] = float(np.mean([r["rows"][i]["auroc"] for r in d64]))

rows = []
for c, t in enumerate(TARGETS):
    y_tr = labs_tr[:, c].astype(np.float64)
    y_va = labs_va[:, c].astype(bool)
    # least-squares linear probe, closed form (our own maths, no library model)
    coef, *_ = np.linalg.lstsq(Ftr, y_tr, rcond=None)
    score = Fva @ coef
    pa = auroc(score, y_va)
    ma = model_auroc.get(t["name"], float("nan"))
    if np.isnan(pa):
        verdict = "NO VALIDATION POSITIVES"
    elif pa >= 0.90:
        verdict = "linearly readable -> model underperforms"
    elif pa >= 0.70:
        verdict = "partly readable"
    else:
        verdict = "NOT linearly readable -> target/architecture issue"
    print(f"{t['name']:<24} {y_tr.mean()*100:>7.2f}% {pa:>12.4f} {ma:>12.4f}  {verdict}")
    rows.append({"name": t["name"], "support": float(y_tr.mean()), "probe_auroc": float(pa),
                 "model_auroc": float(ma), "verdict": verdict})

# redundancy: how much does each concept overlap the others?
print("\nREDUNDANCY — Jaccard overlap of each concept's window set with the others")
print("-" * 100)
S = labs_tr.astype(bool)
for c, t in enumerate(TARGETS):
    jac = []
    for d in range(len(TARGETS)):
        if d == c:
            continue
        inter = (S[:, c] & S[:, d]).sum()
        union = (S[:, c] | S[:, d]).sum()
        jac.append(inter / max(union, 1))
    mx = int(np.argmax(jac))
    print(f"{t['name']:<24} max overlap {max(jac):.3f} with {TARGETS[mx]['name']:<24} "
          f"mean {np.mean(jac):.3f}")

json.dump(rows, open(os.path.join(R, "g1_probe.json"), "w"), indent=1)
print(f"\nsaved runs/t3/g1_probe.json")
print("\nNOTE: the probe uses word PRESENCE only. A concept that needs word ORDER or position can")
print("      score low here while still being learnable by the model, so a low probe score is")
print("      evidence about the target's readability, not proof that the concept is absent.")
