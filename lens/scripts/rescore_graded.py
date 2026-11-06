"""rescore_graded.py — re-score existing checkpoints under METER-06. No retraining.

METER-06 is pre-registered in METER-06-GRADED.md, written before this runs. It exists because the
binary firing floor ("output > 0.5") is the wrong question for a GRADED concept, whose target is
the fraction of the window it occupies. Seven concepts in FIX-03 ranked at AUROC 1.0000 and were
failed by that floor while outputting 0.001-0.011 everywhere.

Both meters are reported side by side. Nothing is overwritten; results go to new files.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402
from lens.concepts import Concept                          # noqa: E402
from glassnet import tiny_data                             # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "meter06")
os.makedirs(OUT, exist_ok=True)

DATA = tiny_data.build(win=8, vocab_cap=1500)
Xva = np.asarray(DATA["Xva"])
z = json.load(open(os.path.join(PROJ, "runs", "t3", "meaningful_targets_ext.json")))
CONCEPTS = [Concept(c["name"], c["ids"], support=float(c["support"])) for c in z["qualifying"]]

# pre-registered sample: the same shuffled 3000 the other audits used
AUD = np.random.default_rng(99).permutation(len(Xva))[:3000]
X = Xva[AUD]

ARMS = [
    ("lc=0.1 binary", "blocker/readable.npz"),
    ("lc=0.3 binary", "fix01_lc0.3/readable.npz"),
    ("lc=0.5 binary", "fix01_lc0.5/readable.npz"),
    ("lc=1.0 binary", "fix01_lc1/readable.npz"),
    ("two-phase", "twophase/twophase.npz"),
    ("FIX-02 soft MSE", "fix02_soft_lc0.3/readable.npz"),
    ("FIX-03 soft BCE", "fix03_softbce_lc0.3/readable.npz"),
]

print("=" * 108)
print("METER-06 — support-matched top-q firing, applied to EXISTING checkpoints (no retraining)")
print("=" * 108)
print("  METER-05 (original): fires = fraction of outputs above 0.5.  Correct for BINARY concepts.")
print("  METER-06 (this one): fires = overlap between the neuron's top-q outputs and the true")
print("                       windows, where q = the concept's declared support. Bar: >= 2 * support.")
print()

rows = []
for tag, rel in ARMS:
    p = os.path.join(PROJ, "lens", "runs", rel)
    if not os.path.exists(p):
        print(f"  {tag:<20} MISSING {rel}")
        continue
    net = lens.load(p, device="cpu")
    h = net.forward_arrays(X)["concepts"]
    out_rows = []
    for i, c in enumerate(CONCEPTS):
        tgt = c.labels(X)
        n1, n0 = int(tgt.sum()), int((~tgt).sum())
        score = h[:, i]
        if n1 == 0 or n0 == 0:
            out_rows.append({"name": c.name, "auroc": None, "m5_fires": None, "m6_fires": None,
                             "m5": False, "m6": False})
            continue
        order = np.argsort(score)
        ranks = np.empty(len(score))
        ranks[order] = np.arange(1, len(score) + 1)
        au = float((ranks[tgt].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        m5 = float((score > 0.5).mean())
        # METER-06: the top-q by neuron, where q = the declared support
        q = float(c.support)
        s = n1 / len(score)                          # the ACTUAL positive rate in this sample
        k = max(1, int(round(q * len(score))))
        top = np.zeros(len(score), dtype=bool)
        top[np.argsort(-score)[:k]] = True
        overlap = float((top & tgt).mean())          # joint rate
        # The random baseline for the JOINT rate is q * s, not q. The first version divided by q
        # alone, which capped a PERFECT concept at 1.0 against a bar of 2.0 -- so every concept
        # failed, including the nine that pass under the old meter. That was an arithmetic error
        # in the metric, not a result; the bar (2x random) was never changed.
        chance = max(q * s, 1e-12)
        m6_fires = overlap / chance
        out_rows.append({"name": c.name, "auroc": au, "m5_fires": m5,
                         "m6_fires": m6_fires, "support": q,
                         "m5": bool(au >= 0.55 and m5 >= 0.02),
                         "m6": bool(au >= 0.55 and m6_fires >= 2.0)})
    n5 = sum(1 for r in out_rows if r["m5"])
    n6 = sum(1 for r in out_rows if r["m6"])
    rows.append({"arm": tag, "n_pass_m5": n5, "n_pass_m6": n6, "n": len(out_rows),
                 "concepts": out_rows})
    print(f"  {tag:<20} METER-05 {n5}/{len(out_rows)}   METER-06 {n6}/{len(out_rows)}")

print()
print("=" * 108)
print("PER-CONCEPT, EVERY ARM — both meters side by side")
print("=" * 108)
hdr = f"  {'arm':<18}{'concept':<24}{'AUROC':>8}{'M5 fires':>10}{'M5':>4}{'M6 x':>8}{'M6':>4}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in rows:
    for c in r["concepts"]:
        au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
        m5 = "None" if c["m5_fires"] is None else f"{c['m5_fires']:.4f}"
        m6 = "None" if c.get("m6_fires") is None else f"{c['m6_fires']:.2f}"
        print(f"  {r['arm']:<18}{c['name']:<24}{au:>8}{m5:>10}"
              f"{('P' if c['m5'] else 'F'):>4}{m6:>8}{('P' if c['m6'] else 'F'):>4}")

with open(os.path.join(OUT, "rescore.json"), "w") as f:
    json.dump(rows, f, indent=1)
print(f"\nsaved {OUT}/rescore.json")
print("\nNOTE: a concept passing METER-06 and failing METER-05 means it RANKS correctly but does")
print("not saturate. Both numbers are reported together and neither is quoted alone.")
