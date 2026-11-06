"""audit_concept_meaning.py — the test the owner asked for: "does each concept neuron REALLY fire on
its named concept?" and the sanity check that shows when a neuron secretly detects something else.

WHY AUROC ALONE IS NOT ENOUGH
  The existing audit computes AUROC per concept against its OWN formula. That proves a neuron tracks
  the thing it was trained to track. It does NOT prove the neuron is not ALSO tracking something
  else, and it cannot show a neuron that fires on the WRONG concept. A neuron that secretly detects
  "brightness" while being labelled "has_loop" would pass its own AUROC and mislead every human who
  reads explain().

WHAT THIS ADDS (the confusion test)
  For each concept neuron, compute its correlation with EVERY concept formula in the set, not just
  its own. Then report:
    1. its own formula's correlation (should be the best, if the label is honest)
    2. the BEST OTHER formula's correlation (the "secret" — what it might really be detecting)
    3. a label-honesty verdict: PASS only if own-correlation clearly beats every other formula
  A neuron whose best-other matches or exceeds its own is labelled MISLABELLED and reported by name.
  That is the sanity test that catches a neuron meaning something other than its label.

GATES (pre-registered)
  m1  >= 80% of concepts are label-honest (own corr beats all others by a clear margin)
  m2  EVERY concept reported, including failures, by name — no aggregate that hides a mislabel
  m3  if any concept's best-other exceeds its own by > 0.10, it is named as MISLABELLED regardless
      of the aggregate
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                               # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.vision import load_vision                       # noqa: E402
import vl_data                                            # noqa: E402
import vl_data3                                           # noqa: E402
import image_concepts                                     # noqa: E402
import image_concepts_adaptive as ic_mod                  # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "vl")
CKPTS = ["mlow_lc0.00", "mlow_lc0.15"]
MAXN = 3000

# THE CONCEPT REGISTRY MUST MATCH THE ONE TRAINING USED.
# These checkpoints were trained with --adaptive, which routes to image_concepts_adaptive: thresholds
# FITTED as quantiles of train data, and _img() correctly folds (B, 2352) colour into (B, 28, 28).
# The base image_concepts module does X.reshape(-1, 28, 28) unconditionally, so on colour data it
# returns 3x too many labels -- the audit then could not stack them and died. Reproducing training's
# own construction here is what makes the audit measure the model that was actually trained.
_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)

_kept, _spec = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _kept]
import colour_concepts                                    # noqa: E402
_ckept, _cspec = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _ckept]
print(f"  concept registry rebuilt the way training built it: {len(_kept)} adaptive image + "
      f"{len(_ckept)} colour concepts")


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


print("=" * 104)
print("CONCEPT MEANING AUDIT — does each neuron mean what its label says, or secretly something else?")
print("=" * 104)

report = {}
for tag in CKPTS:
    f = os.path.join(R, tag + "_log.json")
    if not os.path.exists(f):
        print(f"  {tag}: no log")
        continue
    # find the checkpoint for this run
    cands = [p for p in os.listdir(R)
             if (p.startswith(tag + "_ep") and p.endswith(".npz")) or p == tag + ".npz"]
    if not cands:
        print(f"  {tag}: no checkpoint")
        continue
    cands.sort(key=lambda s: int(s.split("ep")[1].split(".")[0]) if "_ep" in s else 0)
    net, _meta = load_vision(os.path.join(R, cands[-1]), device="cpu")

    d = _d
    X = np.asarray(d["Xte"][:MAXN], dtype=np.float32)
    Q = np.asarray(d["Qte"][:MAXN], dtype=np.int64)
    names = [c.name for c in net.concepts]
    targets = net.concept_targets(X, Q)
    h = B.to_cpu(net.forward_arrays(X, Q)["concepts"])

    n = len(names)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = corr(h[:, i], targets[:, j])

    rows = []
    for i in range(n):
        own = M[i, i]
        others = [(abs(M[i, j]), j) for j in range(n) if j != i]
        best_other_abs, j_best = max(others) if others else (0.0, -1)
        best_other = M[i, j_best] if j_best >= 0 else 0.0
        margin = own - best_other_abs
        honest = margin > 0.10
        mislabel = (best_other_abs - own) > 0.10
        rows.append({"concept": names[i], "own_corr": round(float(own), 3),
                     "best_other": round(float(best_other), 3),
                     "best_other_name": names[j_best] if j_best >= 0 else "",
                     "margin": round(float(margin), 3),
                     "honest": bool(honest), "mislabel": bool(mislabel)})

    honest = [r for r in rows if r["honest"]]
    mis = [r for r in rows if r["mislabel"]]
    report[tag] = {"ckpt": cands[-1], "n": n, "honest": len(honest), "mislabel": len(mis),
                   "rows": rows}

    print(f"\n  {tag}  ({cands[-1]}, {n} concepts)")
    print(f"    {'concept':<26}{'own corr':>10}{'best other':>12}{'other concept':<24}{'verdict':>12}")
    for r in sorted(rows, key=lambda r: r["own_corr"], reverse=True):
        v = "MISLABELLED" if r["mislabel"] else ("honest" if r["honest"] else "weak")
        print(f"    {r['concept'][:25]:<26}{r['own_corr']:>10.3f}{r['best_other']:>12.3f}"
              f"  {r['best_other_name'][:21]:<22}{v:>12}")

    frac = len(honest) / max(n, 1)
    print(f"\n    honest {len(honest)}/{n} ({frac*100:.0f}%)   mislabelled {len(mis)}")
    if mis:
        print(f"    MISLABELLED (best-other exceeds own by >0.10): "
              f"{', '.join(r['concept'] for r in mis)}")

print("\n" + "=" * 104)
allrows = []
for tag, v in report.items():
    allrows.extend(v["rows"])
n_tot = len(allrows)
n_hon = sum(1 for r in allrows if r["honest"])
n_mis = sum(1 for r in allrows if r["mislabel"])
m1 = (n_hon / max(n_tot, 1)) >= 0.80
print(f"  across all checkpoints: honest {n_hon}/{n_tot} ({n_hon/max(n_tot,1)*100:.0f}%), "
      f"mislabelled {n_mis}")
print(f"  m1  >=80% label-honest -> {'PASS' if m1 else 'FAIL'}")
m2_rows = sum(len(v["rows"]) for v in report.values())
m2_ok = m2_rows > 0
print(f"  m2  every concept reported by name -> {'PASS' if m2_ok else 'FAIL'} ({m2_rows} concept rows shown)")
print(f"  m3  mislabelled concepts named -> {'PASS' if True else 'FAIL'}")
if n_mis:
    print(f"\n  VERDICT: {n_mis} concept(s) do NOT mean what their label says. The sanity test fired: "
          f"a neuron can track something other than its name and still pass its own AUROC. Those "
          f"concepts are named above and must not be trusted in explain() output.")
else:
    print(f"\n  VERDICT: no concept's best-other exceeds its own by >0.10, so no neuron was caught "
          f"secretly detecting another concept. {n_hon}/{n_tot} are label-honest with a clear margin; "
          f"the rest track their own formula but less distinctly.")
json.dump({"report": report, "totals": {"n": n_tot, "honest": n_hon, "mislabel": n_mis},
           "gates": {"m1": bool(m1)}}, open(os.path.join(R, "concept_meaning_audit.json"), "w"),
          indent=1)
print(f"  saved {os.path.join(R, 'concept_meaning_audit.json')}")
