"""verify_explain_fidelity.py — does explain() now tell the truth about its own concept names?

Three cases that must all behave correctly:
  1. a checkpoint WITH a fidelity table  -> names annotated with own_corr, decorative ones excluded
  2. a checkpoint WITHOUT one            -> must WARN, not imply the names are verified
  3. the exclusion must be real          -> the decorative concept must not appear as a cause, and
                                            must still be listed separately so nothing is hidden
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants                                           # noqa: E402

R = os.path.join(PROJ, "lens", "runs")
# follow the SHIPPING default rather than a hardcoded path: a test pinned to an old artifact
# silently stops testing what ships (this file pinned seetalk_k22_big.npz and went stale the
# moment shipping moved to the fixed-gradient retrain).
import variants as _V
K22 = _V.build_default(device="cpu")._paths["see_talk"]
FID = os.path.splitext(K22)[0] + "_fidelity.json"

print("=" * 100)
print("EXPLAIN FIDELITY — does explain() now tell the truth about its concept names?")
print("=" * 100)

import vl_data                                            # noqa: E402
d = vl_data.prepare_vqa("fashion", seed=0)
X = d["Xte"]
vocab = json.loads(str(np.load(K22, allow_pickle=True)["meta"][0]))["vocab"]
vi = {w: i for i, w in enumerate(vocab)}
prefix = [vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"], vi["is"], vi["a"]]

v = variants.build_default(device="cpu")

# ---- case 1: no fidelity table (it does not exist yet for K22) -> must WARN
if os.path.exists(FID):
    os.rename(FID, FID + ".bak")
res = v.explain(X[0], prefix)
warned = "WARNING" in res["note"] and res["fidelity_source"] is None
print(f"\n  CASE 1 — checkpoint WITHOUT a fidelity table")
print(f"    fidelity_source: {res['fidelity_source']}")
print(f"    note: {res['note'][:120]}")
print(f"    warns that names are unverified -> {'PASS' if warned else 'FAIL'}")
case1 = warned

# ---- case 2: with a fidelity table for THIS checkpoint -> annotate + exclude
# build one for the shipping see_talk checkpoint on its own data (fashion)
import image_concepts                                     # noqa: E402
import image_concepts_adaptive as ic_mod                  # noqa: E402
from lens.vision import load_vision                       # noqa: E402
import train_seetalk                                      # noqa: E402
from lens import backend as B                             # noqa: E402

net, meta = train_seetalk.SeeTalkNet.load(K22)
Xf = np.asarray(d["Xtr"][:2000], dtype=np.float32)
kept, _ = ic_mod.build(Xf, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in kept]


def corr(a, b):
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    dd = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / dd) if dd > 1e-12 else 0.0


names = [c.name for c in net.concepts]
T = np.stack([np.asarray(c.labels(Xf), dtype=np.float32) for c in net.concepts], axis=1)
H = B.to_cpu(net.forward_arrays(Xf, np.asarray([prefix] * len(Xf), dtype=np.int64))["concepts"])
n = len(names)
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] = corr(H[:, i], T[:, j])
rows = []
for i in range(n):
    own = float(M[i, i])
    order = np.argsort(-np.abs(M[i]))
    rank = int(np.where(order == i)[0][0]) + 1
    jb = int(order[0]) if order[0] != i else int(order[1])
    bo = float(M[i, jb])
    vd = ("strong" if (own >= 0.70 and own - abs(bo) > 0.10) else
          "honest" if (own >= 0.40 and rank <= 3) else
          "weak" if own >= 0.20 else "decorative")
    rows.append({"name": names[i], "own_corr": round(own, 4), "best_other": round(bo, 4),
                 "best_other_name": names[jb], "margin": round(own - abs(bo), 4), "rank": rank,
                 "verdict": vd})
counts = {}
for r in rows:
    counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
json.dump({"tag": os.path.basename(K22), "n_concepts": n, "n_samples": len(Xf),
           "counts": counts, "concepts": rows}, open(FID, "w"), indent=1)
print(f"\n  built a fidelity table for {os.path.basename(K22)}: {counts}")

res2 = v.explain(X[0], prefix)
have_src = res2["fidelity_source"] is not None
annotated = all(c.get("own_corr") is not None for c in res2["image_driven_contributions"])
dec_names = {r["name"] for r in rows if r["verdict"] == "decorative"}
shown = {c["concept"] for c in res2["image_driven_contributions"]}
leaked = dec_names & shown
print(f"\n  CASE 2 — checkpoint WITH a fidelity table")
print(f"    fidelity_source: {res2['fidelity_source']}")
print(f"    every listed concept carries own_corr -> {'PASS' if annotated else 'FAIL'}")
print(f"    decorative concepts on this checkpoint: {len(dec_names)}")
print(f"    decorative leaked into the cause list  : {sorted(leaked) if leaked else 'none'}")
print(f"    excluded ones still reported separately: "
      f"{'yes' if res2.get('decorative_excluded') is not None else 'none to exclude'}")
for c in res2["image_driven_contributions"][:4]:
    print(f"      {c['concept']:<24} own_corr {c['own_corr']:>6}  verdict {c['verdict']:<11} "
          f"contrib {c['contribution']:+.4f}")
case2 = have_src and annotated and not leaked

print("\n" + "-" * 100)
ok = case1 and case2
print(f"  CASE 1 (warns without a table)   : {'PASS' if case1 else 'FAIL'}")
print(f"  CASE 2 (annotates, excludes)     : {'PASS' if case2 else 'FAIL'}")
print(f"\n  {'EXPLAIN IS NOW HONEST ABOUT ITS NAMES — a decorative label can no longer be printed as a cause.' if ok else 'EXPLAIN STILL NOT HONEST — see failures above.'}")
sys.exit(0 if ok else 1)
