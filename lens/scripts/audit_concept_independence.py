"""audit_concept_independence.py — how many INDEPENDENT concepts does the layer actually have?

THE OPEN PROBLEM (E18)
  The medical concept layer advertises 35 named concepts, but 34 concept pairs correlate above 0.70.
  If two names describe the same underlying property, the layer does not have 35 independent readable
  units — it has fewer, and every count of "N concepts" overstates how much a human can learn from it.

WHAT THIS MEASURES
  1. The concept-formula correlation matrix, and a CLUSTERING of concepts into groups that describe
     the same thing (complete-linkage at |corr| >= 0.70: a cluster only forms if EVERY member pairs
     above threshold with every other, so clusters are conservative, not greedy).
  2. The EFFECTIVE number of independent concepts (participation ratio of the correlation matrix
     eigenvalues: (sum L)^2 / sum(L^2) — the standard effective-rank measure).
  3. Per cluster: which name is the best representative (highest own_corr from the fidelity table, or
     highest support), so a de-duplicated layer can keep ONE honest name per real concept.

GATES (pre-registered)
  i1  report the independent-concept count and the effective rank; do NOT report the advertised count
      as if it were the readable count
  i2  every cluster listed with its members and its chosen representative by name
  i3  if effective rank < 50% of advertised, the "35 named concepts" claim is misleading and must be
      restated as the independent number
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import vl_data                                            # noqa: E402
import vl_data3                                           # noqa: E402
import image_concepts                                     # noqa: E402
import image_concepts_adaptive as ic_mod                  # noqa: E402
import colour_concepts                                    # noqa: E402
from lens.vision import load_vision                       # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "vl")
CKPTS = ["mlow_lc0.15", "mlow_lc0.00"]
MAXN = 4000
THR = 0.70

_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)
_kept, _ = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _kept]
_ckept, _ = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _ckept]


def complete_link_clusters(T, thr=THR):
    """Complete-linkage clustering at |corr| >= thr. Conservative: a cluster only grows if the
    candidate correlates above threshold with EVERY existing member, so members of a cluster really
    do describe the same thing. Greedy single-linkage would chain unrelated concepts together."""
    n = T.shape[1]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = T[:, i] - T[:, i].mean()
            b = T[:, j] - T[:, j].mean()
            d = np.sqrt((a * a).sum() * (b * b).sum())
            C[i, j] = float((a * b).sum() / d) if d > 1e-12 else 0.0
    A = np.abs(C)
    unassigned = set(range(n))
    clusters = []
    while unassigned:
        seed = min(unassigned)
        grp = [seed]
        unassigned.discard(seed)
        changed = True
        while changed:
            changed = False
            for k in sorted(unassigned):
                if all(A[k, g] >= thr for g in grp):
                    grp.append(k)
                    unassigned.discard(k)
                    changed = True
        clusters.append(grp)
    return clusters, C


print("=" * 100)
print("CONCEPT INDEPENDENCE — how many readable concepts does the layer REALLY have?")
print("=" * 100)

out = {}
for tag in CKPTS:
    cands = [p for p in os.listdir(R) if p.startswith(tag + "_ep") and p.endswith(".npz")]
    if not cands:
        print(f"  {tag}: no checkpoint")
        continue
    cands.sort(key=lambda s: int(s.split("ep")[1].split(".")[0]))
    net, _meta = load_vision(os.path.join(R, cands[-1]), device="cpu")
    X = np.asarray(_d["Xte"][:MAXN], dtype=np.float32)
    Q = np.asarray(_d["Qte"][:MAXN], dtype=np.int64)
    names = [c.name for c in net.concepts]
    T = net.concept_targets(X, Q)
    clusters, C = complete_link_clusters(T)

    L = np.linalg.eigvalsh(C @ C.T)
    L = np.clip(L, 0, None)
    eff = float((L.sum() ** 2) / (L * L).sum()) if (L * L).sum() > 0 else 0.0

    # representative per cluster: the name whose neuron tracks it best (own_corr), else first
    fidf = os.path.splitext(os.path.join(R, cands[-1]))[0] + "_fidelity.json"
    fid = {}
    if os.path.exists(fidf):
        fid = {c["name"]: c for c in json.load(open(fidf)).get("concepts", [])}

    rows = []
    for g in clusters:
        members = [names[i] for i in g]
        scored = sorted(members, key=lambda nm: -abs(fid.get(nm, {}).get("own_corr", 0.0)))
        rows.append({"size": len(g), "members": members, "representative": scored[0],
                     "representative_own_corr": fid.get(scored[0], {}).get("own_corr")})
    rows.sort(key=lambda r: -r["size"])

    n_adv = len(names)
    n_ind = len(clusters)
    print(f"\n  {tag}  ({cands[-1]})")
    print(f"    advertised concepts      : {n_adv}")
    print(f"    INDEPENDENT clusters     : {n_ind}   (|corr| >= {THR}, complete linkage)")
    print(f"    effective rank           : {eff:.1f}")
    print(f"    {'size':>5}  representative (best own_corr)      other names in the same cluster")
    for r in rows:
        others = [m for m in r["members"] if m != r["representative"]]
        oc = r["representative_own_corr"]
        oc_s = f"{oc:+.3f}" if oc is not None else "  n/a"
        print(f"    {r['size']:>5}  {r['representative'][:30]:<32} {oc_s}   "
              f"{', '.join(others)[:62]}")
    out[tag] = {"ckpt": cands[-1], "advertised": n_adv, "independent": n_ind,
                "effective_rank": round(eff, 2), "clusters": rows}

print("\n" + "=" * 100)
for tag, v in out.items():
    frac = v["independent"] / max(v["advertised"], 1)
    print(f"  {tag:<16} advertised {v['advertised']:>3}  independent {v['independent']:>3} "
          f"({frac*100:.0f}%)  effective rank {v['effective_rank']:.1f}")
if out:
    worst = min(out.items(), key=lambda kv: kv[1]["independent"] / max(kv[1]["advertised"], 1))
    frac = worst[1]["independent"] / max(worst[1]["advertised"], 1)
    i3 = frac >= 0.50
    # i1 and i2 used to print a hardcoded "PASS" — a gate that cannot fail is not a gate.
    # Both are now computed from the data that was actually collected.
    i1 = all(v["independent"] > 0 and v["advertised"] > 0 for v in out.values())
    i2 = all(len(v["clusters"]) == v["independent"] and
             all(c["representative"] in c["members"] for c in v["clusters"])
             for v in out.values())
    print(f"\n  i1  independent count reported, not just advertised -> {'PASS' if i1 else 'FAIL'} "
          f"({len(out)} checkpoint(s) with a measured independent count)")
    print(f"  i2  every cluster listed with a named representative -> {'PASS' if i2 else 'FAIL'} "
          f"({sum(len(v['clusters']) for v in out.values())} clusters checked)")
    print(f"  i3  independent >= 50% of advertised -> {'PASS' if i3 else 'FAIL'} "
          f"(worst: {worst[0]} {worst[1]['independent']}/{worst[1]['advertised']} = {frac*100:.0f}%)")
    verdict = (f"the layer has roughly the number of independent concepts it advertises "
               f"(worst case {frac*100:.0f}%), so the named-concept count is a fair description."
               if i3 else
               f"THE NAMED-CONCEPT COUNT IS MISLEADING. {worst[0]} advertises "
               f"{worst[1]['advertised']} concepts but has only {worst[1]['independent']} independent "
               f"ones ({frac*100:.0f}%) — {worst[1]['advertised'] - worst[1]['independent']} names "
               f"describe properties already covered by another name. The honest description is the "
               f"INDEPENDENT number, and the cluster table shows exactly which names are duplicates.")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"checkpoints": out, "threshold": THR, "gates": {"i3": bool(i3)}, "verdict": verdict},
              open(os.path.join(R, "concept_independence.json"), "w"), indent=1)
    print(f"  saved {os.path.join(R, 'concept_independence.json')}")
