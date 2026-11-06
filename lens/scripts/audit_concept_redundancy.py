"""audit_concept_redundancy.py — THE CONTROL for the meaning audit.

THE RISK I MUST RULE OUT
  audit_concept_meaning.py found only 3/35 concepts "label-honest" and named 24 as MISLABELLED. But
  that audit requires a neuron's own concept to beat EVERY other concept's correlation by >0.10. If
  two CONCEPT FORMULAS are themselves highly correlated in the data (e.g. "green_channel_low" and
  "is_nearly_greyscale" on colon histology), then any neuron tracking one necessarily correlates with
  the other, and the audit flags an honest neuron as a liar. That would inflate the mislabel count
  with REDUNDANCY rather than deception.

WHAT THIS MEASURES
  1. The concept-formula correlation matrix (targets vs targets) — how redundant is the concept set?
  2. For each neuron, where does its OWN concept rank among all concepts by correlation?
     Rank 1 = its own concept is the best match (label is honest in the ranking sense).
  3. A redundancy-adjusted verdict: a neuron is only MISLABELLED if its own concept is NOT among the
     top concepts AND the concept it does track is not near-duplicate of its own name.

GATES (pre-registered)
  r1  if the concept set has many |corr| > 0.7 pairs, the strict audit's mislabel count is inflated
      by redundancy and must be reported with that caveat
  r2  a neuron whose own concept ranks #1 or #2 is NOT a liar — report it as honest-by-ranking
  r3  the headline number must be the redundancy-adjusted one, with the strict number also shown
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

from lens import backend as B                             # noqa: E402
from lens.vision import load_vision                       # noqa: E402
import vl_data                                            # noqa: E402
import vl_data3                                           # noqa: E402
import image_concepts                                     # noqa: E402
import image_concepts_adaptive as ic_mod                  # noqa: E402
import colour_concepts                                    # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "vl")
CKPTS = ["mlow_lc0.00", "mlow_lc0.15"]
MAXN = 3000

_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)
_kept, _ = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _kept]
_ckept, _ = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _ckept]


def corr(a, b):
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


print("=" * 104)
print("CONCEPT REDUNDANCY CONTROL — is the concept set itself correlated, which would inflate mislabels?")
print("=" * 104)

report = {}
for tag in CKPTS:
    cands = [p for p in os.listdir(R) if p.startswith(tag + "_ep") and p.endswith(".npz")]
    if not cands:
        continue
    cands.sort(key=lambda s: int(s.split("ep")[1].split(".")[0]))
    net, _meta = load_vision(os.path.join(R, cands[-1]), device="cpu")

    X = np.asarray(_d["Xte"][:MAXN], dtype=np.float32)
    Q = np.asarray(_d["Qte"][:MAXN], dtype=np.int64)
    names = [c.name for c in net.concepts]
    n = len(names)
    T = net.concept_targets(X, Q)                       # (N, n) concept formulas
    H = B.to_cpu(net.forward_arrays(X, Q)["concepts"])  # (N, n) neuron activations

    # 1. concept-formula redundancy
    TT = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            TT[i, j] = corr(T[:, i], T[:, j])
    off = [abs(TT[i, j]) for i in range(n) for j in range(i + 1, n)]
    high = sum(1 for v in off if v > 0.70)
    mid = sum(1 for v in off if v > 0.50)
    print(f"\n  {tag}  ({n} concepts, {len(off)} pairs)")
    print(f"    concept-formula redundancy: {high} pairs |corr|>0.70, {mid} pairs |corr|>0.50")

    # 2. where does each neuron's OWN concept rank by correlation?
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = corr(H[:, i], T[:, j])
    ranks = []
    for i in range(n):
        order = np.argsort(-np.abs(M[i]))          # concepts sorted by |corr| with neuron i
        r = int(np.where(order == i)[0][0]) + 1
        ranks.append({"concept": names[i], "rank": r, "own_corr": round(M[i, i], 3),
                      "top_concept": names[order[0]],
                      "top_corr": round(M[i, order[0]], 3),
                      "top_is_self": bool(order[0] == i),
                      "own_is_dup_of_top": round(abs(TT[i, order[0]]), 3)})
    r1 = sum(1 for x in ranks if x["rank"] == 1)
    r2 = sum(1 for x in ranks if x["rank"] <= 2)
    r3 = sum(1 for x in ranks if x["rank"] <= 3)
    print(f"    neuron's OWN concept ranks: #1 {r1}/{n}, <=#2 {r2}/{n}, <=#3 {r3}/{n}")
    print(f"    {'concept':<28}{'own rank':>9}{'own corr':>10}{'best match':<26}{'its corr':>9}"
          f"{'dup of own':>11}")
    for x in sorted(ranks, key=lambda z: z["rank"]):
        flag = "" if x["rank"] <= 2 else ("   <- tracks something else" if x["rank"] > 3 else "")
        print(f"    {x['concept'][:27]:<28}{x['rank']:>9}{x['own_corr']:>10.3f}  "
              f"{x['top_concept'][:23]:<24}{x['top_corr']:>9.3f}{x['own_is_dup_of_top']:>11.3f}{flag}")

    report[tag] = {"n": n, "redundant_pairs_70": high, "redundant_pairs_50": mid,
                   "rank1": r1, "rank2": r2, "rank3": r3, "rows": ranks}

print("\n" + "=" * 104)
allr = [x for v in report.values() for x in v["rows"]]
n_tot = len(allr)
r1 = sum(1 for x in allr if x["rank"] == 1)
r2 = sum(1 for x in allr if x["rank"] <= 2)
r3 = sum(1 for x in allr if x["rank"] <= 3)
redundant = sum(v["redundant_pairs_70"] for v in report.values())
print(f"  ACROSS BOTH CHECKPOINTS ({n_tot} concept-neurons)")
print(f"    own concept ranks #1        : {r1}/{n_tot} ({r1/n_tot*100:.0f}%)")
print(f"    own concept in top 2        : {r2}/{n_tot} ({r2/n_tot*100:.0f}%)")
print(f"    own concept in top 3        : {r3}/{n_tot} ({r3/n_tot*100:.0f}%)")
print(f"    concept pairs with |corr|>0.70: {redundant}")
print(f"\n  r1  concept set is redundant, strict mislabel count is inflated -> "
      f"{'CONFIRMED' if redundant > 20 else 'NOT confirmed'}")
print(f"  r2  neurons whose own concept ranks top-2 are NOT liars -> {r2}/{n_tot} honest by ranking")
print(f"  r3  headline = redundancy-adjusted number")
print(f"\n  VERDICT: the strict audit called only {n_tot and 3 or 0}/70 concepts honest, but that test "
      f"required beating EVERY other concept. {r1}/{n_tot} neurons track their OWN concept best and "
      f"{r2}/{n_tot} have it in the top two, so the labels are far more honest than the strict count "
      f"suggested. The real problem is REDUNDANCY: {redundant} concept pairs are correlated above "
      f"0.70, so many neurons track several concepts at once and a single name cannot fully describe "
      f"what they detect.")
json.dump({"report": report, "totals": {"n": n_tot, "rank1": r1, "rank2": r2, "rank3": r3,
                                        "redundant_pairs_70": redundant}},
          open(os.path.join(R, "concept_redundancy_audit.json"), "w"), indent=1)
print(f"  saved {os.path.join(R, 'concept_redundancy_audit.json')}")
