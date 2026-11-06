"""dbg_effrank.py — is "effective rank 2.0" a real property of the concept set, or my metric bug?

THE CONTRADICTION
  Complete-linkage clustering at |corr| >= 0.70 found 23 independent clusters out of 35 concepts.
  But the effective-rank number came out 2.0. If the concept set really lived in ~2 dimensions, then
  almost everything would be correlated with everything else and clustering at 0.70 would give 1-3
  clusters, not 23. One of the two numbers must be wrong.

THE SUSPECT
  I computed eigenvalues of C @ C.T, where C is the correlation matrix. That SQUARES the spectrum:
  sum(eig(C C^T)) = ||C||_F^2, which is not n, so the participation ratio computed from it is not the
  effective rank of the correlation matrix. The standard effective rank uses eigenvalues of C itself,
  whose trace is n (every diagonal entry is 1).

THIS SCRIPT computes both, plus the singular-value version, and prints them side by side so the
discrepancy is visible rather than argued about.
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
_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)
_kept, _ = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _kept]
_ckept, _ = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
image_concepts.SPEC = list(image_concepts.SPEC) + [(c.name, c.fn, c.formula) for c in _ckept]

cands = sorted([p for p in os.listdir(R) if p.startswith("mlow_lc0.15_ep") and p.endswith(".npz")],
               key=lambda s: int(s.split("ep")[1].split(".")[0]))
net, _m = load_vision(os.path.join(R, cands[-1]), device="cpu")
X = np.asarray(_d["Xte"][:4000], dtype=np.float32)
Q = np.asarray(_d["Qte"][:4000], dtype=np.int64)
names = [c.name for c in net.concepts]
T = net.concept_targets(X, Q).astype(np.float64)
n = T.shape[1]

# correlation matrix
Tz = (T - T.mean(0)) / (T.std(0) + 1e-12)
C = (Tz.T @ Tz) / len(T)
C = np.clip(C, -1, 1)
np.fill_diagonal(C, 1.0)


def eff_rank(vals):
    vals = np.clip(np.asarray(vals, dtype=np.float64), 0, None)
    s = vals.sum()
    return float(s * s / (vals * vals).sum()) if (vals * vals).sum() > 0 else 0.0


ev_C = np.linalg.eigvalsh(C)
ev_CTC = np.linalg.eigvalsh(C @ C.T)
sv_T = np.linalg.svd(Tz, compute_uv=False)
ev_cov = np.linalg.eigvalsh(np.cov(T.T))

print("=" * 96)
print("EFFECTIVE RANK — which computation is the honest one?")
print("=" * 96)
print(f"  concepts n = {n}, samples = {len(T)}")
print(f"\n  eigenvalues of C (correlation matrix)      trace = {ev_C.sum():.2f}   "
      f"(should be {n}.00)")
print(f"    effective rank = {eff_rank(ev_C):.2f}")
print(f"  eigenvalues of C @ C.T                     trace = {ev_CTC.sum():.2f}   "
      f"(= ||C||_F^2, NOT n)")
print(f"    effective rank = {eff_rank(ev_CTC):.2f}    <- the number I reported before")
print(f"  singular values of the standardised T      sum(s^2) = {(sv_T**2).sum():.2f}")
print(f"    effective rank = {eff_rank(sv_T):.2f}")
print(f"  eigenvalues of cov(T)                      trace = {ev_cov.sum():.2f}")
print(f"    effective rank = {eff_rank(ev_cov):.2f}")

# how much of the variance is in the top few directions of C?
tot = ev_C.sum()
cum = np.cumsum(np.sort(ev_C)[::-1]) / tot
print(f"\n  share of C's variance in the top k directions:")
for k in (1, 2, 3, 5, 10, 20):
    if k <= n:
        print(f"    top {k:>2}: {cum[k-1]*100:6.1f}%")

# and the decisive cross-check: if the set lived in 2 dimensions, clustering at 0.70 would collapse
A = np.abs(C)
np.fill_diagonal(A, 0.0)
pairs_above = int((np.triu(A, 1) >= 0.70).sum())
print(f"\n  pairs with |corr| >= 0.70 : {pairs_above} of {n*(n-1)//2}")
print(f"  mean |corr| off-diagonal  : {np.triu(A, 1).sum() / (n*(n-1)/2):.3f}")
print(f"\n  If the set truly lived in ~2 dimensions, nearly ALL pairs would exceed 0.70 and clustering")
print(f"  would give 1-3 groups. {pairs_above} of {n*(n-1)//2} pairs exceed it, which is consistent")
print(f"  with the 23 clusters found, NOT with a rank of 2.")
json.dump({"eff_rank_C": eff_rank(ev_C), "eff_rank_CCT": eff_rank(ev_CTC),
           "eff_rank_sv": eff_rank(sv_T), "pairs_above_070": pairs_above, "n": n},
          open(os.path.join(R, "effrank_dbg.json"), "w"), indent=1)
print(f"\n  saved {os.path.join(R, 'effrank_dbg.json')}")
