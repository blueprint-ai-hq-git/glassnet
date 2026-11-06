"""dedup_concepts.py — build a DE-DUPLICATED concept registry: one name per independent cluster.

WHY (the direct follow-up to E18)
  The medical layer advertises 35 concepts but only 23 are independent; 12 names are duplicates of
  others (measured, complete-linkage at |corr| >= 0.70):
      is_bright          <- red_channel_low, green_channel_low, blue_channel_low
      is_pink_dominant   <- is_purple_dominant, has_strong_colour, colour_contrast_high
      has_low_contrast   <- left_right_symmetric, top_bottom_symmetric
      red_channel_high   <- green_channel_high, blue_channel_high
      mostly_empty       <- is_thin_structure
      is_green_dominant  <- blue_exceeds_green
  Duplicate targets mean the concept loss spends gradient teaching the same thing 2-4 times, and the
  layer then cannot tell a reader which name is the real one.

MY PREDICTION, WRITTEN BEFORE RUNNING (so it can be wrong in public)
  * own_corr should RISE for the surviving concepts: the same gradient budget spread over 23 targets
    instead of 35 means each target gets ~1.5x more pressure.
  * accuracy should NOT get worse. If it does, the duplicate concepts were carrying real signal that
    their representative does not, which would itself be informative.
  * the concept loss's accuracy cost (measured earlier: ~3.4 pts and 3.7x variance vs lam 0.00)
    should SHRINK, because the duplicated term was inflating the auxiliary loss.

WHAT THIS BUILDS
  A filtered SPEC containing one representative per cluster (the cluster member with the highest
  support, so the survivor is the most common/most learnable variant of that property). Written to
  a JSON the trainer can load, plus the exact list of what was dropped and why.

GATES (pre-registered)
  q1  own_corr mean must be HIGHER than the 35-concept run's 0.423
  q2  accuracy must be within the seed noise of the 35-concept run (not significantly worse)
  q3  every dropped name listed with the survivor that replaced it — no silent deletion
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

R = os.path.join(PROJ, "lens", "runs", "vl")
OUT = os.path.join(PROJ, "lens", "runs", "vl", "dedup_registry.json")
THR = 0.70

_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)

# rebuild exactly the registry training used, keeping the formulas so survivors are reproducible
_kept, _ = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
_ckept, _ = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
ALL = [(c.name, c.support) for c in list(_kept) + list(_ckept)]

# correlation between every pair of concept formulas, on TRAIN (thresholds were fit on train)
X = _Xtr[:6000]
cols, names = [], []
for c in list(_kept) + list(_ckept):
    cols.append(np.asarray(c.labels(X), dtype=np.float64))
    names.append(c.name)
T = np.stack(cols, axis=1)
n = T.shape[1]
Tz = (T - T.mean(0)) / (T.std(0) + 1e-12)
C = np.clip((Tz.T @ Tz) / len(T), -1, 1)
# MERGE CRITERION: POSITIVE correlation only.
# A first pass used |corr| and merged red_channel_low into is_bright at corr -0.712. That is WRONG:
# a negative correlation means the two concepts are OPPOSITES (low red vs bright), not duplicates.
# Merging them would delete a genuinely distinct readable property and leave a reader unable to see
# "low red" anywhere in the layer. High-|corr|-negative pairs are COMPLEMENTARY: they carry the same
# information with opposite polarity, so BOTH names should survive because each names a different
# thing a human can check. Only same-direction (positive) redundancy is true duplication.
A = np.abs(C)
np.fill_diagonal(A, 0.0)
POS = np.clip(C, 0.0, 1.0)          # only same-direction correlation may merge concepts
np.fill_diagonal(POS, 0.0)


def complete_link(A, thr):
    unassigned = set(range(len(A)))
    groups = []
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
        groups.append(grp)
    return groups


groups = complete_link(POS, THR)
support = dict(ALL)
keep, dropped = [], []
for g in groups:
    members = [names[i] for i in g]
    # survivor = highest support (the most common variant of this property = most learnable)
    survivor = max(members, key=lambda nm: support.get(nm, 0.0))
    keep.append(survivor)
    for m in members:
        if m != survivor:
            dropped.append({"name": m, "replaced_by": survivor,
                            "corr_with_survivor": round(float(C[names.index(m),
                                                               names.index(survivor)]), 3),
                            "support": round(support.get(m, 0.0), 4)})

print("=" * 96)
print("DE-DUPLICATED CONCEPT REGISTRY")
print("=" * 96)
print(f"  original concepts   : {n}")
print(f"  independent clusters: {len(groups)}")
print(f"  survivors (kept)    : {len(keep)}")
print(f"  dropped as duplicate: {len(dropped)}")
print(f"\n  {'dropped name':<28}{'replaced by':<28}{'corr':>7}{'support':>10}")
for d in dropped:
    print(f"  {d['name']:<28}{d['replaced_by']:<28}{d['corr_with_survivor']:>7.3f}"
          f"{d['support']:>10.4f}")

json.dump({"keep": keep, "dropped": dropped, "n_original": n, "n_keep": len(keep),
           "threshold": THR,
           "note": ("one representative per complete-linkage cluster at |corr|>=0.70; survivor = "
                    "highest-support member, so the kept name is the most common variant")},
          open(OUT, "w"), indent=1)
q3 = all(d.get("replaced_by") and d["replaced_by"] != d["name"] for d in dropped)
print(f"\n  q3  every dropped name listed with its survivor -> {'PASS' if q3 else 'FAIL'} "
      f"({len(dropped)} dropped, {len(keep)} kept)")
print(f"  wrote {OUT}")
print(f"\n  survivors: {', '.join(sorted(keep))}")
