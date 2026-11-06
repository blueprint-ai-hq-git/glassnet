"""dedup_thresholds.py — correlation-based de-duplication has a trap: two concepts can correlate
strongly in THIS dataset while meaning different things. Measure how the merge list changes with the
threshold, so the merge rule can be chosen on evidence instead of taste.

THE TRAP (spotted while reviewing the 0.70 merge list)
    has_low_contrast -> left_right_symmetric   corr 0.860
    is_dark          -> mostly_empty           corr 0.718
  "low contrast" and "left-right symmetric" are DIFFERENT properties. They correlate here only because
  PathMNIST's low-contrast tiles also tend to be symmetric. Merging them deletes a name a reader might
  want. So the merge threshold must be strict enough that only NEAR-INTERCHANGEABLE concepts merge.

WHAT THIS PRINTS
  For thresholds 0.70 / 0.80 / 0.90 / 0.95: how many concepts merge, and the exact merge list, so the
  choice is visible. The strict registries keep more names (safer); the loose one maximises de-dup.
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
import image_concepts_adaptive as ic_mod                  # noqa: E402
import colour_concepts                                    # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "vl")
_base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
_d = vl_data3.prepare(_base, "medical", seed=0)
_Xtr = np.asarray(_d["Xtr"], dtype=np.float32)
_kept, _ = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
_ckept, _ = colour_concepts.build(_Xtr, min_support=0.02, verbose=False)
ALL = list(_kept) + list(_ckept)
support = {c.name: c.support for c in ALL}

X = _Xtr[:6000]
T = np.stack([np.asarray(c.labels(X), dtype=np.float64) for c in ALL], axis=1)
names = [c.name for c in ALL]
n = T.shape[1]
Tz = (T - T.mean(0)) / (T.std(0) + 1e-12)
C = np.clip((Tz.T @ Tz) / len(T), -1, 1)
POS = np.clip(C, 0.0, 1.0)
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


print("=" * 96)
print("MERGE THRESHOLD — how strict must 'duplicate' be?")
print("=" * 96)
print(f"  {n} concepts, correlation measured on 6000 train samples\n")

out = {}
for thr in (0.70, 0.80, 0.90, 0.95):
    groups = complete_link(POS, thr)
    merges = []
    for g in groups:
        members = [names[i] for i in g]
        if len(members) < 2:
            continue
        survivor = max(members, key=lambda nm: support.get(nm, 0.0))
        for m in members:
            if m != survivor:
                merges.append((m, survivor, float(C[names.index(m), names.index(survivor)])))
    out[str(thr)] = {"n_keep": len(groups), "n_merged": len(merges),
                     "merges": [{"dropped": m, "kept": s, "corr": round(c, 3)}
                                for m, s, c in merges]}
    print(f"  threshold {thr}: keep {len(groups):>3} of {n}, merge {len(merges)}")
    for m, s, c in merges:
        print(f"      {m:<26} -> {s:<26} corr {c:.3f}")
    print()

json.dump(out, open(os.path.join(R, "dedup_thresholds.json"), "w"), indent=1)
print("=" * 96)
print("  READ THIS BEFORE CHOOSING: the 0.70 list merges properties that are conceptually DIFFERENT")
print("  (has_low_contrast vs left_right_symmetric at 0.86; is_dark vs mostly_empty at 0.72). Those")
print("  pairs correlate in PathMNIST but name different things a human might want to see. A strict")
print("  0.95 threshold merges only near-interchangeable names. The registry used for training is the")
print("  0.95 one: it de-duplicates what is genuinely duplicated without deleting distinct meanings.")
print(f"  saved {os.path.join(R, 'dedup_thresholds.json')}")
