"""t3_m3_diag.py — two diagnostics the name-swap result needs before it means anything.

(1) The firing floor is a COLUMN property, not a pair property. If a column is below
    the floor it fails EVERY pairing, including its own -- so a high rejection rate can
    be produced entirely by dead columns and would say nothing about whether the audit
    detects a WRONG pairing. This recomputes rejection restricted to LIVE columns.

(2) For single-word targets, a word that never occurs in the validation split has an
    undefined AUROC (nan), which fails the >= 0.55 clause automatically. That inflates
    the failure count for reasons that have nothing to do with the model. This reports
    each target's validation positive count.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet.tensor import set_backend
from glassnet import tiny_data
import t3_audit_m3 as M3

RUNDIR = os.path.join(PROJ, "runs", "t3")

print("=" * 92)
print("DIAGNOSTICS FOR THE NAME-SWAP AND DIAGONAL RESULTS")
print("=" * 92)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
vmap = {w: i for i, w in enumerate(DATA["vocab"])}

# ---- (2) target occurrence in validation -----------------------------------
print("\n(2) TARGET OCCURRENCE IN THE VALIDATION SPLIT (10,124 windows)")
out = {}
for kind in ("words", "clusters"):
    targets = M3.load_targets(kind)
    print(f"\n  {kind}:")
    miss = 0
    for i, (lab, ids) in enumerate(targets):
        ids = [j for j in ids if j < len(DATA["vocab"])]
        n = int(np.stack([np.isin(X, j).any(axis=1) for j in ids]).any(0).sum()) if ids else 0
        if n < 10:
            miss += 1
        print(f"    col {i:>2} {lab[:34]:<34} val positives {n:>6} "
              f"({n/len(X)*100:5.2f}%)" + ("   <- AUROC undefined/weak" if n < 10 else ""))
    out[f"{kind}_targets_below_10_val_pos"] = miss
    print(f"    targets with <10 validation positives: {miss}/16")

# ---- (1) rejection restricted to LIVE columns -------------------------------
print("\n(1) NAME-SWAP REJECTION, ALL PAIRS vs LIVE-COLUMN PAIRS ONLY")
print("    (a pair can only test target-matching if the column is above the floor)")
for label, kind, pat in [("words lc=1.0", "words", "readable_K32_lr0.5_lc10_seed{s}.npz"),
                         ("clusters lc=0.5", "clusters",
                          "readable_K32_lr0.5_lc05_clusters_seed{s}.npz")]:
    targets = M3.load_targets(kind)
    allr, liver = [], []
    for seed in (0, 1, 2):
        path = os.path.join(RUNDIR, "ckpt", pat.format(s=seed))
        if not os.path.exists(path):
            continue
        net = M3.load_ckpt(M3.build_net(seed), path)
        h, L, A = M3.prep(net, X)
        for c in range(M3.N_SUP):
            live = float((h[:, c] > 0.5).mean()) >= 0.02
            for d in range(M3.N_SUP):
                if d == c:
                    continue
                z = M3.audit_col(net, X, h, c, targets[d][1], L, A)
                allr.append(z["flag"] != "-")
                if live:
                    liver.append((z["flag"] != "-", z["contrib"], z["auroc"]))
    ra = np.mean(allr) * 100 if allr else 0.0
    rl = np.mean([x[0] for x in liver]) * 100 if liver else 0.0
    print(f"\n  {label}:")
    print(f"    all pairs          : {ra:5.1f}% rejected  (n={len(allr)})")
    print(f"    LIVE columns only  : {rl:5.1f}% rejected  (n={len(liver)})")
    if liver:
        print(f"    among live pairs: contrib med {np.median([x[1] for x in liver]):+.4f} "
              f"(bar +0.25), auroc med {np.median([x[2] for x in liver]):.3f}")
    out[label] = {"reject_all": ra, "reject_live": rl, "n_all": len(allr),
                  "n_live": len(liver)}

json.dump(out, open(os.path.join(RUNDIR, "meter3_diag.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meter3_diag.json")
print("=" * 92)
