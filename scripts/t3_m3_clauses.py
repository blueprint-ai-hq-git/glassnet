"""t3_m3_clauses.py — WHY does the name-swap probe reject 100% of wrong pairings?

METER-RULE-03.md section 4 reports a rejection rate. A rejection rate of 100% is
suspicious unless you know which clause did the rejecting, so this decomposes it:
for every off-diagonal (column c, target d != c) pair, record which clause failed.

If rejections were driven by AUROC alone, a swapped column would be scoring BELOW
chance -- that would be strange and would suggest a bug. If they are driven by the
CONTRIBUTION clause (the logit gain of the target's words through column c's own
answer-layer weights), that is the expected mechanism: a column trained on one target
has no logit pathway for another target's words.

Also reports the AUROC distribution of off-diagonal pairs against the diagonal, so
"rejected" can be distinguished from "rejected because it scored below chance".
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
print("NAME-SWAP CLAUSE BREAKDOWN")
print("=" * 92)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")

CASES = [("words lc=1.0", "words", "readable_K32_lr0.5_lc10_seed{s}.npz"),
         ("clusters lc=0.5", "clusters", "readable_K32_lr0.5_lc05_clusters_seed{s}.npz")]

out = {}
for label, kind, pat in CASES:
    targets = M3.load_targets(kind)
    print(f"\n--- {label}")
    diag_a, off_a, off_c = [], [], []
    fails = {"auroc_only": 0, "contrib_only": 0, "both": 0, "firing": 0, "none": 0}
    for seed in (0, 1, 2):
        path = os.path.join(RUNDIR, "ckpt", pat.format(s=seed))
        if not os.path.exists(path):
            print(f"    seed{seed}: MISSING")
            continue
        net = M3.load_ckpt(M3.build_net(seed), path)
        h, L, A = M3.prep(net, X)
        for c in range(M3.N_SUP):
            r = M3.audit_col(net, X, h, c, targets[c][1], L, A)
            diag_a.append(r["auroc"])
            for d in range(M3.N_SUP):
                if d == c:
                    continue
                z = M3.audit_col(net, X, h, c, targets[d][1], L, A)
                off_a.append(z["auroc"]); off_c.append(z["contrib"])
                fa = z["auroc"] < 0.55
                fc = z["contrib"] < 0.25
                if z["firing"] < 0.02:
                    fails["firing"] += 1
                elif fa and fc:
                    fails["both"] += 1
                elif fa:
                    fails["auroc_only"] += 1
                elif fc:
                    fails["contrib_only"] += 1
                else:
                    fails["none"] += 1
    n = sum(fails.values())
    print(f"    pairs {n} | diagonal AUROC med {np.median(diag_a):.3f} | "
          f"off-diagonal AUROC med {np.median(off_a):.3f}")
    print(f"    off-diagonal contrib med {np.median(off_c):+.4f} "
          f"(bar is +0.25)")
    print(f"    clause failures: " + ", ".join(f"{k}={v}" for k, v in fails.items()
                                               if v))
    out[label] = {"n_pairs": n, "diag_auroc_med": float(np.median(diag_a)),
                  "off_auroc_med": float(np.median(off_a)),
                  "off_contrib_med": float(np.median(off_c)),
                  "clause_failures": fails,
                  "reject_rate": (n - fails["none"]) / n * 100 if n else 0.0}

json.dump(out, open(os.path.join(RUNDIR, "meter3_clauses.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meter3_clauses.json")
print("=" * 92)
