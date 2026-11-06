"""t3_a2_faildiag.py — why do 6 of 16 taught clusters fail Meter-03? Per column.

Owner asked for this diagnosis explicitly. Prediction written first in TARGET-A2.md §5:
the failures are dominated by the FIRING FLOOR (a column-level property), not by AUROC.

Re-score of existing checkpoints (clusters lc=0.5, lr=0.5, 3 seeds). No training.
Reports per taught column: firing rate, contribution, AUROC against its own
pre-registered target, the verdict, and WHICH clause failed.
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
PAT = "readable_K32_lr0.5_lc05_clusters_seed{s}.npz"
N_SUP = 16

print("=" * 108)
print("WHY THE 6 TAUGHT CLUSTERS FAIL — per column, per clause (Meter-03, 2% floor)")
print("=" * 108)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
targets = M3.load_targets("clusters")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters.json")))
supports = [c["support"] for c in CLU["clusters"]]
member_counts = [len(c["words"]) for c in CLU["clusters"]]

per_col = {c: [] for c in range(N_SUP)}
for seed in (0, 1, 2):
    path = os.path.join(RUNDIR, "ckpt", PAT.format(s=seed))
    if not os.path.exists(path):
        print(f"  seed{seed}: MISSING")
        continue
    net = M3.load_ckpt(M3.build_net(seed), path)
    h, L, A = M3.prep(net, X)
    for c in range(N_SUP):
        per_col[c].append(M3.audit_col(net, X, h, c, targets[c][1], L, A))

print(f"\n{'col':>3} {'train sup':>9} {'mem':>4} {'firing':>8} {'contrib':>9} "
      f"{'auroc':>7} {'verdict':>8}  failing clause")
tally = {"firing": 0, "auroc": 0, "contrib": 0, "pass": 0}
rows_out = []
for c in range(N_SUP):
    rs = per_col[c]
    if not rs:
        continue
    fr = float(np.mean([r["firing"] for r in rs]))
    co = float(np.mean([r["contrib"] for r in rs]))
    au = float(np.mean([r["auroc"] for r in rs]))
    # which clause fails, using the same precedence the meter uses
    if fr < 0.02:
        clause = "FIRING (<2%)"
        tally["firing"] += 1
    elif au < 0.55:
        clause = "AUROC (<0.55)"
        tally["auroc"] += 1
    elif co < 0.25:
        clause = "CONTRIB (<0.25)"
        tally["contrib"] += 1
    else:
        clause = "-"
        tally["pass"] += 1
    lab = "_".join(CLU["clusters"][c]["words"][:3])
    print(f"{c:>3} {supports[c]:>9.4f} {member_counts[c]:>4} {fr:>8.4f} "
          f"{co:>+9.4f} {au:>7.4f} {('-' if clause == '-' else 'BROKEN'):>8}  "
          f"{clause:<14} {lab[:28]}")
    rows_out.append({"col": c, "support": supports[c], "members": member_counts[c],
                     "firing": fr, "contrib": co, "auroc": au,
                     "clause": clause})

print(f"\nfailure tally over the 16 taught columns: "
      f"FIRING={tally['firing']}, AUROC={tally['auroc']}, CONTRIB={tally['contrib']}, "
      f"pass={tally['pass']}")
n_fail = tally["firing"] + tally["auroc"] + tally["contrib"]
print(f"failing columns: {n_fail}/16")
print(f"PREDICTION (TARGET-A2.md §5: firing-dominated) -> "
      f"{'CONFIRMED' if tally['firing'] >= max(tally['auroc'], tally['contrib'], 1) else 'NOT CONFIRMED'}")
# does failure track support? (thin targets are harder)
f_sup = [r["support"] for r in rows_out if r["clause"] != "-"]
p_sup = [r["support"] for r in rows_out if r["clause"] == "-"]
if f_sup and p_sup:
    print(f"median TRAIN support: failing {np.median(f_sup):.4f} vs "
          f"passing {np.median(p_sup):.4f}")
print(f"median firing: failing cols {np.median([r['firing'] for r in rows_out if r['clause']!='-']):.4f} "
      f"vs passing {np.median([r['firing'] for r in rows_out if r['clause']=='-']):.4f}")

json.dump({"rows": rows_out, "tally": tally, "n_fail": n_fail},
          open(os.path.join(RUNDIR, "a2_faildiag.json"), "w"), indent=1)
print("\nsaved runs/t3/a2_faildiag.json")
print("=" * 108)
