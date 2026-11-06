"""t3_m3_probe_a2.py — the A2 lie probe under Meter-03.

Same test as t3_m3_probe.py but for the A2 target set: the probe column (2) is frozen
before it learns anything, so it is NEVER supervised on its pre-registered target. Under
Meter-03 it is audited against target 2 -- a target it was never trained on -- and must be
CAUGHT while LIVE (firing >= 2%), i.e. not caught merely by the firing floor.

Also reports its best-scoring target of all 32, so the catch cannot be an unlucky pairing.
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
COL = 2
CK = "lieprobe_K32_lr0.5_lc05_clusters32_seed1.npz"

print("=" * 96)
print("A2 LIE PROBE (Meter-03) — frozen column vs a target it was NEVER trained on")
print("=" * 96)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
targets = M3.load_targets("clusters32")

path = os.path.join(RUNDIR, "ckpt", CK)
if not os.path.exists(path):
    print(f"  MISSING probe checkpoint {CK}")
    raise SystemExit(1)

net = M3.load_ckpt(M3.build_net(1), path)
h, L, A = M3.prep(net, X)
own = M3.audit_col(net, X, h, COL, targets[COL][1], L, A)
allc = [M3.audit_col(net, X, h, COL, targets[d][1], L, A) for d in range(len(targets))]
best_i = int(np.argmax([z["auroc"] if z["auroc"] == z["auroc"] else -1 for z in allc]))
best = allc[best_i]
caught = own["flag"] != "-"
live = own["firing"] >= 0.02

print(f"  probe column {COL}, own target '{targets[COL][0]}'")
print(f"    auroc {own['auroc']:.4f} | contrib {own['contrib']:+.4f} | "
      f"firing {own['firing']:.4f} | verdict {own['flag']}")
print(f"    best of {len(targets)} targets ('{targets[best_i][0]}'): "
      f"auroc {best['auroc']:.4f} contrib {best['contrib']:+.4f} -> {best['flag']}")
print(f"  A2-2 CAUGHT while LIVE: {caught and live}  (caught={caught}, live={live})")
print(f"    caught on its BEST target too: {best['flag'] != '-'}")
clauses = {"auroc<0.55": own["auroc"] < 0.55, "contrib<0.25": own["contrib"] < 0.25,
           "firing<2%": own["firing"] < 0.02}
print("    clause failures: " + ", ".join(f"{k}={'FAIL' if v else 'pass'}"
                                          for k, v in clauses.items()))

json.dump({"col": COL, "own_target": targets[COL][0], "own": own, "caught": caught,
           "live": live, "A2_2_pass": bool(caught and live),
           "best_target": targets[best_i][0], "best": best,
           "best_is_caught": best["flag"] != "-", "clause_failures": clauses},
          open(os.path.join(RUNDIR, "a2_lieprobe.json"), "w"), indent=1)
print("\nsaved runs/t3/a2_lieprobe.json")
print("=" * 96)
