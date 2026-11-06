"""t3_m3_probe_x.py — the lie probe under Meter-03, for any target set (env-driven).

Generalises t3_m3_probe_a2.py so A2 and A3 use identical probe logic:
  T3P_KIND  target set: clusters16 | clusters32 | clusters | words
  T3P_K      number of concept columns
  T3P_NSUP   number of supervised columns
  T3P_CK     probe checkpoint filename (may contain no {s})
  T3P_COL    the frozen probe column
  T3P_SEED   the seed whose net architecture to rebuild

The probe column is frozen before it learns anything, so it is NEVER supervised on its
pre-registered target. Under Meter-03 it is audited against that target and must be CAUGHT
while LIVE (firing >= 2%), i.e. not caught merely by the firing floor. Also reports its
best-scoring target of all of them, so the catch cannot be an unlucky pairing.
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
KIND = os.environ.get("T3P_KIND", "clusters16")
K = int(os.environ.get("T3P_K", "16"))
NSUP = int(os.environ.get("T3P_NSUP", "10"))
CK = os.environ["T3P_CK"]
COL = int(os.environ.get("T3P_COL", "2"))
SEED = int(os.environ.get("T3P_SEED", "1"))
OUT = os.environ.get("T3P_OUT", "a3_lieprobe.json")

print("=" * 96)
print(f"LIE PROBE (Meter-03) — {KIND}, frozen column vs a target it was NEVER trained on")
print("=" * 96)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
targets = M3.load_targets(KIND)
print(f"  target set {KIND}: {len(targets)} pre-registered targets, {NSUP} supervised")

path = os.path.join(RUNDIR, "ckpt", CK)
if not os.path.exists(path):
    print(f"  MISSING probe checkpoint {CK}")
    raise SystemExit(1)

net = M3.load_ckpt(M3.build_net(SEED), path)
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
print(f"  CAUGHT while LIVE: {caught and live}  (caught={caught}, live={live})")
print(f"    caught on its BEST target too: {best['flag'] != '-'}")
clauses = {"auroc<0.55": own["auroc"] < 0.55, "contrib<0.25": own["contrib"] < 0.25,
           "firing<2%": own["firing"] < 0.02}
print("    clause failures: " + ", ".join(f"{k}={'FAIL' if v else 'pass'}"
                                          for k, v in clauses.items()))

json.dump({"kind": KIND, "col": COL, "own_target": targets[COL][0], "own": own,
           "caught": caught, "live": live, "pass": bool(caught and live),
           "best_target": targets[best_i][0], "best": best,
           "best_is_caught": best["flag"] != "-", "clause_failures": clauses},
          open(os.path.join(RUNDIR, OUT), "w"), indent=1)
print(f"\nsaved runs/t3/{OUT}")
print("=" * 96)
