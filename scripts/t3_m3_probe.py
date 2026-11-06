"""t3_m3_probe.py — Meter-03 lie probe: audit the FROZEN column against a target it
was NEVER trained on.

METER-RULE-03.md section 2. The probe column trains normally until its validation
firing reaches 2%, then its gradients are frozen, so it is never trained on any
target. Under Meter-03 it is audited against the PRE-REGISTERED target for its column
index -- a target it was never trained on -- and must be CAUGHT.

Also reports, for contrast:
  * the Meter-02 verdict (namer-derived name), read from the existing audit JSON,
  * the probe's best-matching target out of all 16 (a generous upper bound): if the
    probe cannot pass even its BEST target, the catch is not an unlucky pairing.
  * the per-clause breakdown (which clause did the catching).
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
import run_t2_iter003 as M
import t3_audit_m3 as M3

RUNDIR = os.path.join(PROJ, "runs", "t3")

print("=" * 96)
print("METER-03 LIE PROBE — frozen column vs a target it was NEVER trained on")
print("=" * 96)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
targets = M3.load_targets("words")
set_backend("numpy")

CASES = [
    ("supervised lc=1.0 lr=0.5", "lieprobe_K32_lr0.5_lc10_seed1.npz",
     "audit_K32_lr05_pa_lc10_ff002.json"),
    # NOTE: the baseline tag is "_lc00", not "_lc0": the name is built from
    # str(0.0).replace(".", "") == "00".
    ("baseline lc=0 lr=1.0", "lieprobe_K32_lr1.0_lc00_seed1.npz",
     "audit_K32_lr10_pa_ff002.json"),
]

out = {}
for label, ckname, m2json in CASES:
    ck = os.path.join(RUNDIR, "ckpt", ckname)
    print(f"\n--- {label}")
    if not os.path.exists(ck):
        print(f"    MISSING probe checkpoint {ckname}")
        continue
    net = M3.load_ckpt(M3.build_net(1), ck)
    h, L, A = M3.prep(net, X)
    col = 2
    own = M3.audit_col(net, X, h, col, targets[col][1], L, A)
    allc = [M3.audit_col(net, X, h, col, targets[d][1], L, A) for d in range(16)]
    best = max(allc, key=lambda z: z["auroc"] if z["auroc"] == z["auroc"] else -1)
    best_d = int(np.argmax([z["auroc"] if z["auroc"] == z["auroc"] else -1 for z in allc]))
    caught = own["flag"] != "-"
    m2 = {}
    p2 = os.path.join(RUNDIR, m2json)
    if os.path.exists(p2):
        m2 = json.load(open(p2)).get("lie", {})
    print(f"    own target ({targets[col][0]}):  auroc {own['auroc']:.4f} "
          f"contrib {own['contrib']:+.4f} firing {own['firing']:.4f} -> {own['flag']}")
    print(f"    BEST of 16 targets ({targets[best_d][0]}): auroc {best['auroc']:.4f} "
          f"contrib {best['contrib']:+.4f} -> {best['flag']}")
    print(f"    METER-03 verdict: {'CAUGHT' if caught else 'MISSED'}")
    print(f"    (Meter-02 said: auroc {m2.get('auroc')} contrib {m2.get('contrib')} "
          f"firing {m2.get('firing')} -> {'CAUGHT' if m2.get('caught') else 'MISSED'})")
    out[label] = {"own_target": targets[col][0], "own": own, "caught": caught,
                  "best_target": targets[best_d][0], "best": best,
                  "best_is_caught": best["flag"] != "-",
                  "meter02": {k: m2.get(k) for k in ("auroc", "contrib", "firing",
                                                     "caught", "live",
                                                     "caught_by_floor_only")}}
    # per-clause: WHY was it caught (or not)?
    clauses = {"auroc": own["auroc"] < 0.55, "contrib": own["contrib"] < 0.25,
               "firing": own["firing"] < 0.02}
    print(f"    clause failures: " + ", ".join(f"{k}={'FAIL' if v else 'pass'}"
                                               for k, v in clauses.items()))
    out[label]["clause_failures"] = clauses

json.dump(out, open(os.path.join(RUNDIR, "meter3_lieprobe.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meter3_lieprobe.json")
print("=" * 96)
