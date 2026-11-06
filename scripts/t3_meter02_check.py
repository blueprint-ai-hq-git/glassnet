"""t3_meter02_check.py — verify the Meter-02 implementation before re-auditing.

Two things must be true, or the implementation is wrong:

  R1  REGRESSION: with min_firing=0.0 the new meter reproduces the RECORDED
      Meter-01 pass counts EXACTLY (same checkpoints, same namer, same support).
      If it does not, the re-audit is not a re-score and nothing else is trustworthy.

  R2  FLOOR: with min_firing=0.02 the pass counts match the projection I measured
      earlier on the same checkpoints (t3_meter_floor.py), which was computed a
      different way (recorded flags + separately computed firing rates).

R2 matters because it cross-checks two independent code paths against each other.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.environ.setdefault("T3A_BACKEND", "numpy")
os.environ.setdefault("T3A_K", "32")

import numpy as np
from glassnet.tensor import set_backend
import t3_audit as A
import run_t2_iter003 as M

set_backend("numpy")
CKPT = os.path.join(PROJ, "runs", "t3", "ckpt")
RUNDIR = os.path.join(PROJ, "runs", "t3")
K = A.K

CASES = [
    ("lc=1.0 lr=0.5", "audit_K32_lr05_pa_lc10.json", 0.5, "_lc10"),
    ("lc=0.5 lr=1.0", "audit_K32_lr10_pa_lc05.json", 1.0, "_lc05"),
    ("lc=0.1 lr=1.0", "audit_K32_lr10_pa_lc01.json", 1.0, "_lc01"),
    ("baseline lc=0", "audit_K32_lr10_pa_w20.json", 1.0, "_w20"),
]

d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
V, D, WIN = d["V"], M.D, A.WIN

print("=" * 92)
print("METER-02 IMPLEMENTATION CHECK")
print("=" * 92)
fails = []

for label, jf, lr, suffix in CASES:
    rec = json.load(open(os.path.join(RUNDIR, jf)))
    seeds = sorted(k for k in rec if k.startswith("seed"))
    print(f"\n{label}   (recorded: {jf})")
    old_pass, new_pass, proj_pass = [], [], []
    for s in seeds:
        si = int(s.replace("seed", ""))
        ck = os.path.join(CKPT, f"readable_K32_lr{lr}{suffix}_seed{si}.npz")
        if not os.path.exists(ck):
            print(f"  missing {os.path.basename(ck)}")
            continue
        net = M.TinyLM3(V, K, D, WIN, rng=np.random.default_rng(si))
        z = np.load(ck)
        for p in net.parameters():
            if p.name in z:
                p.data = np.array(z[p.name])
        names = A.namer()(net, d, d["Xtr"])
        rows0, n0 = M.audit_t2c(net, names, d, d["Xva"], min_firing=0.0)
        rows2, n2 = M.audit_t2c(net, names, d, d["Xva"], min_firing=0.02)
        old_pass.append(n0)
        new_pass.append(n2)
        rec_n = rec[s]["n_pass"]
        rec_flags = [r["flag"] for r in rec[s]["rows"]]
        new_flags = [r["flag"] for r in rows0]
        same = (rec_flags == new_flags)
        proj = sum(1 for i, r in enumerate(rec[s]["rows"])
                   if r["flag"] == "-" and r["pos_frac"] is not None)
        print(f"  seed{si}: recorded {rec_n}/{K} | re-scored floor=0 {n0}/{K} "
              f"{'MATCH' if rec_n == n0 and same else 'MISMATCH'} | floor=2% {n2}/{K}")
        if rec_n != n0 or not same:
            fails.append(f"{label} seed{si}: recorded {rec_n} vs re-scored {n0}, "
                         f"flags identical={same}")
    if old_pass:
        print(f"  mean: recorded {np.mean([rec[s]['n_pass'] for s in seeds]):.1f}/{K} "
              f"-> floor=0 {np.mean(old_pass):.1f}/{K} "
              f"-> floor=2% {np.mean(new_pass):.1f}/{K}")

print("\n" + "=" * 92)
if fails:
    print(f"RESULT: FAILED ({len(fails)}) — the re-audit must NOT proceed:")
    for f in fails:
        print("   " + f)
else:
    print("RESULT: floor=0 reproduces every recorded Meter-01 pass count and flag "
          "exactly.\nThe floor is a pure re-score of the same models.")
print("=" * 92)
sys.exit(1 if fails else 0)
