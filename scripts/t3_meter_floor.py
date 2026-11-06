"""t3_meter_floor.py — what would a STRICTER meter do?

The lie probe exposed a hole: a nearly dead column (fires on 1.15% of windows) can
be named to match exactly its own few firings and then passes with AUROC 1.0. The
same hole lets a handful of dead columns into the supervised pass fraction.

This does NOT change any rule. It only MEASURES, on the existing checkpoints, what
the pass fractions would be if the meter ALSO required the column to actually fire
on at least X% of validation windows. Reported for the owner to decide on.

Columns are matched to audit rows by index, so the AUROC/contribution/flag come
from the already-recorded audit JSONs (no retraining, no re-naming).
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

CASES = [
    ("baseline lc=0   lr=1.0 w2.0", "audit_K32_lr10_pa_w20.json", 1.0, "_w20", None),
    ("lc=0.1   lr=1.0           ", "audit_K32_lr10_pa_lc01.json", 1.0, "_lc01", 16),
    ("lc=0.5   lr=1.0           ", "audit_K32_lr10_pa_lc05.json", 1.0, "_lc05", 16),
    ("lc=1.0   lr=0.5           ", "audit_K32_lr05_pa_lc10.json", 0.5, "_lc10", 16),
]
FLOORS = [0.0, 0.02, 0.05, 0.10]

d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
V, K, D, WIN = d["V"], A.K, M.D, A.WIN

print("=" * 100)
print("WHAT A STRICTER METER WOULD DO — pass fraction if the column must also FIRE")
print("(no rule changed; this only measures. Bars stay AUROC>=0.55 AND contrib>=0.25)")
print("=" * 100)

for label, jf, lr, suffix, n_sup in CASES:
    jp = os.path.join(RUNDIR, jf)
    if not os.path.exists(jp):
        print(f"\n{label}: MISSING {jf}")
        continue
    aud = json.load(open(jp))
    seeds = [k for k in aud if k.startswith("seed")]
    print(f"\n{label}   (audit {jf})")
    print(f"  {'floor':>7} {'pass/K':>10} {'frac':>8}"
          + ("   SUP    UNSUP" if n_sup else ""))
    for floor in FLOORS:
        per_seed, sup_p, uns_p, n_sup_tot, n_uns_tot = [], 0, 0, 0, 0
        for s in seeds:
            si = int(s.replace("seed", ""))
            ck = os.path.join(CKPT, f"readable_K32_lr{lr}{suffix}_seed{si}.npz")
            if not os.path.exists(ck):
                print(f"    !! missing ckpt {os.path.basename(ck)}")
                continue
            net = M.TinyLM3(V, K, D, WIN, rng=np.random.default_rng(si))
            z = np.load(ck)
            for p in net.parameters():
                if p.name in z:
                    p.data = np.array(z[p.name])
            h = np.asarray(net.forward(d["Xva"])["concepts"].data)
            fire = (h > 0.5).mean(axis=0)          # firing rate per column
            rows = aud[s]["rows"]
            npass = 0
            for i, r in enumerate(rows):
                if r["flag"] != "-":
                    continue
                if fire[i] < floor:
                    continue
                npass += 1
                if n_sup:
                    if i < n_sup:
                        sup_p += 1
                    else:
                        uns_p += 1
            if n_sup:
                n_sup_tot += n_sup
                n_uns_tot += K - n_sup
            per_seed.append(npass)
        if not per_seed:
            continue
        mp = float(np.mean(per_seed))
        extra = ""
        if n_sup:
            extra = (f"   {sup_p/n_sup_tot*100:5.1f}% "
                     f"{uns_p/n_uns_tot*100:6.1f}%")
        print(f"  {floor*100:>6.0f}% {mp:>7.1f}/{K:<3} {mp/K*100:>7.1f}%{extra}")

print("\n" + "=" * 100)
print("Columns are matched by index; AUROC/contrib/flag are the RECORDED audit values.")
print("A floor of 0% is the meter exactly as it stands today.")
print("=" * 100)
