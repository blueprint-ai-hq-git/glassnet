"""t3_a2_faildiag32.py — why does A2-1 fail? Per column, per clause, plus the hold-out.

A2-1 (>= 50% of taught concepts pass) came in at 33.3%. This reports, for each of the 24
taught columns and each of the 8 held-out columns: TRAIN support of its target, firing
rate, contribution, AUROC against its own pre-registered target, and WHICH clause failed.

The point is to distinguish "the concept was not learned" (AUROC/contribution) from "the
concept was learned but fires too rarely for the pre-registered floor" (firing) -- the
same distinction that explained the 6 failures in the 16-target run.
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
PAT = "readable_K32_lr0.5_lc05_clusters32_seed{s}.npz"
N_TAUGHT = 24
K = 32

print("=" * 116)
print("A2-1 FAILURE DIAGNOSIS — 24 taught columns + 8 held-out, per clause (Meter-03, 2% floor)")
print("=" * 116)

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
targets = M3.load_targets("clusters32")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters32.json")))
supports = [c["support"] for c in CLU["clusters"]]

per_col = {c: [] for c in range(K)}
for seed in (0, 1, 2):
    path = os.path.join(RUNDIR, "ckpt", PAT.format(s=seed))
    if not os.path.exists(path):
        print(f"  seed{seed}: MISSING")
        continue
    net = M3.load_ckpt(M3.build_net(seed), path)
    h, L, A = M3.prep(net, X)
    for c in range(K):
        per_col[c].append(M3.audit_col(net, X, h, c, targets[c][1], L, A))


def clause_of(fr, au, co):
    if fr < 0.02:
        return "FIRING (<2%)"
    if au < 0.55:
        return "AUROC (<0.55)"
    if co < 0.25:
        return "CONTRIB (<0.25)"
    return "-"


def report(lo, hi, title):
    tally = {"firing": 0, "auroc": 0, "contrib": 0, "pass": 0}
    rows = []
    print(f"\n--- {title}")
    print(f"{'col':>3} {'support':>8} {'firing':>8} {'contrib':>9} {'auroc':>7} "
          f"{'verdict':>8}  clause            target")
    for c in range(lo, hi):
        rs = per_col[c]
        if not rs:
            continue
        fr = float(np.mean([r["firing"] for r in rs]))
        co = float(np.mean([r["contrib"] for r in rs]))
        au = float(np.mean([r["auroc"] for r in rs]))
        cl = clause_of(fr, au, co)
        if cl == "-":
            tally["pass"] += 1
        elif cl.startswith("FIRING"):
            tally["firing"] += 1
        elif cl.startswith("AUROC"):
            tally["auroc"] += 1
        else:
            tally["contrib"] += 1
        lab = "_".join(CLU["clusters"][c]["words"][:3])
        print(f"{c:>3} {supports[c]:>8.4f} {fr:>8.4f} {co:>+9.4f} {au:>7.4f} "
              f"{('-' if cl == '-' else 'BROKEN'):>8}  {cl:<16} {lab[:22]}")
        rows.append({"col": c, "support": supports[c], "firing": fr, "contrib": co,
                     "auroc": au, "clause": cl})
    n = hi - lo
    print(f"    tally over {n} columns: FIRING={tally['firing']} AUROC={tally['auroc']} "
          f"CONTRIB={tally['contrib']} pass={tally['pass']} "
          f"-> pass rate {tally['pass']/n*100:.1f}%")
    return tally, rows


t_taught, r_taught = report(0, N_TAUGHT, "TAUGHT columns 0-23 (A2-1 uses these)")
t_held, r_held = report(N_TAUGHT, K, "HELD-OUT columns 24-31 (the hold-out check)")

print(f"\nA2-1 requirement: >= 50% of taught concepts pass")
print(f"A2-1 actual: {t_taught['pass']}/{N_TAUGHT} = {t_taught['pass']/N_TAUGHT*100:.1f}%"
      f" -> {'PASS' if t_taught['pass']/N_TAUGHT >= 0.5 else 'FAIL'}")
print(f"hold-out: {t_held['pass']}/{K-N_TAUGHT} untaught targets ever passed")

sup_taught = [supports[c] for c in range(N_TAUGHT)]
print(f"\ntaught-target TRAIN supports: {min(sup_taught):.4f} .. {max(sup_taught):.4f} "
      f"(median {np.median(sup_taught):.4f})")
above = sum(1 for s in sup_taught if s >= 0.02)
print(f"taught targets clearing the 2% TRAIN-support level: {above}/{N_TAUGHT}")
f_sup = [r["support"] for r in r_taught if r["clause"] != "-"]
p_sup = [r["support"] for r in r_taught if r["clause"] == "-"]
if f_sup and p_sup:
    print(f"median support: failing {np.median(f_sup):.4f} vs passing {np.median(p_sup):.4f}")
if f_sup:
    print(f"median firing of failing columns: {np.median([r['firing'] for r in r_taught if r['clause']!='-']):.4f}")
if p_sup:
    print(f"median firing of passing columns: {np.median([r['firing'] for r in r_taught if r['clause']=='-']):.4f}")
# how many would pass at the 1% sensitivity floor (reported as sensitivity, NOT the rule)
alt = sum(1 for r in r_taught if r["firing"] >= 0.01 and r["auroc"] >= 0.55
          and r["contrib"] >= 0.25)
print(f"\nSENSITIVITY ONLY (not the rule): at a 1% firing floor, taught pass would be "
      f"{alt}/{N_TAUGHT} = {alt/N_TAUGHT*100:.1f}%")

json.dump({"taught_tally": t_taught, "held_tally": t_held,
           "taught_rows": r_taught, "held_rows": r_held,
           "a2_1_pass": bool(t_taught["pass"] / N_TAUGHT >= 0.5),
           "taught_above_2pct_support": above,
           "sensitivity_1pct_floor_pass": alt},
          open(os.path.join(RUNDIR, "a2_faildiag32.json"), "w"), indent=1)
print("\nsaved runs/t3/a2_faildiag32.json")
print("=" * 116)
