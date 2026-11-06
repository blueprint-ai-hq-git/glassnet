"""t3_audit_hole.py — how much of the supervised pass fraction is "dead columns"?

The lie probe showed a NEARLY DEAD frozen column (fires on 1.15% of windows)
passing the meter with AUROC 1.0, because the namer picked a word-set present in
exactly those few windows. That is a hole in the METER, not just in the probe:
any column that fires rarely can be given a name matched to its own firings.

The audit rows carry pos_frac = fraction of VAL windows containing the named
words. For a rare-firing column this equals its firing rate. This script reads the
EXISTING audit JSONs (no retraining) and reports, among PASSING rows, how many have
a very small pos_frac — i.e. how much of the pass fraction rests on columns that
fire on a sliver of the data.
"""
import json
import os

import numpy as np

RUNDIR = "/run/media/yahya/HARD DISK 4/GlassNet/runs/t3"
CASES = [
    ("baseline  lam_concept=0  ", "audit_K32_lr10_pa_w20.json", None),
    ("lc=0.1    lr=1.0         ", "audit_K32_lr10_pa_lc01.json", 16),
    ("lc=0.5    lr=1.0         ", "audit_K32_lr10_pa_lc05.json", 16),
    ("lc=1.0    lr=0.5         ", "audit_K32_lr05_pa_lc10.json", 16),
]

print("=" * 92)
print("AUDIT METER HOLE — pass fraction vs how rarely the passing column fires")
print("=" * 92)
print(f"  {'case':<26} {'pass':>7} {'of which pos_frac<2%':>22} {'pos_frac<5%':>13} "
      f"{'med pos_frac(pass)':>20}")

for label, fname, n_sup in CASES:
    p = os.path.join(RUNDIR, fname)
    if not os.path.exists(p):
        print(f"  {label:<26}  MISSING {fname}")
        continue
    a = json.load(open(p))
    seeds = [k for k in a if k.startswith("seed")]
    K = a[seeds[0]]["n_total"]
    rows = []
    for s in seeds:
        for i, r in enumerate(a[s]["rows"]):
            r = dict(r)
            r["col"] = i
            r["sup"] = (n_sup is not None and i < n_sup)
            rows.append(r)
    pas = [r for r in rows if r["flag"] == "-"]
    pf = np.array([r["pos_frac"] for r in pas if r.get("pos_frac") is not None])
    n_pass_per_seed = np.mean([a[s]["n_pass"] for s in seeds])
    if len(pf) == 0:
        print(f"  {label:<26} {n_pass_per_seed:>6.1f}/{K}   (no passing rows)")
        continue
    tiny2 = int((pf < 0.02).sum())
    tiny5 = int((pf < 0.05).sum())
    print(f"  {label:<26} {n_pass_per_seed:>4.1f}/{K:<3} "
          f"{tiny2:>7}/{len(pf):<14} {tiny5:>6}/{len(pf):<6} "
          f"{np.median(pf)*100:>18.2f}%")

print("\n  (pos_frac = fraction of VAL windows containing the named words. A column")
print("   that fires on ~1% of windows can be named to match those firings exactly,")
print("   so a low pos_frac among PASSING rows means the pass rests on rare firings.)")

print("\n" + "=" * 92)
print("SAME QUESTION, SPLIT BY SUPERVISED / UNSUPERVISED COLUMNS (lc=1.0, lr=0.5)")
print("=" * 92)
a = json.load(open(os.path.join(RUNDIR, "audit_K32_lr05_pa_lc10.json")))
seeds = [k for k in a if k.startswith("seed")]
rows = []
for s in seeds:
    for i, r in enumerate(a[s]["rows"]):
        r = dict(r)
        r["col"] = i
        r["sup"] = i < 16
        rows.append(r)
for grp in (True, False):
    g = [r for r in rows if r["sup"] == grp]
    pas = [r for r in g if r["flag"] == "-"]
    pf = np.array([r["pos_frac"] for r in pas if r.get("pos_frac") is not None])
    name = "SUPERVISED  " if grp else "UNSUPERVISED"
    if len(pf) == 0:
        print(f"  {name}  no passing rows")
        continue
    print(f"  {name}  pass {len(pas)}/{len(g)}  "
          f"median pos_frac {np.median(pf)*100:.2f}%  "
          f"pos_frac<2%: {int((pf<0.02).sum())}/{len(pf)}  "
          f"pos_frac<5%: {int((pf<0.05).sum())}/{len(pf)}")
print("=" * 92)
