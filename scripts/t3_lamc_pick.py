"""t3_lamc_pick.py — apply the FROZEN joint selection rule to the lam_concept
tuning runs (lowest validation perplexity; tie-break accuracy within 1% relative;
then lower lr). Identical rule to SELECTION-RULE.md, applied to the readable only
(the normal is unaffected by lam_concept).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
import numpy as np

RUNDIR = os.path.join(PROJ, "runs", "t3")
LAMCS = ["0.1", "0.5", "1.0"]
LRS = ["0.5", "1.0"]
REL_PPL_TOL, ACC_TOL = 0.01, 0.05


def pick(rows):
    rows = sorted(rows, key=lambda r: r["val_ppl"])
    best = rows[0]
    tied = [r for r in rows
            if (r["val_ppl"] - best["val_ppl"]) / (best["val_ppl"] + 1e-12) <= REL_PPL_TOL]
    if len(tied) == 1:
        return best, f"lowest val ppl ({best['val_ppl']:.2f}), no tie"
    best_acc = max(tied, key=lambda r: r["val"])
    acc_tied = [r for r in tied if best_acc["val"] - r["val"] <= ACC_TOL]
    if len(acc_tied) == 1:
        return best_acc, f"ppl tie -> higher acc {best_acc['val']:.2f}%"
    chosen = min(acc_tied, key=lambda r: r["lr"])
    return chosen, "ppl AND acc tied -> lower lr"


tun = {}
for f in glob.glob(os.path.join(RUNDIR, "models_K32_lc*_lr*_tune.json")):
    # lam_concept comes from the FILENAME (the tag), not from the record: older
    # records predate the field, and a KeyError here silently emptied the whole
    # selection (which then silently skipped the confirm and audit phases).
    import re
    mm = re.search(r"_lc([0-9.]+)_lr([0-9.]+)_", os.path.basename(f))
    if not mm:
        continue
    lc_tag, lr_tag = mm.group(1), mm.group(2)
    for r in json.load(open(f)):
        if r["model"] == "readable":
            tun[(lc_tag, str(r["lr"]))] = r

print("lam_concept TUNING (readable, K=32, 1 seed, validation only)")
print(f"  {'lam_c':>6} {'lr':>5} {'params':>10} {'acc':>7} {'ppl':>8}")
for lc in LAMCS:
    for lr in LRS:
        r = tun.get((lc, lr))
        if not r:
            print(f"  {lc:>6} {lr:>5}   MISSING"); continue
        print(f"  {lc:>6} {lr:>5} {r['params']:>10,} {r['val']:>6.2f}% {r['val_ppl']:>8.2f}")

sel = {}
print("\nFROZEN JOINT SELECTION RULE APPLIED (readable)")
for lc in LAMCS:
    rows = [tun[(lc, lr)] for lr in LRS if (lc, lr) in tun]
    if not rows:
        continue
    chosen, why = pick(rows)
    sel[lc] = {"lam_concept": float(lc), "lr": chosen["lr"], "val": chosen["val"],
               "val_ppl": chosen["val_ppl"], "why": why}
    print(f"  lam_concept={lc:>4} -> lr {chosen['lr']}  acc {chosen['val']:.2f}%  "
          f"ppl {chosen['val_ppl']:.2f}   [{why}]")

json.dump(sel, open(os.path.join(RUNDIR, "lamc_selection.json"), "w"), indent=1)
print(f"\nsaved {os.path.join(RUNDIR, 'lamc_selection.json')}")
