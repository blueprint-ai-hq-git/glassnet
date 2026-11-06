"""t3_clu_pick.py — apply the FROZEN joint selection rule to the CLUSTER tuning runs.

Same rule as SELECTION-RULE.md / t3_lamc_pick.py: lowest validation perplexity;
tie-break accuracy within 1% relative; then lower lr. Applied to the readable only.
Reads lam_concept from the FILENAME tag (older records may lack the field).
"""
import glob
import json
import os
import re
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
RUNDIR = os.path.join(PROJ, "runs", "t3")
import numpy as np

LAMCS = ["0.1", "0.5", "1.0"]
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
for f in glob.glob(os.path.join(RUNDIR, "models_K32_clu*_lr*_tune.json")):
    mm = re.search(r"_clu([0-9.]+)_lr([0-9.]+)_", os.path.basename(f))
    if not mm:
        continue
    lc_tag = mm.group(1)
    for r in json.load(open(f)):
        if r["model"] == "readable":
            tun[(lc_tag, str(r["lr"]))] = r

print("CLUSTER TARGETS — TUNING (readable, K=32, 1 seed, validation only)")
print(f"  {'lam_c':>6} {'lr':>5} {'params':>10} {'acc':>7} {'ppl':>8}")
for lc in LAMCS:
    for lr in ("0.5", "1.0"):
        r = tun.get((lc, lr))
        if not r:
            print(f"  {lc:>6} {lr:>5}   MISSING")
            continue
        print(f"  {lc:>6} {lr:>5} {r['params']:>10,} {r['val']:>6.2f}% "
              f"{r['val_ppl']:>8.2f}")

sel = {}
print("\nFROZEN JOINT SELECTION RULE APPLIED (readable)")
for lc in LAMCS:
    rows = [tun[(lc, lr)] for lr in ("0.5", "1.0") if (lc, lr) in tun]
    if not rows:
        continue
    chosen, why = pick(rows)
    sel[lc] = {"lam_concept": float(lc), "lr": chosen["lr"], "val": chosen["val"],
               "val_ppl": chosen["val_ppl"], "why": why}
    print(f"  lam_concept={lc:>4} -> lr {chosen['lr']}  acc {chosen['val']:.2f}%  "
          f"ppl {chosen['val_ppl']:.2f}   [{why}]")

json.dump(sel, open(os.path.join(RUNDIR, "clu_selection.json"), "w"), indent=1)
print(f"\nsaved {os.path.join(RUNDIR, 'clu_selection.json')}")
