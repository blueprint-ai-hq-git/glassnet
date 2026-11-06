"""t3_ksweep_pick.py — step 2 phase 2 prep: apply the FROZEN joint selection
rule (SELECTION-RULE.md) to the K-sweep tuning results.

Rule, applied identically to both models and every K:
  1. lowest mean validation perplexity
  2. tie-break: higher mean validation accuracy, if perplexities are within
     1.0% relative
  3. tie-break: lower learning rate, if accuracy is also within 0.05 pts

Writes runs/t3/ksweep_selection.json with the chosen lr per (K, model) plus the
full tuning table, so the selection is auditable and cannot be quietly re-picked.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
import numpy as np

RUNDIR = os.path.join(PROJ, "runs", "t3")
KS = [8, 16, 24, 32]
LRS = [0.3, 0.5, 1.0]
REL_PPL_TOL = 0.01      # 1.0% relative perplexity -> tie
ACC_TOL = 0.05          # 0.05 pts accuracy -> still tied


def load_tuning():
    out = {}
    for f in glob.glob(os.path.join(RUNDIR, "models_K*_lr*_tune.json")):
        for r in json.load(open(f)):
            out[(r["K"], r["model"], r["lr"])] = r
    return out


def pick(rows):
    """rows: list of dicts with 'lr','val','val_ppl'. Returns (chosen, why)."""
    rows = sorted(rows, key=lambda r: r["val_ppl"])
    best = rows[0]
    tied = [r for r in rows if (r["val_ppl"] - best["val_ppl"]) / (best["val_ppl"] + 1e-12)
            <= REL_PPL_TOL]
    if len(tied) == 1:
        return best, f"lowest val ppl ({best['val_ppl']:.2f}), no tie"
    best_acc = max(tied, key=lambda r: r["val"])
    acc_tied = [r for r in tied if best_acc["val"] - r["val"] <= ACC_TOL]
    if len(acc_tied) == 1:
        return best_acc, (f"ppl tie within {REL_PPL_TOL*100:.0f}% "
                          f"({[round(r['val_ppl'],2) for r in tied]}) -> "
                          f"higher acc {best_acc['val']:.2f}%")
    chosen = min(acc_tied, key=lambda r: r["lr"])
    return chosen, (f"ppl AND acc tied ({[round(r['val'],2) for r in acc_tied]} "
                    f"acc, {[round(r['val_ppl'],2) for r in acc_tied]} ppl) -> lower lr")


def main():
    tun = load_tuning()
    print("K-SWEEP TUNING TABLE (validation, 1 seed, equal budget both models)")
    print(f"  {'K':>3} {'model':9s} {'lr':>4} {'params':>10} {'acc':>7} {'ppl':>8} {'best_ep':>7}")
    for K in KS:
        for model in ("readable", "normal"):
            for lr in LRS:
                r = tun.get((K, model, lr))
                if not r:
                    print(f"  {K:>3} {model:9s} {lr:>4}  MISSING")
                    continue
                print(f"  {K:>3} {model:9s} {lr:>4} {r['params']:>10,} "
                      f"{r['val']:>6.2f}% {r['val_ppl']:>8.2f} {r['best_epoch']:>7}")

    sel = {}
    print("\nFROZEN JOINT SELECTION RULE APPLIED")
    for K in KS:
        for model in ("readable", "normal"):
            rows = [tun[(K, model, lr)] for lr in LRS if (K, model, lr) in tun]
            if not rows:
                continue
            chosen, why = pick(rows)
            sel[f"K{K}_{model}"] = {"K": K, "model": model, "lr": chosen["lr"],
                                    "val": chosen["val"], "val_ppl": chosen["val_ppl"],
                                    "params": chosen["params"], "why": why}
            print(f"  K={K:>3} {model:9s} -> lr {chosen['lr']}  "
                  f"acc {chosen['val']:.2f}%  ppl {chosen['val_ppl']:.2f}   [{why}]")

    # matched-parameter check
    print("\nMATCHED PARAMETERS CHECK (readable K == normal hidden)")
    for K in KS:
        a = sel.get(f"K{K}_readable"); b = sel.get(f"K{K}_normal")
        if a and b:
            ok = a["params"] == b["params"]
            print(f"  K={K:>3}: readable {a['params']:,} vs normal {b['params']:,} -> "
                  f"{'MATCHED' if ok else 'MISMATCH'}")

    json.dump(sel, open(os.path.join(RUNDIR, "ksweep_selection.json"), "w"), indent=1)
    print(f"\nsaved {os.path.join(RUNDIR, 'ksweep_selection.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
