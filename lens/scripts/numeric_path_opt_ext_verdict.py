#!/usr/bin/env python3
"""numeric_path_opt_ext_verdict.py - the P41 verdict: the same decay over a 100x step-size range.

Bar declared before the run in lens/runs/NUMERIC-PATH-OPT-EXT-PREDICTIONS.txt: RelativeGrad with decay=0.999,
momentum 0, arm B unseen exact >= 90% with 3/3 seeds finite at EACH of lr 0.1, 0.2, 0.5.
Reads the rows tagged 'ext' from lens/runs/numeric_path_decay.jsonl. Prints the compared values beside the
verdict (a number without its comparison is how a wrong conclusion gets written down).
"""
import json
import os

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
PATH = os.environ.get("OUT_JSONL", f"{ROOT}/lens/runs/numeric_path_decay.jsonl")
LRS = [0.1, 0.2, 0.5]


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    ext = [r for r in rows if r.get("tag") == "ext" and r.get("arm", "B") == "B"]
    print(f"  rows tagged ext: {len(ext)}   file: {PATH}")
    ok_all = True
    print(f"  {'lr':>6s}  {'seeds':>5s} {'finite':>6s}  {'unseen exact':>12s}  {'MAE':>8s}  "
          f"{'train exact':>11s}   admissible (>=90%, 3/3 finite)")
    for lr in LRS:
        sel = [r for r in ext if abs(r["lr"] - lr) < 1e-12]
        if not sel:
            print(f"  {lr:>6g}  no rows")
            ok_all = False
            continue
        fin = sum(1 for r in sel if r["finite"])
        ue = float(np.mean([r["unseen_exact"] for r in sel]))
        te = float(np.mean([r["train_exact"] for r in sel]))
        mae = float(np.nanmean([r["unseen_mae"] for r in sel])) if fin else float("nan")
        ok = fin == len(sel) and ue >= 0.90
        ok_all = ok_all and ok
        print(f"  {lr:>6g}  {len(sel):>5d} {fin:>3d}/{len(sel):<2d}  {ue*100:>11.1f}%  {mae:>8.4f}  "
              f"{te*100:>10.1f}%   {'YES' if ok else 'no'}")
    print(f"  P41 the same decay holds over a 100x step-size range : {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
