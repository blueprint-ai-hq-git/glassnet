"""t3_name_precision.py — METER-04 R2: name precision by ABLATION, on EXISTING checkpoints.

For each taught concept column and each word in its pre-registered name:
  * find the column's firing windows (activation > 0.5, the Meter-03 rule)
  * A0 = the column's mean activation over those windows (its own scale)
  * zero that word's embedding row, recompute -> A_w
  * relative drop d_w = (A0 - A_w) / A0
  * the word is USED iff d_w >= 0.10 (pre-registered in METER-04-NAME-PRECISION.md)

name precision = fraction of name words that are USED. R2 passes iff it is 1.0.

No training. Old (Meter-03) vs new (Meter-03 + R2) reported side by side.
Secondary measures (contribution share, detection) are reported but are NOT the pass rule.
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
N_TAUGHT, K = 10, 16
DROP_BAR = 0.10          # pre-registered in METER-04 section 1
FIRING = 0.5             # the audit's firing rule

sel = json.load(open(os.path.join(RUNDIR, "a3_selection.json")))
LR, LC = sel["lr"], sel["lc"]
CK = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_clusters16_seed0.npz"

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
V = DATA["V"]
set_backend("numpy")
targets = M3.load_targets("clusters16")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters16.json")))

net = M3.load_ckpt(M3.build_net(0), os.path.join(RUNDIR, "ckpt", CK))
print("=" * 118)
print(f"METER-04 R2 — NAME PRECISION BY ABLATION   (model {CK}, no training)")
print(f"rule: a name word is USED iff ablating it costs >= {DROP_BAR:.0%} of the column's own "
      f"firing activation")
print("=" * 118)

h0 = np.asarray(net.forward(X)["concepts"].data)
emb = net.emb.data
emb_backup = emb.copy()

rows = []
for c in range(N_TAUGHT):
    words = CLU["clusters"][c]["words"]
    ids = [int(i) for i in CLU["clusters"][c]["ids"]]
    act = h0[:, c]
    fired = act > FIRING
    n_fire = int(fired.sum())
    if n_fire == 0:
        print(f"\n[{c}] name {'_'.join(words)[:44]}  -> NO FIRING WINDOWS (unauditable)")
        rows.append({"col": c, "words": words, "n_fire": 0, "precision": 0.0,
                     "used": [], "unused": words, "drops": {}})
        continue
    A0 = float(act[fired].mean())

    drops = {}
    for w, wid in zip(words, ids):
        emb_backup_row = emb[wid].copy()
        emb[wid] = 0.0                                  # ablate this word everywhere
        hw = np.asarray(net.forward(X)["concepts"].data)[:, c]
        emb[wid] = emb_backup_row                       # restore immediately
        Aw = float(hw[fired].mean())
        drops[w] = float((A0 - Aw) / A0) if A0 > 0 else 0.0

    used = [w for w in words if drops[w] >= DROP_BAR]
    unused = [w for w in words if drops[w] < DROP_BAR]
    prec = len(used) / len(words)

    print(f"\n[{c}] name: {'_'.join(words)}")
    print(f"    firing windows {n_fire}/{len(X)} | mean firing activation A0 = {A0:.4f}")
    for w in words:
        mark = "USED  " if drops[w] >= DROP_BAR else "unused"
        print(f"      {mark}  {w:<12} drop {drops[w]:+.4f}")
    print(f"    name precision = {len(used)}/{len(words)} = {prec:.2f}  -> "
          f"R2 {'PASS' if prec == 1.0 else 'FAIL'}"
          + (f"   (unused: {', '.join(unused)})" if unused else ""))

    rows.append({"col": c, "words": words, "n_fire": n_fire, "A0": A0,
                 "precision": prec, "used": used, "unused": unused, "drops": drops})

emb[:] = emb_backup

n_r2 = sum(1 for r in rows if r["precision"] == 1.0 and r["n_fire"] > 0)
print("\n" + "=" * 118)
print("OLD vs NEW")
print("=" * 118)
a3 = json.load(open(os.path.join(RUNDIR, "a3_eval.json")))
print(f"  OLD (Meter-03 alone)          : {a3['taught']['pass']}/{N_TAUGHT} taught concepts "
      f"pass = {a3['taught']['pass']/N_TAUGHT*100:.1f}%")
print(f"  NEW (Meter-03 + R2)           : {n_r2}/{N_TAUGHT} taught concepts pass = "
      f"{n_r2/N_TAUGHT*100:.1f}%")
print(f"  concepts whose name has an UNUSED word: "
      f"{sum(1 for r in rows if r['unused'])}/{N_TAUGHT}")
print("\n  Note: R2 can only REMOVE passes. A rise would mean the implementation is wrong.")

json.dump({"model": CK, "drop_bar": DROP_BAR, "firing_rule": FIRING, "rows": rows,
           "old_pass": a3["taught"]["pass"], "new_pass": n_r2, "n_taught": N_TAUGHT},
          open(os.path.join(RUNDIR, "name_precision.json"), "w"), indent=1)
print("\nsaved runs/t3/name_precision.json")
print("=" * 118)
