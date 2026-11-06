"""finish_t2_004.py — completes the T2-004 record (training finished; the
naming/audit/json step was externally killed). Evaluates the best-val
checkpoint, names + audits seed 0 with the standing meter, and writes
results.json with the PRE-REGISTERED probe test:
  probe success = seed-0 VAL >= 20.0 AND audit >= 8/24.
VALIDATION only; the test split is never read.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from run_t2_iter003 import (K, D, WIN, ITERDIR, TinyLM3, acc_of,
                            auto_name_t2c, audit_t2c)

d = tiny_data.build(win=WIN)
prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()) / len(d["ytr"])
ck = os.path.join(ITERDIR, "final_seed0.npz")

net = TinyLM3(d["V"], K, D, WIN)
z = np.load(ck)
net.emb.data[:] = z["emb"]
net.concept.W.data[:] = z["cW"]
net.concept.b.data[:] = z["cb"]
net.answer.W.data[:] = z["aW"]
net.answer.b.data[:] = z["ab"]

v = acc_of(net.forward(d["Xva"])["logits"], d["yva"]) * 100
ta = acc_of(net.forward(d["Xtr"][:16384])["logits"], d["ytr"][:16384]) * 100
print(f"seed0 best-val checkpoint: VAL {v:.2f}% | train(16k) {ta:.2f}%",
      flush=True)

names = auto_name_t2c(net, d, d["Xtr"])
rows, n_pass = audit_t2c(net, names, d, d["Xva"])
for r in rows:
    print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
          f"contrib={r['contrib']:.3f} {r['flag']}", flush=True)
print(f"audit pass {n_pass}/{K}", flush=True)

okA = bool(v >= 20.0)
okB = bool(n_pass >= 8)
print(f"[t2-004] probe test: A(val>=20.0)={okA} B(audit>=8/24)={okB} "
      f"-> {'PROBE CLEARS' if (okA and okB) else 'PROBE FAILS'} (192 ep)",
      flush=True)

out = {"iteration": "t2-004", "task": "tiny_shakespeare",
       "mode": "concat-window-readable", "scope": "probe, single seed",
       "K": K, "D": D, "WIN": WIN, "epochs": 192, "lr": 0.1,
       "lam_act": 0.2, "lam_w": 1.0,
       "change": "epochs 96 -> 192 (convergence probe; concat config never "
                "budget-swept)",
       "prior": prior, "val_seed0": v, "train_acc": ta,
       "readable_params": 724401,
       "probe_bar_val": 20.0, "probe_bar_audit": 8,
       "okA": okA, "okB": okB,
       "verdict": "PROBE CLEARS" if (okA and okB) else "PROBE FAILS",
       "audit_pass": n_pass, "audit_total": K, "names": names,
       "note": "best-val checkpoint (18.31 at ~ep65); final-epoch val was "
               "lower (15.77 at ep192) — the concat window overfits "
               "Shakespeare beyond ~epoch 65"}
json.dump(out, open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
print("saved runs/t2/iter004/results.json", flush=True)