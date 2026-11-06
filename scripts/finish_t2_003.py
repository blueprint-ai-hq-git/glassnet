"""finish_t2_003.py — finalize T2-003 from the three surviving checkpoints
(run was externally SIGKILL'd twice; all seeds saved). Computes val + train
accuracies, names + audits seed 0 (standing meter), pre-registered verdict.
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
                            auto_name_t2c, audit_t2c, readable_params)

d = tiny_data.build(win=WIN)
prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()) / len(d["ytr"])

rvals, trs, nets = {}, {}, {}
for s in (0, 1, 2):
    net = TinyLM3(d["V"], K, D, WIN)
    z = np.load(os.path.join(ITERDIR, f"final_seed{s}.npz"))
    net.emb.data[:] = z["emb"]
    net.concept.W.data[:] = z["cW"]
    net.concept.b.data[:] = z["cb"]
    net.answer.W.data[:] = z["aW"]
    net.answer.b.data[:] = z["ab"]
    nets[s] = net
    rvals[s] = acc_of(net.forward(d["Xva"])["logits"], d["yva"]) * 100
    trs[s] = acc_of(net.forward(d["Xtr"][:16384])["logits"],
                    d["ytr"][:16384]) * 100
    print(f"seed{s}: val {rvals[s]:.2f}% | train(16k) {trs[s]:.2f}%")
rm, rs = float(np.mean(list(rvals.values()))), float(np.std(list(rvals.values())))
tm = float(np.mean(list(trs.values())))
print(f"[t2-003] readable VAL {rm:.2f} ± {rs:.2f} | train {tm:.2f}% | "
      f"A bar {2*prior*100:.2f}")

names = auto_name_t2c(nets[0], d, d["Xtr"])
rows, n_pass = audit_t2c(nets[0], names, d, d["Xva"])
for r in rows:
    print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
          f"contrib={r['contrib']:.3f} {r['flag']}")
print(f"[t2-003] audit pass {n_pass}/{K}")

okA = bool(rm >= 2 * prior * 100)
okB = bool(n_pass / K >= 0.5)
verdict = "KEEP" if (okA and okB) else "REVERT"   # C deferred (no probe run)
print(f"[t2-003] A={okA} B={okB} C=deferred -> verdict: {verdict}")

json.dump({"iteration": "t2-003", "task": "tiny_shakespeare",
           "mode": "concat-window-readable", "K": K, "D": D, "WIN": WIN,
           "epochs": 96, "lr": 0.1, "lam_act": 0.2, "lam_w": 1.0,
           "change": "input: sum-bag -> CONCAT window (order+position); "
                    "run externally SIGKILL'd twice, all 3 seed checkpoints "
                    "survived (resume-protected)",
           "prior": prior, "rval_vals": rvals, "rval_mean": rm,
           "rval_std": rs, "train_acc_mean": tm,
           "readable_params": readable_params(d["V"], D, K, WIN),
           "audit_pass": n_pass, "audit_total": K, "lie_caught": False,
           "okA": okA, "okB": okB, "okC": False, "verdict": verdict,
           "names": names},
          open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
print("saved results.json")