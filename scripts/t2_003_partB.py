"""t2_003_partB.py — T2-003 naming + audit + verdict on seed-0 checkpoint.
Short (<30 s). Reads partA.json for accs; writes the full results.json."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from run_t2_iter003 import K, D, WIN, ITERDIR, TinyLM3, auto_name_t2c, audit_t2c

d = tiny_data.build(win=WIN)
pa = json.load(open(os.path.join(ITERDIR, "partA.json")))

net = TinyLM3(d["V"], K, D, WIN)
z = np.load(os.path.join(ITERDIR, "final_seed0.npz"))
net.emb.data[:] = z["emb"]
net.concept.W.data[:] = z["cW"]
net.concept.b.data[:] = z["cb"]
net.answer.W.data[:] = z["aW"]
net.answer.b.data[:] = z["ab"]

names = auto_name_t2c(net, d, d["Xtr"])
rows, n_pass = audit_t2c(net, names, d, d["Xva"])
for r in rows:
    print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
          f"contrib={r['contrib']:.3f} {r['flag']}", flush=True)
print(f"[t2-003] audit pass {n_pass}/{K}", flush=True)

okA = bool(pa["okA"])
okB = bool(n_pass / K >= 0.5)
verdict = "KEEP" if (okA and okB) else "REVERT"   # C deferred (probe not run)
print(f"[t2-003] A={okA} B={okB} C=deferred -> verdict: {verdict}", flush=True)
out = {"iteration": "t2-003", "task": "tiny_shakespeare",
       "mode": "concat-window-readable", "K": K, "D": D, "WIN": WIN,
       "epochs": 96, "lr": 0.1, "lam_act": 0.2, "lam_w": 1.0,
       "change": "input: sum-bag -> CONCAT window (order+position); run "
                "externally SIGKILL'd twice; all 3 seed checkpoints survived",
       "prior": pa["prior"], "rval_vals": pa["rval_vals"],
       "rval_mean": pa["rval_mean"], "rval_std": pa["rval_std"],
       "train_acc_mean": pa["train_acc_mean"],
       "readable_params": 724401,
       "audit_pass": n_pass, "audit_total": K, "lie_caught": False,
       "okA": okA, "okB": okB, "okC": False, "verdict": verdict,
       "names": names}
json.dump(out, open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
print("results.json saved", flush=True)