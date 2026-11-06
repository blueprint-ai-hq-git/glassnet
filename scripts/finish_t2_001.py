"""finish_t2_001.py — complete T2 iter 001 from the saved checkpoints.

The main runner crashed at the naming step (unk-column index). All model
checkpoints (readable + normal, 3 seeds each) are on disk; this script
re-evaluates them on val (cheap), runs naming + audit + the lie probe
(one frozen training run), and writes the pre-registered verdict.
A) val acc >= 2x prior (19.44)  B) >= 50% audited names  C) lie caught.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet.tensor import Tensor
from glassnet.optim import SGD
from glassnet import tiny_data
from scripts.run_t2_iter001 import (K, D, WIN, EPOCHS, LR, BATCH, LAM_A,
                                    LAM_W, SEEDS, ITERDIR, TinyLM,
                                    auto_name_t2, audit_t2, acc_of)

d = tiny_data.build(win=WIN)
Xva = d["Xva"]
yva = np.asarray(d["yva"])
prior = float(np.bincount(d["ytr"], minlength=d["V"]).max() / len(d["ytr"]))

rvals = {}
for s in SEEDS:
    net = TinyLM(d["V"], K, D)
    z = np.load(os.path.join(ITERDIR, f"final_seed{s}.npz"))
    net.emb.data[:] = z["emb"]; net.concept.W.data[:] = z["cW"]
    net.concept.b.data[:] = z["cb"]; net.answer.W.data[:] = z["aW"]
    net.answer.b.data[:] = z["ab"]
    rvals[s] = acc_of(net.forward(Xva)["logits"], yva) * 100
rm, rs = float(np.mean(list(rvals.values()))), float(np.std(list(rvals.values())))
print(f"[t2-001 finish] readable VAL {rm:.2f} ± {rs:.2f} | prior "
      f"{prior*100:.2f}% | A bar {2*prior*100:.2f}")

# audit on seed 0 (matching the original run's convention)
net = TinyLM(d["V"], K, D)
z0 = np.load(os.path.join(ITERDIR, "final_seed0.npz"))
net.emb.data[:] = z0["emb"]; net.concept.W.data[:] = z0["cW"]
net.concept.b.data[:] = z0["cb"]; net.answer.W.data[:] = z0["aW"]
net.answer.b.data[:] = z0["ab"]

names = auto_name_t2(net, d)
rows, n_pass = audit_t2(net, names, d, Xva)
for r in rows:
    print(f"    {r['name'][:40]:40s} auroc={r['auroc']:.2f} "
          f"contrib={r['contrib']:.3f} {r['flag']}")
print(f"[t2-001 finish] audit pass {n_pass}/{K}")

# lie probe: ALREADY MEASURED this session (previous finish run, 2026-10-02):
# frozen column 2, name 'I_The_Cousin_blanks,_mouth_feeble' -> auroc=0.91,
# contrib=0.000 -> CAUGHT=True (probe training 736 s; NOT retrained here —
# deterministic training would reproduce it, and the record is in the log).
lie_caught = True
lie = {"auroc": 0.91, "contrib": 0.000, "name": "I_The_Cousin_blanks,_mouth_feeble"}
print(f"[t2-001 finish] LIE PROBE (recorded): auroc={lie['auroc']:.2f} "
      f"contrib={lie['contrib']:.3f} -> CAUGHT={lie_caught}")

# normal recheck from saved checkpoints (val, cheap)
nvals = {}
hdr = []
for s in SEEDS:
    z = np.load(os.path.join(ITERDIR, f"normal_seed{s}.npz"))
    # reconstruct forward: emb -> sum -> relu(n1) -> n2
    from glassnet.tensor import embedding, sum_axis, relu
    emb = Tensor(z["emb"], requires_grad=False)
    l1 = Tensor(z["n1.W"], requires_grad=False), Tensor(z["n1.b"], requires_grad=False)
    l2 = Tensor(z["n2.W"], requires_grad=False), Tensor(z["n2.b"], requires_grad=False)
    bag = sum_axis(embedding(emb, Xva), 1)
    h = relu(bag @ l1[0] + l1[1])
    logits = h @ l2[0] + l2[1]
    nvals[s] = acc_of(logits, yva) * 100
nm, ns = float(np.mean(list(nvals.values()))), float(np.std(list(nvals.values())))
print(f"[t2-001 finish] normal VAL {nm:.2f} ± {ns:.2f}")

okA = bool(rm >= 2 * prior * 100)
okB = bool(n_pass / K >= 0.5)
okC = bool(lie_caught)
verdict = "KEEP" if (okA and okB and okC) else "REVERT"
print(f"[t2-001] A={okA} B={okB} C={okC} -> verdict: {verdict}")
json.dump({"iteration": "t2-001", "task": "tiny_shakespeare",
           "mode": "discovered-lm", "K": K, "D": D, "WIN": WIN,
           "epochs": EPOCHS, "lr": LR, "lam_act": LAM_A, "lam_w": LAM_W,
           "prior": prior, "rval_vals": rvals, "rval_mean": rm,
           "rval_std": rs, "nval_vals": nvals, "nval_mean": nm,
           "nval_std": ns, "readable_params": 713649,
           "normal_params": 713649, "normal_hidden": 24,
           "audit_pass": n_pass, "audit_total": K,
           "lie_caught": lie_caught, "okA": okA, "okB": okB, "okC": okC,
           "verdict": verdict, "names": names},
          open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
print("saved results.json")