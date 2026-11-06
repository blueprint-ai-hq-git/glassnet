"""finish_t2_probe.py — generic finisher for T2 probe runs. Reads the
iteration dir from T2_ITER, loads the best-val checkpoint, reports VAL +
train accuracy, names + audits with the standing meter, counts dead /
never-firing concepts, writes results.json. VALIDATION only.

Probe bars come from env (T2_BAR_VAL, T2_BAR_AUDIT, T2_BAR_DEAD) so each
iteration's PRE-REGISTERED test is applied literally, never re-chosen
after seeing numbers.
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

ITNAME = os.environ.get("T2_ITER", "iter005")
BAR_VAL = float(os.environ.get("T2_BAR_VAL", "19.0"))
BAR_AUDIT = int(os.environ.get("T2_BAR_AUDIT", "8"))
BAR_DEAD = int(os.environ.get("T2_BAR_DEAD", "1"))
EPOCHS_RUN = int(os.environ.get("T2_EPOCHS", "96"))
LAM_A_ENV = os.environ.get("T2_LAM_A", "0.2")

d = tiny_data.build(win=WIN)
prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()) / len(d["ytr"])

net = TinyLM3(d["V"], K, D, WIN)
z = np.load(os.path.join(ITERDIR, "final_seed0.npz"))
net.emb.data[:] = z["emb"]
net.concept.W.data[:] = z["cW"]
net.concept.b.data[:] = z["cb"]
net.answer.W.data[:] = z["aW"]
net.answer.b.data[:] = z["ab"]

v = acc_of(net.forward(d["Xva"])["logits"], d["yva"]) * 100
ta = acc_of(net.forward(d["Xtr"][:16384])["logits"], d["ytr"][:16384]) * 100
print(f"seed0: VAL {v:.2f}% | train(16k) {ta:.2f}%", flush=True)

names = auto_name_t2c(net, d, d["Xtr"])
rows, n_pass = audit_t2c(net, names, d, d["Xva"])
dead = sum(1 for r in rows
           if r["flag"] == "NO-NAME" or r["auroc"] != r["auroc"])
for r in rows:
    print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
          f"contrib={r['contrib']:.3f} {r['flag']}", flush=True)
print(f"audit pass {n_pass}/{K} | dead/NO-NAME {dead}/{K}", flush=True)

okA = bool(v >= BAR_VAL)
okB = bool(n_pass >= BAR_AUDIT)
okC = bool(dead <= BAR_DEAD)
verdict = "PROBE CLEARS" if (okA and okB and okC) else "PROBE FAILS"
print(f"[{ITNAME}] A(val>={BAR_VAL})={okA} B(audit>={BAR_AUDIT})={okB} "
      f"C(dead<={BAR_DEAD})={okC} -> {verdict}", flush=True)

json.dump({"iteration": ITNAME, "task": "tiny_shakespeare",
           "mode": "concat-window-readable", "scope": "probe, single seed",
           "K": K, "D": D, "WIN": WIN, "epochs": EPOCHS_RUN,
           "lr": 0.1, "lam_act": float(LAM_A_ENV), "lam_w": 1.0,
           "prior": prior, "val_seed0": v, "train_acc": ta,
           "readable_params": 724401,
           "bar_val": BAR_VAL, "bar_audit": BAR_AUDIT, "bar_dead": BAR_DEAD,
           "okA": okA, "okB": okB, "okC": okC, "verdict": verdict,
           "audit_pass": n_pass, "audit_total": K, "dead_neurons": dead,
           "names": names},
          open(os.path.join(ITERDIR, "results.json"), "w"), indent=1)
print(f"saved runs/t2/{ITNAME}/results.json", flush=True)