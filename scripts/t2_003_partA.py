"""t2_003_partA.py — T2-003 accuracies from the surviving checkpoints.
Short (<30 s) so external process killers don't reach it; writes the
partial results JSON immediately, flushed."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import numpy as np

from glassnet import tiny_data
from run_t2_iter003 import K, D, WIN, ITERDIR, TinyLM3, acc_of

d = tiny_data.build(win=WIN)
prior = float(np.bincount(d["ytr"], minlength=d["V"]).max()) / len(d["ytr"])
rvals, trs = {}, {}
for s in (0, 1, 2):
    net = TinyLM3(d["V"], K, D, WIN)
    z = np.load(os.path.join(ITERDIR, f"final_seed{s}.npz"))
    net.emb.data[:] = z["emb"]
    net.concept.W.data[:] = z["cW"]
    net.concept.b.data[:] = z["cb"]
    net.answer.W.data[:] = z["aW"]
    net.answer.b.data[:] = z["ab"]
    rvals[s] = acc_of(net.forward(d["Xva"])["logits"], d["yva"]) * 100
    trs[s] = acc_of(net.forward(d["Xtr"][:16384])["logits"],
                    d["ytr"][:16384]) * 100
    print(f"seed{s}: val {rvals[s]:.2f}% | train(16k) {trs[s]:.2f}%",
          flush=True)
rm = float(np.mean(list(rvals.values())))
rs = float(np.std(list(rvals.values())))
tm = float(np.mean(list(trs.values())))
print(f"[t2-003] VAL {rm:.2f} ± {rs:.2f} | train {tm:.2f}% | "
      f"A bar {2*prior*100:.2f} | okA={rm >= 2*prior*100}", flush=True)
json.dump({"partA": True, "prior": prior, "rval_vals": rvals,
           "rval_mean": rm, "rval_std": rs, "train_acc_mean": tm,
           "okA": rm >= 2 * prior * 100},
          open(os.path.join(ITERDIR, "partA.json"), "w"), indent=1)
print("partA.json saved", flush=True)