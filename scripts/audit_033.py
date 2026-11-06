"""audit_033.py — re-run the standing audit on the iter-033 (rung-3 KEEP)
model to itemize the 3 failed neurons with exact numbers. No training.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet import text_data as TD
from glassnet.discover import DiscoverNet, audit_discover, auto_name
from glassnet.loop_lib import load_split, RUNS

Xtr, ytr, Xval, yval, Xte, yte = load_split("banking77")
_, _, _, _, _, _, t_tr, t_va, t_te, uniq = TD.load()
vocab = TD.load_vocab()

res = json.load(open(os.path.join(RUNS, "iter033", "results.json")))
net = DiscoverNet.load(os.path.join(RUNS, "iter033", "final_seed0.npz"))
# names as recorded at iter 033 (recompute the same FEAT_NAMES=BOW-only)
names, examples = auto_name(net, vocab, Xtr, t_tr)
rows, n_pass = audit_discover(net, names, Xval, t_va)
print(f"iter033 audit re-run: {n_pass}/24 pass")
for r in rows:
    flag = r["flag"]
    if flag != "-":
        print(f"  BROKEN n{r['neuron']:02d} {r['name'][:44]:44s} "
              f"auroc={r['auroc']:.2f} contrib={r['contrib']:.3f} "
              f"pos_frac={r['pos_frac']:.3f} quiet={r['quiet_mean']:.3f} "
              f"loud={r['loud_mean']:.3f}")
print("saved nets untouched; naming matches iter-033 (BOW features only)")