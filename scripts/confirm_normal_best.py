"""confirm_normal_best.py — 3-seed confirmation at the fair sweep's winning
normal config (ep=192, lr=0.2, bs=64, h=23), with PROPER per-seed weight
init + shuffle (the 27-config sweep used a fixed init seed and a varying
shuffle; disclose this). VALIDATION ONLY — the test set is never read.
This produces the mean ± std that goes into the fair rung-3 comparison
against the readable's 86.08 ± 0.60 (iter 033, 3 seeds, VAL).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet.loop_lib import load_split
from glassnet.tensor import relu, Tensor
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "runs", "rung3_normal_tune", "confirm_best.json")

EP, LR, BS, H = 192, 0.2, 64, 23
SEEDS = (0, 1, 2)

Xtr, ytr, Xva, yva, Xte, yte = load_split("banking77")


def run(seed):
    rng = np.random.default_rng(seed)
    l1 = Dense(1175, H, rng=rng)
    l2 = Dense(H, 77, rng=rng)
    opt = SGD(l1.parameters() + l2.parameters(), lr=LR)
    rngd = np.random.default_rng(seed * 131 + 7)
    for _ in range(EP):
        perm = rngd.permutation(len(Xtr))
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            loss = cross_entropy(l2(relu(l1(Xtr[idx]))), ytr[idx])
            loss.backward()
            opt.step()
    v = float(np.mean(l2(relu(l1(Xva))).data.argmax(1) == yva)) * 100
    return v


vals = {}
for s in SEEDS:
    t0 = time.time()
    vals[s] = run(s)
    print(f"  seed{s}: val {vals[s]:.2f}% ({time.time()-t0:.0f}s)", flush=True)

m = float(np.mean(list(vals.values())))
sd = float(np.std(list(vals.values())))
readable_m, readable_sd = 86.08, 0.60
gap = readable_m - m
pooled = float(np.sqrt(readable_sd ** 2 + sd ** 2))
print(f"tuned-normal (best config, 3 seeds): {m:.2f} ± {sd:.2f}")
print(f"readable (iter 033, 3 seeds):        {readable_m:.2f} ± {readable_sd:.2f}")
print(f"gap (readable - normal): {gap:+.2f} | pooled std {pooled:.2f} "
      f"| 2x pooled {2*pooled:.2f}")
if gap > 2 * pooled:
    verdict = "readable win"
elif gap > 0:
    verdict = "no difference (win smaller than seed noise)"
else:
    verdict = "normal wins"
print(f"VERDICT: {verdict}")

json.dump({"ep": EP, "lr": LR, "bs": BS, "h": H, "params": 1253 * H + 77,
           "seeds": vals, "normal_mean": m, "normal_std": sd,
           "readable_mean": readable_m, "readable_std": readable_sd,
           "gap": gap, "pooled_std": pooled, "verdict": verdict,
           "note": "3-seed confirmation with proper per-seed init+shuffle; "
                   "VALIDATION only, test never read"},
          open(OUT, "w"), indent=1)
print("saved confirm_best.json")