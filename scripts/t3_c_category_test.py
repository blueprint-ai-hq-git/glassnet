"""t3_c_category_test.py — diagnose the R2 = 0/6 result.

R2 (per-word ablation) asks "is every word in the name individually load-bearing?". That is
the RIGHT question for a name like KING_What_you,_RICHARD, where KING and RICHARD are specific
claims the neuron never uses. It may be the WRONG question for a CATEGORY name like
is_a_negation, where the words are examples of the idea rather than individual claims.

This measures the alternative: ablate the WHOLE category at once, and compare the drop against
a random set of the same size (the control). No bar is moved and R2's 0/6 stands as the
recorded result; this is a diagnosis, and any new rule must be pre-registered separately.
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

R = os.path.join(PROJ, "runs", "t3")
sel = json.load(open(os.path.join(R, "c_selection.json")))
LR, LC = sel["lr"], sel["lc"]
CK = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_meaningful_c_seed0.npz"
N_TAUGHT = 6
N_CONTROL = 20

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]
V = DATA["V"]
set_backend("numpy")
tj = json.load(open(os.path.join(R, "meaningful_targets.json")))["targets"]
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", CK))
h0 = np.asarray(net.forward(X)["concepts"].data)
emb = net.emb.data
backup = emb.copy()
rng = np.random.default_rng(0)

print("=" * 108)
print("CATEGORY ABLATION — ablate the WHOLE name set at once vs a random set of the same size")
print("=" * 108)
print(f"{'col':>3} {'concept':<22} {'members':>8} {'drop_all':>9} {'random mean':>12} "
      f"{'random sd':>10} {'vs control':>11}")
out = []
for c in range(N_TAUGHT):
    ids = [i for i in tj[c]["ids"] if 0 <= i < V]
    act = h0[:, c]
    fired = act > 0.5
    A0 = float(act[fired].mean())
    if A0 <= 0:
        continue

    def drop_for(idset):
        for i in idset:
            emb[i] = 0.0
        hw = np.asarray(net.forward(X)["concepts"].data)[:, c]
        for i in idset:
            emb[i] = backup[i]
        return (A0 - float(hw[fired].mean())) / A0

    d_all = drop_for(ids)
    rand = [drop_for(rng.choice(V - 1, size=len(ids), replace=False)) for _ in range(N_CONTROL)]
    z = (d_all - float(np.mean(rand))) / (float(np.std(rand)) + 1e-9)
    print(f"{c:>3} {tj[c]['name']:<22} {len(ids):>8} {d_all:>9.4f} {np.mean(rand):>12.4f} "
          f"{np.std(rand):>10.4f} {z:>+11.2f}")
    out.append({"col": c, "name": tj[c]["name"], "n_members": len(ids),
                "drop_all": float(d_all), "rand_mean": float(np.mean(rand)),
                "rand_sd": float(np.std(rand)), "z": float(z)})
emb[:] = backup

print("\n  reading: 'drop_all' is how much of the column's firing activation disappears when its")
print("  whole named set is removed. The random control removes the same NUMBER of unrelated")
print("  words, so it captures how much damage size alone does. z = how far above that the")
print("  concept's own set sits.")
print("\n  NOTE: this is a DIAGNOSIS, not a new bar. R2's pre-registered 0/6 stands as the")
print("  recorded Target C result; any new rule must be pre-registered before it is used.")
json.dump({"model": CK, "control_draws": N_CONTROL, "rows": out},
          open(os.path.join(R, "c_category_test.json"), "w"), indent=1)
print("\nsaved runs/t3/c_category_test.json")
