"""f_diag_concepts.py — WHY did the concepts die?

Answers the owner's question with a measurement instead of a story. Trains a readable student
and logs, every epoch:
  * the LM loss
  * the AUX concept loss (which my F trainer never logged -- that is the gap being closed)
  * the live concept AUROC against the pre-registered targets

Run twice: on the GENERATED corpus and on REAL data. Same architecture, same lr, same epochs.
If the concept loss falls on real data but not on generated data, the answer is "the supervision
had no signal". If it falls in BOTH but the AUROC still dies, the answer is different and I will
say so.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np

SOURCE = os.environ.get("D_SOURCE", "generated")   # generated | real
RULE = os.environ.get("D_RULE", "all")
EPOCHS = int(os.environ.get("D_EPOCHS", "6"))
LR = float(os.environ.get("D_LR", "1.0"))
LC = float(os.environ.get("D_LC", "0.1"))
BATCH = 128

os.environ.setdefault("T2_CONCEPT_SET", "meaningful_ext")
os.environ.setdefault("T2_N_TAUGHT", "9")
os.environ.setdefault("T2_LAM_A", "0.2")
os.environ.setdefault("T2_LAM_W", "1.0")
os.environ["T2_LAM_CONCEPT"] = str(LC)

from glassnet.tensor import set_backend
from glassnet import tiny_data
from glassnet.optim import SGD
import run_t2_iter003 as M

R = os.path.join(PROJ, "runs", "t3")
set_backend("numpy")

DATA = tiny_data.build(win=8, vocab_cap=1500)
Xva, Yva = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"])
V = DATA["V"]

if SOURCE == "generated":
    c = np.load(os.path.join(R, f"f_corpus_{RULE}.npz"))
    Xtr, Ytr = c["X"], c["y"]
else:
    Xtr, Ytr = np.asarray(DATA["Xtr"]), np.asarray(DATA["ytr"])

tj = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]
names = [t["name"] for t in tj]
ids9 = [t["ids"] for t in tj]


def auroc(score, target01):
    order = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[order] = np.arange(1, len(score) + 1)
    n1, n0 = target01.sum(), (~target01).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[target01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print("=" * 96)
print(f"WHY DID THE CONCEPTS DIE?  source={SOURCE}  rule={RULE}  lr={LR}  lam_concept={LC}")
print("=" * 96)
print(f"training windows: {len(Xtr)}   validation: {len(Xva)} (REAL)")

rng = np.random.default_rng(0)
net = M.TinyLM3(V, 9, 64, 8, rng=rng, lam_concept=LC)
opt = SGD(net.parameters(), LR)

# what the concept targets look like in THIS training data -- if the target words barely occur,
# there is nothing for the concept loss to learn
occ = []
for ids in ids9:
    m = np.isin(Xtr, ids).any(1)
    occ.append(float(m.mean()))
print("\nfraction of TRAINING windows containing each concept's target words:")
for n, o in zip(names, occ):
    print(f"    {n:<22} {o*100:6.2f}%")

rows = []
for ep in range(1, EPOCHS + 1):
    idx = rng.permutation(len(Xtr))
    lm_sum, aux_sum, nb = 0.0, 0.0, 0
    for s in range(0, len(Xtr), BATCH):
        b = idx[s:s + BATCH]
        opt.zero_grad()
        loss, parts = net.loss(Xtr[b], Ytr[b])
        loss.backward()
        opt.step()
        lm_sum += float(parts.get("lm", 0.0)) if isinstance(parts, dict) else 0.0
        aux_sum += float(parts.get("aux", 0.0)) if isinstance(parts, dict) else 0.0
        nb += 1
    # live concept AUROC on REAL validation
    h = np.asarray(net.forward(Xva)["concepts"].data)
    aus = []
    for c in range(9):
        tgt = np.isin(Xva, ids9[c]).any(1).astype(bool)
        aus.append(auroc(h[:, c], tgt))
    print(f"  ep {ep}: lm {lm_sum/max(nb,1):8.4f}  aux {aux_sum/max(nb,1):8.4f}  "
          f"mean AUROC {np.nanmean(aus):.4f}  (max {np.nanmax(aus):.4f})")
    rows.append({"epoch": ep, "lm": lm_sum / max(nb, 1), "aux": aux_sum / max(nb, 1),
                 "auroc_mean": float(np.nanmean(aus)), "auroc_max": float(np.nanmax(aus))})

json.dump({"source": SOURCE, "rule": RULE, "lr": LR, "lam_concept": LC,
           "target_occurrence_in_training": dict(zip(names, occ)), "rows": rows},
          open(os.path.join(R, f"f_diag_{SOURCE}.json"), "w"), indent=1)
print(f"\nsaved runs/t3/f_diag_{SOURCE}.json")
