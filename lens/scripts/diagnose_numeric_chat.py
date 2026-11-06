#!/usr/bin/env python3
"""diagnose_numeric_chat.py - find the EXACT step and tensor where the numeric term goes non-finite.

Prints the loss terms and max|param| per tensor for every step. No result is produced here.
Env: DIAG_STEPS (default 25), DIAG_LAM, DIAG_LR, DIAG_MATCH (1 = per-tensor matched lr, 0 = one lr).
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
os.chdir(PROJ)
os.environ["CHAT_DATA_DIR"] = os.path.join(PROJ, "data", "chat", "prepared_mix_ctrl_honest")

import chat_data
from lens import backend as B
from lens.concepts import Concept
from lens.model import LensNet
from lens.numeric import magnitude_channel, numeric_targets

W, D, SCALE = 32, 128, 100.0
BATCH = int(os.environ.get("DIAG_BATCH", "256"))
lam = float(os.environ.get("DIAG_LAM", "1.0"))
lr = float(os.environ.get("DIAG_LR", "0.2"))
steps = int(os.environ.get("DIAG_STEPS", "25"))
match = os.environ.get("DIAG_MATCH", "1") == "1"

d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
Xtr, Ytr = chat_data.make_windows(ids, bounds, split, W, which=0, max_windows=8000, seed=0)
id_to_token = {i: t for i, t in enumerate(vocab)}
mag_tr, numeric_tr = magnitude_channel(Xtr, id_to_token, scale=SCALE)
keep_tr = 1.0 - numeric_tr
raw, ymask = numeric_targets(vocab, Ytr, scale=1.0)
tmean, tstd = float(raw[ymask > 0.5].mean()), float(raw[ymask > 0.5].std())
ynum = ((raw - tmean) / tstd).astype(np.float32)
print(f"targets mean {tmean:.3f} std {tstd:.3f} masked {int(ymask.sum())}/{len(ymask)} "
      f"lam {lam} lr {lr} match_lr {match}")

net = LensNet(vocab_size=len(vocab),
              concepts=[Concept("is_a_number",
                                [i for i, w in enumerate(vocab) if w.replace('.', '').isdigit()])],
              d=D, window=W, seed=0, lam_concept=0.0, lam_act=0.0, lam_w=0.0, numeric=True)

# one batch for the lr map
idx0 = np.arange(BATCH)
for p in net.parameters():
    p.grad = None
tot0, _ = net.loss(Xtr[idx0], Ytr[idx0], use_concept_loss=False, mag=mag_tr[idx0], magmask=keep_tr[idx0],
                   ynum=ynum[idx0], ynum_mask=ymask[idx0], lam_num=lam)
tot0.backward()
norms, lang = {}, []
for p in net.parameters():
    if p.grad is not None:
        nn = float(np.linalg.norm(np.asarray(B.to_cpu(p.grad), dtype=np.float64)))
        norms[p.name] = nn
        if not p.name.startswith(("mag.", "num.")):
            lang.append(nn)
med = float(np.median(lang))
lrmap = {k: lr * med / max(v, 1e-12) for k, v in norms.items() if k.startswith(("mag.", "num."))}
# STABILITY BOUND for the quantity head: it is a LINEAR regression on the magnitudes, so the step must
# satisfy lr < 2/lambda_max of the (2/n) Q^T Q Hessian. Measured on the masked rows of one batch.
mv = ymask[:BATCH] > 0.5
Q = np.asarray(mag_tr[:BATCH][mv], dtype=np.float64)
Q = np.concatenate([Q, np.ones((len(Q), 1))], axis=1) if len(Q) else Q
if len(Q):
    H = (2.0 / len(Q)) * (Q.T @ Q)
    lam_max = float(np.linalg.eigvalsh(H)[-1])
    stable_lr = 0.5 / lam_max
    print(f"quantity-head Hessian: n_masked {len(Q)}  lambda_max {lam_max:.4g}  "
          f"stability lr < {2.0/lam_max:.4g}  -> using {stable_lr:.4g}")
    lrmap["num.Wn"] = stable_lr
    lrmap["num.bn"] = stable_lr
print("per-tensor lr:", {k: round(v, 6) for k, v in lrmap.items()}, " median lang ||g||", round(med, 5))
print("||g||:", {k: round(v, 5) for k, v in norms.items()})

names = [p.name for p in net.parameters()]


def mxs():
    return {p.name: float(np.abs(np.asarray(B.to_cpu(p.data))).max()) for p in net.parameters()}


rng = np.random.default_rng(0)
order = rng.permutation(len(Xtr))
hdr = "step total      ce        num     " + " ".join(f"{n[:9]:>9s}" for n in names)
print(hdr)
for s in range(0, steps * BATCH, BATCH):
    idx = order[s:s + BATCH]
    for p in net.parameters():
        p.grad = None
    total, parts = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=False,
                            mag=mag_tr[idx], magmask=keep_tr[idx],
                            ynum=ynum[idx], ynum_mask=ymask[idx], lam_num=lam)
    total.backward()
    m = mxs()
    gmax = {p.name: float(np.abs(np.asarray(B.to_cpu(p.grad))).max()) if p.grad is not None else 0.0
            for p in net.parameters()}
    print(f"{s//BATCH:>4} {float(B.to_cpu(total.data)):>10.4g} {parts.get('ce', 0):>9.4g} "
          f"{parts.get('num_mse', 0):>9.4g} " + " ".join(f"{m[n]:>9.3g}" for n in names))
    if s == 0:
        print("       ||g||max  " + " " * 22 + " ".join(f"{gmax[n]:>9.3g}" for n in names))
    for p in net.parameters():
        if p.grad is not None:
            p.data = p.data - (lrmap.get(p.name, lr) if match else lr) * p.grad
