"""diag_lie_probe.py — WHY did the lie probe stop being caught?

The probe freezes concept column 2 (gradient zeroed, weights stay at init). At
lam_concept=0 it was CAUGHT (AUROC 0.5057); with supervision it is MISSED
(0.99-1.0). Frozen weights never move, so the only path to informativeness is the
INPUT: x = flatten(emb[ids]) — and the auxiliary loss trains the embeddings too.

Measures, on VALIDATION, using the audit's OWN meter (length-residualized AUROC
vs "window contains a named word"):
  1. the frozen column's firing statistics
  2. the audit's meter for its own name (reproduces the audit)
  3. the meter for RANDOM word-sets of the SAME ARITY as the name
     -> if the name sits inside that control distribution, the meter is not
        measuring anything specific about this column
     -> if the name is a clear outlier, the column genuinely tracks that word

Run twice:  T2_LAM_CONCEPT=0   and   T2_LAM_CONCEPT=1.0 T3A_LR=0.5
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

os.environ.setdefault("T3A_K", "32")
os.environ.setdefault("T3A_SEEDS", "1")
os.environ.setdefault("T3A_EPOCHS", "96")
os.environ.setdefault("T3A_MIN_EPOCH", "30")
os.environ.setdefault("T3A_PATIENCE", "15")
os.environ.setdefault("T3A_LIE_COL", "2")
os.environ.setdefault("T3A_BACKEND", "cupy")

import t3_audit as A
from glassnet.tensor import set_backend
import run_t2_iter003 as M

LIE_COL = int(os.environ.get("T3A_LIE_COL", "2"))
LAMC = M.LAM_CONCEPT
LR = float(os.environ.get("T3A_LR", "0.5"))

print("=" * 80)
print(f"LIE PROBE DIAGNOSTIC   lam_concept={LAMC}  lr={LR}  frozen col={LIE_COL}")
print("=" * 80)

d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
set_backend("cupy")
net, best, stopped = A.train_readable(d, 1, freeze_col=LIE_COL, tag="diagprobe")
A.to_cpu(net)
set_backend("numpy")
print(f"\ntrained: val {best['val']:.2f}%  ppl {np.exp(best['ce']):.2f}  "
      f"stopped at epoch {stopped}")

names = A.namer()(net, d, d["Xtr"])
nm = names[LIE_COL]
rows, n_pass = A.audit_t2c(net, names, d, d["Xva"])
row = rows[LIE_COL]
words = sorted(set(nm.split("_")) - {"neuron"})
print(f"\nfrozen column {LIE_COL} name = '{nm}'  ({len(words)} words: {words})")
print(f"[AUDIT METER] auroc {row['auroc']:.4f}  contrib {row['contrib']:+.4f}  "
      f"flag {row['flag']}  pos_frac {row['pos_frac']:.4f}")

X = np.asarray(d["Xva"])
h = np.asarray(net.forward(X)["concepts"].data)
col = h[:, LIE_COL]
print(f"\n[FIRING] val: mean {col.mean():.4f} std {col.std():.4f} "
      f"min {col.min():.4f} max {col.max():.4f}")
print(f"         frac>0.5 {(col > 0.5).mean():.4f}  frac>0.9 {(col > 0.9).mean():.4f}  "
      f"frac<0.1 {(col < 0.1).mean():.4f}")
print(f"         emb norm {np.linalg.norm(np.asarray(net.emb.data)):.3f}")

# ---- reproduce the meter's AUROC and run the arity-matched control ----
L = np.log1p(np.asarray([(x != d["V"] - 1).sum() for x in X]))
Ab = np.vstack([np.ones_like(L), L]).T
coef, *_ = np.linalg.lstsq(Ab, col, rcond=None)
resid = col - Ab @ coef

vmap = {w: i for i, w in enumerate(d["vocab"])}
own_ids = [vmap[w] for w in words if w in vmap]
if own_ids:
    own_contains = np.stack([np.isin(X, i).any(axis=1) for i in own_ids]).any(0)
    print(f"\n[OWN NAME] ids {own_ids}  pos_frac {own_contains.mean():.4f}  "
          f"n_pos {int(own_contains.sum())}/{len(X)}  "
          f"auroc(recomputed) {M._auroc(resid, own_contains):.4f}")

rng = np.random.default_rng(0)
arity = max(len(own_ids), 1)
cand_words = [w for w in d["vocab"] if w not in words]
aucs, posfracs = [], []
for _ in range(300):
    pick = rng.choice(len(cand_words), size=arity, replace=False)
    ids = [vmap[cand_words[j]] for j in pick]
    c = np.stack([np.isin(X, i).any(axis=1) for i in ids]).any(0)
    a = M._auroc(resid, c)
    if not np.isnan(a):
        aucs.append(float(a))
        posfracs.append(float(c.mean()))
aucs = np.array(aucs)
print(f"\n[CONTROL] same column, {len(aucs)} RANDOM {arity}-word sets:")
print(f"  AUROC  median {np.median(aucs):.4f}  90th {np.percentile(aucs,90):.4f}  "
      f"max {aucs.max():.4f}")
print(f"  frac of random sets reaching >= 0.55 : {(aucs >= 0.55).mean():.4f}")
print(f"  frac reaching >= 0.90 : {(aucs >= 0.90).mean():.4f}")
if own_ids:
    own_a = M._auroc(resid, own_contains)
    pct = (aucs < own_a).mean() * 100
    print(f"\n  OWN NAME auroc {own_a:.4f} sits at the {pct:.1f}th percentile "
          f"of random word-sets")
print("=" * 80)
