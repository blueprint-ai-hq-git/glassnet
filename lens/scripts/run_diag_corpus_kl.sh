#!/bin/bash
# WHY does a merged-corpus run score at chance while splitfix runs score ~60%?
# The window-cap bug was rejected as the cause. Vocabulary mismatch between corpora was found (so
# the 63.85% reference is NOT comparable to the matrix) -- this measures each corpus INTERNALLY:
# how far the training target distribution is from the validation target distribution, and how much
# of the val set comes from each source. If the merged corpus's val genuinely diverges from its
# train, that is the answer; if it matches, the model is the suspect.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
for CORPUS in prepared_ext prepared_splitfix; do
  echo "=== $CORPUS"
  CHAT_DATA_DIR="$PWD/data/chat/$CORPUS" .venv-gpu/bin/python3 -u - <<'PY'
import os, sys
import numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import chat_data as cd

d = cd.build()
ids, bounds, split, vocab = d["ids"], d["bounds"], d["split"], d["vocab"]
Xtr, Ytr = cd.make_windows(ids, bounds, split, 32, which=0)
Xva, Yva = cd.make_windows(ids, bounds, split, 32, which=1)
V = len(vocab)

def dist(y):
    return np.bincount(y, minlength=V).astype(np.float64) / len(y)
def kl(p, q):
    p, q = np.maximum(p, 1e-12), np.maximum(q, 1e-12)
    return float((p * np.log(p / q)).sum())

sel = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 1_500_000), replace=False)
dtr, dva = dist(Ytr[sel]), dist(Yva)
print(f"  targets: KL(train || val) {kl(dtr, dva):.4f} nats   "
      f"union support {int(((dtr > 0) | (dva > 0)).sum())} of {V}")
# per-source sanity: are val windows' contexts longer/shorter than train windows'?
print(f"  window length train {Xtr.shape[1]} val {Xva.shape[1]} (fixed by design)")
# how often is the target token one that is rare in TRAIN (i.e. effectively unlearnable)?
freq_tr = np.bincount(Ytr[sel], minlength=V)
rare = freq_tr < max(1, len(sel) // 1_000_000)
print(f"  val targets whose token occurred <1e-6 of train targets: "
      f"{float(rare[Yva].mean())*100:.2f}%")
# a trivially strong baseline: copy the previous token (repetition) -- tells us the val's own
# self-predictability under the same windowing
print(f"  'copy the previous token' baseline on val: {float((Xva[:, -1] == Yva).mean())*100:.2f}%")
print(f"  'copy the previous token' baseline on train: {float((Xtr[sel][:, -1] == Ytr[sel]).mean())*100:.2f}%")
PY
done