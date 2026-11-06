#!/bin/bash
# PROOF OF THE WINDOW-CAP BUG (no training needed).
# Claim: a PREFIX cap on the merged corpus trained on the first data source(s) only, so training
# targets and validation targets came from different distributions -- which is why clean-val accuracy
# sat at chance while train loss fell.
# Method: build the real windows, take the first 1.5M (the old behaviour) and a uniform random 1.5M
# (the new behaviour), and compare each set's TARGET token distribution against the VALIDATION
# target distribution (KL divergence), plus the unknown-token rate. Same data, same code -- only the
# selection rule differs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
CHAT_DATA_DIR="$PWD/data/chat/prepared_ext" .venv-gpu/bin/python3 -u - <<'PY'
import json
import sys
import numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import chat_data as cd

CAP = 1_500_000
d = cd.build()
ids, bounds, split, vocab = d["ids"], d["bounds"], d["split"], d["vocab"]
Xtr, Ytr = cd.make_windows(ids, bounds, split, 32, which=0)
Xva, Yva = cd.make_windows(ids, bounds, split, 32, which=1)
V = len(vocab)
unk = vocab.index("<unk>") if "<unk>" in vocab else 0
print(f"  corpus prepared_ext: {len(Xtr):,} train windows, {len(Xva):,} val windows, cap {CAP:,}")

def dist(y):
    return np.bincount(y, minlength=V).astype(np.float64) / len(y)

def kl(p, q):
    p, q = np.maximum(p, 1e-12), np.maximum(q, 1e-12)
    return float((p * np.log(p / q)).sum())

dv = dist(Yva)
rows = []
for name, sel in (("PREFIX (old behaviour)", np.arange(min(CAP, len(Xtr)))),
                  ("UNIFORM (fixed)", np.random.default_rng(0).choice(len(Xtr), CAP, replace=False))):
    y = Ytr[sel]
    dp = dist(y)
    rows.append((name, float((y == unk).mean()), kl(dp, dv), float((sel > 0.75 * len(Xtr)).mean())))
    print(f"  {name:<22} unk-rate {rows[-1][1]*100:5.2f}%   KL(train targets || val targets) "
          f"{rows[-1][2]:.4f}   windows from the last quarter of the corpus {rows[-1][3]*100:5.1f}%")
print(f"  val reference: unk-rate {float((Yva == unk).mean())*100:.2f}%")
print()
print("  VERDICT:", "the two selections really do differ in distribution (bug confirmed)"
      if rows[0][2] > 2 * rows[1][2] else "no meaningful difference between the selections")
PY