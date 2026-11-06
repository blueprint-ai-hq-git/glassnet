#!/bin/bash
# DECISIVE DIAGNOSTIC: can ANYTHING predict the merged corpus's val targets?
# On the merged corpus a from-scratch model drives train loss down (0.69) while val perplexity
# EXPLODES (880,707, worse than the 12,000-word uniform baseline) and val accuracy sits at chance.
# Two explanations remain: (a) the merged val windows/labels are misaligned, or (b) the model is
# simply failing. A bigram counts model trained on the TRAIN split settles it: it needs no training
# loop, so if the bigram also cannot beat chance on val, the VAL TARGETS are the problem -- and if
# the bigram does well on val, the val data is fine and the model is what failed.
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
W = 32
tr = cd.make_windows(ids, bounds, split, W, which=0)
va = cd.make_windows(ids, bounds, split, W, which=1)
Xtr, Ytr = tr[0], tr[1]
Xva, Yva = va[0], va[1]

# bigram counts from TRAIN tokens only
V = len(vocab)
train_tokens = np.concatenate([ids[bounds[j]:bounds[j + 1]] for j in range(len(bounds) - 1)
                              if split[j] == 0])
cnt = np.zeros((V, V), dtype=np.int32)
x, y = train_tokens[:-1], train_tokens[1:]
# accumulate in chunks to keep memory sane
for s in range(0, len(x), 2_000_000):
    np.add.at(cnt, (x[s:s + 2_000_000], y[s:s + 2_000_000]), 1)
row = cnt.sum(axis=1)
most_common = int(cnt.sum(axis=0).argmax())

def acc_of(targets, prev):
    """Accuracy of the bigram's top continuation per previous token, plus the constant baseline."""
    pred = cnt.argmax(axis=1)[prev] if len(prev) else np.zeros(0, dtype=np.int64)
    seen = row[prev] > 0
    pred = np.where(seen, pred, most_common)
    return float((pred == targets).mean()) if len(targets) else float("nan")

# for every window position, the 'previous token' is the token before the target
prev_tr = Xtr[:, -1]
prev_va = Xva[:, -1]
print(f"  vocab {V}   train windows {len(Xtr):,}   val windows {len(Xva):,}")
print(f"  bigram (fitted on train)     train-target acc {acc_of(Ytr, prev_tr)*100:6.2f}%   "
      f"val-target acc {acc_of(Yva, prev_va)*100:6.2f}%")
print(f"  constant most-frequent token train-target acc "
      f"{(Ytr == most_common).mean()*100:6.2f}%   val-target acc {(Yva == most_common).mean()*100:6.2f}%")
print(f"  uniform chance                       {(1/V)*100:6.2f}%")
# does the val target distribution match the train target distribution?
def ent(counts):
    p = counts / max(counts.sum(), 1)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())

e_tr = ent(np.bincount(Ytr, minlength=V).astype(np.float64))
e_va = ent(np.bincount(Yva, minlength=V).astype(np.float64))
print(f"  target token entropy: train {e_tr:.2f} nats   val {e_va:.2f} nats")
PY
done