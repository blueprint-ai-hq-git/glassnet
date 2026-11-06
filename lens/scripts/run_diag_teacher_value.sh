#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u - <<'PY' 2>&1 | tail -30
import os, sys, json
import numpy as np
PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ); sys.path.insert(0, PROJ + "/lens/scripts")
import lens.tensor as T
T.set_device("cpu")
import train_seetalk as ts

data = ts.build_caption_data(n=10000, seed=0)
teacher, meta = ts.SeeTalkNet.load(PROJ + "/lens/runs/seetalk/seetalk_teacher_d128.npz")
s_vocab, t_vocab = data["vocab"], meta["vocab"]
print(f"  vocab order identical: {t_vocab == s_vocab}  (set identical: {sorted(t_vocab)==sorted(s_vocab)})")
remap = np.array([t_vocab.index(w) for w in s_vocab])

rng0 = np.random.default_rng(7)
order = rng0.permutation(len(data["Xtr"]))
rows10 = np.sort(order[:max(1, int(0.1*len(order)))])
lg = teacher.forward_arrays(data["Xtr"][rows10], data["Str"][rows10])["logits"][:, remap]
q = np.exp(lg - lg.max(axis=1, keepdims=True)); q /= q.sum(axis=1, keepdims=True)
Y = data["Ytr"][rows10]
top1 = q.max(axis=1); ent = -(q*np.log(q+1e-12)).sum(axis=1)
truep = q[np.arange(len(Y)), Y]
argmax = q.argmax(axis=1)
agree = (argmax == Y).mean()
print(f"  teacher on the 10% STUDENT rows ({len(Y):,} rows):")
print(f"    mean top1 {top1.mean():.3f}   mean entropy {ent.mean():.3f} nats "
      f"(uniform log{len(s_vocab)} = {np.log(len(s_vocab)):.3f})")
print(f"    argmax AGREES with the real label on {agree*100:.1f}% of rows")
print(f"    probability mass on the TRUE word: mean {truep.mean():.3f}, "
      f"median {np.median(truep):.3f}")
# the key comparison: teacher soft mass on truth vs the one-hot real label
# one-hot = 1.0 on truth by construction. The teacher can only ADD through the distribution on
# rows where its argmax is WRONG: how much of the truth does it still spread there?
wrong = argmax != Y
print(f"    on its {wrong.mean()*100:.1f}% WRONG rows: mean prob on the true word "
      f"{truep[wrong].mean():.3f}")
# and the student-pool truth itself: is the label even learnable from 10%? A is 66.5 val
print()
print("  INTERPRETATION (numbers, not vibes):")
print(f"    A one-hot real label puts 1.000 on the truth on EVERY row.")
print(f"    The teacher puts {truep.mean():.3f} on average — LESS mass on the truth than the real")
print(f"    label itself. The 10% real labels ARE the stronger signal: on this task the")
print(f"    student's targets are deterministic functions of the image (the class noun), so")
print(f"    ground truth strictly dominates any teacher estimate of it. Distillation helps when")
print(f"    labels are NOISY, ambiguous, or expensive; none of the three is true here.")
PY
