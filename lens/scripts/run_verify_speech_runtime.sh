#!/bin/bash
# Retrain (ship now carries mu/sd), then measure accuracy THROUGH THE RUNTIME PATH
# (lens.speech.load + WordNet.predict) -- an independent re-measurement of the shipped artifact,
# not the trainer's own report.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u lens/scripts/train_speech.py > /tmp/train_speech2.log 2>&1
echo "RC=$?"; grep -E "shipped pruned|after pruning|epoch  60" /tmp/train_speech2.log
.venv-gpu/bin/python3 -u - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
import lens.speech as S
ship = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/speech/speech_readable_s0.npz"
net = S.load(ship)
d = np.load("/run/media/yahya/HARD DISK 4/GlassNet/data/speech/speech_v1.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split"]
va = split == 1
h, r, lg = net.forward(X[va])
acc = float((lg.argmax(axis=1) == y[va]).mean())
base = float(max(np.bincount(y[~va])) / (~va).sum())
print(f"  RUNTIME-PATH accuracy on the official val split: {acc*100:.2f}%  "
      f"(majority baseline {base*100:.2f}%)  n={va.sum():,}")
# one worked example, end to end, in the runtime's own explanation format
import wave
wav = "/run/media/yahya/HARD DISK 4/GlassNet/data/external/speech_commands_raw/extracted/yes/00f0204f_nohash_0.wav"
x = S.read_wav(wav)
print("  example clip yes/00f0204f_nohash_0.wav:")
print("   ", S.explain(net, x).replace("\n", "\n    "))
print(f"  ship concepts: {net.concepts}")
print(f"  connections per rule: {float((net.W2 != 0).sum())/net.W2.shape[1]:.2f}")
PY