#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 - <<'PY'
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "lens/scripts")
import lens.tensor as T; T.set_device("cpu")
import train_seetalk as ts
for f in ("seetalk_small2", "seetalk_teacher_d128", "seetalk_big"):
    p = f"lens/runs/seetalk/{f}.npz"
    try:
        net, meta = ts.SeeTalkNet.load(p)
        print(f"  {f}: LOADS OK, K={len(net.concepts)}, vocab {len(meta['vocab'])} words")
    except Exception as e:
        print(f"  {f}: LOAD FAILED {repr(e)[:90]}")
PY
