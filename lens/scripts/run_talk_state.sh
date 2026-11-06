#!/bin/bash
# Discovery: the talk trainer's CLI + the teacher ship's exact meta, so the BIGGER ladder is
# defined against the real baseline (same data, same decoder, bigger capacity).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
grep -n "add_argument" lens/scripts/train_chat.py | head -16
echo "=== teacher meta:"
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json
z = np.load("lens/runs/chat/t11_talk41_ep192.npz", allow_pickle=True)
m = json.loads(str(z["meta"][0]))
print({k: m.get(k) for k in ("d", "layers", "epochs", "vocab", "window", "params", "val_acc", "best")})
print("meta keys:", sorted(m.keys())[:16])
PY