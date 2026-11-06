#!/bin/bash
# Baseline architecture: derive the talk teacher's exact shapes from its checkpoint (old format,
# pre-meta). The ladder must be built against THIS model, not against a guess.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np
z = np.load("lens/runs/chat/t11_talk41_ep192.npz")
n = 0
for k in sorted(z.files):
    a = z[k]
    n += a.size
    print(f"{k:<16} {str(a.shape):<14} {a.dtype}")
print("total params:", n)
PY
echo "=== talk audit entry:"
sed -n '/"talk"/,/}/p' lens/runs/strict_text_audit.json | head -9