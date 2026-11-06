#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 - <<'PY' 2>&1 | tail -10
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "lens/scripts")
import variants as VA
v = VA.build_default(device="cpu")
net, _ = v.models["talk"]
print("  talk model attrs: V", getattr(net, "V", None), "W", getattr(net, "W", None))
r = v.talk_sentence("Human: hello , how are you ?\n\nAssistant:", n_new=25, seed=0)
print("  talk_sentence:", r["reply"][:120])
r2 = v.talk_sentence("Human: what is your name ?\n\nAssistant:", n_new=25, seed=1)
print("  talk_sentence:", r2["reply"][:120])
PY
