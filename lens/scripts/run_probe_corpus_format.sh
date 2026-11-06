#!/bin/bash
# What shape is the talk corpus in? (special tokens, meta fields, and how the trainer decides
# which tokens are "assistant") -- the converter for external chat data must match this exactly.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json
v = json.load(open("data/chat/prepared/vocab_v12000.json"))
print("vocab size", len(v))
print("first 30:", v[:30])
print("special-looking:", [t for t in v if t.startswith("<") or t.startswith("##")][:12])
m = json.load(open("data/chat/prepared/meta_v12000.json"))
print("meta keys:", list(m.keys()))
print("scalars:", {k: x for k, x in m.items() if not isinstance(x, (list, dict))})
PY
echo "=== assistant masking / window building in chat_data.py:"
grep -nE "assistant|<u>|<a>|mask|role" lens/scripts/chat_data.py | head -14