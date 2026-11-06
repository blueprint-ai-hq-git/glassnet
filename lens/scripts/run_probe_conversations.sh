#!/bin/bash
# Decode a few REAL conversations from the talk corpus so the external-data converter reproduces
# the exact convention (role markers, separators, casing) instead of guessing.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json
ids = np.load("data/chat/prepared/ids_v12000.npy")
b = np.load("data/chat/prepared/bounds_v12000.npy")
v = json.load(open("data/chat/prepared/vocab_v12000.json"))
s = np.load("data/chat/prepared/split_v12000.npy")
for j in (0, 1, 2, 5):
    t = ids[b[j]:b[j+1]]
    print(f"--- conv[{j}] split={s[j]} len={len(t)}")
    print("   ", " ".join(v[i] for i in t[:48]))
    print("   ...", " ".join(v[i] for i in t[-20:]))
print("--- role marker counts across the corpus (first 200k tokens):")
from collections import Counter
c = Counter(ids[:200000].tolist())
print("    'human' (id", v.index("human"), ") =", c[v.index("human")], "  'assistant' (id",
      v.index("assistant"), ") =", c[v.index("assistant")])
PY