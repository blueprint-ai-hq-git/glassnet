#!/bin/bash
# SANITY CHECK the curated corpus VS the original: decode the same conversation index from both
# and print them side by side. A 25-point collapse is either a real data effect or corrupted
# arrays; this tells us which, before any conclusion is written down.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json
def peek(dirname, js):
    ids = np.load(f"data/chat/{dirname}/ids_v12000.npy")
    b = np.load(f"data/chat/{dirname}/bounds_v12000.npy")
    vocab = json.load(open(f"data/chat/prepared/vocab_v12000.json"))
    out = []
    for j in range(js, js + 2):
        toks = [vocab[i] for i in ids[b[j]:b[j+1]]]
        out.append((j, len(toks), " ".join(toks[:40])))
    return out, len(ids), len(b) - 1
for name in ("prepared", "prepared_clean"):
    rows, ntok, nconv = peek(name, 100)
    print(f"--- {name}: {nconv} convs, {ntok:,} tokens")
    for j, ln, txt in rows:
        print(f"    conv[{j}] len {ln}: {txt[:150]}")
PY
echo "=== chain status:"
ps aux | grep -E "run_chain_attn|train_chat.py" | grep -v grep | awk '{print $2, $3"%CPU", substr($0, index($0,$11), 90)}' | head -3