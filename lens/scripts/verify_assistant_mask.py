"""verify_assistant_mask.py — prove the assistant-only mask marks the RIGHT tokens before training
on it. A mask that is silently empty (or silently everything) would train a model on nothing or on
the same thing as before, and both would look like a normal run.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_data                                          # noqa: E402

d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]

is_asst = np.zeros(len(ids), dtype=np.float32)
in_asst = False
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    toks = [vocab[i] for i in ids[a0:b0]]
    k = 0
    while k < len(toks):
        if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = True
            k += 2
            continue
        if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_asst = False
            k += 2
            continue
        if in_asst:
            is_asst[a0 + k] = 1.0
        k += 1

print("=" * 92)
print("ASSISTANT-ONLY MASK VERIFICATION")
print("=" * 92)
print(f"  corpus tokens            : {len(ids):,}")
print(f"  marked assistant         : {int(is_asst.sum()):,} ({is_asst.mean()*100:.1f}%)")
print(f"  marked human/other       : {int((1-is_asst).sum()):,} ({(1-is_asst).mean()*100:.1f}%)")

# show a real conversation with the marking, so the split is visible
print("\n  A REAL CONVERSATION, token by token (A = assistant, H = human):")
shown = 0
for j in range(len(bounds) - 1):
    if shown >= 1:
        break
    if split[j] != 0:
        continue
    a0, b0 = bounds[j], bounds[j + 1]
    seg = ids[a0:b0][:70]
    line = " ".join(f"{vocab[i]}" for i in seg)
    marks = " ".join("A" if is_asst[a0 + k] else "H" for k in range(len(seg)))
    print(f"    text : {line[:150]}")
    print(f"    mask : {marks[:150]}")
    shown += 1

# the mask must mark SOME and not ALL — either extreme means the marker detection is broken
frac = is_asst.mean()
ok = 0.05 < frac < 0.95
print("\n" + "-" * 92)
print(f"  assistant share {frac*100:.1f}% — must be strictly between 5% and 95%")
print(f"  {'PASS: the mask marks a real subset, not nothing and not everything.' if ok else 'FAIL: mask is degenerate — do NOT train with it.'}")
sys.exit(0 if ok else 1)
