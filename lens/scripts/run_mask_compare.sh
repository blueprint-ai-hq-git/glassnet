#!/bin/bash
# WHICH MASK IS RIGHT? Two implementations of "is this token assistant text?" exist:
#   (A) the project meter's walk (inside run_eval_clean.sh)  -> reports 2,227 assistant targets
#   (B) my cross_eval's walk                                  -> reports 3,724
# Same corpus, same val split. This prints both counts and the first positions where they disagree,
# with the surrounding words, so the difference is visible instead of argued about.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
CHAT_DATA_DIR="$PWD/data/chat/prepared_splitfix" .venv-gpu/bin/python3 -u - <<'PY'
import os, sys
import numpy as np
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import chat_data

d = chat_data.build()
ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
WIN = 32
Xva, Yva = chat_data.make_windows(ids, bounds, split, WIN, which=1)
print(f"  val windows {len(Xva):,}")

# (A) the project's walk, copied verbatim from run_eval_clean.sh
mA = np.zeros(len(ids), dtype=np.float32)
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    toks = [vocab[i] for i in ids[a0:b0]]
    in_a, k = False, 0
    while k < len(toks):
        if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_a, k = True, k + 2
            continue
        if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
            in_a, k = False, k + 2
            continue
        if in_a:
            mA[a0 + k] = 1.0
        k += 1

# (B) my cross_eval's walk, per conversation, flags kept in step with the tokens
mB = np.zeros(len(ids), dtype=np.float32)
for j in range(len(bounds) - 1):
    a0, b0 = bounds[j], bounds[j + 1]
    words = [vocab[i] for i in ids[a0:b0]]
    flags, in_a, k = [], False, 0
    while k < len(words):
        if words[k] == "assistant" and k + 1 < len(words) and words[k + 1] == ":":
            in_a = True; flags.extend([False, False]); k += 2; continue
        if words[k] == "human" and k + 1 < len(words) and words[k + 1] == ":":
            in_a = False; flags.extend([False, False]); k += 2; continue
        flags.append(in_a); k += 1
    for off, f in enumerate(flags):
        if f:
            mB[a0 + off] = 1.0

print(f"  tokens marked assistant:  project {int(mA.sum()):,}   mine {int(mB.sum()):,}")
print(f"  val targets marked:       project {int(mA[Yva].sum()):,}   mine {int(mB[Yva].sum()):,}")
diff = np.where(mA != mB)[0]
print(f"  positions where they disagree: {len(diff):,}")
shown = 0
for p in diff:
    if shown >= 5:
        break
    lo, hi = max(0, p - 8), min(len(ids), p + 4)
    ctx = " ".join(vocab[i] for i in ids[lo:hi])
    print(f"    at {p}: project={mA[p]:.0f} mine={mB[p]:.0f}   ...{ctx}...")
    shown += 1
PY