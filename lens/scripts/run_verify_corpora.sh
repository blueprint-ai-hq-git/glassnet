#!/bin/bash
# INDEPENDENT verification of the three new corpora (never trust the builder's own log):
#   1. leak audit: val conversations whose 100-token window also occurs in train -> must be 0
#   2. vocab identity: all three must share the SAME 12,000-word list, otherwise "one common
#      yardstick" is a false claim and cross-corpus numbers would be meaningless
#   3. sanity: conversation counts, token counts, split sizes, no degenerate conversations
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import hashlib, json, os
import numpy as np

CORPORA = ["prepared_ext", "prepared_ext_chat", "prepared_ext_big"]
vocab_hashes = {}
for name in CORPORA:
    d = f"data/chat/{name}"
    if not os.path.isdir(d):
        print(f"  {name}: MISSING"); continue
    ids = np.load(f"{d}/ids_v12000.npy")
    b = np.load(f"{d}/bounds_v12000.npy")
    sp = np.load(f"{d}/split_v12000.npy")
    vocab = json.load(open(f"{d}/vocab_v12000.json"))
    vh = hashlib.md5(json.dumps(vocab).encode()).hexdigest()[:10]
    vocab_hashes[name] = vh
    train_h, moved = set(), 0
    for j in range(len(b) - 1):
        if sp[j] != 0:
            continue
        t = ids[b[j]:b[j + 1]]
        if len(t) >= 110:
            train_h.add(hashlib.md5(t[10:110].tobytes()).hexdigest())
    leaked = 0
    for j in range(len(b) - 1):
        if sp[j] != 1:
            continue
        t = ids[b[j]:b[j + 1]]
        if len(t) >= 110 and hashlib.md5(t[10:110].tobytes()).hexdigest() in train_h:
            leaked += 1
    short = int((np.diff(b) < 8).sum())
    print(f"  {name:<20} tokens {len(ids):>11,}  convs {len(b):>7,}  val {int((sp==1).sum()):>6,}  "
          f"LEAKED VAL WINDOWS {leaked}  short convs {short}  vocab md5 {vh}")
print()
uniq = set(vocab_hashes.values())
if len(uniq) == 1:
    print(f"  vocab identity: ALL SHARE THE SAME word list (md5 {list(uniq)[0]}) -> the merged val is "
          f"a valid common yardstick for every matrix run")
else:
    print(f"  vocab identity: MISMATCH across corpora ({vocab_hashes}) -> cross-corpus comparison "
          f"would be INVALID; matrix numbers must not be compared until this is fixed")
PY