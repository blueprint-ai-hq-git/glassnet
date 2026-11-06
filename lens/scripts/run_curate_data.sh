#!/bin/bash
# CURATOR v1 — ONE change: deduplicate conversations that share an identical 100-token window.
# Also quantifies the val leak (how many VAL convs have a twin in TRAIN) BEFORE and AFTER.
# Output: data/chat/prepared_clean/ids+bounds+split+meta (new dir; originals untouched).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json, hashlib, os
src = "data/chat/prepared"
ids = np.load(f"{src}/ids_v12000.npy")
b   = np.load(f"{src}/bounds_v12000.npy")
s   = np.load(f"{src}/split_v12000.npy")
meta = json.load(open(f"{src}/meta_v12000.json"))
n = len(b) - 1

def winhash(j):
    t = ids[b[j]:b[j+1]]
    if len(t) < 110:
        return None
    return hashlib.md5(t[10:110].tobytes()).hexdigest()

train_seen, val_leak = {}, 0
for j in range(n):
    h = winhash(j)
    if h is None:
        continue
    if s[j] == 0:
        train_seen[h] = j
    elif h in train_seen:
        val_leak += 1
print(f"val convs whose 100-token window ALSO appears in train: {val_leak} / {(s==1).sum()}")

# keep: first conv of each hash group; drop later twins (and val twins of train)
keep = np.ones(n, bool)
seen = {}
for j in range(n):
    h = winhash(j)
    if h is None:
        continue
    if h in seen:
        keep[j] = False
    else:
        seen[h] = j
ids_new, bounds_new, split_new = [], [0], []
for j in range(n):
    if keep[j]:
        ids_new.append(ids[b[j]:b[j+1]])
        split_new.append(int(s[j]))
        bounds_new.append(bounds_new[-1] + len(ids_new[-1]))
ids_new = np.concatenate(ids_new).astype(np.int32)
bounds_new = np.asarray(bounds_new, np.int64)
split_new = np.asarray(split_new)
n_new = len(bounds_new) - 1
out = "data/chat/prepared_clean"
os.makedirs(out, exist_ok=True)
np.save(f"{out}/ids_v12000.npy", ids_new)
np.save(f"{out}/bounds_v12000.npy", bounds_new)
np.save(f"{out}/split_v12000.npy", split_new)
json.dump({**meta, "n_conv": n_new, "curated": True,
           "note": "dedup: identical 100-token windows kept once"}, open(f"{out}/meta_v12000.json", "w"))
import shutil
for f in ("vocab_v12000.json",):
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
print(f"curated: {n} -> {n_new} convs  (dropped {n - n_new}); train {(split_new==0).sum()} val {(split_new==1).sum()}")
print(f"tokens: {len(ids)} -> {len(ids_new)}")
PY