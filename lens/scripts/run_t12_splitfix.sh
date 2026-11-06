#!/bin/bash
# T12-SPLITFIX-D512 — the correct leak fix: KEEP ALL 13.3M TRAINING TOKENS, exclude ONLY the
# val convs whose content also sits in train (the curated split proves a leak-free val is
# achievable; the dedupe experiment proved deleting training data was the wrong half of the fix:
# clean-trained 31.26% vs raw-trained 56.07% on the same clean val, -24.8 pts, FALSIFIED).
# ONE change vs the raw d512-full run: the val split is leak-free.
# Builds a prepared_splitfix corpus: full ids/bounds, split_tags: convs with a train twin keep
# their train tag ONLY if they were train; val-tagged convs with a train twin move out of val
# (re-tagged train). Then trains d512 (no attention) at 192 ep and evals on run_eval_clean.
# Bar (pre-registered, binary): > 56.07% assistant-masked on the leak-free val AND >= the raw
# run's trainer val_acc on the same split. If not, capacity+data optima already measured stand.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
set -e
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json, hashlib, os, shutil
src = "data/chat/prepared"
ids = np.load(f"{src}/ids_v12000.npy"); b = np.load(f"{src}/bounds_v12000.npy")
s = np.load(f"{src}/split_v12000.npy")
def wh(j):
    t = ids[b[j]:b[j+1]]
    return None if len(t) < 110 else hashlib.md5(t[10:110].tobytes()).hexdigest()
train_hashes = set()
for j in range(len(b) - 1):
    if s[j] == 0:
        h = wh(j)
        if h: train_hashes.add(h)
moved = 0
s2 = s.copy()
for j in range(len(b) - 1):
    if s[j] == 1 and wh(j) in train_hashes:
        s2[j] = 0; moved += 1          # leaked val conv stays in the corpus, leaves val
out = "data/chat/prepared_splitfix"
os.makedirs(out, exist_ok=True)
for f in ("ids_v12000.npy", "bounds_v12000.npy"):
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
np.save(f"{out}/split_v12000.npy", s2)
meta = json.load(open(f"{src}/meta_v12000.json"))
json.dump({**meta, "splitfix": True, "moved_out_of_val": moved,
           "note": "all tokens kept; val convs with a train twin re-tagged train"}, open(f"{out}/meta_v12000.json", "w"))
shutil.copy(f"{src}/vocab_v12000.json", f"{out}/vocab_v12000.json")
print(f"splitfix corpus ready: {moved} leaked val convs moved to train; tokens unchanged "
      f"({len(ids):,}); val now {(s2==1).sum()} convs")
PY
export CHAT_DATA_DIR="$PWD/data/chat/prepared_splitfix"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
  --d 512 --window 32 --epochs 192 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
  --out t12_splitfix_d512 2>&1 | tail -20
echo "=== leak-free val eval:"
bash lens/scripts/run_eval_clean.sh lens/runs/chat/t12_splitfix_d512_ep192.npz