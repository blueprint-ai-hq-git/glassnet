#!/bin/bash
# TOKENIZER GAP — measured, not assumed (gap 5). How much of the corpus does the word-level
# v12000 vocabulary actually represent? A high <unk>/hash rate would mean the model is partly
# learning placeholders, which would cap accuracy no matter how big or ordered it gets.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json
from collections import Counter
ids = np.load("data/chat/prepared/ids_v12000.npy")
b = np.load("data/chat/prepared/bounds_v12000.npy")
s = np.load("data/chat/prepared/split_v12000.npy")
vocab = json.load(open("data/chat/prepared/vocab_v12000.json"))
vmap = {w: i for i, w in enumerate(vocab)}
cnt = Counter(ids.tolist())
tot = len(ids)
unk_id = vmap.get("<unk>")
unk_rate = cnt.get(unk_id, 0) / tot if unk_id is not None else float("nan")
print(f"tokens {tot:,}  vocab {len(vocab):,}  <unk> id {unk_id}  <unk> share {unk_rate*100:.3f}%")
# how many token POSITIONS are singletons / rare (a proxy for vocabulary pressure)
rare = sum(v for k, v in cnt.items() if v == 1)
rare5 = sum(v for k, v in cnt.items() if v <= 5)
print(f"tokens appearing once: {rare:,} ({rare/tot*100:.2f}%)   <=5 times: {rare5:,} ({rare5/tot*100:.2f}%)")
print(f"distinct ids actually used: {len(cnt):,} of {len(vocab):,}")
# what the rarest 20 look like (junk? names? typos?)
rare_words = [w for w, i in vmap.items() if cnt.get(i, 0) in (0, 1)][:20]
print("rare/zero-count vocab entries (sample):", rare_words)
# is the vocab hash-bucketed? ('hash' entries would signal compression of OOV)
hashy = [w for w in vocab if isinstance(w, str) and w.startswith("h#")][:5]
print("hash-style vocab entries (sample):", hashy if hashy else "none -- plain word vocab")
# per-split: how much of the ASSISTANT text is rare (that is what the model must generate)
PY