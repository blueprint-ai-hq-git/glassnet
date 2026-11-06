#!/bin/bash
# DATA PROFILER — what is actually IN the talk corpus? Before curating anything, measure it:
#   * length distribution (degenerate 1-token convs? absurd 500-token walls of text?)
#   * degenerate loops (same token repeated 8+ times: "ok ok ok ok...", "ha ha ha...")
#   * near-duplicate conversations (identical or ~identical window content inflates training
#     and leaks across the split)
#   * vocabulary coverage per conversation (junk/symbol soup)
# Every number below is printed, nothing is dropped yet — the drop list comes after this.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import numpy as np, json
from collections import Counter
ids = np.load("data/chat/prepared/ids_v12000.npy")
b   = np.load("data/chat/prepared/bounds_v12000.npy")
s   = np.load("data/chat/prepared/split_v12000.npy")
vocab = json.load(open("data/chat/prepared/vocab_v12000.json"))
n = len(b) - 1
lens = np.diff(b)
print(f"conversations: {n}   tokens: {len(ids)}   train convs {(s==0).sum()}   val {(s==1).sum()}")
print(f"conv length: min {lens.min()}  p5 {np.percentile(lens,5):.0f}  median {np.median(lens):.0f}  p95 {np.percentile(lens,95):.0f}  max {lens.max()}")
print(f"single-token convs: {(lens<=2).sum()}   > 300 tokens: {(lens>300).sum()}")

def signal(x):
    seg = x[:800]  # window on long convs
    counts = Counter(seg.tolist())
    top, ntop = counts.most_common(1)[0]
    return top, ntop / max(len(seg), 1)

loop = dup = 0
loop_ids = []
skip = set()
for j in range(n):
    t = ids[b[j]:b[j+1]]
    if len(t) < 4:
        continue
    top, frac = signal(t)
    if frac > 0.4:                       # same token >40% of the conversation
        loop += 1
        if len(loop_ids) < 5:
            loop_ids.append(" ".join(str(vocab[i]) for i in t[:24]))
    if len(t) > 100:
        dup += 1
print(f"\ndegenerate loops (one token >40% of conv): {loop} / {n}")
for x in loop_ids:
    print("   LOOP EXAMPLE:", x[:90])
# near-duplicate window content: hash a chunk of each long conv
import hashlib
chunks = {}
dd = 0
for j in range(n):
    t = ids[b[j]:b[j+1]]
    if len(t) < 40:
        continue
    h = hashlib.md5(t[10:110].tobytes()).hexdigest()
    chunks.setdefault(h, []).append(j)
for h, js in chunks.items():
    if len(js) > 1:
        dd += len(js) - 1
print(f"\nconvs sharing an identical 100-token window: {dd} extra duplicates (train/val leak risk)")
print("CURATION CANDIDATES: drop loops + single-token + >300-token outliers, dedupe window-shared convs")
PY